import torch
from hutils import asymmetric_gcn_norm, ToHetero, ToHomo
import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB, DBLP, HGBDataset
from eval_utils import eval_GNN
from ugc_f import ugc
from torch_geometric.data import HeteroData, Data
import torch.nn.functional as F
from heteroModels import HeteroSGC, HeteroGCN, HeteroGCN2
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from ConsistentHashing import *
import UGC_binwidth_finder
import utils
from fgc_f import fgc
from heatmap import plotHeatmap
import os
import pandas as pd
import load_datasets
from tabulate import tabulate

import sys
sys.path.append("/home/mohit/projects/heteroGC/coarGC/SCAL/Scal/GCN")
from SCAL.Scal.GCN import train as scal

model_dict = {'HeteroSGC': HeteroSGC, 'HeteroGCN': HeteroGCN, 'HeteroGCN2': HeteroGCN2}

model_architecture = {'hidden_channels':64,
                      'num_layers':3}

model_train = {'epochs':1000,
                'lr':0.01,
                'weight_decay':0.0005}

coar_device = 'cpu'

def run_hetero(dataset_name, coar_method, coarsening_ratio, model_name, num_evals, device):
    print("Hi")
    if dataset_name == 'imdb':
        dataset = load_datasets.load_imdb()
    elif dataset_name == 'hdblp':
        dataset = load_datasets.load_dblp()
    elif dataset_name == 'acm':
        dataset = load_datasets.load_acm()
    
    headers1 = ["Dataset", "Coarsening Method", "Coarsening ratio", "GNN Model"]
    table1 = [[dataset_name, coar_method, coarsening_ratio, model_name]]

    headers2 = ["Node type", "Edge types", "Target Node"]
    table2 = [[f"{node_type}" for node_type in dataset.node_types]]
    
    if coar_method == 'base': # No coarsening direct HGNN evaluation
        x_syn_dict, adj_t_syn_dict = {}, {}
        for k, v in dataset.x_dict.items():
            x_syn_dict[k] = dataset.x_dict[k].to(device)
        for k, v in dataset.adj_t_dict.items():
            adj_t_syn_dict[k] = dataset.adj_t_dict[k].to(device)
        
        y_syn, mask_syn = dataset[dataset.target_node].y, dataset[dataset.target_node].train_mask
    
    elif coar_method == 'cugc':
        C_dict = {}
        zero_list_dict ={}
        for node_type in dataset.node_types:
            if hasattr(dataset[node_type], 'x'):
                hasher = ConsistentHashing(input_dim=dataset[node_type].x.shape[1], proj_dim=1000)
                bin_width, _ = UGC_binwidth_finder.find_Binwidth(dataset[node_type], 0.95 if (dataset_name == 'hdblp' and node_type == 'conference') or (dataset_name == 'acm' and node_type == 'subject') else 1-coarsening_ratio)

                Bin_values = hasher.UGC_hashed_values(dataset[node_type], function='dot')
                summary_dict = hasher.UGC_partition([bin_width], Bin_values)
            
                supernode_dict = summary_dict[bin_width]
                
                rr = 1 - len(supernode_dict.keys())/dataset[node_type].x.shape[0]
                reduced_percentage = rr
                
                print(f'{node_type} reduced by: {reduced_percentage} percent. \n Now we have {len(supernode_dict.keys())} supernodes; Starting nodes were: {dataset[node_type].x.shape[0]}')
        
                # Send the data back to cpu
                dataset[node_type].to(coar_device)

                C = torch.zeros(dataset[node_type].x.shape[0] , len(supernode_dict.keys()))
                zero_list = torch.ones(len(supernode_dict.keys()), dtype=torch.bool)

                for super_idx, node_list in enumerate(supernode_dict.values()):
                    for node in node_list:
                        C[node][super_idx] = 1
                        if node_type == dataset.target_node:
                            zero_list[super_idx] = zero_list[super_idx] and (not (dataset[dataset.target_node].train_mask)[node])
                
                C_diag = torch.sum(C, dim=0)
                P = torch.sparse.mm(C,(torch.diag(torch.pow(C_diag, -1/2))))
                P = P.to_sparse().to_dense()

                C_dict[node_type], zero_list_dict[node_type] = F.normalize(P, p=1.0, dim=1), zero_list
        
        # Count edges and form the adj of overall coarsened 
        new_data = HeteroData()

        clusters = {}
        for key in C_dict.keys():
            clusters[key] = torch.argmax(C_dict[key], dim=1)

        for edge_type in dataset.edge_types:
            src_type, rel, dst_type = edge_type
            original_edges = dataset[edge_type].edge_index
            
            src_map = clusters[src_type]
            dst_map = clusters[dst_type]
            
            super_src = src_map[original_edges[0]]
            super_dst = dst_map[original_edges[1]]
            
            super_edges = torch.stack([super_src, super_dst])
            unique_edges, counts = torch.unique(
                super_edges, 
                dim=1, 
                return_counts=True
            )

            new_data[src_type, rel, dst_type].edge_index = unique_edges
            new_data[src_type, rel, dst_type].edge_weight = counts.float()


        for node_type in dataset.node_types:
            if hasattr(dataset[node_type], 'x'):
                new_data[node_type].x = C_dict[node_type].T @ dataset[node_type].x
            if node_type in dataset.target_node:
                myY = F.one_hot(dataset[node_type].y)
                myY[~dataset[node_type].train_mask] = torch.tensor([0 for _ in range(myY.shape[1])], dtype=myY.dtype)
                interm = C_dict[node_type].T @ myY.to(torch.float32)
                new_data[node_type].y = torch.argmax(interm, dim=1)

        ndata = new_data
        x_syn_dict, adj_t_syn_dict = {}, {}
        for node_type in dataset.node_types:
            x_syn_dict[node_type] = ndata[node_type].x.to(device)

        for edge_type in dataset.edge_types:
            adjmat = torch.zeros(ndata[edge_type[2]].x.shape[0], ndata[edge_type[0]].x.shape[0])
            for index, (src, dst) in enumerate(ndata[edge_type].edge_index.T):
                adjmat[dst, src]=ndata[edge_type].edge_weight[index]
            adj_t_syn_dict[edge_type] = asymmetric_gcn_norm(adjmat).to(device)

        y_syn = ndata[dataset.target_node].y.to(device)
        mask_syn = ~zero_list_dict[dataset.target_node].to(device)
        print(mask_syn.count_nonzero())
    
    else:
        homDataset = ToHomo(dataset)
        if coar_method == 'fugc':
            hasher = ConsistentHashing(input_dim=homDataset.x.shape[1], proj_dim=1000)
            bin_width, _ = UGC_binwidth_finder.find_Binwidth(homDataset, 1-coarsening_ratio)

            Bin_values = hasher.UGC_hashed_values(homDataset, function='dot')
            summary_dict = hasher.UGC_partition([bin_width], Bin_values)
        
            supernode_dict = summary_dict[bin_width]
            
            rr = 1 - len(supernode_dict.keys())/homDataset.x.shape[0]
            reduced_percentage = rr
            
            #supernode_dict.keys()--> new supernode formed in coarsened graph
            print(f'Graph reduced by: {reduced_percentage} percent. \n Now we have {len(supernode_dict.keys())} supernodes; Starting nodes were: {homDataset.x.shape[0]}')

            #Send back to cpu
            homDataset.to(coar_device)

            C = torch.zeros(homDataset.x.shape[0] , len(supernode_dict.keys()))
            zero_list = torch.ones(len(supernode_dict.keys()), dtype=torch.bool)

            for super_idx, node_list in enumerate(supernode_dict.values()):
                for node in node_list:
                    C[node][super_idx] = 1
                    zero_list[super_idx] = zero_list[super_idx] and (not (homDataset.train_mask)[node])
            
            C_diag = torch.sum(C, dim=0)
            P = torch.sparse.mm(C,(torch.diag(torch.pow(C_diag, -1/2))))
            P = P.to_sparse().to_dense()

            P = P.to(homDataset.edge_index.device)
            adj_c = P.T @ to_dense_adj(homDataset.edge_index)[0] @ P
            data_coarsen = Data(edge_index=dense_to_sparse(adj_c)[0].to(torch.int64), train_mask=~zero_list)
        
        else:
            print("Hello ji")
            data_coarsen, P = scal.scal_coarsen([homDataset], coarsening_ratio, coar_method, dataset_name)
        
        nodetype_y = torch.cat([i*torch.ones(dataset[node_type].x.shape[0]) for i, node_type in enumerate(dataset.node_types)])
        type_map = {i: node_type for i, node_type in enumerate(dataset.node_types)}
        type_mapinv = {v:k for k, v in type_map.items()}
        newData = ToHetero(dataset, homDataset, data_coarsen, P, nodetype_y, type_map, type_mapinv, dataset.num_classes, dataset.target_node, coar_method)

        x_syn_dict, adj_t_syn_dict = {}, {}
        for node_type in dataset.node_types:
            x_syn_dict[node_type] = newData[node_type].x.to(device)

        for edge_type in dataset.edge_types:
            adjmat = torch.zeros(newData[edge_type[2]].x.shape[0], newData[edge_type[0]].x.shape[0])
            for index, (src, dst) in enumerate(newData[edge_type].edge_index.T):
                adjmat[dst, src]=1
            adj_t_syn_dict[edge_type] = asymmetric_gcn_norm(adjmat).to(device)
        
        y_syn = newData[dataset.target_node].y.to(device)
        mask_syn = newData[dataset.target_node].train_mask.to(device)
        print(mask_syn.count_nonzero())
    
    for k, v in dataset.x_dict.items():
        dataset.x_dict[k] = dataset.x_dict[k].to(device)
    for k, v in dataset.adj_t_dict.items():
        dataset.adj_t_dict[k] = dataset.adj_t_dict[k].to(device)
    
    # Node Classification task
    model_name = model_dict[model_name]
    acc, f1_micro, f1_macro = eval_GNN(num_evalue=num_evals, 
                                           x_dict=dataset.x_dict, 
                                           adj_t_dict=dataset.adj_t_dict, 
                                           node_types=dataset.node_types, 
                                           edge_types=dataset.edge_types, 
                                           y=dataset[dataset.target_node].y.to(device), 
                                           target_node_type=dataset.target_node, 
                                           num_classes=dataset.num_classes, 
                                           train_mask=dataset[dataset.target_node].train_mask.to(device), 
                                           test_mask=dataset[dataset.target_node].test_mask.to(device), 
                                           val_mask=dataset[dataset.target_node].val_mask.to(device), 
                                           x_syn_dict=x_syn_dict, 
                                           adj_t_syn_dict=adj_t_syn_dict, 
                                           y_syn=y_syn.to(device), 
                                           mask_syn=mask_syn.to(device), 
                                           model_name=model_name, model_architecture=model_architecture, model_train=model_train)
    print(acc)