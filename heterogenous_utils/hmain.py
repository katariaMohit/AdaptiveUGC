## This could be the potential main file
import torch
from hutils import asymmetric_gcn_norm, ToHetero
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

import sys
sys.path.append("/home/mohit/projects/heteroGC/coarGC/SCAL/Scal/GCN")
from SCAL.Scal.GCN import train as scal

model_dict = {'HeteroSGC': HeteroSGC, 'HeteroGCN': HeteroGCN, 'HeteroGCN2': HeteroGCN2}

model_architecture = {'hidden_channels':64,
                      'num_layers':3}

model_train = {'epochs':1000,
                'lr':0.01,
                'weight_decay':0.0005}

def update_table_accuracy(method, dataset, acc, model, wcoar=False, path="results_table_accuracy_heterogenous.csv"):
    row = {"dataset": dataset, "method": method, "acc": acc, "model": model, "base_dataset": wcoar}

    # Check if file exists and is not empty
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    df.to_csv(path, index=False)
    print(f"Results saved: {method} on {dataset} with {model}")

def run_imdb(coar_method, coarsening_ratio, mypath, num_evals, model_name):
    model_name = model_dict[model_name]
    dataset = IMDB(root=mypath, transform=T.ToSparseTensor(remove_edge_index=False))
    node_types, edge_types = dataset[0].metadata()
    target_node = 'movie'
    num_classes = 3
    mydata = utils.rand_train_test_idx(dataset[0][target_node], 0.6, 0.2)
    x_dict, adj_t_dict= {}, {}
    for node_type in node_types:
        x_dict[node_type] = dataset[0][node_type].x.to('cuda:1')
    for edge_type in edge_types:
        adj_t_dict[edge_type] = asymmetric_gcn_norm(dataset[0][edge_type].adj_t).to('cuda:1')
    y = dataset[0][target_node].y.to('cuda:1')
    train_mask = mydata.train_mask.to('cuda:1')
    val_mask =  mydata.val_mask.to('cuda:1')
    test_mask = mydata.test_mask.to('cuda:1')


    if coar_method == 'base':
        acc, f1_micro, f1_macro = eval_GNN(num_evals, x_dict, adj_t_dict, node_types, edge_types, y, target_node, num_classes, train_mask, test_mask, val_mask, x_dict, adj_t_dict, y, train_mask, model_name, model_architecture, model_train)
        print(acc)
    elif coar_method in ['cugc', 'caugc']:
        # myaccuracy = []
        # for coarsening_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        proportions = {'movie': 0.57, 'director': 0.23, 'actor': 0.34}
        prop2 = {'movie': 0.00155, 'director': 0.0104, 'actor': 0.00245}
        C_dict = {}
        zero_list_dict ={}
        for node_type in node_types:
            if hasattr(dataset[0][node_type], 'x'):
                if coar_method == 'cugc':
                    # C_dict[node_type], zero_list_dict[node_type] = ugc(dataset[0][node_type].x, prop2[node_type], 1000, train_mask if node_type == 'movie' else None)
                    hasher = ConsistentHashing(input_dim=dataset[0][node_type].x.shape[1], proj_dim=1000)
                    bin_width, _ = UGC_binwidth_finder.find_Binwidth(dataset[0][node_type], 1-coarsening_ratio)

                    Bin_values = hasher.UGC_hashed_values(dataset[0][node_type].to('cuda:1'), function='dot')
                    summary_dict = hasher.UGC_partition([bin_width], Bin_values)
                
                    supernode_dict = summary_dict[bin_width]
                    
                    rr = 1 - len(supernode_dict.keys())/dataset[0][node_type].x.shape[0]
                    reduced_percentage = rr
                    
                    #supernode_dict.keys()--> new supernode formed in coarsened graph
                    print(f'Graph reduced by: {reduced_percentage} percent. \n Now we have {len(supernode_dict.keys())} supernodes; Starting nodes were: {dataset[0][node_type].x.shape[0]}')
            

                    C = torch.zeros(dataset[0][node_type].x.shape[0] , len(supernode_dict.keys()))
                    zero_list = torch.ones(len(supernode_dict.keys()), dtype=torch.bool)

                    for super_idx, node_list in enumerate(supernode_dict.values()):
                        for node in node_list:
                            C[node][super_idx] = 1
                            if node_type == 'movie':
                                zero_list[super_idx] = zero_list[super_idx] and (not (train_mask)[node])
                    
                    C_diag = torch.sum(C, dim=0)
                    P = torch.sparse.mm(C,(torch.diag(torch.pow(C_diag, -1/2))))
                    P = P.to_sparse().to_dense().to('cuda:1')

                    # P, zero_list = ugc(X, 0.00165, 1000, train_mask)
                    C_dict[node_type], zero_list_dict[node_type] = P, zero_list.to('cuda:1')
                else:
                    hasher = ConsistentHashing(input_dim=dataset[0][node_type].x.shape[1], proj_dim=1000)
                    if node_type == 'movie':
                        print('hi')
                        temp_data = Data(x=dataset[0]['movie'].x, y=dataset[0]['movie'].y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
                    else:
                        temp_data = dataset[0][node_type]
                    supernode_dict = hasher.forward(temp_data)
                    supernode_dict, _, ordered_mean_values = hasher.rank_and_sort_supernodes(supernode_dict, temp_data.x)
                    num_supernodes = int((proportions[node_type])*temp_data.x.shape[0])
                    print(num_supernodes)
                    P, supernode_dict, zero_list = hasher.coarsen_ring_parallel_gpu(supernode_dict, temp_data, num_supernodes)
                    C_dict[node_type], zero_list_dict[node_type] = P, zero_list
                    print((~zero_list_dict['movie']).count_nonzero())
    
        new_data = HeteroData()

        clusters = {}
        for key in C_dict.keys():
            clusters[key] = torch.argmax(C_dict[key], dim=1)

        # Process each edge type
        for edge_type in edge_types:
            src_type, rel, dst_type = edge_type
            original_edges = dataset[0][edge_type].edge_index
            
            # Get cluster mappings
            src_map = clusters[src_type]
            dst_map = clusters[dst_type]
            
            # Convert original nodes to supernodes
            super_src = src_map[original_edges[0]]
            super_dst = dst_map[original_edges[1]]
            
            # Create and count superedges
            super_edges = torch.stack([super_src, super_dst])
            unique_edges, counts = torch.unique(
                super_edges, 
                dim=1, 
                return_counts=True
            )
            # Store in new graph
            new_data[src_type, rel, dst_type].edge_index = unique_edges
            new_data[src_type, rel, dst_type].edge_weight = counts.float()


        for node_type in node_types:
            if hasattr(dataset[0][node_type], 'x'):
                new_data[node_type].x = C_dict[node_type].T.to('cuda:1') @ dataset[0][node_type].x.to('cuda:1')
            if node_type in target_node:
                myY = F.one_hot(dataset[0][node_type].y).to('cuda:1')
                myY[~train_mask] = torch.tensor([0 for _ in range(myY.shape[1])], dtype=myY.dtype).to('cuda:1')
                interm = (C_dict[node_type].T @ myY.to(torch.float32))
                new_data[node_type].y = torch.argmax(interm,dim=1)

        ndata = new_data
        x_syn_dict, adj_t_syn_dict = {}, {}
        for node_type in node_types:
            x_syn_dict[node_type] = ndata[node_type].x.to('cuda:1')

        ndata = T.ToSparseTensor(remove_edge_index=False)(ndata)
        for edge_type in edge_types:
            # adjmat = torch.zeros(ndata[edge_type[2]].x.shape[0], ndata[edge_type[0]].x.shape[0])
            # for index, (src, dst) in enumerate(ndata[edge_type].edge_index.T):
            #     adjmat[dst, src]=ndata[edge_type].edge_weight[index]
            # print(adjmat[:15, :15])
            adj_t_syn_dict[edge_type] = asymmetric_gcn_norm(ndata[edge_type].adj_t).to('cuda:1')

        y_syn = ndata['movie'].y.to('cuda:1')
        mask_syn = ~zero_list_dict['movie'].to('cuda:1')
        print(mask_syn.count_nonzero())

        ## Heat map plots
        # heatY = torch.cat((torch.zeros(4278), torch.ones(2081), 2*torch.ones(5257)))
        # heatC = torch.zeros(11616, ndata['movie'].x.shape[0]+ndata['director'].x.shape[0]+ndata['actor'].x.shape[0])
        # prev, curr=0, 0
        # prevy, curry=0, 0
        # for node_type in node_types:
        #     curr += dataset[0][node_type].x.shape[0]
        #     curry += ndata[node_type].x.shape[0]
        #     heatC[prev:curr, prevy:curry] = C_dict[node_type]
        #     prev += dataset[0][node_type].x.shape[0]
        #     prevy += ndata[node_type].x.shape[0]
        # plotHeatmap(heatC, heatY, filename=f"imdb_{coar_method}.png")

        for model in ['HeteroSGC', 'HeteroGCN', 'HeteroGCN2']:
        # for model in ['HeteroGCN2']:
            model_name = model_dict[model]
            acc, f1_micro, f1_macro = eval_GNN(num_evals, x_dict, adj_t_dict, node_types, edge_types, y, target_node, num_classes, train_mask, test_mask, val_mask, x_syn_dict, adj_t_syn_dict, y_syn, mask_syn, model_name, model_architecture, model_train)
            print(f"Using {model}, accuracy is {acc[0]*100}%")
            update_table_accuracy(coar_method, 'IMDB', acc[0]*100, model_name)
    #         myaccuracy.append(acc[0])
    # print(myaccuracy)
    elif coar_method in ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron', 'fugc', 'faugc', 'fgc', 'lagc']:
        dataset = IMDB(root=mypath)
        X = torch.cat((dataset[0]['movie'].x, dataset[0]['director'].x, dataset[0]['actor'].x), dim=0)
        y = torch.cat((dataset[0]['movie'].y, 3*torch.ones(2081,), 4*torch.ones(5257,)), dim=0)
        y = y.to(torch.int64)
        edge1 = torch.cat((torch.zeros(4278, 1), torch.ones(4278,1)*4278), dim=1) + dataset[0]['movie', 'to', 'director'].edge_index.T
        edge2 = torch.cat((torch.zeros(12828, 1), torch.ones(12828,1)*(4278+2081)), dim=1) + dataset[0]['movie', 'to', 'actor'].edge_index.T
        edge3 = torch.cat((torch.ones(4278,1)*4278, torch.zeros(4278, 1)), dim=1) + dataset[0]['director', 'to', 'movie'].edge_index.T
        edge4 = torch.cat((torch.ones(12828,1)*(4278+2081), torch.zeros(12828, 1)), dim=1) + dataset[0]['actor', 'to', 'movie'].edge_index.T
        edge = torch.cat((edge1, edge2, edge3, edge4), dim=0)
        mydata = utils.rand_train_test_idx(dataset[0]['movie'], 0.6, 0.2)
        train_mask=torch.cat((mydata.train_mask, torch.ones(11616-4278, dtype=bool))).to('cuda:1')
        test_mask=torch.cat((mydata.test_mask, torch.ones(11616-4278, dtype=bool))).to('cuda:1')
        val_mask=torch.cat((mydata.val_mask, torch.ones(11616-4278, dtype=bool))).to('cuda:1')

        dataset1 = [Data(x=X, y=y, edge_index=edge.T.to(torch.int64), train_mask=train_mask.to('cpu'), test_mask=test_mask.to('cpu'), val_mask=val_mask.to('cpu'))]

        # myratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        # myaccuracy = []
        # for coarsening_ratio in myratios:
        if coar_method == 'fugc':
            # P, zero_list = ugc(X, 0.00075, 1000, train_mask)
            hasher = ConsistentHashing(input_dim=dataset1[0].x.shape[1], proj_dim=1000)
            bin_width, _ = UGC_binwidth_finder.find_Binwidth(dataset1[0], 1-coarsening_ratio)

            Bin_values = hasher.UGC_hashed_values(dataset1[0], function='dot')
            summary_dict = hasher.UGC_partition([bin_width], Bin_values)
        
            supernode_dict = summary_dict[bin_width]
            
            rr = 1 - len(supernode_dict.keys())/dataset1[0].x.shape[0]
            reduced_percentage = rr
            
            #supernode_dict.keys()--> new supernode formed in coarsened graph
            print(f'Graph reduced by: {reduced_percentage} percent. \n Now we have {len(supernode_dict.keys())} supernodes; Starting nodes were: {dataset1[0].x.shape[0]}')
    

            C = torch.zeros(dataset1[0].x.shape[0] , len(supernode_dict.keys()))
            zero_list = torch.ones(len(supernode_dict.keys()), dtype=torch.bool)

            for super_idx, node_list in enumerate(supernode_dict.values()):
                for node in node_list:
                    C[node][super_idx] = 1
                    zero_list[super_idx] = zero_list[super_idx] and (not (dataset1[0].train_mask)[node])
            
            C_diag = torch.sum(C, dim=0)
            P = torch.sparse.mm(C,(torch.diag(torch.pow(C_diag, -1/2))))
            P = P.to_sparse().to_dense()

            # P, zero_list = ugc(X, 0.00165, 1000, train_mask)
            P = P.to(dataset1[0].edge_index.device).to('cuda:1')
            adj_c = P.T @ to_dense_adj(dataset1[0].edge_index).to('cuda:1')[0] @ P
            data_coarsen = Data(edge_index=dense_to_sparse(adj_c)[0].to(torch.int64).to('cuda:1'), train_mask=~zero_list)
        elif coar_method == 'faugc':
            hasher = ConsistentHashing(input_dim=X.shape[1], proj_dim=1000)
            supernode_dict = hasher.forward(dataset1[0])
            supernode_dict, _, ordered_mean_values = hasher.rank_and_sort_supernodes(supernode_dict, dataset1[0].x)
            num_supernodes = int((coarsening_ratio)*dataset1[0].x.shape[0])
            P, supernode_dict, zero_list = hasher.coarsen_ring_parallel_gpu(supernode_dict, dataset1[0], num_supernodes)
            device = dataset1[0].edge_index.device
            P = P.to(device)
            adj_c = P.T @ to_dense_adj(dataset1[0].edge_index) @ P
            # myY = F.one_hot(y,5).to(torch.float).to(device)
            # myY[~dataset1[0].train_mask] = torch.tensor([0 for _ in range(myY.shape[1])], dtype=myY.dtype).to(device)
            data_coarsen = Data(edge_index=dense_to_sparse(adj_c)[0].to(torch.int64), train_mask=~zero_list)
            # print(data_coarsen.train_mask.count_nonzero())
        elif coar_method == 'fgc':
            X, P = fgc(dataset1, coarsening_ratio)
            adj_c = P.T @ to_dense_adj(dataset1[0].edge_index).to(P.device) @ P
            new_mask = (P.T @ F.one_hot(dataset1[0].train_mask.to(torch.int64).to(P.device), 2).to(torch.float))[:, 1] > 0
            data_coarsen = Data(x=X.to(dataset1[0].x.dtype), edge_index=dense_to_sparse(adj_c)[0].to(torch.int64), train_mask=new_mask)
            print(data_coarsen)
        else: 
            data_coarsen, P = scal.scal_coarsen(dataset1, coarsening_ratio, coar_method, 'imdb')
            print(data_coarsen)
            print(P.shape, P.count_nonzero())

            # print(data_coarsen.y)
            # myY = F.one_hot(dataset1[0].y, 5).to(torch.float)
            # myY[~dataset1[0].train_mask] = torch.tensor([0 for _ in range(myY.shape[1])], dtype=myY.dtype)

            # data_coarsen.y = torch.argmax(P.T @ myY, dim=1)

        # vals, couts = torch.unique(data_coarsen.y, return_counts=True) 
        # print(vals)
        # print(couts)

        # nodetype_y = torch.cat((torch.zeros(4278), torch.ones(2081), 2*torch.ones(5257)))
        # vals, couts = torch.unique(torch.argmax(P.T @ F.one_hot(nodetype_y.to(torch.int64), 3).to(torch.float), dim=1), return_counts=True)
        # vals1, couts1 = torch.unique(data_coarsen.y, return_counts=True)
        # print(vals)
        # print(couts)
        # print(vals1)
        # print(couts1)
        

        nodetype_y = torch.cat((torch.zeros(4278), torch.ones(2081), 2*torch.ones(5257)))
        type_map = {0: 'movie', 1: 'director', 2: 'actor'}
        type_mapinv = {'movie': 0, 'director': 1, 'actor': 2}
        newData = ToHetero(dataset[0], dataset1[0], data_coarsen, P, nodetype_y, type_map, type_mapinv, 3, 'movie', coar_method, ifFGC_LAGC=coar_method == 'fgc' or coar_method == 'lagc') #data_orig, data_coarse, C, nodetype_y, type_map
        # print(torch.unique(newData['movie'].y, return_counts=True)[0])
        # print(vals))


        newData['movie', 'to', 'director'].edge_index = (newData['movie', 'to', 'director'].edge_index.T - torch.cat((torch.zeros(newData['movie', 'to', 'director'].edge_index.shape[1], 1), torch.ones(newData['movie', 'to', 'director'].edge_index.shape[1],1)*newData['movie'].x.shape[0]), dim=1)).T.to(dataset[0]['movie', 'to', 'director'].edge_index.dtype)
        newData['movie', 'to', 'actor'].edge_index = (newData['movie', 'to', 'actor'].edge_index.T - torch.cat((torch.zeros(newData['movie', 'to', 'actor'].edge_index.shape[1], 1), torch.ones(newData['movie', 'to', 'actor'].edge_index.shape[1],1)*(newData['movie'].x.shape[0]+newData['director'].x.shape[0])), dim=1)).T.to(dataset[0]['movie', 'to', 'director'].edge_index.dtype)
        newData['director', 'to', 'movie'].edge_index = (newData['director', 'to', 'movie'].edge_index.T - torch.cat((torch.ones(newData['director', 'to', 'movie'].edge_index.shape[1], 1)*newData['movie'].x.shape[0], torch.zeros(newData['director', 'to', 'movie'].edge_index.shape[1],1)), dim=1)).T.to(dataset[0]['movie', 'to', 'director'].edge_index.dtype)
        newData['actor', 'to', 'movie'].edge_index = (newData['actor', 'to', 'movie'].edge_index.T - torch.cat((torch.ones(newData['actor', 'to', 'movie'].edge_index.shape[1], 1)*(newData['movie'].x.shape[0]+newData['director'].x.shape[0]), torch.zeros(newData['actor', 'to', 'movie'].edge_index.shape[1],1)), dim=1)).T.to(dataset[0]['movie', 'to', 'director'].edge_index.dtype)

        x_syn_dict, adj_t_syn_dict = {}, {}
        for node_type in node_types:
            x_syn_dict[node_type] = newData[node_type].x.to('cuda:1')

        for edge_type in edge_types:
            adjmat = torch.zeros(newData[edge_type[2]].x.shape[0], newData[edge_type[0]].x.shape[0])
            for index, (src, dst) in enumerate(newData[edge_type].edge_index.T):
                adjmat[dst, src]=1
            adj_t_syn_dict[edge_type] = asymmetric_gcn_norm(adjmat).to('cuda:1')

        y_syn = newData['movie'].y.to('cuda:1')
        mask_syn = newData['movie'].train_mask.to('cuda:1')
        print(mask_syn.count_nonzero())

        ## Heat map plots
        # heatY = torch.cat((torch.zeros(4278), torch.ones(2081), 2*torch.ones(5257)))
        # heatC = P
        # plotHeatmap(heatC, heatY, filename=f"imdb_{coar_method}.png")

        for models in ['HeteroSGC', 'HeteroGCN', 'HeteroGCN2']:
        # for models in ['HeteroGCN']:
            model_name = model_dict[models]
            acc = eval_GNN(num_evals, x_dict, adj_t_dict, node_types, edge_types, dataset[0]['movie'].y.to('cuda:1'), target_node, num_classes, mydata.train_mask.to('cuda:1'), mydata.test_mask.to('cuda:1'), mydata.val_mask.to('cuda:1'), x_syn_dict, adj_t_syn_dict, y_syn, mask_syn, model_name, model_architecture, model_train)
            print(f"Using {models}, accuracy is {acc[0][0]*100}%")
            update_table_accuracy(coar_method, 'IMDB', acc[0][0]*100, model_name)
    #         myaccuracy.append(sum(acc[0]*100)/3)
    # print(myaccuracy)
        

    elif coar_method in 'fgc':
        pass

    elif coar_method in 'lagc':
        pass

    elif coar_method in 'gcond':
        pass

    return


def run_dblp(coar_method, coarsening_ratio, mypath, num_evals, model_name):
    model_name = model_dict[model_name]
    dataset0 = DBLP(mypath)
    mydata = utils.rand_train_test_idx(dataset0[0]['author'], 0.6, 0.2)
    node_types, edge_types = dataset0[0].metadata()
    dataset = HeteroData()
    for node_type in node_types:
        if hasattr(dataset0[0][node_type], 'x'):
            dataset[node_type].x = dataset0[0][node_type].x
            if hasattr(dataset0[0][node_type], 'train_mask'):
                dataset[node_type].train_mask = mydata.train_mask
                dataset[node_type].test_mask = mydata.test_mask
                dataset[node_type].val_mask = mydata.val_mask
        else:
            dataset[node_type].x = torch.ones(dataset0[0][node_type].num_nodes, 1, dtype=torch.float)
        
        if hasattr(dataset0[0][node_type], 'y'):
            dataset[node_type].y = dataset0[0][node_type].y
    
    for edge_type in edge_types:
        dataset[edge_type].edge_index = dataset0[0][edge_type].edge_index
    
    dataset = [T.ToSparseTensor(remove_edge_index=False)(dataset)]
                
    target_node = 'author'
    num_classes = 4
    x_dict, adj_t_dict= {}, {}
    for node_type in node_types:
        x_dict[node_type] = dataset[0][node_type].x.to('cuda:1')
    for edge_type in edge_types:
        adj_t_dict[edge_type] = asymmetric_gcn_norm(dataset[0][edge_type].adj_t).to('cuda:1')
    y = dataset[0][target_node].y
    train_mask = dataset[0][target_node].train_mask
    val_mask =  dataset[0][target_node].val_mask
    test_mask = dataset[0][target_node].test_mask

    if coar_method == 'base':
        for model in ['HeteroSGC', 'HeteroGCN', 'HeteroGCN2']:
            model_name = model_dict[model]
            acc, f1_micro, f1_macro = eval_GNN(num_evals, x_dict, adj_t_dict, node_types, edge_types, y, target_node, num_classes, train_mask, test_mask, val_mask, x_dict, adj_t_dict, y, train_mask, model_name, model_architecture, model_train)
            print(f"Using {model}, accuracy is {acc[0]*100}%")
    elif coar_method == 'cugc' or coar_method == 'caugc':
        # myaccuracy = []
        # for coarsening_ratio in [0.45]:
        proportions = {'author': 0.4, 'paper': 0.4, 'term': 0.4, 'conference': 0.05}
        C_dict = {}
        zero_list_dict ={}
        for node_type in node_types:
            if hasattr(dataset[0][node_type], 'x'):
                if coar_method == 'cugc':
                    hasher = ConsistentHashing(input_dim=dataset[0][node_type].x.shape[1], proj_dim=1000)
                    bin_width, _ = UGC_binwidth_finder.find_Binwidth(dataset[0][node_type], 0.95 if node_type=='conference' else 1-coarsening_ratio)

                    Bin_values = hasher.UGC_hashed_values(dataset[0][node_type], function='dot')
                    summary_dict = hasher.UGC_partition([bin_width], Bin_values)
                
                    supernode_dict = summary_dict[bin_width]
                    
                    rr = 1 - len(supernode_dict.keys())/dataset[0][node_type].x.shape[0]
                    reduced_percentage = rr
                    
                    #supernode_dict.keys()--> new supernode formed in coarsened graph
                    print(f'Graph reduced by: {reduced_percentage} percent. \n Now we have {len(supernode_dict.keys())} supernodes; Starting nodes were: {dataset[0][node_type].x.shape[0]}')
            

                    C = torch.zeros(dataset[0][node_type].x.shape[0] , len(supernode_dict.keys()))
                    zero_list = torch.ones(len(supernode_dict.keys()), dtype=torch.bool)

                    for super_idx, node_list in enumerate(supernode_dict.values()):
                        for node in node_list:
                            C[node][super_idx] = 1
                            if node_type == 'author':
                                zero_list[super_idx] = zero_list[super_idx] and (not (dataset[0][node_type].train_mask)[node])
                    
                    C_diag = torch.sum(C, dim=0)
                    P = torch.sparse.mm(C,(torch.diag(torch.pow(C_diag, -1/2))))
                    P = P.to_sparse().to_dense()

                    # P, zero_list = ugc(X, 0.00165, 1000, train_mask)
                    C_dict[node_type], zero_list_dict[node_type] = P, zero_list
                else:
                    hasher = ConsistentHashing(input_dim=dataset[0][node_type].x.shape[1], proj_dim=1000)
                    supernode_dict = hasher.forward(dataset[0][node_type])
                    supernode_dict, _, ordered_mean_values = hasher.rank_and_sort_supernodes(supernode_dict, dataset[0][node_type].x)
                    num_supernodes = int((proportions[node_type])*dataset[0][node_type].x.shape[0])
                    P, supernode_dict, zero_list = hasher.coarsen_ring_parallel_gpu(supernode_dict, dataset[0][node_type], num_supernodes)
                    C_dict[node_type], zero_list_dict[node_type] = P, zero_list
        ## Count number of edges for nodes and add them to respective supernodes adj
        new_data = HeteroData()

        clusters = {}
        for key in C_dict.keys():
            clusters[key] = torch.argmax(C_dict[key], dim=1)

        # Process each edge type
        for edge_type in edge_types:
            src_type, rel, dst_type = edge_type
            original_edges = dataset[0][edge_type].edge_index
            
            # Get cluster mappings
            src_map = clusters[src_type]
            dst_map = clusters[dst_type]
            
            # Convert original nodes to supernodes
            super_src = src_map[original_edges[0]]
            super_dst = dst_map[original_edges[1]]
            
            # Create and count superedges
            super_edges = torch.stack([super_src, super_dst])
            unique_edges, counts = torch.unique(
                super_edges, 
                dim=1, 
                return_counts=True
            )
            # Store in new graph
            new_data[src_type, rel, dst_type].edge_index = unique_edges
            new_data[src_type, rel, dst_type].edge_weight = counts.float()


        for node_type in node_types:
            if hasattr(dataset[0][node_type], 'x'):
                new_data[node_type].x = C_dict[node_type].T.to(dataset[0][node_type].x.device) @ dataset[0][node_type].x
            if node_type in target_node:
                myY = F.one_hot(dataset[0][node_type].y)
                myY[~dataset[0][node_type].train_mask] = torch.tensor([0 for _ in range(myY.shape[1])], dtype=myY.dtype, device=myY.device)
                interm = (C_dict[node_type].T @ myY.to(torch.float32).to(C_dict[node_type].T.device))
                print((~zero_list_dict['author'][(interm == 0).all(dim=1)]).count_nonzero())
                new_data[node_type].y = torch.argmax(interm, dim=1)
        ndata = new_data
        x_syn_dict, adj_t_syn_dict = {}, {}
        for node_type in node_types:
            x_syn_dict[node_type] = ndata[node_type].x.to('cuda:1')

        for edge_type in edge_types:
            adjmat = torch.zeros(ndata[edge_type[2]].x.shape[0], ndata[edge_type[0]].x.shape[0])
            for index, (src, dst) in enumerate(ndata[edge_type].edge_index.T):
                adjmat[dst, src]=ndata[edge_type].edge_weight[index]
            adj_t_syn_dict[edge_type] = asymmetric_gcn_norm(adjmat).to('cuda:1')

        y_syn = ndata['author'].y.to('cuda:1')
        mask_syn = ~zero_list_dict['author'].to('cuda:1')
        print(mask_syn.count_nonzero())

        ## Heat map plots
        # heatY = torch.cat((torch.zeros(4057), torch.ones(14328), 2*torch.ones(7723), 3*torch.ones(20)))
        # heatC = torch.zeros(26128, ndata['author'].x.shape[0]+ndata['paper'].x.shape[0]+ndata['term'].x.shape[0]+ndata['conference'].x.shape[0])
        # prev, curr=0, 0
        # prevy, curry=0, 0
        # for node_type in node_types:
        #     curr += dataset[0][node_type].x.shape[0]
        #     curry += ndata[node_type].x.shape[0]
        #     heatC[prev:curr, prevy:curry] = C_dict[node_type]
        #     prev += dataset[0][node_type].x.shape[0]
        #     prevy += ndata[node_type].x.shape[0]
        # plotHeatmap(heatC, heatY, filename=f"dblp_{coar_method}.png")

        for model in ['HeteroSGC', 'HeteroGCN', 'HeteroGCN2']:
        # for model in ['HeteroGCN']:
            model_name = model_dict[model]
            acc, f1_micro, f1_macro = eval_GNN(1, x_dict, adj_t_dict, node_types, edge_types, y.to('cuda:1'), target_node, num_classes, train_mask.to('cuda:1'), test_mask.to('cuda:1'), val_mask.to('cuda:1'), x_syn_dict, adj_t_syn_dict, y_syn, mask_syn, model_name, model_architecture, model_train)
            print(f"Using {model}, accuracy is {acc[0]*100}%")
            update_table_accuracy(coar_method, 'DBLP', acc[0]*100, model_name)
        #     myaccuracy.append(acc[0]*100)
        # print(myaccuracy)

    elif coar_method in ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron', 'fugc', 'faugc']:
        total_nodes = 0
        for node_type in node_types:
            total_nodes += dataset[0][node_type].x.shape[0]

        X = torch.zeros(total_nodes, 4231)
        curr = 0
        prev = 0
        for node_type in node_types:
            curr += dataset[0][node_type].x.shape[0]
            X[prev:curr, :dataset[0][node_type].x.shape[1]] = dataset[0][node_type].x
            prev += dataset[0][node_type].x.shape[0]

        y = torch.cat((dataset[0]['author'].y, 4*torch.ones(14328), 5*torch.ones(7723), 6*torch.ones(20)), dim=0)
        y = y.to(torch.int64)

        edge1 = torch.cat((torch.zeros(19645, 1), torch.ones(19645,1)*4057), dim=1) + dataset[0]['author', 'to', 'paper'].edge_index.T
        edge2 = torch.cat((torch.ones(19645, 1)*4057, torch.zeros(19645, 1)), dim=1) + dataset[0]['paper', 'to', 'author'].edge_index.T
        edge3 = torch.cat((torch.ones(85810, 1)*4057, torch.ones(85810, 1)*(4057+14328)), dim=1) + dataset[0]['paper', 'to', 'term'].edge_index.T
        edge4 = torch.cat((torch.ones(14328, 1)*4057, torch.ones(14328, 1)*(4057+14328+7723)), dim=1) + dataset[0]['paper', 'to', 'conference'].edge_index.T
        edge5 = torch.cat((torch.ones(85810, 1)*(4057+14328), torch.ones(85810, 1)*4057), dim=1) + dataset[0]['term', 'to', 'paper'].edge_index.T
        edge6 = torch.cat((torch.ones(14328, 1)*(4057+14328+7723), torch.ones(14328, 1)*4057), dim=1) + dataset[0]['conference', 'to', 'paper'].edge_index.T
        edge = torch.cat((edge1, edge2, edge3, edge4, edge5, edge6), dim=0)
        mydata = utils.rand_train_test_idx(dataset[0]['author'], 0.6, 0.2)
        train_mask=torch.cat((train_mask, torch.ones(26128-4057, dtype=bool)))
        test_mask=torch.cat((test_mask, torch.ones(26128-4057, dtype=bool)))
        val_mask=torch.cat((val_mask, torch.ones(26128-4057, dtype=bool)))

        dataset1 = [Data(x=X, y=y, edge_index=edge.T.to(torch.int64), train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)]
        print(dataset1[0])
        # myaccuracy = []
        # for coarsening_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        if coar_method == 'fugc':
            hasher = ConsistentHashing(input_dim=dataset1[0].x.shape[1], proj_dim=1000)
            bin_width, _ = UGC_binwidth_finder.find_Binwidth(dataset1[0], 1-coarsening_ratio)

            Bin_values = hasher.UGC_hashed_values(dataset1[0], function='dot')
            summary_dict = hasher.UGC_partition([bin_width], Bin_values)
        
            supernode_dict = summary_dict[bin_width]
            
            rr = 1 - len(supernode_dict.keys())/dataset1[0].x.shape[0]
            reduced_percentage = rr
            
            #supernode_dict.keys()--> new supernode formed in coarsened graph
            print(f'Graph reduced by: {reduced_percentage} percent. \n Now we have {len(supernode_dict.keys())} supernodes; Starting nodes were: {dataset1[0].x.shape[0]}')
    

            C = torch.zeros(dataset1[0].x.shape[0] , len(supernode_dict.keys()))
            zero_list = torch.ones(len(supernode_dict.keys()), dtype=torch.bool)

            for super_idx, node_list in enumerate(supernode_dict.values()):
                for node in node_list:
                    C[node][super_idx] = 1
                    zero_list[super_idx] = zero_list[super_idx] and (not (dataset1[0].train_mask)[node])
            
            C_diag = torch.sum(C, dim=0)
            P = torch.sparse.mm(C,(torch.diag(torch.pow(C_diag, -1/2))))
            P = P.to_sparse().to_dense()

            # P, zero_list = ugc(X, 0.00165, 1000, train_mask)
            P = P.to(dataset1[0].edge_index.device)
            adj_c = P.T @ to_dense_adj(dataset1[0].edge_index) @ P
            data_coarsen = Data(edge_index=dense_to_sparse(adj_c)[0].to(torch.int64), train_mask=~zero_list)
        elif coar_method == 'faugc':
            hasher = ConsistentHashing(input_dim=X.shape[1], proj_dim=1000)
            supernode_dict = hasher.forward(dataset1[0])
            supernode_dict, _, ordered_mean_values = hasher.rank_and_sort_supernodes(supernode_dict, dataset1[0].x)
            num_supernodes = int((coarsening_ratio)*dataset1[0].x.shape[0])
            P, supernode_dict, zero_list = hasher.coarsen_ring_parallel_gpu(supernode_dict, dataset1[0], num_supernodes)
            device = dataset1[0].edge_index.device
            P = P.to(device)
            adj_c = P.T @ to_dense_adj(dataset1[0].edge_index) @ P
            data_coarsen = Data(edge_index=dense_to_sparse(adj_c)[0].to(torch.int64), train_mask=~zero_list)
        else:
            data_coarsen, P = scal.scal_coarsen(dataset1, coarsening_ratio, coar_method, 'dblp')

        nodetype_y = torch.cat((torch.zeros(4057), torch.ones(14328), 2*torch.ones(7723), 3*torch.ones(20)))
        type_map = {0: 'author', 1: 'paper', 2: 'term', 3: 'conference'}
        type_mapinv = {v:k for k, v in type_map.items()}
        newData = ToHetero(dataset[0], dataset1[0], data_coarsen, P, nodetype_y, type_map, type_mapinv, 4, 'author', coar_method)


        newData['author', 'to', 'paper'].edge_index = (newData['author', 'to', 'paper'].edge_index.T - torch.cat((torch.zeros(newData['author', 'to', 'paper'].edge_index.shape[1], 1), torch.ones(newData['author', 'to', 'paper'].edge_index.shape[1],1)*newData['author'].x.shape[0]), dim=1)).T.to(newData['author', 'to', 'paper'].edge_index.dtype)
        newData['paper', 'to', 'author'].edge_index = (newData['paper', 'to', 'author'].edge_index.T - torch.cat((torch.ones(newData['paper', 'to', 'author'].edge_index.shape[1], 1)*newData['author'].x.shape[0], torch.zeros(newData['paper', 'to', 'author'].edge_index.shape[1],1)), dim=1)).T.to(newData['author', 'to', 'paper'].edge_index.dtype)
        newData['paper', 'to', 'term'].edge_index = (newData['paper', 'to', 'term'].edge_index.T - torch.cat((torch.ones(newData['paper', 'to', 'term'].edge_index.shape[1], 1)*newData['author'].x.shape[0], torch.ones(newData['paper', 'to', 'term'].edge_index.shape[1],1)*(newData['paper'].x.shape[0]+newData['author'].x.shape[0])), dim=1)).T.to(newData['author', 'to', 'paper'].edge_index.dtype)
        newData['term', 'to', 'paper'].edge_index = (newData['term', 'to', 'paper'].edge_index.T - torch.cat((torch.ones(newData['term', 'to', 'paper'].edge_index.shape[1], 1)*(newData['author'].x.shape[0]+newData['paper'].x.shape[0]), torch.ones(newData['term', 'to', 'paper'].edge_index.shape[1],1)*(newData['author'].x.shape[0])), dim=1)).T.to(newData['author', 'to', 'paper'].edge_index.dtype)
        if newData['paper', 'to', 'conference'].edge_index.shape[0]>0:
            newData['paper', 'to', 'conference'].edge_index = (newData['paper', 'to', 'conference'].edge_index.T - torch.cat((torch.ones(newData['paper', 'to', 'conference'].edge_index.shape[1], 1)*newData['author'].x.shape[0], torch.ones(newData['paper', 'to', 'conference'].edge_index.shape[1], 1)*(newData['author'].x.shape[0]+newData['paper'].x.shape[0]+newData['term'].x.shape[0])), dim=1)).T.to(newData['author', 'to', 'paper'].edge_index.dtype)
            newData['conference', 'to', 'paper'].edge_index = (newData['conference', 'to', 'paper'].edge_index.T - torch.cat((torch.ones(newData['conference', 'to', 'paper'].edge_index.shape[1], 1)*(newData['author'].x.shape[0]+newData['paper'].x.shape[0]+newData['term'].x.shape[0]), torch.ones(newData['conference', 'to', 'paper'].edge_index.shape[1],1)*(newData['author'].x.shape[0])), dim=1)).T.to(newData['author', 'to', 'paper'].edge_index.dtype)
        
        for node_type in node_types:
            newData[node_type].x = newData[node_type].x[:, :dataset[0][node_type].x.shape[1]]

        # for edge_type in edge_types:
        #     print(edge_type[0], torch.min(newData[edge_type].edge_index[0]), torch.max(newData[edge_type].edge_index[0]), newData[edge_type].edge_index.shape)
        #     print(edge_type[2], torch.min(newData[edge_type].edge_index[1]), torch.max(newData[edge_type].edge_index[1]))
        
        device = 'cuda:1'
        x_syn_dict, adj_t_syn_dict = {}, {}
        for node_type in node_types:
            x_syn_dict[node_type] = newData[node_type].x.to(device)

        for edge_type in edge_types:
            adjmat = torch.zeros(newData[edge_type[2]].x.shape[0], newData[edge_type[0]].x.shape[0])
            for index, (src, dst) in enumerate(newData[edge_type].edge_index.T):
                adjmat[dst, src]=1
            adj_t_syn_dict[edge_type] = asymmetric_gcn_norm(adjmat).to(device)
        
        y_syn = newData['author'].y.to(device)
        mask_syn = newData['author'].train_mask.to(device)
        dataset[0] = dataset[0].to(device)
        print(mask_syn.count_nonzero())

        ## Heat map plots
        # heatY = torch.cat((torch.zeros(4057), torch.ones(14328), 2*torch.ones(7723), 3*torch.ones(20)))
        # heatC = P
        # plotHeatmap(heatC, heatY, filename=f"dblp_{coar_method}.png")


        for models in ['HeteroSGC', 'HeteroGCN', 'HeteroGCN2']:
        # for models in ['HeteroGCN']:
            model_name = model_dict[models]
            acc = eval_GNN(num_evals, x_dict, adj_t_dict, node_types, edge_types, dataset[0]['author'].y.to('cuda:1'), target_node, num_classes, dataset[0]['author'].train_mask.to('cuda:1'), dataset[0]['author'].test_mask.to('cuda:1'), dataset[0]['author'].val_mask.to('cuda:1'), x_syn_dict, adj_t_syn_dict, y_syn, mask_syn, model_name, model_architecture, model_train)
            print(f"Using {models}, accuracy is {acc[0][0]*100}%")
            update_table_accuracy(coar_method, 'DBLP', acc[0][0]*100, model_name)
        #     myaccuracy.append(sum(acc[0])*100/3)
        # print(myaccuracy)

    elif coar_method in 'fgc':
        pass

    elif coar_method in 'lagc':
        pass

    elif coar_method in 'gcond':
        pass
    
    return


def run_acm(coar_method, coarsening_ratio, mypath, num_evals, model_name):
    model_name = model_dict[model_name]
    dataset0 = HGBDataset(mypath, name="ACM", transform=T.ToSparseTensor(remove_edge_index=False))
    dataset = [dataset0[0]]
    node_types, edge_types = dataset[0].metadata()
    target_node = 'paper'
    num_classes = 3
    dataset[0]['term'].x = torch.eye(dataset[0]['term'].num_nodes, 1902)
    x_dict, adj_t_dict= {}, {}
    for node_type in node_types:
        x_dict[node_type] = dataset[0][node_type].x.to('cuda:1')
    for edge_type in edge_types:
        adj_t_dict[edge_type] = asymmetric_gcn_norm(dataset[0][edge_type].adj_t).to('cuda:1')
    y = dataset[0][target_node].y.to('cuda:1')
    
    mydata = utils.rand_train_test_idx(dataset0[0]['paper'], 0.6, 0.2)
    dataset[0]['paper'].train_mask = mydata.train_mask.to('cuda:1')
    dataset[0]['paper'].test_mask = mydata.test_mask.to('cuda:1')
    dataset[0]['paper'].val_mask = mydata.val_mask.to('cuda:1')

    if coar_method == 'base':
        for model in ['HeteroSGC', 'HeteroGCN', 'HeteroGCN2']:
            model_name = model_dict[model]
            acc, f1_micro, f1_macro = eval_GNN(num_evals, x_dict, adj_t_dict, node_types, edge_types, y, target_node, num_classes, mydata.train_mask, mydata.test_mask, mydata.val_mask, x_dict, adj_t_dict, y, mydata.train_mask, model_name, model_architecture, model_train)
            print(f"Using {model}, accuracy is {acc[0]*100}%")

    elif coar_method == 'cugc' or coar_method == 'caugc' or coar_method == 'cvan':
        proportions = {'paper': coarsening_ratio, 'author': coarsening_ratio, 'subject': coarsening_ratio, 'term': coarsening_ratio}
        C_dict = {}
        zero_list_dict ={}
        for node_type in node_types:
            if hasattr(dataset[0][node_type], 'x'):
                if coar_method == 'cugc':
                    hasher = ConsistentHashing(input_dim=dataset[0][node_type].x.shape[1], proj_dim=1000)
                    bin_width, _ = UGC_binwidth_finder.find_Binwidth(dataset[0][node_type], 1-proportions[node_type])

                    Bin_values = hasher.UGC_hashed_values(dataset[0][node_type], function='dot')
                    summary_dict = hasher.UGC_partition([bin_width], Bin_values)
                
                    supernode_dict = summary_dict[bin_width]
                    
                    rr = 1 - len(supernode_dict.keys())/dataset[0][node_type].x.shape[0]
                    reduced_percentage = rr
                    
                    #supernode_dict.keys()--> new supernode formed in coarsened graph
                    print(f'Graph reduced by: {reduced_percentage} percent. \n Now we have {len(supernode_dict.keys())} supernodes; Starting nodes were: {dataset[0][node_type].x.shape[0]}')
            

                    C = torch.zeros(dataset[0][node_type].x.shape[0] , len(supernode_dict.keys()))
                    zero_list = torch.ones(len(supernode_dict.keys()), dtype=torch.bool)

                    for super_idx, node_list in enumerate(supernode_dict.values()):
                        for node in node_list:
                            C[node][super_idx] = 1
                            if node_type == 'paper':
                                zero_list[super_idx] = zero_list[super_idx] and (not (dataset[0][node_type].train_mask)[node])
                    
                    C_diag = torch.sum(C, dim=0)
                    P = torch.sparse.mm(C,(torch.diag(torch.pow(C_diag, -1/2))))
                    P = P.to_sparse().to_dense()

                    # P, zero_list = ugc(X, 0.00165, 1000, train_mask)
                    C_dict[node_type], zero_list_dict[node_type] = P, zero_list
                
                elif coar_method == 'caugc':
                    hasher = ConsistentHashing(input_dim=dataset[0][node_type].x.shape[1], proj_dim=1000)
                    supernode_dict = hasher.forward(dataset[0][node_type])
                    supernode_dict, _, ordered_mean_values = hasher.rank_and_sort_supernodes(supernode_dict, dataset[0][node_type].x)
                    num_supernodes = int((proportions[node_type])*dataset[0][node_type].x.shape[0])
                    P, supernode_dict, zero_list = hasher.coarsen_ring_parallel_gpu(supernode_dict, dataset[0][node_type], num_supernodes)
                    C_dict[node_type], zero_list_dict[node_type] = P, zero_list
                # else:
                    # dataset_new = [dataset[0][node_type].clone().to('cpu')]
                    # dataset_new[0].edge_index = torch.tensor([[0], [0]], dtype=dataset[0]['paper', 'to', 'author'].edge_index.dtype)
                    # if node_type != target_node:
                    #     dataset_new[0].y = torch.ones(dataset_new[0].x.shape[0])
                    #     dataset_new[0]
                    # data_coarsen, P = scal.scal_coarsen(dataset_new, coarsening_ratio, coar_method, 'acm')
                    # C_dict[node_type], zero_list_dict[node_type] = P, ~data_coarsen.train_mask
        ## Count number of edges for nodes and add them to respective supernodes adj
        new_data = HeteroData()

        clusters = {}
        for key in C_dict.keys():
            clusters[key] = torch.argmax(C_dict[key], dim=1)

        # Process each edge type
        for edge_type in edge_types:
            src_type, rel, dst_type = edge_type
            original_edges = dataset[0][edge_type].edge_index
            
            # Get cluster mappings
            src_map = clusters[src_type]
            dst_map = clusters[dst_type]
            
            # Convert original nodes to supernodes
            super_src = src_map[original_edges[0]]
            super_dst = dst_map[original_edges[1]]
            
            # Create and count superedges
            super_edges = torch.stack([super_src, super_dst])
            unique_edges, counts = torch.unique(
                super_edges, 
                dim=1, 
                return_counts=True
            )
            # Store in new graph
            new_data[src_type, rel, dst_type].edge_index = unique_edges
            new_data[src_type, rel, dst_type].edge_weight = counts.float()


        for node_type in node_types:
            if hasattr(dataset[0][node_type], 'x'):

                new_data[node_type].x = C_dict[node_type].T @ dataset[0][node_type].x.to(C_dict[node_type].T.device)
            if node_type in target_node:
                myY = F.one_hot(dataset[0][node_type].y)
                myY[~dataset[0][node_type].train_mask] = torch.tensor([0 for _ in range(myY.shape[1])], dtype=myY.dtype, device=myY.device)
                interm = (C_dict[node_type].T @ myY.to(torch.float32).to(C_dict[node_type].T.device))
                new_data[node_type].y = torch.argmax(interm, dim=1)
        ndata = new_data
        x_syn_dict, adj_t_syn_dict = {}, {}
        for node_type in node_types:
            x_syn_dict[node_type] = ndata[node_type].x.to('cuda:1')

        for edge_type in edge_types[::-1]:
            adjmat = torch.zeros(ndata[edge_type[2]].x.shape[0], ndata[edge_type[0]].x.shape[0])
            for index, (src, dst) in enumerate(ndata[edge_type].edge_index.T):
                adjmat[dst, src]=1
            adj_t_syn_dict[edge_type] = asymmetric_gcn_norm(adjmat).to('cuda:1')

        y_syn = ndata['paper'].y.to('cuda:1')
        mask_syn = ~zero_list_dict['paper'].to('cuda:1')
        print(mask_syn.count_nonzero())

        for model in ['HeteroSGC', 'HeteroGCN', 'HeteroGCN2']:
            model_name = model_dict[model]
            acc, f1_micro, f1_macro = eval_GNN(num_evals, x_dict, adj_t_dict, node_types, edge_types, y, target_node, num_classes, dataset[0]['paper'].train_mask, dataset[0]['paper'].test_mask, dataset[0]['paper'].val_mask, x_syn_dict, adj_t_syn_dict, y_syn, mask_syn, model_name, model_architecture, model_train)
            print(f"Using {model}, accuracy is {acc[0]*100}%")


    elif coar_method in ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron', 'fugc', 'faugc']:
        dataset[0]['paper', 'to', 'paper'].edge_index = torch.unique(torch.cat((dataset[0]['paper', 'cite', 'paper'].edge_index.T, dataset[0]['paper', 'ref', 'paper'].edge_index.T)), dim=0).T.to(dataset[0]['paper', 'ref', 'paper'].edge_index.dtype)
        del dataset[0]['paper', 'cite', 'paper']
        del dataset[0]['paper', 'ref', 'paper']
        dataset[0] = T.ToSparseTensor(remove_edge_index=False)(dataset[0])

        total_nodes = 0
        for node_type in node_types:
            total_nodes += dataset[0][node_type].x.shape[0]

        X = torch.zeros(total_nodes, 1902)
        curr = 0
        prev = 0
        for node_type in node_types:
            curr += dataset[0][node_type].x.shape[0]
            X[prev:curr, :dataset[0][node_type].x.shape[1]] = dataset[0][node_type].x
            prev += dataset[0][node_type].x.shape[0]
        
        y = torch.cat((dataset[0]['paper'].y, 3*torch.ones(5959), 4*torch.ones(56), 5*torch.ones(1902)), dim=0)
        y = y.to(torch.int64)

        edge1 = dataset[0]['paper', 'to', 'paper'].edge_index.T
        edge3 = torch.cat((torch.zeros(9949, 1), torch.ones(9949, 1)*(3025)), dim=1) + dataset[0]['paper', 'to', 'author'].edge_index.T
        edge4 = torch.cat((torch.ones(9949, 1)*3025, torch.zeros(9949, 1)), dim=1) + dataset[0]['author', 'to', 'paper'].edge_index.T
        edge5 = torch.cat((torch.zeros(3025, 1), torch.ones(3025, 1)*(3025+5959)), dim=1) + dataset[0]['paper', 'to', 'subject'].edge_index.T
        edge6 = torch.cat((torch.ones(3025, 1)*(3025+5959), torch.zeros(3025, 1)), dim=1) + dataset[0]['subject', 'to', 'paper'].edge_index.T
        edge7 = torch.cat((torch.zeros(255619, 1), torch.ones(255619, 1)*(3025+5959+56)), dim=1) + dataset[0]['paper', 'to', 'term'].edge_index.T
        edge8 = torch.cat((torch.ones(255619, 1)*(3025+5959+56), torch.zeros(255619, 1)), dim=1) + dataset[0]['term', 'to', 'paper'].edge_index.T
        edge = torch.cat((edge1, edge3, edge4, edge5, edge6, edge7, edge8), dim=0)
        
        train_mask=torch.cat((dataset[0]['paper'].train_mask, torch.ones(10942-3025, dtype=bool, device='cuda:1')))
        test_mask=torch.cat((dataset[0]['paper'].test_mask, torch.ones(10942-3025, dtype=bool, device='cuda:1')))
        val_mask=torch.cat((dataset[0]['paper'].val_mask, torch.ones(10942-3025, dtype=bool, device='cuda:1')))
        dataset1 = [Data(x=X, y=y, edge_index=edge.T.to(torch.int64), train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)]
        print(dataset1[0])

        if coar_method == 'fugc':
            hasher = ConsistentHashing(input_dim=dataset1[0].x.shape[1], proj_dim=1000)
            bin_width, _ = UGC_binwidth_finder.find_Binwidth(dataset1[0], 1-coarsening_ratio)

            Bin_values = hasher.UGC_hashed_values(dataset1[0], function='dot')
            summary_dict = hasher.UGC_partition([bin_width], Bin_values)
        
            supernode_dict = summary_dict[bin_width]
            
            rr = 1 - len(supernode_dict.keys())/dataset1[0].x.shape[0]
            reduced_percentage = rr
            
            #supernode_dict.keys()--> new supernode formed in coarsened graph
            print(f'Graph reduced by: {reduced_percentage} percent. \n Now we have {len(supernode_dict.keys())} supernodes; Starting nodes were: {dataset1[0].x.shape[0]}')
      

            C = torch.zeros(dataset1[0].x.shape[0] , len(supernode_dict.keys()))
            zero_list = torch.ones(len(supernode_dict.keys()), dtype=torch.bool)

            for super_idx, node_list in enumerate(supernode_dict.values()):
                for node in node_list:
                    C[node][super_idx] = 1
                    zero_list[super_idx] = zero_list[super_idx] and (not (dataset1[0].train_mask)[node])
            
            C_diag = torch.sum(C, dim=0)
            P = torch.sparse.mm(C,(torch.diag(torch.pow(C_diag, -1/2))))
            P = P.to_sparse().to_dense()

            # P, zero_list = ugc(X, 0.00165, 1000, train_mask)
            P = P.to(dataset1[0].edge_index.device)
            adj_c = P.T @ to_dense_adj(dataset1[0].edge_index) @ P
            data_coarsen = Data(edge_index=dense_to_sparse(adj_c)[0].to(torch.int64), train_mask=~zero_list)
        
        elif coar_method == 'faugc':
            hasher = ConsistentHashing(input_dim=X.shape[1], proj_dim=1000)
            supernode_dict = hasher.forward(dataset1[0])
            supernode_dict, _, ordered_mean_values = hasher.rank_and_sort_supernodes(supernode_dict, dataset1[0].x)
            num_supernodes = int((coarsening_ratio)*dataset1[0].x.shape[0])
            P, supernode_dict, zero_list = hasher.coarsen_ring_parallel_gpu(supernode_dict, dataset1[0], num_supernodes)
            device = dataset1[0].edge_index.device
            P = P.to(device)
            adj_c = P.T @ to_dense_adj(dataset1[0].edge_index) @ P
            data_coarsen = Data(edge_index=dense_to_sparse(adj_c)[0].to(torch.int64), train_mask=~zero_list)

        else:
            data_coarsen, P = scal.scal_coarsen(dataset1, coarsening_ratio, coar_method, 'acm')

        nodetype_y = torch.cat((torch.zeros(3025), torch.ones(5959), 2*torch.ones(56), 3*torch.ones(1902)))
        type_map = {0: 'paper', 1: 'author', 2: 'subject', 3: 'term'}
        type_mapinv = {v:k for k, v in type_map.items()}
        newData = ToHetero(dataset[0], dataset1[0], data_coarsen, P, nodetype_y, type_map, type_mapinv, 3, 'paper', coar_method)
        print(newData)

        newData['paper', 'to', 'author'].edge_index = (newData['paper', 'to', 'author'].edge_index.T - torch.cat((torch.zeros(newData['paper', 'to', 'author'].edge_index.shape[1], 1), newData['paper'].x.shape[0]*torch.ones(newData['paper', 'to', 'author'].edge_index.shape[1], 1)), dim=1)).T.to(newData['paper', 'to', 'author'].edge_index.dtype)
        newData['author', 'to', 'paper'].edge_index = (newData['author', 'to', 'paper'].edge_index.T - torch.cat((newData['paper'].x.shape[0]*torch.ones(newData['author', 'to', 'paper'].edge_index.shape[1], 1), torch.zeros(newData['author', 'to', 'paper'].edge_index.shape[1], 1)), dim=1)).T.to(newData['paper', 'to', 'author'].edge_index.dtype)
        newData['paper', 'to', 'subject'].edge_index = (newData['paper', 'to', 'subject'].edge_index.T - torch.cat((torch.zeros(newData['paper', 'to', 'subject'].edge_index.shape[1], 1), (newData['paper'].x.shape[0]+newData['author'].x.shape[0])*torch.ones(newData['paper', 'to', 'subject'].edge_index.shape[1], 1)), dim=1)).T.to(newData['paper', 'to', 'author'].edge_index.dtype)
        newData['subject', 'to', 'paper'].edge_index = (newData['subject', 'to', 'paper'].edge_index.T - torch.cat(((newData['paper'].x.shape[0]+newData['author'].x.shape[0])*torch.ones(newData['subject', 'to', 'paper'].edge_index.shape[1], 1), torch.zeros(newData['subject', 'to', 'paper'].edge_index.shape[1], 1)), dim=1)).T.to(newData['paper', 'to', 'author'].edge_index.dtype)
        newData['paper', 'to', 'term'].edge_index = (newData['paper', 'to', 'term'].edge_index.T - torch.cat((torch.zeros(newData['paper', 'to', 'term'].edge_index.shape[1], 1), (newData['paper'].x.shape[0]+newData['author'].x.shape[0]+newData['subject'].x.shape[0])*torch.ones(newData['paper', 'to', 'term'].edge_index.shape[1], 1)), dim=1)).T.to(newData['paper', 'to', 'author'].edge_index.dtype)
        newData['term', 'to', 'paper'].edge_index = (newData['term', 'to', 'paper'].edge_index.T - torch.cat(((newData['paper'].x.shape[0]+newData['author'].x.shape[0]+newData['subject'].x.shape[0])*torch.ones(newData['term', 'to', 'paper'].edge_index.shape[1], 1), torch.zeros(newData['term', 'to', 'paper'].edge_index.shape[1], 1)), dim=1)).T.to(newData['paper', 'to', 'author'].edge_index.dtype)

        for node_type in node_types:
            newData[node_type].x = newData[node_type].x[:, :dataset[0][node_type].x.shape[1]]

        # for edge_type in dataset[0].metadata()[1]:
        #     print(edge_type[0], torch.min(newData[edge_type].edge_index[0]), torch.max(newData[edge_type].edge_index[0]), newData[edge_type].edge_index.shape)
        #     print(edge_type[2], torch.min(newData[edge_type].edge_index[1]), torch.max(newData[edge_type].edge_index[1]))
        

        x_syn_dict, adj_t_syn_dict = {}, {}
        for node_type in node_types:
            x_syn_dict[node_type] = newData[node_type].x.to('cuda:1')

        for edge_type in dataset[0].metadata()[1]:
            adjmat = torch.zeros(newData[edge_type[2]].x.shape[0], newData[edge_type[0]].x.shape[0])
            for index, (src, dst) in enumerate(newData[edge_type].edge_index.T):
                adjmat[dst, src]=1
            adj_t_syn_dict[edge_type] = asymmetric_gcn_norm(adjmat).to('cuda:1')
        
        y_syn = newData['paper'].y.to('cuda:1')
        mask_syn = newData['paper'].train_mask.to('cuda:1')
        print(mask_syn.count_nonzero())
        dataset[0] = dataset[0].to('cuda:1')
        for models in ['HeteroSGC', 'HeteroGCN', 'HeteroGCN2']:
            model_name = model_dict[models]
            acc = eval_GNN(num_evals, x_dict, adj_t_dict, node_types, dataset[0].metadata()[1], dataset[0]['paper'].y, target_node, num_classes, dataset[0]['paper'].train_mask, dataset[0]['paper'].test_mask, dataset[0]['paper'].val_mask, x_syn_dict, adj_t_syn_dict, y_syn, mask_syn, model_name, model_architecture, model_train)
            print(f"Using {models}, accuracy is {acc[0][0]*100}%")
            update_table_accuracy(coar_method, 'ACM', acc[0][0]*100, model_name)


    elif coar_method in 'fgc':
        pass

    elif coar_method in 'lagc':
        pass

    elif coar_method in 'gcond':
        pass
    
    return

