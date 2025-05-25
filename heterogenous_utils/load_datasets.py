import torch
from torch_geometric.datasets import IMDB, DBLP, HGBDataset
import torch_geometric.transforms as T

from hutils import asymmetric_gcn_norm, ToHetero, ToHomo

import sys
sys.path.append("/home/mohit/projects/heteroGC/coarGC/")
import utils

def load_imdb():
    dataset0 = IMDB(root='hdatasets/IMDB', transform=T.ToSparseTensor(remove_edge_index=False))
    dataset = dataset0[0]
    node_types, edge_types = dataset.metadata()
    dataset.target_node = 'movie'
    dataset.num_classes = 3
    # Create train, test, val splits
    mydata = utils.rand_train_test_idx(dataset[dataset.target_node], 0.6, 0.2)
    dataset[dataset.target_node].train_mask = mydata.train_mask
    dataset[dataset.target_node].test_mask = mydata.test_mask
    dataset[dataset.target_node].val_mask = mydata.val_mask

    x_dict, adj_t_dict = {}, {}
    for node_type in node_types:
        x_dict[node_type] = dataset[node_type].x
    for edge_type in edge_types:
        adj_t_dict[edge_type] = asymmetric_gcn_norm(dataset[edge_type].adj_t)

    dataset.x_dict = x_dict
    dataset.adj_t_dict = adj_t_dict
    dataset.node_types = node_types
    dataset.edge_types = edge_types

    return dataset


def load_dblp():
    dataset0 = DBLP(root='hdatasets/DBLP', transform=T.ToSparseTensor(remove_edge_index=False))
    dataset = dataset0[0]

    node_types, edge_types = dataset.metadata()
    dataset.target_node = 'author'
    dataset.num_classes = 4
    # Create train, test, val splits
    mydata = utils.rand_train_test_idx(dataset[dataset.target_node], 0.6, 0.2)
    dataset[dataset.target_node].train_mask = mydata.train_mask
    dataset[dataset.target_node].test_mask = mydata.test_mask
    dataset[dataset.target_node].val_mask = mydata.val_mask

    dataset['conference'].x = torch.ones(dataset['conference'].num_nodes, 1, dtype=torch.float)
    del dataset['conference'].num_nodes

    x_dict, adj_t_dict = {}, {}
    for node_type in node_types:
        x_dict[node_type] = dataset[node_type].x
    for edge_type in edge_types:
        adj_t_dict[edge_type] = asymmetric_gcn_norm(dataset[edge_type].adj_t)

    dataset.x_dict = x_dict
    dataset.adj_t_dict = adj_t_dict
    dataset.node_types = node_types
    dataset.edge_types = edge_types

    return dataset

def load_acm():
    dataset0 = HGBDataset(root='hdatasets/ACM', name='ACM', transform=T.ToSparseTensor(remove_edge_index=False))
    dataset = dataset0[0]
    
    dataset.target_node = 'paper'
    dataset.num_classes = 3
    # Create train, test, val splits
    mydata = utils.rand_train_test_idx(dataset[dataset.target_node], 0.6, 0.2)
    dataset[dataset.target_node].train_mask = mydata.train_mask
    dataset[dataset.target_node].test_mask = mydata.test_mask
    dataset[dataset.target_node].val_mask = mydata.val_mask

    dataset['term'].x = torch.eye(dataset['term'].num_nodes, 1902)
    dataset['paper', 'to', 'paper'].edge_index = torch.unique(torch.cat((dataset['paper', 'cite', 'paper'].edge_index.T, dataset['paper', 'ref', 'paper'].edge_index.T)), dim=0).T.to(dataset['paper', 'ref', 'paper'].edge_index.dtype)
    del dataset['paper', 'cite', 'paper']
    del dataset['paper', 'ref', 'paper']
    dataset = T.ToSparseTensor(remove_edge_index=False)(dataset)

    node_types, edge_types = dataset.metadata()

    x_dict, adj_t_dict = {}, {}
    for node_type in node_types:
        x_dict[node_type] = dataset[node_type].x
    for edge_type in edge_types:

        adj_t_dict[edge_type] = asymmetric_gcn_norm(dataset[edge_type].adj_t)

    dataset.x_dict = x_dict
    dataset.adj_t_dict = adj_t_dict
    dataset.node_types = node_types
    dataset.edge_types = edge_types

    return dataset
