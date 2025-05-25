import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from locale import currency
import math
from pickle import FALSE
from re import L
from unicodedata import name
import numpy as np
import random
import torch
import torch.nn.functional as F
import networkx as nx
import torch_geometric
from scatter_letters import sl

import seaborn as sns
from sklearn.manifold import TSNE

from scipy.spatial.distance import cdist

from torch_geometric.utils import to_dense_adj, dense_to_sparse, get_laplacian
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Reddit2
from torch_geometric.datasets import Yelp
from torch_geometric.datasets import AmazonProducts
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import AMiner
from torch_geometric.datasets import OGB_MAG
from sklearn.neighbors import NearestNeighbors

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import scipy.io

import os
import json
import scipy as sp
from scipy.sparse import csr_matrix
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
#import tensorflow as tf
import argparse
import time
from scipy.spatial.distance import pdist
from itertools import chain
import os.path as osp
import copy

import pygsp
from ConsistentHashing import *
import UGC_binwidth_finder
import spectral_properties

import sys
sys.path.append("/home/mohit/projects/heteroGC/coarGC/SCAL/Scal/GCN")

import sys
sys.path.append("/home/mohit/projects/heteroGC/coarGC/heterogenous_utils")

import fgc_f
from SCAL.Scal.GCN import train as scal
import pandas as pd

from torch_geometric.utils import to_undirected
from torch_geometric.utils import dense_to_sparse
import utils
import UGC_bin_widths
from models.heterophilic_models import parse as heterophilic_parse
import train
import train_coarsen
from heterophlic_data import dataset as more_heterophilic_datasets
from heterophlic_data import data_utils as heterophilic_data_utils
from torch_scatter import scatter
import torch_geometric.transforms as T
import pycuda.driver as cuda
import pycuda.autoinit
from scipy.special import erf
from sklearn.metrics.pairwise import euclidean_distances

##Imports for heterogenous datasets
from heterogenous_utils.hmain import run_imdb, run_dblp, run_acm
from heterogenous_utils.hmain2 import run_hetero

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class MyDataset(Dataset):
    def __init__(self, data, num_classes):
        super().__init__()
        self.data_list = [data]  # Store as a list for indexing
        self.n_classes = num_classes

    def len(self):
        return len(self.data_list)  # Correct implementation

    def get(self, idx):
        if idx < 0 or idx >= len(self.data_list):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.data_list)}")
        return self.data_list[idx]

    def indices(self):
        return range(self.len())
    
def validate_projection_proximity_torch(projected_vals, embeddings, num_of_projectors, num_pairs=1000, epsilons=torch.linspace(0.001, 0.01, 3)):
    device = embeddings.device if isinstance(embeddings, torch.Tensor) else torch.device("cpu")

    projected_vals = projected_vals.float().to(device)
    embeddings = embeddings.float().to(device)
    N = projected_vals.shape[0]
    results = []

    pairs = torch.randint(0, N, (num_pairs, 2), device=device)

    for eps in epsilons:
        # Get projected values for both nodes in each pair
        hx = projected_vals[pairs[:, 0]]
        hy = projected_vals[pairs[:, 1]]

        # Empirical: check how many have |h(x) - h(y)| <= eps
        diff = torch.abs(hx - hy)
        
        empirical_prob = (diff <= eps).float().mean().item()

        # Compute Euclidean distance between node embeddings
        x = embeddings[pairs[:, 0]]
        y = embeddings[pairs[:, 1]]
        dists = torch.norm(x - y, dim=1)

        # Avoid division by zero
        valid = dists > 0
        safe_dists = dists[valid]
        if safe_dists.numel() > 0:
            theor_vals = eps.item() / (torch.sqrt(2 * torch.tensor(num_of_projectors, dtype=torch.float32, device=device)) * safe_dists)
            theoretical_prob = erf(theor_vals.cpu().numpy()).mean()
        else:
            theoretical_prob = 0.0

        results.append((eps.item(), empirical_prob, theoretical_prob))

    return torch.tensor(results)
    



def validate_projection_proximity(projected_vals, embeddings, num_of_projectors, num_pairs=1000, epsilons=np.linspace(0.1, 3.0, 20)):
    projected_vals = projected_vals.float().to('cpu')
    embeddings = embeddings.float().to('cpu')

    projected_vals = np.array(projected_vals)
    N = len(projected_vals)
    results = []

    # Randomly sample node pairs
    pairs = np.random.choice(N, size=(num_pairs, 2), replace=True)

    for eps in epsilons:
        count = 0
        theor_probs = []
        for i, j in pairs:
            hx, hy = projected_vals[i], projected_vals[j]
            dist = np.linalg.norm(embeddings[i] - embeddings[j])

            # empirical check
            if np.abs(hx - hy) <= eps:
                count += 1

            # theoretical probability for this pair
            if dist > 0:
                theor_probs.append(erf(eps / (np.sqrt(2 * num_of_projectors) * dist)))

        empirical_prob = count / num_pairs
        theoretical_prob = np.mean(theor_probs)
        results.append((eps, empirical_prob, theoretical_prob))

    return np.array(results)
    

def parse_args():
    parser = argparse.ArgumentParser(description='Coarsened Graph Training')
    parser.add_argument('--full_dataset',type=bool,required=False,default=False,help="Checking accuracy on original dataset.")
    parser.add_argument('--dataset',type=str,required=False,default='cora',help="Dataset name")
    parser.add_argument('--edge_index_path',type=str,required=False,default='None',help="Give path of edge index file")
    parser.add_argument('--label_path',type=str,required=False,default='None',help="Give path of label file")
    parser.add_argument('--node_feat_path',type=str,required=False,default='None',help="Give path of node feature file")
    parser.add_argument('--add_adj_to_node_features',type=bool,required=False,default=True,help="Adding Adjacency matrix one hot vectors in node features")
    parser.add_argument('--epochs',type=int,required=False, default=500,help="Number of epochs to train the coarsened graph")
    parser.add_argument('--lr',type=float,required=False,default=0.007,help="Learning Rate")
    parser.add_argument('--decay',type=float,required=False,default=0.0005,help="Learning Rate Decay")
    parser.add_argument('--seed',type=int,required=False,default=42,help="Seed")
    parser.add_argument('--ratio',type=int,required=False,default=50,help='reduction ratio list, example (30,50,70)')
    parser.add_argument('--dataset_not_in_torch_geometric',type=bool,required=False,default=False,help='Turn true if your dataset is not in the torch geometric. We will create geometric dataset first')
    parser.add_argument('--num_classes',type=int,required=False,default=-1,help='You should give value here if new instance of torch_geometric dataset is being created.')
    parser.add_argument('--number_of_projectors',type=int,required=False,default=500,help='Total number of projectors we want while Doing LSH.')
    parser.add_argument('--out_of_sample',type=int,required=False,default=0,help='UGC2.0 should be supporting this. out_of_sample in percent (from 0 to 1) of dataset')
    parser.add_argument('--feature_size',type=int,required=False,default=-1,help='You should give value here if new instance of torch_geometric dataset is being created.')
    parser.add_argument('--hash_function',type=str,required=False,default='dot',help='Hash Function choices 1). Dot 2). L1-norm 3). L2-norm')
    parser.add_argument('--projectors_distribution',type=str,required=False,default='uniform',help='1). uniform 2). normal. coming soon.... 3). VAEs in this case need to give learned mean and sigma also.')
    parser.add_argument('--random_coarsening',type=bool,required=False,default=False,help='True for random coarsening.')
    parser.add_argument('--visualize_graph',type=bool,required=False,default=False,help='True for graph visualization.')
    parser.add_argument('--induce_adverserial_edges',type=bool,required=False,default=False,help='True for adding noise in the graph edges.')
    parser.add_argument('--tsne_visualization',type=bool,required=False,default=False,help='tsne_visualization')
    parser.add_argument('--calculate_spectral_errors',type=bool,required=False,default=False,help='calculate_spectral_errors')
    parser.add_argument('--hidden_units',type=int,required=False,default=64,help='hidden_units of GCN')
    parser.add_argument('--gsp_graphs',type=bool,required=False,default=False,help='making graphs from Graph Signal Processing lib')
    parser.add_argument('--scatter_alphabets',type=str,required=False,default="None",help='making graphs from names and alphabets')
    parser.add_argument('--alpha',type=float,required=False,default=0.1,help='Heterophilic factor')
    parser.add_argument('--model_type',type=str,required=False,default='gcn',help='model type')
    parser.add_argument('--start_coarsen_method',type=str,required=False,default='UGC',help='start coarsen method')
    parser.add_argument('--multiple_ugc',type=bool,required=False,default=False,help='use only UGC for multiple hashings')
    parser.add_argument('--train_coarsen',type=bool,required=False,default=False,help='use only UGC for multiple hashings')
    parser.add_argument('--wwl_features',type=bool,required=False,default=False,help='augment with WWL features?')

    #######################################################################

    #### heterophilic parser
    
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--method', '-m', type=str, default='link')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    parser.add_argument('--gcn2_alpha', type=float, default=.1,
                        help='alpha for gcn2')
    parser.add_argument('--theta', type=float, default=.5,
                        help='theta for gcn2')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],
                        help='jumping knowledge type')
    parser.add_argument('--rocauc', action='store_true',
                        help='set the eval function to rocauc')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers in h2gcn')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--adam', action='store_true', help='use adam instead of adamW')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')
    parser.add_argument('--sampling', action='store_true', help='use neighbor sampling')
    parser.add_argument('--inner_activation', action='store_true', help='Whether linkV3 uses inner activation')
    parser.add_argument('--inner_dropout', action='store_true', help='Whether linkV3 uses inner dropout')
    parser.add_argument("--SGD", action='store_true', help='Use SGD as optimizer')
    parser.add_argument('--link_init_layers_A', type=int, default=1)
    parser.add_argument('--link_init_layers_X', type=int, default=1)

    ######################################################################

    ## Heterogenous Parsers
    parser.add_argument('--hetero_r', type=float, required=False, default=1.0, help="Coarsening ratio for heterogenous datasets")
    parser.add_argument('--hgnn', type=str, default='HeteroSGC', required=False, help="Heterogenous GNN model")
    parser.add_argument('--nevals', type=int, default=1, required=False, help='Number of evaluations per run for HGNN training')
    

    # args = parser.parse_args()
    return parser

def get_key(val, g_coarsened):
  KEYS = []
  for key, value in g_coarsened.items():
    if val == value:
      KEYS.append(key)
  return len(KEYS),KEYS

def update_spectral_properties(method, dataset, coar_ratio, he_error, re_construct_error, diri_energy, eigen_error, path="results_spectral_properties.csv"):
    if isinstance(he_error, torch.Tensor):
        he_error = round(he_error.item(), 4)
    else:
        he_error = round(float(he_error), 4)

    if isinstance(re_construct_error, torch.Tensor):
        re_construct_error = round(re_construct_error.item(), 4)
    else:
        re_construct_error = round(float(re_construct_error), 4)

    if isinstance(diri_energy, torch.Tensor):
        diri_energy = round(diri_energy.item(), 4)
    else:
        diri_energy = round(float(diri_energy), 4)

    if isinstance(eigen_error, torch.Tensor):
        eigen_error = round(eigen_error.item(), 4)
    else:
        eigen_error = round(float(eigen_error), 4)

    # if isinstance(eigen_error, torch.Tensor):
    #     eigen_error = eigen_error.detach().cpu().numpy()
    # eigen_error_str = ",".join(f"{v:.4f}" for v in eigen_error)

    row = {
        "method": method,
        "dataset": dataset,
        "coar_ratio": coar_ratio,
        "he_error": he_error,
        "re_construct_error": re_construct_error,
        "diri_energy": diri_energy,
        "eigen_error": [eigen_error]
    }

    # Append or create DataFrame
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)
    print(f"Spectral Results saved: {method} on {dataset}")

def update_table_accuracy(method, dataset, acc, model, wcoar=False, path="results_table_accuracy_homophilic.csv"):
    row = {"method": method, "dataset": dataset, "acc": acc, "model": model, "base_dataset": wcoar}

    # Check if file exists and is not empty
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True) #Hi
    else:
        df = pd.DataFrame([row])
    
    df.to_csv(path, index=False)
    print(f"Results saved: {method} on {dataset} with {model}")


def update_table(method, dataset, value, path="results_table.csv"):
    row = {"method": method, "dataset": dataset, "value": value}

    # Check if file exists and is not empty
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    df.to_csv(path, index=False)
    print(f"Results saved: {method} on {dataset} = {value}")


if __name__ == "__main__":
    time1 = time.time()
    current_parser = parse_args()
    args = current_parser.parse_args()
    utils.fix_seeds(args.seed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    for i in range(cuda.Device.count()):
        dev = cuda.Device(i)
        ctx = dev.make_context()
        free, total = cuda.mem_get_info()
        print(f"Device {i}: {dev.name()} | Free: {free // (1024**2)} MB / Total: {total // (1024**2)} MB")
        ctx.pop()

####################################### LARGE DATASETS##################
    if args.dataset in  ['reddit22222']:
        large_datasets = True
        
        if args.dataset == 'ogbn_products':
            dataset_name = 'ogbn-products'
        elif args.dataset == 'ogbn_arxiv':
            dataset_name = 'ogbn-arxiv'
        else:
            dataset_name = args.dataset

        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_name)
        path = "large_datasets"

        data_full, data = utils.read_data(dataset_name, path)


        # utils.run_gcn(data, device)

        edge_index = torch.LongTensor(data_full.adj.nonzero())
        if sp.sparse.issparse(data_full.features):
            x = torch.FloatTensor(data_full.features.todense()).float()
        else:
            x = torch.FloatTensor(data_full.features).float()
        y = torch.LongTensor(data_full.labels)

        train_mask = torch.tensor(data_full.idx_train)
        val_mask =  torch.tensor(data_full.idx_val)
        test_mask =  torch.tensor(data_full.idx_test)

        data = Data(x=x, edge_index=edge_index, y=y)

        print(len(train_mask), len(test_mask), len(val_mask))

        data.train_mask = torch_geometric.utils.index_to_mask(train_mask, size=data.x.shape[0])
        data.val_mask = torch_geometric.utils.index_to_mask(val_mask, size=data.x.shape[0]) 
        data.test_mask = torch_geometric.utils.index_to_mask(test_mask, size=data.x.shape[0])

        num_classes = len(torch.unique(data.y))
        dataset = MyDataset(data, num_classes)
        
        feature_size = data.x.shape[1]
        num_nodes = data.x.shape[0]
        
    # elif args.dataset == 'ogbn_products':
    #     large_datasets = True
    #     dataset_name = 'ogbn-products'
    #     path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_name)
    #     path = "large_datasets"
        
    #     data_full, data = utils.read_data(dataset_name, path)

    #     ## to run ogbn_products on gcn
    #     # utils.run_gcn(data, device)

    #     edge_index = torch.LongTensor(data_full.adj.nonzero())
    #     if sp.sparse.issparse(data_full.features):
    #         x = torch.FloatTensor(data_full.features.todense()).float()
    #     else:
    #         x = torch.FloatTensor(data_full.features).float()
    #     y = torch.LongTensor(data_full.labels)


    #     train_mask = torch.tensor(data_full.idx_train)
    #     val_mask =  torch.tensor(data_full.idx_val)
    #     test_mask =  torch.tensor(data_full.idx_test)

    #     data = Data(x=x, edge_index=edge_index, y=y)

    #     print(len(train_mask), len(test_mask), len(val_mask))

    #     data.train_mask = torch_geometric.utils.index_to_mask(train_mask, size=data.x.shape[0])
    #     data.val_mask = torch_geometric.utils.index_to_mask(val_mask, size=data.x.shape[0]) 
    #     data.test_mask = torch_geometric.utils.index_to_mask(test_mask, size=data.x.shape[0])

    #     num_classes = len(torch.unique(data.y))
    #     dataset = MyDataset(data, num_classes)

    #     feature_size = data.x.shape[1]
    #     num_nodes = data.x.shape[0]

    # elif args.dataset in  ['flickr', 'reddit']:
    #     large_datasets = True
    #     dataset_name = args.dataset
    #     path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_name) 
    #     path = "large_datasets"

    #     data_full, data = utils.read_data(dataset_name, path)

    #     # utils.run_gcn(data, device)

    #     edge_index = torch.LongTensor(data_full.adj.nonzero())
    #     if sp.sparse.issparse(data_full.features):
    #         x = torch.FloatTensor(data_full.features.todense()).float()
    #     else:
    #         x = torch.FloatTensor(data_full.features).float()
    #     y = torch.LongTensor(data_full.labels)


    #     train_mask = torch.tensor(data_full.idx_train)
    #     val_mask =  torch.tensor(data_full.idx_val)
    #     test_mask =  torch.tensor(data_full.idx_test)

    #     data = Data(x=x, edge_index=edge_index, y=y)

    #     print(len(train_mask), len(test_mask), len(val_mask))

    #     data.train_mask = torch_geometric.utils.index_to_mask(train_mask, size=data.x.shape[0])
    #     data.val_mask = torch_geometric.utils.index_to_mask(val_mask, size=data.x.shape[0]) 
    #     data.test_mask = torch_geometric.utils.index_to_mask(test_mask, size=data.x.shape[0])

    #     num_classes = len(torch.unique(data.y))
    #     dataset = MyDataset(data, num_classes)

    #     feature_size = data.x.shape[1]
    #     num_nodes = data.x.shape[0]

    # elif args.dataset == 'reddit':
    #     large_datasets = True
    #     dataset_name = 'reddit'
    #     path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_name)
    #     path = "large_datasets"

    #     data_full, data = utils.read_data(dataset_name, path)

    #     # utils.run_gcn(data, device)

    #     edge_index = torch.LongTensor(data_full.adj.nonzero())
    #     if sp.sparse.issparse(data_full.features):
    #         x = torch.FloatTensor(data_full.features.todense()).float()
    #     else:
    #         x = torch.FloatTensor(data_full.features).float()
    #     y = torch.LongTensor(data_full.labels)


    #     train_mask = torch.tensor(data_full.idx_train)
    #     val_mask =  torch.tensor(data_full.idx_val)
    #     test_mask =  torch.tensor(data_full.idx_test)

    #     data = Data(x=x, edge_index=edge_index, y=y)

    #     print(len(train_mask), len(test_mask), len(val_mask))

    #     data.train_mask = torch_geometric.utils.index_to_mask(train_mask, size=data.x.shape[0])
    #     data.val_mask = torch_geometric.utils.index_to_mask(val_mask, size=data.x.shape[0]) 
    #     data.test_mask = torch_geometric.utils.index_to_mask(test_mask, size=data.x.shape[0])

    #     num_classes = len(torch.unique(data.y))
    #     dataset = MyDataset(data, num_classes)

    #     feature_size = data.x.shape[1]
    #     num_nodes = data.x.shape[0]

    elif args.dataset == "yelp":
        large_datasets = True
        dataset_name = 'yelp'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_name)
        dataset = Yelp(path, transform=T.NormalizeFeatures())
        data = dataset[0]

        # utils.run_gcn(data, device)

####################################### HOMOPHILIC DATASETS#####

    elif args.dataset == 'flickr':
        large_datasets = True
        dataset = Flickr(root = 'data/Flickr')
        data = dataset[0]

        #### why working good for this split ?? (by default less number of train nodes ???)
        data = utils.rand_train_test_idx(data, train_prop=.6, valid_prop=.2)

    elif args.dataset == 'reddit':
        large_datasets = True
        dataset = Reddit2(root = 'data/Reddit')
        data = dataset[0]

        #### why working good for this split ?? (by default less number of train nodes ???)
        data = utils.rand_train_test_idx(data, train_prop=.6, valid_prop=.2)

    elif args.dataset == 'yelp':
        large_datasets = True
        dataset = Yelp(root = 'data/Yelp')
        data = dataset[0]

        #### why working good for this split ?? (by default less number of train nodes ???)
        data = utils.rand_train_test_idx(data, train_prop=.6, valid_prop=.2)

    elif args.dataset == 'citeseer':
        large_datasets = False
        dataset = Planetoid(root = 'data/CiteSeer', name = 'CiteSeer')
        data = dataset[0]

        #### why working good for this split ?? (by default less number of train nodes ???)
        data = utils.rand_train_test_idx(data, train_prop=.6, valid_prop=.2)

    elif args.dataset == 'cora':
        large_datasets = False
        dataset = Planetoid(root = 'data/Cora', name = 'Cora')
        data = dataset[0]

        #### why working good for this split ?? (by default less number of train nodes ???)
        data = utils.rand_train_test_idx(data, train_prop=.6, valid_prop=.2)

    elif args.dataset == 'pubmed':
        large_datasets = False
        dataset = Planetoid(root = 'data/PubMed', name = 'PubMed')
        data = dataset[0]

        #### why working good for this split ?? (by default less number of train nodes ???)
        data = utils.rand_train_test_idx(data, train_prop=.6, valid_prop=.2)

    elif args.dataset == 'physics':
        large_datasets = False
        dataset = Coauthor(root = 'data/Physics', name = 'Physics')
        print(dataset)
        data = dataset[0]

        #### why working good for this split ??
        data = utils.rand_train_test_idx(data, train_prop=.6, valid_prop=.2)

        dataset = [Data(x=data.x, y=data.y, edge_index=data.edge_index, train_mask = data.train_mask, val_mask = data.val_mask, test_mask = data.val_mask)]
        num_classes = len(torch.unique(data.y))
        feature_size = data.x.shape[1]
        num_nodes = data.x.shape[0]

    elif args.dataset == 'dblp':
        large_datasets = False
        dataset = CitationFull(root = 'data/DBLP', name = 'DBLP')
        data = dataset[0]

        data = utils.rand_train_test_idx(data, train_prop=.6, valid_prop=.2)
        
        dataset = [Data(x=data.x, y=data.y, edge_index=data.edge_index, train_mask = data.train_mask, val_mask = data.val_mask, test_mask = data.val_mask)]
        num_classes = len(torch.unique(data.y))
        feature_size = data.x.shape[1]
        num_nodes = data.x.shape[0]

    

    elif args.dataset == 'cs':
        large_datasets = False
        dataset = Coauthor(root = 'data/CS', name = 'CS')
        data = dataset[0]

        data = utils.rand_train_test_idx(data, train_prop=.6, valid_prop=.2)

        dataset = [Data(x=data.x, y=data.y, edge_index=data.edge_index, train_mask = data.train_mask, val_mask = data.val_mask, test_mask = data.val_mask)]
        num_classes = len(torch.unique(data.y))
        feature_size = data.x.shape[1]
        num_nodes = data.x.shape[0]
####################################### HETROPHILIC DATASETS#####

    elif args.dataset in ['twitch-e', 'wiki', 'deezer-europe', 'ogbn-proteins', 'yelp-chi' ,
                           'fb100', 'chameleon', 'squirrel', 'wisconsin', 'cornell', 'film', 'texas',
                             'pokec', 'arxiv-year', 'snap-patents', 'genius', 'twitch-gamer', 'ogbn-arxiv', 'ogbn-products']:
        
        if args.dataset in ['genius', 'pokec', 'twitch-gamer', 'deezer-europe', 'ogbn-arxiv', 'ogbn-products']:
            large_datasets = True
        else:
            large_datasets = False
        
        dataset = more_heterophilic_datasets.load_nc_dataset(args.dataset, args.sub_dataset)
        
        if len(dataset.label.shape) == 1:
            dataset.label = dataset.label.unsqueeze(1)
        # dataset.label = dataset.label.to(device)
        # dataset.label = dataset.label

        # if args.rand_split or args.dataset in ['ogbn-proteins', 'wiki']:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                        for _ in range(args.runs)]
        # else:
        #     split_idx_lst = heterophilic_data_utils.load_fixed_splits(args.dataset, args.sub_dataset)

        if args.dataset == 'ogbn-proteins':
            if args.method == 'mlp' or args.method == 'cs':
                dataset.graph['node_feat'] = scatter(dataset.graph['edge_feat'], dataset.graph['edge_index'][0],
                    dim=0, dim_size=dataset.graph['num_nodes'], reduce='mean')
            else:
                dataset.graph['edge_index'] = heterophilic_data_utils.to_sparse_tensor(dataset.graph['edge_index'],
                    dataset.graph['edge_feat'], dataset.graph['num_nodes'])
                dataset.graph['node_feat'] = dataset.graph['edge_index'].mean(dim=1)
                dataset.graph['edge_index'].set_value_(None)
            dataset.graph['edge_feat'] = None

        if not args.directed and args.dataset != 'ogbn-proteins':
            dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

        # dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        #     dataset.graph['edge_index'].to(
        #         device), dataset.graph['node_feat'].to(device)

        # dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        #     dataset.graph['edge_index'], dataset.graph['node_feat']
        
        train_loader, subgraph_loader = None, None

        split_idx = split_idx_lst[0]

        data = Data(x = dataset.graph['node_feat'], edge_index = dataset.graph['edge_index'], y = dataset.label)
        data.train_mask = torch_geometric.utils.index_to_mask(torch.tensor(split_idx['train']), size=data.x.shape[0])
        data.val_mask = torch_geometric.utils.index_to_mask(torch.tensor(split_idx['valid']), size=data.x.shape[0]) 
        data.test_mask = torch_geometric.utils.index_to_mask(torch.tensor(split_idx['test']), size=data.x.shape[0])

        num_classes = len(torch.unique(data.y))
        dataset = MyDataset(data, num_classes)
        
        feature_size = data.x.shape[1]
        num_nodes = data.x.shape[0]
        print(dataset[0])

####################################### HETEROGENEOUS DATASETS#########################
    elif args.dataset in ['imdb', 'hdblp', 'acm']:
        run_hetero(args.dataset, args.start_coarsen_method, args.hetero_r, args.hgnn, args.nevals, 'cuda:1')
        exit(1)
    
        
    else:
        print("are you sure about the dataset name ??")
        exit(1)



##################################### training on full dataset ##############
    feature_size = data.x.shape[1]
    num_nodes = data.x.shape[0]
    num_classes = len(torch.unique(data.y))
    print("num_classes", num_classes)

    if args.full_dataset == True:
        full_train_time0 = time.time()
        data = data.to(device=device)

        if args.dataset in  ['reddit22222']:
            utils.run_gcn(data, device)
        
        elif args.dataset in ['twitch-e', 'wiki', 'deezer-europe', 'ogbn-proteins', 'yelp-chi' ,
                           'fb100', 'chameleon', 'squirrel', 'wisconsin', 'cornell', 'film', 'texas',
                             'pokec', 'arxiv-year', 'snap-patents', 'genius', 'twitch-gamer', 'ogbn-arxiv', 'ogbn-products']:
        
            model = heterophilic_parse.parse_method(args, data, num_nodes, dataset.num_classes, feature_size, device)
            # print(data.y.shape)
            train_acc, val_acc, test_acc = train.train_heterophilic_models(model, args, data, device)

            if args.dataset == 'fb100':
                dataset_name =  args.sub_dataset
            else:
                dataset_name =  args.dataset
            
            update_table_accuracy('Base', dataset_name, test_acc, args.method, path="results_table_accuracy_heterophilic.csv")
        else:
            incorrect_index, acc = train.train_on_UGC_models(data, num_classes,feature_size,args.hidden_units,args.lr,args.decay,args.epochs, device, args.method)
            update_table_accuracy('Base', args.dataset, acc, args.method, path="results_table_accuracy_homophilic_mohit.csv")


        # exit(1)

    # data.y = data.y.squeeze()
    # incorrect_index = train.train_on_UGC_models(data, dataset.num_classes,feature_size,args.hidden_units,args.lr,args.decay,args.epochs, device, args.model_type)
        

#############################################################################

    if large_datasets == False and args.start_coarsen_method == "UGC":
        print("making augmented feature matrix")
        data.x = (1-args.alpha)*data.x
        g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0]
        g_adj = args.alpha*g_adj
        
        data.x = torch.cat((data.x, g_adj), dim = 1)
        feature_size = data.x.shape[1]# + data.num_nodes

    if args.start_coarsen_method == "UGC" and args.multiple_ugc:
        print("all first UGC")
        no_of_pojectors = 1000 #int(np.ceil(np.log(data.x.shape[0])))
        hasher = ConsistentHashing(input_dim=data.x.shape[1], proj_dim=no_of_pojectors)

        ugc_multiple_hashing0 = time.time() #time_taken_initially
        
        valid_ratios_list = [0.6, 0.55, 0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
        # valid_ratios_list = [0.5]
        
        for coarsening_ratio in valid_ratios_list:
            ugc_bin_time0 = time.time()
            bin_width, _ = UGC_binwidth_finder.find_Binwidth(data, coarsening_ratio)
            ugc_bin_time1 = time.time()
            # print("bin width finding time ", ugc_bin_time1 - ugc_bin_time0)

            Bin_values = hasher.UGC_hashed_values(data, function='dot')
            summary_dict = hasher.UGC_partition([bin_width], Bin_values)
        
            supernode_dict = summary_dict[bin_width]
            
            rr = 1 - len(supernode_dict.keys())/data.x.shape[0]
            reduced_percentage = rr
            
            #supernode_dict.keys()--> new supernode formed in coarsened graph
            print(f'Graph reduced by: {reduced_percentage} percent. \n Now we have {len(supernode_dict.keys())} supernodes; Starting nodes were: {data.x.shape[0]}')
      

            C = torch.zeros(data.x.shape[0] , len(supernode_dict.keys()))
            zero_list = torch.ones(len(supernode_dict.keys()), dtype=torch.bool)

            for super_idx, node_list in enumerate(supernode_dict.values()):
                for node in node_list:
                    C[node][super_idx] = 1
                    zero_list[super_idx] = zero_list[super_idx] and (not (data.train_mask)[node])
        
        ugc_multiple_hashing1 = time.time() #time_taken_after_reduction
        
        #ugc_multiple_hashing_total_time -------> Total time taken
        ugc_multiple_hashing_total_time = ugc_multiple_hashing1 - ugc_multiple_hashing0
        print("Total time taken by UGC ", ugc_multiple_hashing_total_time)

        update_table('ugc', args.dataset, ugc_multiple_hashing_total_time)
        exit(1)
    
    elif args.start_coarsen_method == "UGC":
        print("only first UGC")
        no_of_pojectors = 1000#int(np.ceil(np.log(data.x.shape[0])))

        intial_coarse0 = time.time()
        coarsening_ratio = 0.45
        data = data.to(device=device)
         
        if args.wwl_features == True:
            ## Considering WWL feature to augment feature vector
            hasher = ConsistentHashing(input_dim=2*data.x.shape[1], proj_dim=no_of_pojectors)
            Bin_values = hasher.UGC_hashed_values_with_WWL(data, function='dot')  #function_call: from ConistentHashing.py
            bin_width, _ = UGC_binwidth_finder.find_Binwidth_WWL(data, coarsening_ratio)  #got bin_width
        else:
            hasher = ConsistentHashing(input_dim=data.x.shape[1], proj_dim=no_of_pojectors)
            Bin_values = hasher.UGC_hashed_values(data, function='dot')  #function_call: from ConistentHashing.py
            bin_width, _ = UGC_binwidth_finder.find_Binwidth(data, coarsening_ratio)  #got bin_width

            # bin_width = 0.00002

        summary_dict = hasher.UGC_partition([bin_width], Bin_values)

        supernode_dict = summary_dict[bin_width]
        
        #reduced_perecntage --> graph reduced by
        rr = 1 - len(supernode_dict.keys())/data.x.shape[0]
        reduced_percentage=rr*100 
        
        #supernode_dict.keys()--> new supernode formed in coarsened graph
        #data.x.shape[0] --> initial nodes in original graph
        print(f'Graph reduced by: {reduced_percentage} percent.\n Now we have {len(supernode_dict.keys())} supernode, starting nodes were: {data.x.shape[0]}')
        intial_coarse1 = time.time()

    elif args.start_coarsen_method in ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron']:
        print("going for other SOTA (SCAL) coarsening methods")
        # valid_ratios_list = [0.6, 0.55, 0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
        valid_ratios_list = [0.6, 0.35, 0.10]
        # valid_ratios_list = [0.5]
        scal_time0 = time.time()
        print(dataset[0].x.shape)
        for coarsening_ratio in valid_ratios_list:
            data_coarsen, P = scal.scal_coarsen(dataset, coarsening_ratio, args.start_coarsen_method, args.dataset)  
        scal_time1 = time.time()
        scal_multiple_hashing_total_time = scal_time1 - scal_time0
        print("time taken by scal methods ", scal_multiple_hashing_total_time)

        # update_table(args.start_coarsen_method, args.dataset, scal_multiple_hashing_total_time, "scal_methods_time_results.csv")
        update_table(args.start_coarsen_method, args.dataset, scal_multiple_hashing_total_time)
        exit(1)

        zero_list = torch.ones(data_coarsen.x.shape[0], dtype=torch.bool)
        zero_list[data_coarsen.train_mask] = False
        data_coarsen.y = data_coarsen.y.unsqueeze(1)
        # data.y = data.y.unsqueeze(1)
        data = data.to(device=device)
        data_coarsen = data_coarsen.to(device=device)

        data.y = data.y.squeeze() 
        data_coarsen.y = data_coarsen.y.squeeze() 
        
        # model = heterophilic_parse.parse_method(args, data_coarsen, num_nodes, dataset.num_classes, feature_size, device)
        # train_coarsen.train_heterophilic_models(model, args, data, data_coarsen, zero_list, device)

    elif args.start_coarsen_method == "fgc":
        fgc_0 = time.time()
        cor_feat, P = fgc_f.fgc(dataset, 0.5)
        fgc_1 = time.time()

        fgc_total_time = fgc_1 - fgc_0
        update_table(args.start_coarsen_method, args.dataset, fgc_total_time)

        g_coarse_adj = P.T @ to_dense_adj(dataset[0].edge_index)[0].to(P.device) @ P

        edge_index_corsen = dense_to_sparse(g_coarse_adj)[0]

        Y = np.array(dataset[0].y.cpu())
        Y = utils.one_hot(Y, num_classes).to(device)
        Y[~data.train_mask] = torch.Tensor([0 for _ in range(num_classes)]).to(device = device)

        print(Y.shape, data.y.shape, P.shape)
        labels_coarse = torch.argmax(torch.sparse.mm(torch.t(P).double() , Y.double()).double() , 1).to(device)

        data_coarsen = Data(x=cor_feat, edge_index = edge_index_corsen, y = labels_coarse)
        zero_list = (P.T @ F.one_hot(dataset[0].train_mask.to(torch.int64).to(P.device), 2).to(torch.float))[:, 1] > 0
        zero_list = ~zero_list
        print(zero_list.shape, zero_list.count_nonzero())


        ####???????????????????????
        

    else:
        ## All the nodes are supernodes at this time if we use this forward function to coarsen down at start also
        ## ha_UGC
        intial_coarse0 = time.time()
        no_of_pojectors = 1000     #int(np.ceil(np.log(data.x.shape[0])))
        hasher = ConsistentHashing(input_dim=data.x.shape[1], proj_dim=no_of_pojectors)
        supernode_dict = hasher.forward(data)
        intial_coarse1 = time.time()


    ## only run for consistent hashing
    if (args.start_coarsen_method == "UGC" and args.multiple_ugc != True) or args.start_coarsen_method == 'ha-UGC':
        consistentHash_multiple_hashing0 = time.time()

        ### pre preprocessing before mapping supernodes to ring
        supernode_dict, _, ordered_mean_values = hasher.rank_and_sort_supernodes(supernode_dict, data.x)
        
        ############# checking the probabilistic measure  (theorem Plots)
        # results = validate_projection_proximity_torch(torch.tensor(ordered_mean_values), data.x, no_of_pojectors, num_pairs=1000)

        # print(results[:, 0], results[:, 1])
        # # Plot
        # plt.figure(figsize=(8, 5))

        # ### calculated values just to save results
        # # plt.plot(torch.tensor([0.0010, 0.0055, 0.0100]), torch.tensor([0.2040, 0.7030, 0.8770]), label='Cora', marker='o')
        # # plt.plot(torch.tensor([0.0010, 0.0055, 0.0100]), torch.tensor([0.3230, 0.9840, 1.0000]), label='Citeseer', linestyle='--', marker='o')
        # # plt.plot(torch.tensor([0.0010, 0.0055, 0.0100]), torch.tensor([0.4340, 0.9980, 1.0000]), label='DBLP', linestyle='-.', marker='o')

        # plt.plot(results[:, 0], results[:, 1], label='Empirical', marker='o')

        # # plt.plot(results[:, 0], results[:, 2], label='Theoretical (erf)', linestyle='--')
        # plt.xlabel(r'$\varepsilon$', fontsize=20)
        # plt.ylabel(r'$\Pr[|z| \leq \varepsilon]$', fontsize=20)
        # plt.title('Projection Proximity Validation')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.savefig("probablistic_measure.png")
        # plt.show()

        # exit(1)
        ##################

        # list_num_supernodes_ratio = [0.55, 0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
        list_num_supernodes_ratio = [0.5]
        for ratio in list_num_supernodes_ratio:
            num_supernodes = int((ratio) * data.x.shape[0])
            time0 = time.time()
            C, supernode_dict, zero_list = hasher.coarsen_ring_parallel_gpu(supernode_dict, data, num_supernodes)
        consistentHash_multiple_hashing1 = time.time()

        consistentHash_multiple_hashing_total_time = consistentHash_multiple_hashing1 - consistentHash_multiple_hashing0 + intial_coarse1 - intial_coarse0
        # update_table('new_UGC', args.dataset, consistentHash_multiple_hashing_total_time)
        # exit(1)

        print("total time taken by consistent hash for this coarsening list ", consistentHash_multiple_hashing_total_time)

    # node_counts = np.array(C.sum(axis=0))

    # nonzero_indices = np.where(node_counts > 0)[0]

    # # Sort and select top populated
    # sorted_indices = nonzero_indices[np.argsort(node_counts[nonzero_indices])][::-1]

    # # Take only as many as exist (at most 50)
    # top_k = min(500, len(sorted_indices))
    # top_indices = sorted_indices[:top_k]
    # top_counts = node_counts[top_indices]

    # # Plot
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.bar(np.arange(top_k), top_counts, color='cornflowerblue', edgecolor='black')
    # ax.set_xticks(np.arange(top_k))
    # ax.set_xticklabels(top_indices, rotation=90)
    # ax.set_xlabel('Supernode Index (Top Populated)')
    # ax.set_ylabel('Number of Nodes')
    # ax.set_title(f'Top {top_k} Most Populated Supernodes')
    # ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # plt.tight_layout()
    # if args.multiple_ugc != True:
    #     plt.savefig("balanced mapping distribution.jpg")
    # else:
    #     plt.savefig("mapping distribution.jpg")
    # plt.show()
    # exit(1)
    ################################## We have a partition matrix now. All the code below this will handle node classification accuracy ######
    
    ### error
    # data = data.to(device=device)
    ## --------------------------------------------------
    if args.start_coarsen_method == 'UGC' or args.start_coarsen_method == 'ha-UGC':
        C_diag = torch.sum(C, dim=0)
        P = torch.sparse.mm(C,(torch.diag(torch.pow(C_diag, -1/2))))
        P = P.to_sparse().to(device = device)

        features =  data.x.to(device = device)
        cor_feat = (torch.sparse.mm((torch.t(P)), features))

        i = data.edge_index.to(device = device)
        v = torch.ones(data.edge_index.shape[1]).to(device = device)
        shape = torch.Size([data.x.shape[0],data.x.shape[0]])

        g_adj_tens = torch.sparse_coo_tensor(i, v, size=shape, dtype=torch.float32)
        g_coarse_adj = torch.sparse.mm(torch.t(P) , torch.sparse.mm( g_adj_tens , P))
        
        C_diag_matrix = np.diag(np.array(C_diag.to('cpu'), dtype = np.float32))

        [edges_src, edges_dst] = g_coarse_adj.coalesce().indices()
        edge_features = g_coarse_adj.coalesce().values()
        edge_index_corsen = torch.stack((edges_src, edges_dst))

        ## changing size from (n*1) to (n)
        data.y = data.y.squeeze()    

        Y = np.array(data.y.cpu())
        Y = utils.one_hot(Y, num_classes).to(device)
        Y[~data.train_mask] = torch.Tensor([0 for _ in range(num_classes)]).to(device = device)

        print(Y.shape, data.y.shape, P.shape)
        labels_coarse = torch.argmax(torch.sparse.mm(torch.t(P).double() , Y.double()).double() , 1).to(device)
        
        del C_diag_matrix
        del g_coarse_adj
        del edges_dst
        del i
        del v

        data_coarsen = Data(x=cor_feat, edge_index = edge_index_corsen, y = labels_coarse)
        data_coarsen.edge_attr = edge_features
        ## --------------------------------------------------------------------

    else:
        ## for spectral property calculations
        edge_index_corsen = data_coarsen.edge_index 
        P = P.to(device)
        print(P.shape)
    #------------------
    # ## Epsilion bounds
    
    # epsilion_bound = utils.get_smooth_features(data.edge_index, P, data.x.numpy())
    # print("epsilion_bound ", epsilion_bound)
    # exit(1)
    # #------------------

    # he_error_list = []
    # ree_error_list = []
    # dirichlet_energy_list = []

    if args.calculate_spectral_errors == True:
        if data.x.size(0) < 100:
            number_of_eigen_vectors = (int)(data.x.size(0)/2)
        else:
            number_of_eigen_vectors = 100

        if args.start_coarsen_method == 'UGC':
            if args.multiple_ugc != True:
                method_name = 'Consistent_hash'
            else:
                method_name = 'UGC'
        else:
            method_name = args.start_coarsen_method

        # eigen_plot_name = 'results_and_plots/' + args.dataset + '_' + (str)(math.floor(rr*100)) + '_method_' + (str)(method_name)
        # spectral_properties.plot_most_significant_eigen_values(100,data.edge_index,edge_index_corsen,edge_features,eigen_plot_name)

        he_error = spectral_properties.hyperbolic_error(P.to_dense().T,data.edge_index,edge_index_corsen,data.x)
        # he_error_list.append(he_error)
        print("check hyperbolic error",he_error)
        
        re_construct_error = spectral_properties.reconstruction_error(data.num_nodes,P.to_dense().T,data.edge_index,edge_index_corsen)
        # ree_error_list.append(re_construct_error)
        print("re_construction error ",re_construct_error)
        
        diri_energy = spectral_properties.dirichlet_energy(P.to_dense(),data.edge_index,edge_index_corsen,data.x,data_coarsen.x)
        # dirichlet_energy_list.append(diri_energy)
        print("dirichlet_energy error ",diri_energy)

        eigen_error = spectral_properties.eigen_error(data.edge_index, edge_index_corsen, number_of_eigen_vectors)
        # print("eigen_error ", eigen_error)
        
        coarsening_ratio = math.floor((1 - data_coarsen.x.shape[0]/data.x.shape[0])*100)

        update_spectral_properties(method_name, args.dataset, coarsening_ratio, he_error, re_construct_error, diri_energy, np.mean(eigen_error))
        exit(1)

    print("training on coarsened dataset")

    if args.train_coarsen == True:
        data_coarsen = data_coarsen.to(device=device)

        if args.dataset in  ['reddit22222']:
            utils.run_gcn(data_coarsen, device)
        
        elif args.dataset in ['twitch-e', 'wiki', 'deezer-europe', 'ogbn-proteins', 'yelp-chi' ,
                            'fb100', 'chameleon', 'squirrel', 'wisconsin', 'cornell', 'film', 'texas',
                                'pokec', 'arxiv-year', 'snap-patents', 'genius', 'twitch-gamer', 'ogbn-arxiv', 'ogbn-products']:

            ###### next two lines are added so that it can run on the hetero models
            data_coarsen.y = data_coarsen.y.unsqueeze(1)
            data.y = data.y.unsqueeze(1)
            
            # print(zero_list)
            # print(zero_list.shape)
            total_true = zero_list.sum().item()
            # print(total_true)

            num_nodes = data_coarsen.x.shape[0]
            model = heterophilic_parse.parse_method(args, data_coarsen, num_nodes, dataset.num_classes, feature_size, device)
            train_acc, val_acc, test_acc = train_coarsen.train_heterophilic_models(model, args, data, data_coarsen, zero_list, device)

            if args.start_coarsen_method == 'UGC':
                if args.multiple_ugc != True:
                    method_name = 'Consistent_hash'
                else:
                    method_name = 'UGC'
            else:
                method_name = args.start_coarsen_method

            if args.dataset == 'fb100':
                dataset_name =  args.sub_dataset
            else:
                dataset_name =  args.dataset

            update_table_accuracy(method_name, dataset_name, test_acc, args.method, path="results_table_accuracy_heterophilic.csv")
        else:
            if args.start_coarsen_method == 'UGC':
                if args.multiple_ugc != True:
                    method_name = 'Consistent_hash'
                else:
                    method_name = 'UGC'
            else:
                method_name = args.start_coarsen_method

            method_list = ['gcn', 'sage', 'gin', 'gat', 'ugc']
            # method_list = [args.method]

            for method in method_list:
                incorrect_index, acc = train_coarsen.train_on_UGC_models(data, data_coarsen, num_classes,feature_size,args.hidden_units,args.lr,args.decay,args.epochs, device, zero_list, method)
                update_table_accuracy(method_name, args.dataset, acc, method, path="results_table_accuracy_homophilic_mohit.csv")

    
    # incorrect_index = train_coarsen.train_on_UGC_models(data, data_coarsen, num_classes,feature_size,args.hidden_units,args.lr,args.decay,args.epochs, device, zero_list, args.model_type)
    
    print("ok")
