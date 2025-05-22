import torch
import random
import numpy as np
from torch_geometric.datasets import IMDB
from torch_geometric.data import Data
import utils1
import torch.nn.functional as F

def hashed_values(num_nodes, X, no_of_hash,feature_size,function,out_of_sample,projectors_distribution):

  if projectors_distribution == 'VAEs':
    print("some random intilization is given here for mean and sigma make sure these contain learned values")
    learned_mean = -0.0017
    learned_sigma = 0.29
    Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(learned_mean,learned_sigma)
    # Wl = torch.FloatTensor(vecs)
  elif projectors_distribution == 'karate':
     Wl = [] #torch.FloatTensor(utils.sample_projectors(vecs,no_of_hash,feature_size))
  elif projectors_distribution == 'normal':
    Wl = torch.FloatTensor(no_of_hash, feature_size).normal_(0,1)
  else:
    #uniform
    Wl = torch.FloatTensor(no_of_hash, feature_size).uniform_(0,1)
  
  if out_of_sample != 0:
    num_out_of_sample = (int)(num_nodes*(1 - out_of_sample))
    idx = np.random.randint(num_nodes, size=num_out_of_sample)
    out_of_sampled_data_x = X[idx,:]
  else:
    out_of_sampled_data_x = X

  if function == 'L2-norm':
    Bin_values = torch.cdist(out_of_sampled_data_x, Wl, p = 2)
  elif function == 'L1-norm':
    Bin_values = torch.cdist(out_of_sampled_data_x, Wl, p = 1)
  else:
    #dot
    Bin_values = torch.matmul(out_of_sampled_data_x, Wl.T)
  
  return Bin_values

def partition(list_bin_width,Bin_values,no_of_hash):
    summary_dict = {}
    for bin_width in list_bin_width:
        bias = torch.tensor([random.uniform(-bin_width, bin_width) for i in range(no_of_hash)])#.to(device)
        temp = torch.floor((1/bin_width)*(Bin_values + bias))#.to(device)

        cluster, _ = torch.mode(temp, dim = 1)
        dict_hash_indices = {}
        no_nodes = Bin_values.shape[0]
        for i in range(no_nodes):
            dict_hash_indices[i] = int(cluster[i]) #.to('cpu')
        summary_dict[bin_width] = dict_hash_indices 
 
    return summary_dict


def ugc(X, bin_width, no_of_projectors, train_mask):
    bin_vals = hashed_values(num_nodes=X.shape[0], X=X, no_of_hash=no_of_projectors, feature_size=X.shape[1], function=None, out_of_sample=0, projectors_distribution='normal')

    list_bin_width = [bin_width]
    summary_dict = partition(list_bin_width, bin_vals, 1000)

    current_bin_width_summary = summary_dict[list_bin_width[0]]
    values = current_bin_width_summary.values()
    unique_values = set(values)
    rr = 1 - len(unique_values)/len(values)
    print(f'Graph reduced by: {rr*100} percent.\nWe now have {len(unique_values)} supernode, starting nodes were: {len(values)}')
    dict_blabla ={}
    C_diag = torch.zeros(len(unique_values))#, device= device)
    help_count = 0

    for v in unique_values:
        C_diag[help_count],dict_blabla[help_count] = utils1.get_key(v, current_bin_width_summary)
        help_count += 1

    P_hat = torch.zeros((X.shape[0], len(unique_values)))#, device= device)
    zero_list = torch.ones(len(unique_values), dtype=torch.bool)

    for x in dict_blabla:
        if len(dict_blabla[x]) == 0:
            print("zero element in this supernode", x)
        for y in dict_blabla[x]:
            P_hat[y,x] = 1
            if train_mask != None:
              zero_list[x] = zero_list[x] and (not (train_mask)[y])
        
    P_hat = P_hat.to_sparse()
    #dividing by number of elements in each supernode to get average value 
    P = torch.sparse.mm(P_hat,(torch.diag(torch.pow(C_diag, -1/2))))

    # return F.normalize(P_hat.to_dense(), p=1, dim=1), zero_list
    return P, zero_list
