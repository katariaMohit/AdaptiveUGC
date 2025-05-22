import numpy as np
import torch
# import sys
import deeprobust.graph.utils as utils
import torch.nn.functional as F
from utils_gcond import *
from utils_graphsaint import DataGraphSAINT
# from utils_graphsaint_coarsen import DataGraphSAINT_coarsen
from models_gcond import GCN
import random

keep_ratio = 1.0
normalize_features = True
weight_decay = 5e-4

def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def split(data, num_classes,split_percent):
    indices = []
    num_test = (int)(data.num_nodes * split_percent / num_classes)
    for i in range(num_classes):
        index = (data.y == i).nonzero().reshape(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    
    test_index = torch.cat([i[:num_test] for i in indices], dim=0)
    val_index = torch.cat([i[num_test:int(num_test*1.5)] for i in indices], dim=0)
    train_index = torch.cat([i[int(num_test*1.5):] for i in indices], dim=0)

    # print(train_index)

    data.train_mask = utils.index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = utils.index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = utils.index_to_mask(test_index, size=data.num_nodes)
    return data

def rand_train_test_idx(data, train_prop=.5, valid_prop=.25):
    """ randomly splits label into train/valid/test splits """

    num_nodes = data.x.shape[0]
    train_num = int(num_nodes * train_prop)
    valid_num = int(num_nodes * valid_prop)

    perm = torch.as_tensor(np.random.permutation(num_nodes))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    data.train_mask = index_to_mask(train_indices, size=num_nodes)
    data.val_mask = index_to_mask(val_indices, size=num_nodes)
    data.test_mask = index_to_mask(test_indices, size=num_nodes)

    return data


# def coarsen_pyg2_data(dataset_name, adj_full, idx_train, idx_test, idx_val, feat, feat):
#     data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv', 'ogbn-products']

#     if dataset_name in data_graphsaint:
#         data = DataGraphSAINT_coarsen(dataset_name, adj_full, idx_train, idx_test, idx_val, feat, feat)
#         data_full = data.data_full
#         data = Transd2Ind(data_full, keep_ratio=keep_ratio)
#     else:
#         data_full = get_dataset(dataset_name, normalize_features)
#         data = Transd2Ind(data_full, keep_ratio=keep_ratio)

#     return data_full, data

def read_data(dataset_name, path):
    data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv', 'ogbn-products']

    if dataset_name in data_graphsaint:
        data = DataGraphSAINT(dataset_name, path)
        data_full = data.data_full
        data = Transd2Ind(data_full, keep_ratio=keep_ratio)
    else:
        data_full = get_dataset(dataset_name, normalize_features)
        data = Transd2Ind(data_full, keep_ratio=keep_ratio)

    return data_full, data

def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

def run_gcn(data, device):

    feat_train = data.feat_train
    adj_train = data.adj_train
    labels_train = data.labels_train

    # Setup GCN Model
    model = GCN(nfeat=feat_train.shape[1], nhid=256, nclass=labels_train.max()+1, device=device, weight_decay=weight_decay)
    model = model.to(device)

    model.fit_with_val(feat_train, adj_train, labels_train, data,
                train_iters=500, normalize=True, verbose=False)

    model.eval()
    labels_test = torch.LongTensor(data.labels_test).to(device)
    feat_test, adj_test = data.feat_test, data.adj_test

    embeds = model.predict().detach()
    
    output = model.predict(feat_test, adj_test).to(device)
    labels_test = labels_test.to(device)
    print(output.shape, labels_test.shape)
    loss_test = F.nll_loss(F.log_softmax(output, dim=1), labels_test)
    # loss_test = F.nll_loss(output, labels_test)
    acc_test = utils.accuracy(output, labels_test)
    print("Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()))


def run_gcn_coarsen(data, data_coarsen, device):

    feat_train = data_coarsen.feat_train
    adj_train = data_coarsen.adj_train
    labels_train = data_coarsen.labels_train

    # Setup GCN Model
    model = GCN(nfeat=feat_train.shape[1], nhid=256, nclass=labels_train.max()+1, device=device, weight_decay=weight_decay)
    model = model.to(device)

    model.fit_with_val(feat_train, adj_train, labels_train, data,
                train_iters=500, normalize=True, verbose=False)

    model.eval()
    labels_test = torch.LongTensor(data.labels_test).to(device)
    feat_test, adj_test = data.feat_test, data.adj_test

    embeds = model.predict().detach()
    
    output = model.predict(feat_test, adj_test).to(device)
    labels_test = labels_test.to(device)
    print(output.shape, labels_test.shape)
    loss_test = F.nll_loss(F.log_softmax(output, dim=1), labels_test)
    # loss_test = F.nll_loss(output, labels_test)
    acc_test = utils.accuracy(output, labels_test)
    print("Test set results:",
        "loss= {:.4f}".format(loss_test.item()),
        "accuracy= {:.4f}".format(acc_test.item()))