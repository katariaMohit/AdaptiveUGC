import argparse
import torch.nn.functional as F
import torch
from torch import tensor
from network import Net
import numpy as np
from .utils import load_data, coarsening
import os
from torch_geometric.data import Data

import scipy.sparse as sp

def build_global_P(coarsen_features, data, C_list, candidate):
    num_corsen_nodes = coarsen_features.shape[0]
    original_nodes = data.x.shape[0]

    P = torch.zeros((num_corsen_nodes, original_nodes))
    supernode_indices = 0

    for i, C in enumerate(C_list):
        current_candidate_nodes = np.array(candidate[i].info['orig_idx'])

        for row in C.toarray():
            active_indices = np.where(row > 0)[0]
            global_nodes = current_candidate_nodes[active_indices]

            for node, idx in zip(global_nodes, active_indices):
                if supernode_indices < num_corsen_nodes:
                    P[supernode_indices, node] = row[idx]
                else:
                    print("this should not happen")

            supernode_indices += 1

    return P.T

experiment = 'fixed'  # options: 'fixed', 'random', 'few'
runs = 1
hidden = 64
epochs = 60
early_stopping = 10
lr = 0.01
weight_decay = 0.0005
normalize_features = True


def scal_coarsen(dataset, coarsening_ratio, coarsening_method, dataset_name):


    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("coarsening_ratio ", 1-coarsening_ratio)
    num_features, num_classes, candidate, C_list, Gc_list = coarsening(dataset, 1-coarsening_ratio, coarsening_method)
  
    ### uncomment next lines if you want to run GCN on the coarsened graph to see accuracies

    model = Net(num_features, hidden, num_classes).to(device)
    all_acc = []

    for i in range(runs):

        data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge = load_data(
            dataset, candidate, C_list, Gc_list, experiment)
    #     data = data.to(device)
        coarsen_features = coarsen_features.to(device)
        coarsen_train_labels = coarsen_train_labels.to(device)
        coarsen_train_mask = coarsen_train_mask.to(device)
        coarsen_val_labels = coarsen_val_labels.to(device)
        coarsen_val_mask = coarsen_val_mask.to(device)
        coarsen_edge = coarsen_edge.to(device)

    # C_global = build_global_C_from_keep(candidate, C_list, dataset[0].x.shape[0])
    C_global = build_global_P(coarsen_features, data, C_list, candidate)

    # print(coarsen_val_labels.shape)
    # print(coarsen_train_labels.shape)
    # print(coarsen_train_labels)
    # print(coarsen_val_labels)

    data_coarsen = Data(x=coarsen_features, edge_index = coarsen_edge, y = coarsen_train_labels, train_mask=coarsen_train_mask, val_mask=coarsen_val_mask)
    return data_coarsen, C_global

    #     if normalize_features:
    #         coarsen_features = F.normalize(coarsen_features, p=1)
    #         data.x = F.normalize(data.x, p=1)

    #     model.reset_parameters()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #     best_val_loss = float('inf')
    #     val_loss_history = []
    #     for epoch in range(epochs):

    #         model.train()
    #         optimizer.zero_grad()
    #         out = model(coarsen_features, coarsen_edge)
    #         loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
    #         loss.backward()
    #         optimizer.step()

    #         model.eval()
    #         pred = model(coarsen_features, coarsen_edge)
    #         val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()

    #         if val_loss < best_val_loss and epoch > epochs // 2:
    #             best_val_loss = val_loss
    #             torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')

    #         val_loss_history.append(val_loss)
    #         if early_stopping > 0 and epoch > epochs // 2:
    #             tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
    #             if val_loss > tmp.mean().item():
    #                 break

    #     model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
    #     model.eval()
    #     pred = model(data.x, data.edge_index).max(1)[1]
    #     test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
    #     print(test_acc)
    #     all_acc.append(test_acc)

    # print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))

