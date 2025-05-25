import torch
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch_sparse import sum as sparsesum, mul, SparseTensor
from torch_geometric.data import HeteroData, Data

def ToHetero(data_orig, datamid, data_coarse, C, nodetype_y, type_map, type_mapinv, nclasses, target_node, coar_method, ifFGC_LAGC=False):
    device = C.device
    C = F.normalize(C, p=1.0, dim=1)
    one_hot = F.one_hot(nodetype_y.to(torch.int64), len(data_orig.node_types)).to(device)
    # one_hot[~datamid.train_mask] = torch.tensor([0 for _ in range(one_hot.shape[1])], dtype=one_hot.dtype)
    counts = C.T @ one_hot.to(torch.float)
    coarse_ny = torch.argmax(counts, dim=1).squeeze() #Super nodes type
    

    newY = F.one_hot(datamid.y).to(torch.float).to(device)
    newY[~datamid.train_mask] = torch.tensor([0 for _ in range(newY.shape[1])], dtype=newY.dtype, device=newY.device)
    temp_temp = (C.T @ newY)
    newY = torch.argmax((C.T @ newY)[:, :nclasses], dim=1)

    # Check which rows are all zeros
    zero_rows_mask = (temp_temp == 0).all(dim=1)

    # Handle empty super nodes (if any)
    data_coarse.train_mask[zero_rows_mask] = False

    nodetype_idx = {k: [] for k in type_map.keys()}

    indexCy = [[j, i] for i, j in enumerate(coarse_ny.tolist())]
    for i, j in indexCy:
        nodetype_idx[i].append(j)

    for k, v in nodetype_idx.items():
        nodetype_idx[k] = torch.tensor(v, dtype=torch.int64)


    newData = HeteroData()

    if not ifFGC_LAGC:
        if coar_method in ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron']:
            X = data_coarse.x
        else:
            X = C.T @ datamid.x
    else:
        X = data_coarse.x

    new_indices = sorted([[i[1], n] for n, i in enumerate(sorted(indexCy))])

    # Make X
    curr = 0
    for node_type in data_orig.metadata()[0]:
        tempX = torch.zeros(nodetype_idx[type_mapinv[node_type]].shape[0], X.shape[1])
        for index in nodetype_idx[type_mapinv[node_type]]:
            tempX[new_indices[index.item()][1]-curr] = X[index.item()] 
        newData[node_type].x = tempX
        curr += tempX.shape[0]
    
    # Make Y
    tempY = torch.zeros(nodetype_idx[type_mapinv[target_node]].shape[0]).to(datamid.y.dtype)
    for index in nodetype_idx[type_mapinv[target_node]]:
        tempY[new_indices[index.item()][1]] = newY[index.item()].item()
    newData[target_node].y = tempY
    
    # Make masks
    tempMask = torch.zeros(nodetype_idx[type_mapinv[target_node]].shape[0]).to(datamid.train_mask.dtype)
    for index in nodetype_idx[type_mapinv[target_node]]:
        tempMask[new_indices[index.item()][1]] = data_coarse.train_mask[index.item()].item()
    
    newData[target_node].train_mask = tempMask         

    ei = data_coarse.edge_index

    new_edges_dict = {(edge_type[0], edge_type[2]): [] for edge_type in data_orig.metadata()[1]}
    for i, j in ei.T.tolist():
        if (type_map[indexCy[i][0]], type_map[indexCy[j][0]]) in new_edges_dict:
            new_edges_dict[(type_map[indexCy[i][0]], type_map[indexCy[j][0]])].append([new_indices[i][1], new_indices[j][1]])


    for edge_type in data_orig.metadata()[1]:
        newData[edge_type].edge_index = torch.tensor(new_edges_dict[(edge_type[0], edge_type[2])], dtype=data_orig[edge_type].edge_index.dtype).T

    newIndices = {}
    curr = 0
    for node_type in data_orig.node_types:
        newIndices[node_type] = curr
        curr += newData[node_type].x.shape[0]


    for edge_type in data_orig.edge_types:
        if newData[edge_type].edge_index.shape[0]>0:
            src, dst = edge_type[0], edge_type[2]
            idx = torch.ones(newData[edge_type].edge_index.shape[1], 2) * torch.tensor([newIndices[src], newIndices[dst]])
            newData[edge_type].edge_index = (newData[edge_type].edge_index.T - idx).T.to(newData[edge_type].edge_index.dtype)
    
    for node_type in data_orig.node_types:
            newData[node_type].x = newData[node_type].x[:, :data_orig[node_type].x.shape[1]]
            
    return newData

def ToHomo(dataset):
    total_nodes = sum([dataset[node_type].x.shape[0] for node_type in dataset.node_types])

    max_feat_size = max([dataset[node_type].x.shape[1] for node_type in dataset.node_types])
    X = torch.zeros(total_nodes, max_feat_size)
    curr = 0
    prev = 0
    for node_type in dataset.node_types:
        curr += dataset[node_type].x.shape[0]
        X[prev:curr, :dataset[node_type].x.shape[1]] = dataset[node_type].x
        prev += dataset[node_type].x.shape[0]

    y_list = [dataset[dataset.target_node].y] + [(i-1+dataset.num_classes)*torch.ones(dataset[node_type].x.shape[0]) for i, node_type in enumerate(dataset.node_types) if node_type != dataset.target_node]
    y = torch.cat(y_list, dim=0).to(torch.int64)

    
    curr = 0
    new_indices = {}
    for node_type in dataset.node_types:
        new_indices[node_type] = curr
        curr += dataset[node_type].x.shape[0]


    edge_list = []
    for edge_type in dataset.edge_types:
        src, dst = edge_type[0], edge_type[2]
        idx = torch.ones(dataset[edge_type].edge_index.shape[1], 2) * torch.tensor([new_indices[src], new_indices[dst]])
        myedge = dataset[edge_type].edge_index.T + idx
        edge_list.append(myedge)
    final_edge_list = torch.cat(edge_list).to(torch.int64)

    
    maskEx = torch.ones(total_nodes-dataset[dataset.target_node].x.shape[0], dtype=torch.bool)
    ntrain_mask = torch.cat((dataset[dataset.target_node].train_mask, maskEx))
    ntest_mask = torch.cat((dataset[dataset.target_node].test_mask, maskEx))
    nval_mask = torch.cat((dataset[dataset.target_node].val_mask, maskEx))

    newDataset = Data(x=X, y=y, edge_index=final_edge_list.T, train_mask = ntrain_mask, test_mask = ntest_mask, val_mask = nval_mask)

    return newDataset


def asymmetric_gcn_norm(adj_t):
    if isinstance(adj_t, SparseTensor):
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.)
        deg_src = sparsesum(adj_t, dim=0)+0.00001
        deg_src_inv_sqrt = deg_src.pow_(-0.5)
        deg_src_inv_sqrt.masked_fill_(deg_src_inv_sqrt == float('inf'), 0.)
        deg_dst = sparsesum(adj_t, dim=1)+0.00001
        deg_dst_inv_sqrt = deg_dst.pow_(-0.5)
        deg_dst_inv_sqrt.masked_fill_(deg_dst_inv_sqrt == float('inf'), 0.)
        
        adj_t = mul(adj_t, deg_dst_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_src_inv_sqrt.view(1, -1))
    else:
        deg_src = adj_t.sum(0)+0.00001
        deg_src_inv_sqrt = deg_src.pow_(-0.5)
        deg_src_inv_sqrt.masked_fill_(deg_src_inv_sqrt == float('inf'), 0.)
        deg_dst = adj_t.sum(1)+0.00001
        deg_dst_inv_sqrt = deg_dst.pow_(-0.5)
        deg_dst_inv_sqrt.masked_fill_(deg_dst_inv_sqrt == float('inf'), 0.)
        adj_t = adj_t*deg_dst_inv_sqrt.view(-1, 1)
        adj_t = adj_t*deg_src_inv_sqrt.view(1, -1)
    return adj_t

def evalue_model(model, x_dict, adj_t_dict, y, test_mask):
    model.eval()
    logits = model(x_dict, adj_t_dict)[test_mask]
    labels = y[test_mask].cpu()
    preds = logits.argmax(1).cpu()
    print(labels, preds)
    acc = (labels == preds).sum()/test_mask.sum()
    acc = acc.item()
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    return acc,f1_micro,f1_macro

def train_model(model, opt_parameter,optimizer, x_syn_dict, adj_t_syn_dict, y_syn, mask_syn):
    for epoch in range(1, opt_parameter['epochs_basic_model']+1):
        model.train()
        optimizer.zero_grad()
        out = model(x_syn_dict, adj_t_syn_dict)[mask_syn]
        loss = F.nll_loss(out, y_syn[mask_syn]) # nll_loss  cross_entropy
        loss.backward()
        optimizer.step()

def train_model_ealystop(model, opt_parameter, x_dict, adj_t_dict, y,
                         train_mask, val_mask):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_parameter['lr'], weight_decay=opt_parameter['weight_decay'])

    best_val_acc = 0
    for epoch in tqdm(range(1, opt_parameter['epochs']+1),desc='Training',ncols=80):
        model.train()
        optimizer.zero_grad()
        out = model(x_dict, adj_t_dict)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        pred = model(x_dict, adj_t_dict).argmax(dim=-1)
        val_acc = (pred[val_mask] == y[val_mask]).sum() / val_mask.sum()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            weights = deepcopy(model.state_dict())
    model.load_state_dict(weights)
    return best_val_acc
def lazy_initialize(model, x_dict, adj_t_dict):
    with torch.no_grad(): 
        model(x_dict, adj_t_dict)