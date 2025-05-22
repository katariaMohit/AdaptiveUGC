import torch 
import torch.nn as nn
from torch_geometric.data import NeighborSampler
import torch.nn.functional as F
from tqdm import tqdm
import time
from models.ugc_models import GCN
from models.ugc_models import GraphSage
from models.ugc_models import GAT
from models.ugc_models import GIN
from models.ugc_models import APPNP as APPNP

from heterophlic_data.data_utils import rand_train_test_idx, normalize, gen_normalized_adjs, evaluate, eval_acc, eval_rocauc, to_sparse_tensor, load_fixed_splits

def val(model_type, model, data):
    data = data#.to(device)
    model.eval()
    if model_type not in ['gcn']:
      pred = model(data.x, data.edge_index).argmax(dim=1)
    else:
      pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)
    
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    return acc

def train_on_UGC_models(data, data_coarsen, num_classes, feature_size, hidden_units, learning_rate, decay, epochs, device, zero_list, model_type="gcn"):
    if model_type == 'gin':
        model = GIN.GIN(feature_size, hidden_units, num_classes)
    elif model_type == 'sage':
        model = GraphSage.GraphSAGE(feature_size, hidden_units, num_classes)
    elif model_type == 'gat':
        model = GAT.GAT(feature_size, hidden_units, num_classes)
    elif model_type == 'ugc':
        model = APPNP.Net(feature_size, hidden_units, num_classes)
    else:
        model = GCN.GCN_(feature_size, hidden_units, num_classes)

    model = model.to(device)
    data = data.to(device)
    data_coarsen = data_coarsen.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=decay)

    
    if data_coarsen.edge_attr == None:
        edge_weight = torch.ones(data_coarsen.edge_index.size(1))
        data_coarsen.edge_attr = edge_weight.to(device=device)
        
    for epoch in range(epochs):
        optimizer.zero_grad()
        if model_type not in ['gcn']:
            out = model(data_coarsen.x, data_coarsen.edge_index)
        else:
            # print(data.x.device, data.edge_index.device, data.edge_attr.device)
            out = model(data_coarsen.x, data_coarsen.edge_index,data_coarsen.edge_attr.float())

        pred = out.argmax(1)
        criterion = torch.nn.NLLLoss()
        
        loss = criterion(out[~zero_list], data_coarsen.y[~zero_list]) 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        best_val_acc = 0
        
        val_acc = val(model_type, model, data)
        if best_val_acc < val_acc:
            # torch.save(model, 'full_best_model.pt')
            best_val_acc = val_acc
    
        if epoch % 100 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})'.format(epoch, loss, val_acc, best_val_acc))

    # model = torch.load('full_best_model.pt')
    model.eval()

    if model_type not in ['gcn']:
        pred = model(data.x, data.edge_index).argmax(dim=1)
    else:
        pred = model(data.x, data.edge_index,data.edge_attr).argmax(dim=1)

    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())

    incorrect_indices = (pred[data.test_mask] != data.y[data.test_mask]).nonzero()

    # Convert the indices to a list
    incorrect_indices_list = incorrect_indices.view(-1).tolist()
        
    print('--------------------------')
    print('Accuracy on test data {:.3f}'.format(acc*100))

    return incorrect_indices_list, acc

def train_heterophilic_models(model, args, data, data_coarsen, zero_list, device):
    if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
        criterion = nn.BCEWithLogitsLoss()
        eval_func = eval_rocauc
    else:
        criterion = nn.NLLLoss()
        eval_func = eval_acc

    model.train()
    print('MODEL:', model)

    # split_idx_lst = load_fixed_splits(args.dataset, args.sub_dataset)
    # train_idx, valid_idx, test_idx = rand_train_test_idx(data.y, train_prop=.6, valid_prop=.2, ignore_negative=True)
    # split_idx = {'train': train_idx,
    #                      'valid': valid_idx,
    #                      'test': test_idx}

    # if len(data.y.shape) == 1:
    #     data.y = data.y.unsqueeze(1)

    train_loader, subgraph_loader = None, None
    split_idx = {}
    split_idx['train'] = data.train_mask
    split_idx['valid'] = data.val_mask
    split_idx['test'] = data.test_mask

    for run in range(args.runs):
        # split_idx = split_idx_lst[run]
        # train_idx = split_idx['train'].to(device)
        if args.sampling:
            if args.num_layers == 2:
                sizes = [15, 10]
            elif args.num_layers == 3:
                sizes = [15, 10, 5]
            train_loader = NeighborSampler(data_coarsen.edge_index, node_idx=~zero_list,
                                    sizes=sizes, batch_size=1024,
                                    shuffle=True, num_workers=12)
            subgraph_loader = NeighborSampler(data_coarsen.edge_index, node_idx=None, sizes=[-1],
                                            batch_size=4096, shuffle=False,
                                            num_workers=12)
        
        model.reset_parameters()
        if args.adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.SGD:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, nesterov=args.nesterov, momentum=args.momentum)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val = float('-inf')
        for epoch in range(args.epochs):
            model.train()

            if not args.sampling:
                optimizer.zero_grad()
                out = model(data_coarsen)
                #loss = criterion(out[train_idx], data.y.squeeze(1)[train_idx].type_as(out))
                if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
                    if data_coarsen.y.shape[1] == 1:
                        # change -1 instances to 0 for one-hot transform
                        # data.y[data.y==-1] = 0
                        true_label = F.one_hot(data_coarsen.y, data_coarsen.y.max() + 1).squeeze(1)
                    else:
                        true_label = data_coarsen.y

                    loss = criterion(out[~zero_list], true_label.squeeze(1)[
                                    ~zero_list].to(torch.float))
                else:
                    out = F.log_softmax(out, dim=1)
                    # print("out.shape ", out.shape)
                    loss = criterion(
                        out[~zero_list], data_coarsen.y.squeeze(1)[~zero_list])
                loss.backward()
                optimizer.step()
            else:
                pbar = tqdm(total=zero_list.size(0))
                pbar.set_description(f'Epoch {epoch:02d}')

                for batch_size, n_id, adjs in train_loader:
                    # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                    adjs = [adj.to(device) for adj in adjs]

                    optimizer.zero_grad()
                    out = model(data_coarsen, adjs, data_coarsen.x[n_id])
                    out = F.log_softmax(out, dim=1)
                    loss = criterion(out, data_coarsen.y.squeeze(1)[n_id[:batch_size]])
                    loss.backward()
                    optimizer.step()
                    pbar.update(batch_size)
                pbar.close()
            
            result = evaluate(model, data, split_idx, eval_func, sampling=args.sampling, subgraph_loader=subgraph_loader)
            # logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]
                if args.dataset != 'ogbn-proteins':
                    best_out = F.softmax(result[-1], dim=1)
                else:
                    best_out = result[-1]

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * result[0]:.2f}%, '
                    f'Valid: {100 * result[1]:.2f}%, '
                    f'Test: {100 * result[2]:.2f}%')
                if args.print_prop:
                    pred = out.argmax(dim=-1, keepdim=True)
                    print("Predicted proportions:", pred.unique(return_counts=True)[1].float()/pred.shape[0])
        
    return result[0], result[1], result[2]
