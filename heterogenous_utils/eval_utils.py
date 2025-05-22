import torch
from copy import deepcopy
from tqdm import tqdm
from hutils import lazy_initialize, evalue_model
from sklearn.metrics import f1_score
import torch.nn.functional as F
device = 'cpu'

def eval_GNN(num_evalue, x_dict, adj_t_dict, node_types, edge_types, y, target_node_type, num_classes, train_mask, test_mask, val_mask, x_syn_dict, adj_t_syn_dict, y_syn, mask_syn,
                  model_name, model_architecture, model_train, loss_fn = F.cross_entropy):
    device = train_mask.device
    
    x_syn_dict = {k:v.to(device) for k,v in x_syn_dict.items()}
    adj_t_syn_dict = {k:v.to(device) for k,v in adj_t_syn_dict.items()}
    y_syn = y_syn.to(device)
    mask_syn = mask_syn.to(device)

    
    val_model = model_name(**model_architecture, 
                                out_channels=num_classes, 
                                node_types=node_types, 
                                edge_types=edge_types,
                                target_node_type=target_node_type).to(device)
    lazy_initialize(val_model, x_dict, adj_t_dict)
    
    max_patience = 10
    trig_early_stop = True
    accs, f1_micros, f1_macros = [], [], []

    for i in range(num_evalue):
        best_acc = 0.
        for j in tqdm(range(model_train['epochs']),desc='Training', ncols=80):
            if trig_early_stop:
                patience = 0
                trig_early_stop = False
                val_model.reset_parameters()
                optimizer_val_model = torch.optim.Adam(val_model.parameters(), lr=model_train['lr'])
            val_model.train()
            optimizer_val_model.zero_grad()
            logits_train = val_model(x_syn_dict, adj_t_syn_dict)[mask_syn]
            loss = loss_fn(logits_train, y_syn[mask_syn])# cross_entropy nll_loss
            loss.backward()
            optimizer_val_model.step()
            with torch.no_grad():
                val_model.eval()
                logits_val = val_model(x_dict, adj_t_dict)[val_mask]
                acc = (logits_val.argmax(1) == y[val_mask]).sum()/logits_val.shape[0]
                if acc > best_acc:
                    best_acc = acc
                    patience = 0
                    weights = deepcopy(val_model.state_dict())
                else:
                    patience += 1
                if patience == max_patience:
                    trig_early_stop = True
        #--------------------------------------------------------------------------
        val_model.load_state_dict(weights)
        # acc,f1_micro,f1_macro = evalue_model(val_model, x_dict, adj_t_dict, y, test_mask)
        val_model.eval()
        logits = val_model(x_dict, adj_t_dict)[test_mask]
        labels = y[test_mask].cpu()
        preds = logits.argmax(1).cpu()
        acc = (labels == preds).sum()/test_mask.sum()
        acc = acc.item()
        f1_micro = f1_score(labels, preds, average='micro')
        f1_macro = f1_score(labels, preds, average='macro')
        accs.append(acc)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)
        del weights
        val_model.reset_parameters()
    return accs,f1_micros,f1_macros