import torch
from scipy.stats import rv_continuous
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix, random
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class CustomDistribution(rv_continuous):
    def _rvs(self, size=None, random_state=None):
        return random_state.standard_normal(size) # type: ignore

def get_laplacian(adj):
    b = torch.ones(adj.shape[0], device=device)
    return torch.diag(adj @ b) - adj

def experiment(alpha_param, gamma_param, lambda_param, C, results, exp_iter, X_tilde):
    p = results.p
    k = results.k_global
    n = results.n

    ones = torch.ones((k, k), dtype=torch.float32, device=device)
    J = torch.outer(torch.ones(k, device=device), torch.ones(k, device=device)) / k

    def update(X_tilde, C, i):
        global L
        try:
            C[C < 0] = 0
            CT = C.T
            thetaC = results.theta @ C
            X_tildeT = X_tilde.T
            CX_tilde = C @ X_tilde
            t1 = CT @ thetaC + J
            L = 1 / k

            logdet = -gamma_param*torch.logdet(t1)
            frobenius_norm = (alpha_param / 2) * torch.norm(C @ X_tilde - results.X, p='fro') ** 2
            l21_norm = (lambda_param / 2) * torch.norm(C.T, p=2) ** 2
            trace_term = torch.trace(X_tildeT @ CT @ thetaC @ X_tilde)

            loss = frobenius_norm + l21_norm + logdet + trace_term

            logdet_d = -2*gamma_param*thetaC@torch.linalg.pinv(t1)
            frobnorm_d = alpha_param * (CX_tilde - results.X) @ (X_tildeT)
            l12_d = lambda_param * (C @ ones)
            trace_d = 2*thetaC@X_tilde@X_tildeT

            delf_C = (logdet_d + frobnorm_d + l12_d + trace_d)/L
            Cnew = (C - delf_C).clamp(min=results.thresh)

            X_tilde_new = torch.linalg.pinv(((2/alpha_param)*(C.T @ thetaC)) + C.T @ C) @ CT @ results.X
            Cnew = F.normalize(Cnew, p=1, dim=1)
            X_tilde_new = F.normalize(X_tilde_new, p=1, dim=1)
            return X_tilde_new, Cnew, loss
        
        except torch.linalg.LinAlgError:
            print("SVD did not converge, skipping this update.")
            return X_tilde, C, loss

    for i in tqdm(range(exp_iter)):
        X_tilde, C, loss = update(X_tilde, C, i)
    
    return X_tilde, C, loss.item()

def fitness_function(alpha_param, gamma_param, lambda_param, results):
    X_tilde = torch.tensor(random(k_global, results.n, density=0.15, random_state=1, data_rvs=temp2.rvs).toarray(), dtype=torch.float32, device=device)
    C = torch.tensor(random(results.p, k_global, density=0.15, random_state=1, data_rvs=temp2.rvs).toarray(), dtype=torch.float32, device=device)

    X_tilde_0, C_0, loss = experiment(alpha_param=alpha_param, 
                            gamma_param=gamma_param, 
                            lambda_param=lambda_param,
                            X_tilde=X_tilde, 
                            C=C,
                            results=results, 
                            exp_iter=12)
    return X_tilde_0, C_0, loss

def fgc(dataset, r):
    global r_global, k_global, temp2
    r_global = r


    labels = dataset[0].y.to(device)
    edge_list = dataset[0].edge_index
    NO_OF_EDGES=edge_list.shape[1]
    adj = to_dense_adj(dataset[0].edge_index).to(device)
    adj = adj[0]

    X = dataset[0].x.to_dense().to(device)
    N = X.shape[0]
    NO_OF_CLASSES = len(set(labels.cpu().numpy()))

    sparsity_original = 2 * NO_OF_EDGES / (N * (N - 1))

    nn = int(1 * N)
    X = X[:nn, :]
    adj = adj[:nn, :nn]
    labels = labels[:nn]
    def get_laplacian(adj):
        b = torch.ones(adj.shape[0], device=device)
        return torch.diag(adj @ b) - adj

    theta = get_laplacian(adj).float()
    p = N
    k_global = int(p * r_global)
    n = X.shape[1]
    lr = 1e-5
    thresh = 1e-10

    temp = CustomDistribution(seed=1)
    temp2 = temp()

    results_dict = {
        'dataset': None,
        'X': X,
        'p': p,
        'labels': labels,
        'NO_OF_CLASSES': NO_OF_CLASSES,
        'edge_list': edge_list,
        'NO_OF_EDGES': NO_OF_EDGES,
        'adj': adj,
        'theta': theta,
        'k_global': k_global,
        'n': n,
        'thresh': thresh,
    }

    results = SimpleNamespace(**results_dict)


    alpha_param  = [100, 10, 1, 0.1, 0.01, 0.001]
    gamma_param  = [100, 10, 1, 0.1, 0.01, 0.001]
    lambda_param = [100, 10, 1, 0.1, 0.01, 0.001]

    not_visited_set = np.array(np.meshgrid(alpha_param, gamma_param, lambda_param)).T.reshape(-1, 3)
    not_visited_set = set(map(tuple, not_visited_set))

    not_visited_set = not_visited_set
    currLoss = float('inf')
    currC = torch.tensor([[]])
    currX = torch.tensor([[]])

    for index, s in enumerate(not_visited_set):
        alpha_param, gamma_param, lambda_param = s
        try:
            X_tilde_0, C_0, loss = fitness_function(alpha_param, gamma_param, lambda_param, results)
            if loss < currLoss:
                currLoss = loss
                currC = C_0
                currX = X_tilde_0
        except Exception as e:
            print(e)
    
    print(f"Achieved loss is: {currLoss}")

    return currX, currC

