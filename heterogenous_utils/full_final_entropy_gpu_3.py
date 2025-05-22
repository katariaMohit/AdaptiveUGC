import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix, random
from scipy.stats import rv_continuous
import argparse
import subprocess

# Ensure the code runs on GPU
r_global = 0
dataset_global = ""
device_global = ""
detailed = 0

def get_gpu_with_max_memory():
    try:
        # Query nvidia-smi to get GPU memory information
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        
        # Parse the output and get available memory for each GPU
        memory_free = [int(x) for x in result.strip().split('\n')]
        
        # Get the GPU index with the maximum available memory
        gpu_index = memory_free.index(max(memory_free))
        
        return gpu_index, max(memory_free)
    except Exception as e:
        print(f"Failed to get GPU memory information: {e}")
        return None, None

def set_best_gpu():
    gpu_index, max_memory = get_gpu_with_max_memory()
    if gpu_index is not None:
        torch.cuda.set_device(gpu_index)
        print(f"Using GPU {gpu_index} with {max_memory} MiB free memory.")
        return torch.device(f'cuda:{gpu_index}')
    else:
        print("No suitable GPU found. Using default GPU.")
        return torch.device('cuda')


def check_cuda_availability():
    if torch.cuda.is_available():
        print("CUDA is available on this device.")
        print(f"Number of available CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available on this device.")

def terminal_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--device', type=str, required=True, help='Device')
    parser.add_argument('--r', type=float, required=True, help='Regularization parameter')
    args = parser.parse_args()
    
    global r_global, dataset_global, device_global
    dataset_global = args.dataset
    device_global = args.device
    r_global = args.r

    update_paths()

def update_paths():
    global readings_path, path
    path = os.getcwd()
    readings_path = os.path.join(path, dataset_global, str(r_global))
    os.makedirs(readings_path, exist_ok=True)
    
    print(f"Readings path: {readings_path}")
    ensure_data_frame_exists()

def ensure_data_frame_exists():
    global filepath, filename
    filename = f"parameter({r_global},{dataset_global},{device_global})_final_gpu_readings.csv"
    filepath = os.path.join(readings_path, filename)
    try:
        pd.read_csv(filepath)
    except FileNotFoundError:
        print("Dataframe is being made.")
        df = pd.DataFrame(columns=['alpha_param','beta_param','gamma_param','lambda_param','delta_param','r','k','acc','std'])
        df.to_csv(filepath, index=False)

def save_readings(alpha_param, beta_param, gamma_param, lambda_param, delta_param, r, k, acc, std):
    df = pd.read_csv(filepath)
    df.loc[len(df)] = [alpha_param, beta_param, gamma_param, lambda_param, delta_param, r, k, acc, std]
    df.to_csv(filepath, index=False)

def get_device(device_global):
    if device_global.upper() == 'GPU':
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is not available, but GPU was requested.")
        return set_best_gpu()
    elif device_global.upper() == 'CPU':
        return torch.device('cpu')

if __name__ == "__main__":
    check_cuda_availability()
    terminal_command()
    device = get_device(device_global)
    print(f"Using device: {device}")

data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)
dataset = Planetoid(root='data', name=dataset_global)

edge_list = dataset[0].edge_index
NO_OF_EDGES = edge_list.shape[1]
labels = dataset[0].y.to(device)

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

p = X.shape[0]
k = int(p * r_global)
k_global = int(p * r_global)
n = X.shape[1]
lambda_param = 100
beta_param = 50
alpha_param = 100
gamma_param = 100
lr = 1e-5
thresh = 1e-10

class CustomDistribution(rv_continuous):
    def _rvs(self, size=None, random_state=None):
        return random_state.standard_normal(size)
temp = CustomDistribution(seed=1)
temp2 = temp()
X_tilde = torch.tensor(random(k, n, density=0.25, random_state=1, data_rvs=temp2.rvs).toarray(), dtype=torch.float32, device=device)
C = torch.tensor(random(p, k, density=0.25, random_state=1, data_rvs=temp2.rvs).toarray(), dtype=torch.float32, device=device)

total_indices1 = torch.arange(N, device=device)
shuffled_indices1 = torch.randperm(total_indices1.numel(), device=device)
train_mask = torch.zeros(N, dtype=torch.bool, device=device)
train_mask[:int(0.1 * total_indices1.numel())] = True
print(f"train mask: {train_mask.shape}")
test_mask = torch.ones(N, dtype=torch.bool, device=device)
print(f"test mask: {test_mask.shape}")

def convertScipyToTensor(coo):
    coo = coo.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices).to(device)
    v = torch.FloatTensor(values).to(device)
    shape = coo.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)

def experiment(alpha_param, beta_param, gamma_param, lambda_param, delta_param, C, X_tilde, theta, X, exp_iter):
    p = X.shape[0]
    k = int(p * r_global)
    n = X.shape[1]
    ones = torch.ones((k, k), dtype=torch.float32, device=device)
    J = torch.outer(torch.ones(k, device=device), torch.ones(k, device=device)) / k
    zeros = torch.zeros((p, k), dtype=torch.float32, device=device)
    eye = torch.eye(k, device=device)

    def one_hot(x, class_count):
        return torch.eye(class_count, device=device)[x, :]

    P = one_hot(labels, NO_OF_CLASSES)
    P[train_mask, :] = 0

    def update(X_tilde, C, i):
        global L
        try:
            C[C < 0] = 0
            thetaC = theta @ C
            CT = torch.transpose(C, 0, 1)
            phi = CT @ P
            X_tildeT = torch.transpose(X_tilde, 0, 1)
            CX_tilde = C @ X_tilde
            t1 = CT @ thetaC + J
            term_bracket = torch.linalg.pinv(t1)
            thetacX_tilde = thetaC @ X_tilde
            L = 1 / k

            t1 = -2 * gamma_param * (thetaC @ term_bracket)
            t2 = alpha_param * (CX_tilde - X) @ (X_tildeT)
            t3 = 2 * thetacX_tilde @ (X_tildeT)
            t4 = lambda_param * (C @ ones)
            t5 = 2 * beta_param * (thetaC @ CT @ thetaC)
            gradH = torch.zeros(k, NO_OF_CLASSES, device=device)
            row_sums = torch.zeros(phi.shape[0], device=device)

            for i in range(phi.shape[0]):
                row_sums[i] = torch.sum(phi[i, :])

            for i in range(k):
                for j in range(NO_OF_CLASSES):
                    gradH[i, j] = -(1.44 + torch.log2(phi[i, j] / row_sums[i])) * (row_sums[i] - phi[i, j])

            gradHT = torch.transpose(gradH, 0, 1)
            t6 = P @ gradHT

            T2 = (t1 + t2 + t3 + t4 + t5 + t6) / L
            Cnew = (C - T2).clamp(min=thresh)
            t1 = CT @ thetaC * (2 / alpha_param)
            t2 = CT @ C
            t1 = torch.linalg.pinv(t1 + t2)
            t1 = t1 @ CT
            t1 = t1 @ X
            X_tilde_new = t1
            Cnew = F.normalize(Cnew, p=1, dim=1)
            X_tilde_new = F.normalize(X_tilde_new, p=1, dim=1)
            return X_tilde_new, Cnew
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory, skipping this update.")
            torch.cuda.empty_cache()
            return X_tilde, C
        except torch.linalg.LinAlgError:
            print("SVD did not converge, skipping this update.")
            # Reinitialize X_tilde and C with new random values
            X_tilde = torch.tensor(random(k, n, density=0.15, random_state=1, data_rvs=temp2.rvs).toarray(), dtype=torch.float32, device=device)
            C = torch.tensor(random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs).toarray(), dtype=torch.float32, device=device)
            return X_tilde, C

    for i in tqdm(range(exp_iter)):
        X_tilde, C = update(X_tilde, C, i)
    
    return X_tilde, C

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(X.shape[1], 64)
        self.conv2 = GCNConv(64, NO_OF_CLASSES)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def get_accuracy(C_0, L, X_t_0):
    global labels, NO_OF_CLASSES, k
    all_acc = []
    for _ in [1, 2]:
        C_0_new = torch.zeros_like(C_0)
        C_0_new.scatter_(1, C_0.argmax(dim=1, keepdim=True), 1)

        Lc = C_0_new.T @ L @ C_0_new
        Wc = (-1 * Lc) * (1 - torch.eye(Lc.shape[0], device=device))
        Wc[Wc < 0.1] = 0

        Wc_sparse = Wc.to_sparse_csr()  # Convert to CSR format explicitly

        def one_hot(x, class_count):
            return torch.eye(class_count, device=device)[x, :]

        Y = labels.to(device)
        Y = one_hot(Y, NO_OF_CLASSES)
        Y[train_mask, :] = 0

        P = torch.pinverse(C_0_new)
        labels_coarse = torch.argmax(torch.sparse.mm(P, Y), 1)

        Xt = P @ X.to(device)
        coarsen_features = Xt
        coarsen_train_labels = labels_coarse

        total_indices = torch.arange(k_global, device=device)
        shuffled_indices = torch.randperm(total_indices.numel(), device=device)
        train_size = int(0.8 * total_indices.numel())
        train_indices = shuffled_indices[:train_size]
        val_size = int(0.2 * total_indices.numel())
        val_indices = shuffled_indices[train_size:train_size + val_size]

        data = dataset[0].to(device)

        coarsen_train_mask = train_indices
        coarsen_val_labels = labels_coarse
        coarsen_val_mask = val_indices

        model = Net().to(device)
        lr = 0.01
        decay = 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        model.reset_parameters()
        pathm = path + "/"
        best_val_loss = float('inf')
        no_of_epochs = 500
        no_of_early_stopping = 10
        val_loss_history = []

        for epoch in range(no_of_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(coarsen_features, Wc_sparse)
            loss = F.nll_loss(out[coarsen_train_mask], coarsen_train_labels[coarsen_train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            pred = model(coarsen_features, Wc_sparse)
            val_loss = F.nll_loss(pred[coarsen_val_mask], coarsen_val_labels[coarsen_val_mask]).item()

            if val_loss < best_val_loss and epoch > no_of_epochs // 2:
                best_val_loss = val_loss
                torch.save(model.state_dict(), pathm + 'checkpoint-best-acc.pkl')

            val_loss_history.append(val_loss)
            if no_of_early_stopping > 0 and epoch > no_of_epochs // 2:
                tmp = torch.tensor(val_loss_history[-(no_of_early_stopping + 1):-1], device=device)
                if val_loss > tmp.mean().item():
                    break

        model.load_state_dict(torch.load(pathm + 'checkpoint-best-acc.pkl'))
        model.eval()
        pred = model(data.x.to(device), data.edge_index.to(device)).max(1)[1]
        test_acc = int(pred.eq(data.y.to(device)).sum().item()) / data.y.size(0)
        print("Test_acc", f"{test_acc:.4f}")
        all_acc.append(test_acc)

    print('ave_acc: {:.8f}'.format(torch.mean(torch.tensor(all_acc))), '+/- {:.8f}'.format(torch.std(torch.tensor(all_acc))))
    return torch.mean(torch.tensor(all_acc)), torch.std(torch.tensor(all_acc))

def fitness_function(alpha_param, beta_param, gamma_param, lambda_param, delta_param):
    print(alpha_param, beta_param, gamma_param, lambda_param, delta_param)
    acc, std = torch.tensor(0), torch.tensor(0)  # Initialize acc and std to avoid uninitialized variable error
    start_time = time.time()  # Start time for the test case
    try:
        X_tilde = torch.tensor(random(k, n, density=0.15, random_state=1, data_rvs=temp2.rvs).toarray(), device=device, dtype=torch.float32)
        C = torch.tensor(random(p, k, density=0.15, random_state=1, data_rvs=temp2.rvs).toarray(), device=device, dtype=torch.float32)
        X_t_0, C_0 = experiment(alpha_param, beta_param, gamma_param, lambda_param, delta_param, C, X_tilde, theta, X, exp_iter=10)
        L = theta.clone().detach().to(device).float()  # Fix for the warning
        acc, std = get_accuracy(C_0, L, X_t_0)
    except Exception as e:
        print(e)
    end_time = time.time()  # End time for the test case
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print("\n------\t",f"Time taken: {elapsed_time:.2f} seconds","\t------")
    print(f"\n______ ______ ______ ______ ______ ___End of the test case: {len(not_visited_set) - index}___ ______ ______ ______ ______ ______")
    return acc.item(), std.item()

alpha_param = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
beta_param = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
gamma_param = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
lambda_param = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
delta_param = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

not_visited_set = np.array(np.meshgrid(alpha_param, beta_param, gamma_param, lambda_param, delta_param)).T.reshape(-1, 5)
not_visited_set = set(map(tuple, not_visited_set))

visited_set = pd.read_csv(filepath)
visited_set = visited_set[['alpha_param', 'beta_param', 'gamma_param', 'lambda_param', 'delta_param']].values
visited_set = set(map(tuple, visited_set))

not_visited_set = not_visited_set - visited_set
highest_acc = 0

for index, s in enumerate(not_visited_set):
    print("\n------\t", dataset_global, "\t", r_global, "\t", index, "|", len(not_visited_set), "\t------")
    alpha_param, beta_param, gamma_param, lambda_param, delta_param = s
    try:
        acc, std = fitness_function(alpha_param, beta_param, gamma_param, lambda_param, delta_param)
        if acc > highest_acc:
            highest_acc = acc
        save_readings(alpha_param, beta_param, gamma_param, lambda_param, delta_param, r_global, k_global, acc, std)
    except Exception as e:
        print(e)
