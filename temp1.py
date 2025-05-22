import torch

# Setup
n, d, l = 100, 32, 8
X = torch.randn(n, d, device='cuda')
W = torch.randn(d, l, device='cuda')
b = torch.randn(l, device='cuda')

# Hash projections
hash_vals = torch.floor(X @ W + b)
node_line_values = hash_vals.mean(dim=1)


def coarsen_ring_parallel(ring_pos, num_supernodes):
    """Reduce entries on ring by merging adjacent positions"""
    n = ring_pos.size(0)
    unique_pos, inverse_indices = torch.unique(ring_pos, return_inverse=True, sorted=True)
    p = unique_pos.size(0)

    # Initial coarsening matrix
    C = torch.zeros(n, p, device=ring_pos.device)
    C[torch.arange(n), inverse_indices] = 1

    while p > num_supernodes:
        perm = torch.randperm(p, device=ring_pos.device)
        selected = perm[:p // 2]
        merge_to = (selected + 1) % p

        new_indices = inverse_indices.clone()
        for i, j in zip(selected.tolist(), merge_to.tolist()):
            new_indices[inverse_indices == i] = j

        unique_pos, new_inverse = torch.unique(new_indices, return_inverse=True, sorted=True)
        p = unique_pos.size(0)
        C = torch.zeros(n, p, device=ring_pos.device)
        C[torch.arange(n), new_inverse] = 1
        inverse_indices = new_inverse

    return C, inverse_indices


C, supernode_ids = coarsen_ring_parallel(node_line_values, num_supernodes=5)

print("Coarsening matrix shape:", C.shape)  # [n, p]
print("Supernode assignment:", supernode_ids.shape)
print(C)


#######################


import matplotlib.pyplot as plt
import numpy as np

# Example type proportions for 6 supernodes, each with 3 node types
type_props = np.array([
    [0.23, 0.50, 0.27],
    [0.20, 0.10, 0.70],
    [0.45, 0.32, 0.23],
    [0.27, 0.50, 0.23],
    [0.28, 0.52, 0.20],
    [0.40, 0.45, 0.15],
])

# Define type colors
type_colors = ["#60BD68", "#F15854", "#5DA5DA"]  # Type 0 (green), Type 1 (orange), Type 2 (blue)

# Plotting
fig, ax = plt.subplots(figsize=(8, 4))
bottom = np.zeros(len(type_props))

for t in range(len(type_colors)):
    ax.bar(np.arange(len(type_props)), type_props[:, t], bottom=bottom,
           color=type_colors[t], label=f'Type {t}')
    bottom += type_props[:, t]

ax.set_title("Supernode Type Composition")
ax.set_xlabel("Supernode Index")
ax.set_ylabel("Proportion")
ax.legend(title="Node Types", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
