import torch
import numpy as np
import matplotlib.pyplot as plt

def plotHeatmap(C, y, filename="supernode_composition.png"):
    """
    Plots a stacked bar chart showing node type composition of each supernode.

    Parameters:
    - C (torch.Tensor): [n x p] loading matrix from coarsening (n = num nodes, p = num supernodes)
    - y (torch.Tensor): [n] node type labels, integer values from 0 to num_classes - 1
    - filename (str): Filename to save the plot (as .png)
    """

    # Convert tensors to numpy
    C_np = C.clone().detach().cpu().numpy()
    y_np = y.to(torch.int).clone().detach().cpu().numpy()

    print(C_np.shape, y_np.shape)

    # Infer number of classes and create one-hot encoding
    num_classes = int(y_np.max()) + 1
    y_onehot = np.eye(num_classes)[y_np]  # shape: [n x num_classes]

    # Compute type composition: M = C.T @ y_onehot -> shape: [p x num_classes]
    type_counts = C_np.T @ y_onehot

    # Normalize to get proportions
    type_props = type_counts / type_counts.sum(axis=1, keepdims=True)

    # Plot settings
    type_colors = ["#60BD68", "#F15854", "#5DA5DA", "#FAA43A", "#B276B2", "#DECF3F"]
    type_colors = type_colors[:num_classes]  # truncate if more than 6

    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(type_props.shape[0])

    for t in range(num_classes):
        ax.bar(np.arange(type_props.shape[0]), type_props[:, t], bottom=bottom,
               color=type_colors[t], label=f'Type {t}')
        bottom += type_props[:, t]

    ax.set_title("")
    ax.set_xlabel("Supernode Index")
    ax.set_ylabel("Proportion")
    ax.legend(title="Node Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('heatmaps/'+filename)
    plt.close()
    print(f"Plot saved as: {filename}")