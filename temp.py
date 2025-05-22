import matplotlib.pyplot as plt
import pandas as pd

# Method and capability setup
methods = ['LVN', 'LVE', 'LVC', 'HEM', 'Alg. Dist.', 'Affinity', 'Kron', 'GCond', 'HGCond' , 'FGC', 'UGC', 'AH-UGC']
criteria = [
    '1: Model-Agnostic',
    '2: Linear Time',
    '3: Heterophilic',
    '4: Heterogeneous',
    '5: Adaptive',
    '6: Streaming'
]

data = [
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1]
]

# Convert to DataFrame
df = pd.DataFrame(data, columns=criteria, index=methods)

# Generate capability labels
def get_capability_indices(row):
    return ', '.join([criteria[i].split(':')[0] for i, val in enumerate(row) if val == 1])

capability_labels = df.apply(get_capability_indices, axis=1)
capability_count = df.sum(axis=1)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Draw horizontal bars
bars = ax.barh(df.index, capability_count, color='#4C72B0', edgecolor='black')

# Annotate capability index list
for bar, label in zip(bars, capability_labels):
    ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
            f'[{label}]', va='center', fontsize=11, fontweight='medium')

# Configure plot appearance
ax.set_xlim(0, 7.5)
ax.set_xlabel('Total Capabilities Supported (out of 6)', fontsize=20)
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=20)
ax.tick_params(axis='x', labelsize=10)
ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)

# Horizontal legend row above the plot
legend_text = '  '.join(criteria)
fig.text(0.5, 0.95, legend_text, ha='center', fontsize=13, fontweight='semibold')

plt.tight_layout(rect=[0, 0, 1, 0.92])  # room for legend
plt.savefig("capability_barplot_horizontal_legend.jpg", bbox_inches='tight')
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load your CSV or define manually as before
# df = pd.read_csv("results_spectral_properties.csv")  # or use StringIO if you're loading from a string
# df["eigen_error"] = df["eigen_error"].str.strip("[]").astype(float)

# # List of metrics
# metrics = ["he_error", "re_construct_error", "diri_energy", "eigen_error"]

# # Plot setup
# fig, axes = plt.subplots(2, 2, figsize=(16, 10))
# axes = axes.flatten()

# for i, metric in enumerate(metrics):
#     best_vals = df.groupby("dataset")[metric].min().rename("best")
#     df_metric = df.merge(best_vals, on="dataset")
#     df_metric["delta"] = df_metric[metric] - df_metric["best"]
    
#     sns.barplot(data=df_metric, x="dataset", y="delta", hue="method", ax=axes[i])
#     axes[i].set_title(f"Gap to Best ({metric})", fontsize=14)
#     axes[i].set_ylabel("Gap to Best")
#     axes[i].tick_params(axis='x', rotation=45)
#     axes[i].legend(loc='upper right', fontsize='small')

# fig.suptitle("Performance Gap to Best Method per Metric", fontsize=16)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig("spectral_results.png")
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# from scipy.special import erf
# from sklearn.metrics.pairwise import euclidean_distances

# def random_projection(x, r_vectors):
#     return np.sum([r @ x for r in r_vectors])

# def experiment_projection_proximity(X, num_pairs=1000, ell=50, epsilons=np.linspace(0.1, 3.0, 20)):
#     d = X.shape[1]
#     results = []

#     # Sample random pairs of nodes
#     indices = np.random.choice(len(X), size=(num_pairs, 2), replace=True)
#     for eps in epsilons:
#         count = 0
#         for i, j in indices:
#             x, y = X[i], X[j]
#             dist = np.linalg.norm(x - y)

#             # Generate ℓ random projection vectors
#             r_vectors = [np.random.normal(0, 1, d) for _ in range(ell)]

#             hx = random_projection(x, r_vectors)
#             hy = random_projection(y, r_vectors)

#             if np.abs(hx - hy) <= eps:
#                 count += 1

#         empirical_prob = count / num_pairs
#         theoretical_prob = erf(eps / (np.sqrt(2 * ell) * np.linalg.norm(X[indices[0, 0]] - X[indices[0, 1]])))

#         results.append((eps, empirical_prob, theoretical_prob))

#     return np.array(results)

# # Example: simulate or load graph node embeddings (X)
# # Here we create random 128-dim node embeddings for example
# np.random.seed(42)
# X = np.random.randn(500, 128)  # Replace with real graph node embeddings

# # Run experiment
# results = experiment_projection_proximity(X, num_pairs=1000, ell=50)

# # Plot
# plt.figure(figsize=(8, 5))
# plt.plot(results[:, 0], results[:, 1], label='Empirical', marker='o')
# plt.plot(results[:, 0], results[:, 2], label='Theoretical (erf)', linestyle='--')
# plt.xlabel(r'$\varepsilon$')
# plt.ylabel(r'$\Pr[|h(x) - h(y)| \leq \varepsilon]$')
# plt.title('Projection Proximity Validation')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("temp.png")
# plt.show()

