# AdaptiveUGC
Graph Coarsening (GC) is a prominent graph reduction technique that com2 presses large graphs to enable efficient learning and inference. However, existing
GC methods generate only one coarsened graph per run and must recompute from
scratch for each new coarsening ratio, resulting in unnecessary overhead. Moreover,
most prior approaches are tailored to homogeneous graphs and fail to accommodate
the semantic constraints of heterogeneous graphs, which comprise multiple node
and edge types. To overcome these limitations, we introduce a novel framework
that combines Locality-Sensitive Hashing (LSH) with Consistent Hashing to enable adaptive graph coarsening. Leveraging hashing techniques, our method is
inherently fast and scalable. For heterogeneous graphs, we propose a type-isolated
coarsening strategy that ensures semantic consistency by restricting merges to
nodes of the same type. Our approach is the first unified framework to support both
adaptive and heterogeneous coarsening. Extensive evaluations on 23 real-world
datasets—including homophilic, heterophilic, homogeneous, and heterogeneous
graphs demonstrate that our method achieves superior scalability while preserving
the structural and semantic integrity of the original graph

## Run
To run different timing results as well as node classification (on homophilic, heterophilic datasets):  
- Physics: `python3 main.py --dataset=physics --train_coarsen=True --full_dataset=True --start_coarsen_method=UGC --method=gcn`  
- Cora: `python3 main.py --dataset=cora --train_coarsen=True --full_dataset=True --start_coarsen_method=UGC --method=gcn`  
- Texas: `python3 main.py --dataset=texas --train_coarsen=True --full_dataset=True --start_coarsen_method=UGC --method=gcn`

To run different coarsening methods for node classification on heterogenous datasets:  
- IMDB: `python3 main.py --dataset imdb --start_coarsen_method fugc --hetero_r 0.3`  
- DBLP: `python3 main.py --dataset hdblp --start_coarsen_method fugc --hetero_r 0.3`  
- ACM: `python3 main.py --dataset acm --start_coarsen_method fugc --hetero_r 0.3`