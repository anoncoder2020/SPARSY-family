# **The Power of Subgraphs: Enhancing Efficiency and Effectiveness of Graph Convolutional Networks with Structure-Aware Sparsifiers**
------------------------------------------------------------------------------------------------------------------------------------
**1. Requirements**

networkx==2.2, scipy==1.1.0, setuptools==40.6.3, numpy==1.15.4, tensorflow==1.13.2

**2. For training and testing in NODE-SPARSY on Cora**

please run the file run_cora_node.sh

**3.For training and testing in CYCLE-SPARSY on Cora**

please run the file run_cora_cycle.sh

**4.For training and testing in ClIQUE-SPARSY on Cora**

please run the file run_cora_clique.sh

Note that the number of clusters is correponding the file like 'cora_6cluster.npy' which is from METIS, batch_size<=num_cluster and 0.0<percent_edges<1.0


