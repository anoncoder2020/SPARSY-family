# **The Power of Subgraphs: Enhancing Efficiency and Effectiveness of Graph Convolutional Networks with Structure-Aware Sparsifiers**
------------------------------------------------------------------------------------------------------------------------------------
**1. Requirements**

networkx==2.2, scipy==1.1.0, setuptools==40.6.3, numpy==1.15.4, tensorflow==1.13.2

**2. For training and testing in NODE-SPARSY on Cora**

please run the file run_cora_node.sh
            
    python train_node.py --load_metis_data_train cora_6cluster.npy --num_cluster 6 --batch_size 5 --percent_trained_node 0.052 --top_k_degree 0.05 --percent_edges 0.05 --dataset cora --learning_rate 0.001 --epochs 600 --hidden1 16 --dropout 0.5 --weight_decay 5e-4 --early_stopping 600 --num_layers 2

**3.For training and testing in CYCLE-SPARSY on Cora**

please run the file run_cora_cycle.sh

    python train_cycle.py --load_metis_data_train cora_6cluster.npy --num_cluster 6 --batch_size 5 --percent_trained_node 0.052 --percent_edges 0.01 --dataset cora --learning_rate 0.001 --epochs 600 --hidden1 16 --dropout 0.5 --weight_decay 0.0003 --early_stopping 600 --num_layers 2

**4.For training and testing in ClIQUE-SPARSY on Cora**

please run the file run_cora_clique.sh

Note that the number of clusters is correponding the file like 'cora_6cluster.npy' which is from METIS, batch_size<=num_cluster and 0.0<percent_edges<1.0


