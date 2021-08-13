#!/bin/bash

python train_node.py --load_metis_data_train cora_6cluster.npy --num_cluster 6 --batch_size 5 --percent_trained_node 0.052 --top_k_degree 0.05 --percent_edges 0.05 --dataset cora --learning_rate 0.001 --epochs 600 --hidden1 16 --dropout 0.5 --weight_decay 5e-4 --early_stopping 600 --num_layers 2