#!/bin/bash

python train_cycle.py --load_metis_data_train cora_6cluster.npy --num_cluster 6 --batch_size 5 --percent_trained_node 0.052 --percent_edges 0.01 --dataset cora --learning_rate 0.001 --epochs 600 --hidden1 16 --dropout 0.5 --weight_decay 0.0003 --early_stopping 600 --num_layers 2