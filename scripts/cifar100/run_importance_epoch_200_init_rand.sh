#!/bin/bash

for i in 0.2 0.4 0.6 0.8 1.0; 
    do 
    sbatch aaai_2022/scripts/cifar100/schedule_cifar100_original_model.sh -m cifar100_stochastic -mt moe_stochastic_model -i $i -r 10 -M 20 -E 200; 
       sbatch aaai_2022/scripts/cifar100/schedule_cifar100_original_model.sh -g gate_layers_top_k -k 2 -m cifar100_top_2 -mt moe_top_k_model -i $i -r 10 -M 20 -E 200;
done