#!/bin/bash

for i in 0.2 0.4 0.6 0.8 1.0; 
    do sbatch aaai_2022/scripts/cifar10/schedule_cifar10_original_model.sh -m cifar10_output_mixture -i $i -r 10 -M 10 -E 200; 
       sbatch aaai_2022/scripts/cifar10/schedule_cifar10_original_model.sh -g gate_layers_top_k -k 2 -m cifar10_top_2 -mt moe_top_k_model -i $i -r 10 -M 10 -E 200;
done