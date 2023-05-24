#!/bin/bash

for i in 0.2 0.4 0.6 0.8 1.0; 
    do sbatch aaai_2022/scripts/mnist/schedule_mnist_original_model.sh -m mnist_output_mixture -i $i -r 10 -M 10 -E 100; 
       sbatch aaai_2022/scripts/mnist/schedule_mnist_original_model.sh -g gate_layers_top_k -k 2 -m mnist_top_2 -mt moe_top_k_model -i $i -r 10 -M 10 -E 100;
done