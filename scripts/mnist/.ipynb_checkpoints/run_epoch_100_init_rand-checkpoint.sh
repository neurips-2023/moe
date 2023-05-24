#!/bin/bash

sbatch aaai_2022/scripts/mnist/schedule_mnist_original_model.sh -g gate_layers_top_k -k 1 -m mnist_top_1 -mt moe_top_k_model -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_original_model.sh -g gate_layers_top_k -k 2 -m mnist_top_2 -mt moe_top_k_model -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_original_model.sh -m mnist_output_mixture  -mt moe_expectation_model -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_single_model.sh -r 10 -E 100