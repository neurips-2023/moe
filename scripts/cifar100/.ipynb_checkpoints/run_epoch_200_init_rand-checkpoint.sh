#!/bin/bash

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_original_model.sh -m cifar100_stochastic -mt moe_stochastic_model -r 10 -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_original_model.sh -g gate_layers_top_k -k 1 -m cifar100_top_1 -mt moe_top_k_model -r 10 -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_original_model.sh -g gate_layers_top_k -k 2 -m cifar100_top_2 -mt moe_top_k_model -r 10 -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_original_model.sh -m cifar100_output_mixture  -mt moe_expectation_model -r 10 -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_single_model.sh
