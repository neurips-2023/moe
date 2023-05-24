#!/bin/bash

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_no_gate_model.sh -m cifar10_no_gate_oracle_step_1 -ot moe_stochastic_model -r 10 -M 10 -E 180

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_no_gate_model.sh -m cifar10_no_gate_oracle_step_1 -k 1 -g gate_layers_top_k -ot moe_top_k_model -r 10 -M 10 -E 180

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_no_gate_model.sh -m cifar10_no_gate_oracle_step_1 -k 2 -g gate_layers_top_k  -ot moe_top_k_model  -r 10 -M 10 -E 180
