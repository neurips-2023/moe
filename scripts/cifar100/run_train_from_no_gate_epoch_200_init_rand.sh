#!/bin/bash

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_train_from_no_gate_model.sh -m cifar100_no_gate_oracle_step_1 -ot moe_stochastic_model -r 10 -M 20 -E 160

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_train_from_no_gate_model.sh -m cifar100_no_gate_oracle_step_1 -ot moe_top_k_model -k 2 -g gate_layers_top_k -r 10 -M 20 -E 160



        