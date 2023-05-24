#!/bin/bash

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_from_no_gate_model.sh -m mnist_no_gate_oracle_step_1 -ot moe_stochastic_model -r 10 -M 10 -E 80

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_from_no_gate_model.sh -m mnist_no_gate_oracle_step_1 -ot moe_top_k_model -k 1 -g gate_layers_top_k -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_from_no_gate_model.sh -m mnist_no_gate_oracle_step_1 -ot moe_top_k_model -k 2 -g gate_layers_top_k -r 10 -M 10 -E 100
