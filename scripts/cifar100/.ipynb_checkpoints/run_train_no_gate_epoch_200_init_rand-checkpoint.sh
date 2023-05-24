#!/bin/bash

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_train_no_gate_model.sh -m cifar100_no_gate_oracle_step_1 -mt moe_no_gate_self_information_model -r 10 -M 20 -E 40

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_train_no_gate_model.sh -m cifar100_no_gate_loudest -mt moe_no_gate_entropy_model -r 10 -M 20 -E 200
