#!/bin/bash

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_no_gate_model.sh -m cifar10_no_gate_oracle_step_1 -mt moe_no_gate_self_information_model -r 10 -M 10 -E 20

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_no_gate_model.sh -m cifar10_no_gate_loudest -mt moe_no_gate_entropy_model -r 10 -M 10 -E 200

