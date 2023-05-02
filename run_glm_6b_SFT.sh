#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# nohup CUDA_VISIBLE_DEVICES=0 deepspeed finetuning_lora_sft.py --num_train_epochs 2 --train_batch_size 2 --lora_r 8   > nohup_lora.out 2>&1 &
# DeepSpeed Team
OUTPUT_PATH=./output/0502-3/
mkdir -p $OUTPUT_PATH

deepspeed finetuning_lora_sft.py \
    --num_train_epochs 1 \
    --train_batch_size 2 \
    --lora_r 8 \
   --output_dir $OUTPUT_PATH \
   &> $OUTPUT_PATH/training.log