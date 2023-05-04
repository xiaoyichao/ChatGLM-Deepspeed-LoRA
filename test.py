#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import transformers
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
from data_set_sft import SFTDataSet, coll_fn

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, \
    set_peft_model_state_dict
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
# from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
# from utils.model.model_utils import create_hf_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    # parser.add_argument('--data_path',
    #                     nargs='*',
    #                     default=['Dahoas/rm-static'],
    #                     help='Path to the training dataset. Accepted format:'
    #                     '1) a single data path, 2) multiple datasets in the'
    #                     'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `6,2,2`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    # parser.add_argument("--num_train_epochs",
    #                     type=int,
    #                     default=1,
    #                     help="Total number of training epochs to perform.")
    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     type=int,
    #     default=1,
    #     help=
    #     "Number of updates steps to accumulate before performing a backward/update pass.",
    # )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    # parser.add_argument("--output_dir",
    #                     type=str,
    #                     default=None,
    #                     help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    # parser.add_argument("--local_rank",
    #                     type=int,
    #                     default=-1,
    #                     help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')

    # 自己添加的参数
    parser.add_argument('--train_path', default='/app/ChatGLM-Deepspeed-LoRA/data/test.json', type=str, help='')
    # parser.add_argument('--train_path', default='/app/ChatGLM-Deepspeed-LoRA/data/alpaca_gpt4_data_zh_zx.json', type=str, help='')
    parser.add_argument('--model_dir', default="/root/autodl-tmp/chatglm-6b", type=str, help='')
    parser.add_argument('--num_train_epochs', default=1, type=int, help='')
    parser.add_argument('--train_batch_size', default=2, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--output_dir', default='/app/ChatGLM-Deepspeed-LoRA/output_dir_lora/0503-speed', type=str, help='')
    # parser.add_argument('--log_steps', type=int, default=10, help='')
    parser.add_argument('--max_seq_length', type=int, default=512, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--lora_r', type=int, default=8, help='')


    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    assert not args.offload, "zero-offload is not currently supported but coming soon!"

    torch.distributed.barrier()

    model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_dir)
    config = transformers.AutoConfig.from_pretrained(args.model_dir,trust_remote_code=True)

    # tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # tokenizer.pad_token = tokenizer.eos_token

    # model = create_hf_model(AutoModelForCausalLM,
    #                         args.model_name_or_path,
    #                         tokenizer,
    #                         ds_config,
    #                         disable_dropout=args.disable_dropout)

    Lora_config = LoraConfig(r=args.lora_r,
                        lora_alpha=32,
                        target_modules=["query_key_value"],
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM",
                        inference_mode=False,
                        )

    model = get_peft_model(model, Lora_config)
    model = model.cuda()

    # Prepare the data
    train_phase = 1
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
        sft_only_data_path=args.sft_only_data_path)


    train_dataset = SFTDataSet(args.train_path, tokenizer, config, args.max_seq_length)
    
    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        # eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        # eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=coll_fn,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    # eval_dataloader = DataLoader(eval_dataset,
    #                              collate_fn=default_data_collator,
    #                              sampler=eval_sampler,
    #                              batch_size=args.per_device_eval_batch_size)

    # def evaluation(model, eval_dataloader):
    #     model.eval()
    #     losses = 0
    #     for step, batch in enumerate(eval_dataloader):
    #         batch = to_device(batch, device)
    #         with torch.no_grad():
    #             outputs = model(**batch)

    #         loss = outputs.loss
    #         losses += loss.float()
    #     losses = losses / (step + 1)
    #     try:
    #         perplexity = torch.exp(losses)
    #     except OverflowError:
    #         perplexity = float("inf")
    #     try:
    #         perplexity = get_all_reduce_mean(perplexity).item()
    #     except:
    #         pass
    #     return perplexity

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            model.step()

        # # Evaluate perplexity on the validation set.
        # print_rank_0(
        #     f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
        #     args.global_rank)
        # perplexity = evaluation(model, eval_dataloader)
        # print_rank_0(f"ppl: {perplexity}", args.global_rank)
        # model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        # model = convert_lora_to_linear_layer(model)

        # if args.global_rank == 0:
        #     save_hf_format(model, tokenizer, args)

        # if args.zero_stage == 3:
        #     # For zero stage 3, each gpu only has a part of the model, so we need a special save function
        #     save_zero_three_model(model,
        #                           args.global_rank,
        #                           args.output_dir,
        #                           zero_stage=args.zero_stage)
        save_dir = os.path.join(args.output_dir, f"global_step-{global_step}")
        model_engine.save_pretrained(save_dir, overwrite_output_dir=True)
        copy(os.path.join(args.model_dir, "tokenizer_config.json"), os.path.join(save_dir, "tokenizer_config.json"))
        copy(os.path.join(args.model_dir, "ice_text.model"), os.path.join(save_dir, "ice_text.model"))



if __name__ == "__main__":
    main()