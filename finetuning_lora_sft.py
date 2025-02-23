# -*- coding:utf-8 -*-
"""
    文件说明：给予LoRA算法进行SFT
    CUDA_VISIBLE_DEVICES=0 deepspeed finetuning_lora_sft.py --num_train_epochs 5 --train_batch_size 2 --lora_r 8 

    CUDA_VISIBLE_DEVICES=0 deepspeed finetuning_lora_sft.py --num_train_epochs 2 --train_batch_size 2 --lora_r 8  && shutdown now

    CUDA_VISIBLE_DEVICES=0,1 deepspeed finetuning_lora_sft.py --num_train_epochs 2 --train_batch_size 2 --lora_r 8  && shutdown now

    
"""
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
import torch
import deepspeed
import argparse
from torch.utils.data import RandomSampler, DataLoader
from data_set_sft import SFTDataSet, coll_fn
import os
import transformers
from shutil import copy
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, \
    set_peft_model_state_dict



def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def set_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train_path', default='/app/ChatGLM-Deepspeed-LoRA/data/test.json', type=str, help='')
    parser.add_argument('--train_path', default='/app/ChatGLM-Deepspeed-LoRA/data/alpaca_gpt4_data_zh_zx.json', type=str, help='')
    parser.add_argument('--model_dir', default="/root/autodl-tmp/chatglm-6b", type=str, help='')
    parser.add_argument('--num_train_epochs', default=1, type=int, help='')
    parser.add_argument('--train_batch_size', default=2, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--output_dir', default='/app/ChatGLM-Deepspeed-LoRA/output_dir_lora/0503-speed', type=str, help='')
    parser.add_argument('--log_steps', type=int, default=10, help='')
    parser.add_argument('--max_seq_length', type=int, default=512, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--lora_r', type=int, default=8, help='')
    # parser.add_argument('--master_port', type=int, default=6666, help='')
    return parser.parse_args()


def main():
    args = set_args()

    model = ChatGLMForConditionalGeneration.from_pretrained(args.model_dir)
    tokenizer = ChatGLMTokenizer.from_pretrained(args.model_dir)
    config = transformers.AutoConfig.from_pretrained(args.model_dir,trust_remote_code=True)
    # Lora_config = LoraConfig(
    #             task_type="CAUSAL_LM",
    #             inference_mode=False,
    #             r=args.lora_r,
    #             lora_alpha=32,
    #             lora_dropout=0.1,
    #         )
    
    # Lora_config = LoraConfig(
    #             task_type="CAUSAL_LM",
    #             inference_mode=False,
    #             r=32,
    #             lora_alpha=32,
    #             lora_dropout=0.1,
    #             target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    #         )
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
    

    conf = {"train_micro_batch_size_per_gpu": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5,
                    "betas": [
                        0.9,
                        0.95
                    ],
                    "eps": 1e-8,
                    "weight_decay": 5e-4
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },

            # "zero_optimization": {
            #     "stage": 2,
            #     "offload_optimizer": {
            #         "device": "cpu",
            #         "pin_memory": True
            #     },
            #     "allgather_partitions": True,
            #     "allgather_bucket_size": 2e8,
            #     "reduce_scatter": True,
            #     "reduce_bucket_size": 2e8,
            #     "overlap_comm": True,
            #     "contiguous_gradients": True
            # },

            # "zero_optimization": {
            #     "stage": 3,
            #     "contiguous_gradients": True,
            #     "stage3_max_live_parameters": 1e9,
            #     "stage3_max_reuse_distance": 1e9,
            #     "stage3_prefetch_bucket_size": 1e7,
            #     "stage3_param_persistence_threshold": 1e5,
            #     "reduce_bucket_size": 1e7,
            #     "sub_group_size": 1e9,
            #     "offload_optimizer": {
            #         "device": "cpu"
            #     },
            #     "offload_param": {
            #         "device": "cpu"
            # }
            # },

            "steps_per_print": args.log_steps
            }

    print_trainable_parameters(model)
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

    train_dataset = SFTDataSet(args.train_path, tokenizer, config, args.max_seq_length)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=conf["train_micro_batch_size_per_gpu"],
                                  sampler=RandomSampler(train_dataset),
                                  collate_fn=coll_fn,
                                  drop_last=True,
                                  num_workers=0)

    model_engine, optimizer, _, _ = deepspeed.initialize(config=conf,
                                                         model=model,
                                                         model_parameters=model.parameters())
    model_engine.train()
    global_step = 0
    for i_epoch in range(args.num_train_epochs):
        train_iter = iter(train_dataloader)
        for step, batch in enumerate(train_iter):
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            outputs = model_engine.forward(input_ids=input_ids, labels=labels)
            loss = outputs[0]
            if conf["gradient_accumulation_steps"] > 1:
                loss = loss / conf["gradient_accumulation_steps"]
            model_engine.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (step + 1) % conf["gradient_accumulation_steps"] == 0:
                model_engine.step()
                global_step += 1
            if global_step % args.log_steps == 0:
                print("loss:{}, global_step:{}".format(float(loss.item()), global_step))
        save_dir = os.path.join(args.output_dir, f"global_step-{global_step}")
        model_engine.save_pretrained(save_dir, overwrite_output_dir=True)
        copy(os.path.join(args.model_dir, "tokenizer_config.json"), os.path.join(save_dir, "tokenizer_config.json"))
        copy(os.path.join(args.model_dir, "ice_text.model"), os.path.join(save_dir, "ice_text.model"))


if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 5555 finetuning_lora_sft.py
