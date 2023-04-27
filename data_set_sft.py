# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_set
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/4 14:42
"""
    文件说明：
            
"""
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SFTDataSet(Dataset):
    """数据处理函数"""
    def __init__(self, data_path, tokenizer, config, max_seq_length):
        # prompt_text = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："

        self.all_data = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                example = self.format_example(sample)
                prompt = example["context"]
                target = example["target"]
                prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
                target_ids = tokenizer.encode(
                    target,
                    max_length=max_seq_length,
                    truncation=True,
                    add_special_tokens=False)
                input_ids = prompt_ids + target_ids + [config.eos_token_id]
                format_data = {"input_ids": input_ids, "seq_len": len(prompt_ids)}
                self.all_data.append(format_data)
                            

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance

    def format_example(self, example: dict) -> dict:
        context = f"Instruction: {example['instruction']}\n"
        if example.get("input"):
            context += f"Input: {example['input']}\n"
        context += "Answer: "
        target = example["output"]
        return {"context": context, "target": target}


def coll_fn(batch):
    input_ids_list, labels_list = [], []
    for instance in batch:
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=20003),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=20003)}
