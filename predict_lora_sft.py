
"""
    文件说明：
            
"""
import torch
import json
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from tqdm import tqdm
from data_set_sft import format_example
import time
import os
import argparse
import transformers


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='/app/ChatGLM-Deepspeed-LoRA/data/test.json', type=str, help='')
    parser.add_argument('--device', default='1', type=str, help='')
    parser.add_argument('--ori_model_dir',
                        default="/root/autodl-tmp/chatglm-6b", type=str,
                        help='')
    parser.add_argument('--model_dir',
                        default="/app/ChatGLM-Deepspeed-LoRA/output_dir_lora/global_step-48818", type=str,
                        help='')
    parser.add_argument('--max_seq_length', type=int, default=768, help='')
    return parser.parse_args()


def main():
    args = set_args()
    model = ChatGLMForConditionalGeneration.from_pretrained(args.ori_model_dir)
    tokenizer = ChatGLMTokenizer.from_pretrained(args.ori_model_dir)
    config = transformers.AutoConfig.from_pretrained(args.ori_model_dir,trust_remote_code=True)

    model.eval()
    model = PeftModel.from_pretrained(model, args.model_dir, torch_dtype=torch.float32)
    model.half().to("cuda:{}".format(args.device))
    model.eval()
    save_data = []
    f1 = 0.0
    max_seq_length = args.max_seq_length
    s_time = time.time()
    with open(args.test_path, "r", encoding="utf-8") as fh:
        examples = json.load(fh)
        for sample in tqdm(examples):
            example = format_example(sample)
            prompt = example["context"]
            target = example["target"]
            with torch.no_grad():
                prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
                target_ids = tokenizer.encode(
                    target,
                    max_length=max_seq_length,
                    truncation=True,
                    add_special_tokens=False)
                input_ids = prompt_ids + target_ids + [config.eos_token_id]

                input_ids = torch.tensor([input_ids]).to("cuda:{}".format(args.device))
                generation_kwargs = {
                    "min_length": 5,
                    "max_new_tokens": max_seq_length,
                    "top_p": 0.7,
                    "temperature": 0.95,
                    "do_sample": False,
                    "num_return_sequences": 1,
                }
                response = model.generate(input_ids, **generation_kwargs)
                res = []
                for i_r in range(generation_kwargs["num_return_sequences"]):
                    outputs = response.tolist()[i_r][input_ids.shape[1]:]
                    r = tokenizer.decode(outputs).replace("<eop>", "")
                    res.append(r)
                pre_res = [rr for rr in res[0].split("\n") if len(rr.split("_")) == 3]
                real_res = sample["answer"].split("\n")
                same_res = set(pre_res) & set(real_res)
                if len(set(pre_res)) == 0:
                    p = 0.0
                else:
                    p = len(same_res) / len(set(pre_res))
                r = len(same_res) / len(set(real_res))
                if (p + r) != 0.0:
                    f = 2 * p * r / (p + r)
                else:
                    f = 0.0
                f1 += f
                save_data.append(
                    {"text": sample["text"], "ori_answer": sample["answer"], "gen_answer": res[0], "f1": f})

    e_time = time.time()
    print("总耗时：{}s".format(e_time - s_time))
    print(f1 / 50)
    save_path = os.path.join(args.model_dir, "ft_pt_answer.json")
    fin = open(save_path, "w", encoding="utf-8")
    json.dump(save_data, fin, ensure_ascii=False, indent=4)
    fin.close()


if __name__ == '__main__':
    main()
