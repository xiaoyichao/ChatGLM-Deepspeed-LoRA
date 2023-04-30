import torch
import json
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from data_set_sft import format_example



model = AutoModel.from_pretrained("/root/autodl-tmp/chatglm-6b", trust_remote_code=True, load_in_8bit=False)
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/chatglm-6b", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "/app/ChatGLM-Deepspeed-LoRA/output_dir_lora/global_step-48818")
model = model.half().cuda()
model = model.eval()

instructions = json.load(open("/app/ChatGLM-Deepspeed-LoRA/data/test.json"))

answers = []

with torch.no_grad():
    for idx, item in enumerate(instructions[:]):
        example = format_example(item)
        prompt = example["context"]
        target = example["target"]
        ids = tokenizer.encode(prompt)
        input_ids = torch.LongTensor([ids]).cuda()
        generation_kwargs = {
            "min_length": 5,
            "max_new_tokens": 512,
            "top_p": 0.7,
            "temperature": 1,
            "do_sample": False,
            "num_return_sequences": 1,
        }
        out = model.generate(
            input_ids=input_ids,
            **generation_kwargs

        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(prompt, "").replace("\nEND", "").strip()
        item['infer_answer'] = answer
        print(f"### {idx+1}.instruction:\n", item.get('instruction'), '\n\n')
        print(f"### {idx+1}.infer_answer:\n", item.get('infer_answer'), '\n\n')
        print(f"### {idx+1}.target:\n", item.get('output'), '\n\n')
        
        answers.append({'index': idx, **item})