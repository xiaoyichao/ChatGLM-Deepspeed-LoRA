## 使用Deepspeed 在ChatGLM上基于LoRA方法的微调

LoRA参数已经在项目的output/0503-speed/global_step-24414文件夹中共享。
对比效果

![image](https://github.com/xiaoyichao/ChatGLM-Deepspeed-LoRA/blob/master/images/%E5%AF%B9%E6%AF%94.png)


生成requirements.txt

    pipreqs ./     --force

安装环境

    pip install -r requirements.txt

如何安装peft==0.3.0.dev0

    pip install peft==0.3.0.dev0 -i https://pypi.tuna.tsinghua.edu.cn/simple
    或者
    pip install peft==0.3.0.dev0 -i git+https://ghproxy.com/https://github.com/huggingface/peft.git

如何安装mpi4py

    conda install --channel https://conda.anaconda.org/dhirschfeld mpi4py

如何运行程序

    CUDA_VISIBLE_DEVICES=0 deepspeed finetuning_lora_sft.py --num_train_epochs 2 --train_batch_size 2 --lora_r 8  && shutdown now
    或者 
    nohup bash run_glm_6b_SFT.sh  && shutdown now > nohup.out 2>&1 &
    或者
    nohup bash run_glm_6b_SFT.sh  > nohup.out 2>&1 &

web_demo
    修改./output/0504-speed/global_step-122070文件夹中adapter_config.json
    参数 "inference_mode"为false
    
    python web_demo_lora.py

如何查看程序是否还在运行

    ps -ef|grep web_demo_lora.py
    ps -ef|grep finetuning_lora_sft.py

数据集来源

    https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM

## Reference

> 非常感谢以下作者的无私开源

- https://github.com/liucongg/ChatGLM-Finetuning
- https://github.com/yanqiangmiffy/InstructGLM
- https://github.com/mymusise/ChatGLM-Tuning
- https://huggingface.co/BelleGroup/BELLE-7B-2M
- https://github.com/LianjiaTech/BELLE
- https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN
- https://huggingface.co/datasets/JosephusCheung/GuanacoDataset
- https://guanaco-model.github.io/
- https://github.com/carbonz0/alpaca-chinese-dataset
- https://github.com/THUDM/ChatGLM-6B
- https://huggingface.co/THUDM/chatglm-6b
- https://github.com/lich99/ChatGLM-finetune-LoRA


