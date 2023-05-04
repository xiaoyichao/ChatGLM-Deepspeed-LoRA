import json
import os
import sys

txt_path = "data/knowledge_based_docs.txt"
data_path = "data/knowledge_zx.json"

with open(txt_path,"r") as f:
    json_list = []
    lines = f.readlines()
    num = 0
    for line in lines:
        line = line.strip()
        line = json.loads(line)
        title = line["title"]
        if title is not None:
            if "?" in title or "？" in title:
                remark = line["remark"]
                print("title", title)
                print("remark", remark)
                if_use = input("if use:")
                if  if_use != "0":
                    Instruction = input("Instruction:")
                    Instruction = Instruction.strip()
                    print("请输入Answer，以 Ctrl-D 结束输入：")
                    Answer = sys.stdin.read()
                    if Instruction != Instruction:
                        dict_data = {
                                "instruction": Instruction,
                                "input": "",
                                "output": Answer
                                }
                        json_list.append(dict_data)
                        num+=1
                    else:
                        print("这条数据不要了。")
                os.system('cls' if os.name == 'nt' else 'clear')
                print("可用数据：", num)
                if num>=10:
                    break



with open(data_path, "w") as f:
    json.dump(json_list, f, ensure_ascii=False, indent=2)
    print("数据写入完成")