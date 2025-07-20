import os
import re
import json
import copy
import random
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "" #替换为实际qwen模型路径
print(f"[INFO] Loading Qwen from {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def call_qwen(messages, max_new_tokens=512, temperature=0.7, top_p=0.9):
    try:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(
            [input_text],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        generated_ids = [output_ids[len(inputs.input_ids[0]):] for output_ids in outputs]
        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer.split("assistant")[-1].strip()
    except Exception as e:
        print("调用本地 Qwen 模型时出错:", e)
        return ""

def initialize_messages(system_message_content, initial_msgs_template, classification_defs):
    messages = [{"role": "system", "content": system_message_content}]
    for msg_tmpl in initial_msgs_template:
        content_str = msg_tmpl["content"].replace("{classification_definitions}", json.dumps(classification_defs, ensure_ascii=False))
        messages.append({"role": msg_tmpl["role"], "content": content_str})
        resp = call_qwen(messages, max_new_tokens=256)
        print("初始消息回复:\n", resp)
        print("------------------------------")
    print("-----------初始化完毕-----------")
    return messages

def main():
    with open("../prompt_6_1.json", "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    system_message_content = prompt_data["system_message"]
    classification_defs = prompt_data["classification_definitions"]
    initial_msgs_template = prompt_data["initial_messages"]
    instruction = prompt_data["instruction"]

    with open("../Text_Embedding/fusion_alpha_0.50.json", "r", encoding="utf-8") as ff:
        fusion_data = json.load(ff)
    fusion_map = {item["test_text"]: item["results"] for item in fusion_data}

    df = pd.read_csv("7_1_qwen.csv")
    print(f"[INFO] loaded 7_1_qwen.csv => shape: {df.shape}")

    label_map_num2str = {
        "0": "direct",
        "1": "general",
        "2": "indirect",
        "3": "method",
        "4": "others",
        "5": "reference",
        "6": "term"
    }

    base_messages = initialize_messages(system_message_content, initial_msgs_template, classification_defs)

    results_list = []
    for idx, row in df.iterrows():
        test_text = row["text"]
        predicted_label = row["predicted_label"]
        true_label = row["true_label"]
        if test_text not in fusion_map:
            print(f"[WARN] 缺少向量样本: {test_text[:20]}...")
            continue
        cat_dict = fusion_map[test_text]

        candidate_samples_str = ""
        for lab_num, sample_list in cat_dict.items():
            if label_map_num2str[lab_num].lower() == predicted_label.lower():
                continue
            text_label = label_map_num2str[lab_num]
            top_5 = sample_list[:5]
            candidate_samples_str += f"\n[类别 {text_label} top5样例]:"
            for i, sitem in enumerate(top_5):
                cand_text = sitem["train_text"]
                sim_val = sitem["fused_similarity"]
                candidate_samples_str += f"\n{i+1}. {cand_text} (相似度={sim_val:.3f})"
            candidate_samples_str += "\n"

        real_prompt = instruction.replace("{text}", test_text).replace("{predicted_label}", predicted_label).replace("{candidate_samples}", candidate_samples_str)

        base_ctx = copy.deepcopy(base_messages)
        base_ctx.append({"role": "user", "content": real_prompt})

        runs = []
        for run_id in range(3):
            reply = call_qwen(copy.deepcopy(base_ctx), max_new_tokens=512)
            print(f"\n=== 样本 {idx+1}/{len(df)} 第{run_id+1}次 ===")
            print(reply, "\n")
            match_label = re.search(r"该规范的备选类别为\s*[:：]\s*([^\n]+)", reply)
            alt_cat = match_label.group(1).strip() if match_label else "未知"
            match_conf = re.search(r"混淆点为\s*[:：]\s*([^\n]+)", reply)
            confusion_point_once = match_conf.group(1).strip() if match_conf else "未匹配到"
            runs.append((alt_cat, confusion_point_once))

        vote = {}
        for cat, _ in runs:
            key = cat.lower()
            vote[key] = vote.get(key, 0) + 1
        majority_lower = sorted(vote.items(), key=lambda x: (-x[1], x[0]))[0][0]
        majority_cat = None
        conf_points = []
        for cat, cp in runs:
            if cat.lower() == majority_lower:
                conf_points.append(cp)
                if majority_cat is None:
                    majority_cat = cat
        predicted_label2 = majority_cat
        confusion_point = random.choice(conf_points) if conf_points else "未匹配到"
        final_result = "正确" if true_label.lower() in [predicted_label.lower(), predicted_label2.lower()] else "错误"

        results_list.append({
            "text": test_text,
            "predicted_label": predicted_label,
            "predicted_label2": predicted_label2,
            "true_label": true_label,
            "result": final_result,
            "confusion_point": confusion_point
        })

    out_df = pd.DataFrame(results_list)
    out_csv = "6_1_qwen.csv"
    out_df.to_csv(out_csv, index=False, columns=["text", "predicted_label", "predicted_label2", "true_label", "result", "confusion_point"])
    print(f"[INFO] 输出 {out_csv} 共 {len(out_df)} 条")

if __name__ == "__main__":
    main()
