# 2_1_qwen.py

import os
import re
import json
import copy
import pandas as pd
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
model_path = "" #替换为实际qwen路径
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
        answer_cleaned = answer.split("assistant")[-1].strip()
        return answer_cleaned
    except Exception as e:
        print("调用本地 Qwen 模型时出错:", e)
        return ""

def initialize_messages(system_message_content, initial_msgs_template, classification_defs):
    messages = []
    messages.append({"role": "system", "content": system_message_content})
    for msg_tmpl in initial_msgs_template:
        content_str = msg_tmpl["content"].replace(
            "{classification_definitions}",
            json.dumps(classification_defs, ensure_ascii=False)
        )
        messages.append({"role": msg_tmpl["role"], "content": content_str})
        resp = call_qwen(messages, max_new_tokens=128)
        print("大模型对该条初始消息的回答：\n", resp)
        print("------------------------------")
    print("-----------初始化完毕，开始 2选1 分类-----------")
    return messages

def main():
    prompt_file = "../prompt.json"
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    system_message_content = prompt_data["system_message"]
    classification_defs = prompt_data["classification_definitions"]
    initial_msgs_template = prompt_data["initial_messages"]
    instruction = prompt_data["instruction"]
    result_json_path = "../results/results.json"
    with open(result_json_path, "r", encoding="utf-8") as ff:
        fusion_data = json.load(ff)
    fusion_map = {}
    for item in fusion_data:
        txt = item["test_text"]
        cat_dict = item["results"]
        fusion_map[txt] = cat_dict
    df = pd.read_csv("./false.csv")
    print(f"[INFO] loaded 6_1_qwen.csv => shape: {df.shape}")
    base_messages = initialize_messages(system_message_content, initial_msgs_template, classification_defs)
    results_list = []
    for idx, row in df.iterrows():
        test_title = row.get("title", "")
        test_text = row["text"]
        predicted_label1 = row["predicted_label1"]
        predicted_label2 = row["predicted_label2"]
        true_label = row.get("true_label", "")
        confusion_point = row.get("confusion_point", "")
        if test_text not in fusion_map:
            print(f"[WARN] 未在 result.json 中找到 => {test_text[:20]}... 跳过。")
            continue
        cat_dict = fusion_map[test_text]
        candidate_samples_str = ""
        for candidate_cat in [predicted_label1, predicted_label2]:
            if candidate_cat not in cat_dict:
                print(f"[WARN] 类别 {candidate_cat} 不在 result.json => 跳过显示其相似样本。")
                continue
            sample_list = cat_dict[candidate_cat]
            candidate_samples_str += f"\n[Category: {candidate_cat} | Top-3 samples]"
            for i, sitem in enumerate(sample_list[:3]):
                cand_text = sitem["train_text"]
                sim_val = sitem.get("similarity", 0.0)
                candidate_samples_str += f"\n{i+1}. {cand_text} (similarity={sim_val:.3f})"
            candidate_samples_str += "\n"
        real_prompt = instruction
        real_prompt = real_prompt.replace("{title}", test_title)
        real_prompt = real_prompt.replace("{text}", test_text)
        real_prompt = real_prompt.replace("{predicted_label1}", predicted_label1)
        real_prompt = real_prompt.replace("{predicted_label2}", predicted_label2)
        real_prompt = real_prompt.replace("{candidate_samples}", candidate_samples_str)
        real_prompt = real_prompt.replace("{confusion_point}", confusion_point)
        messages_base = copy.deepcopy(base_messages)
        messages_base.append({"role": "user", "content": real_prompt})
        runs = []
        for _ in range(3):
            reply = call_qwen(copy.deepcopy(messages_base), max_new_tokens=512)
            match_final = re.search(r"specification is\s*[:：]\s*([^\n]+)", reply, flags=re.IGNORECASE)
            final_category = match_final.group(1).strip() if match_final else "未知"
            match_reason = re.search(r"rationale is\s*[:：]\s*([^\n]+)", reply, flags=re.IGNORECASE)
            reason = match_reason.group(1).strip() if match_reason else "未解析到理由"
            runs.append((final_category, reason))
        vote_counts = {}
        for c, r in runs:
            vote_counts[c.lower()] = vote_counts.get(c.lower(), 0) + 1
        majority_label_lower = sorted(vote_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        majority_label = None
        candidate_reasons = []
        for c, r in runs:
            if c.lower() == majority_label_lower:
                candidate_reasons.append(r)
                if majority_label is None:
                    majority_label = c
        final_category = majority_label
        reason = random.choice(candidate_reasons) if candidate_reasons else "未解析到理由"
        final_result = "正确" if final_category.lower() == true_label.lower() else "错误"
        print(f"\n=== 第{idx+1}/{len(df)}条文本 ===")
        print("3次预测：", runs)
        print("最终类别：", final_category, " 理由：", reason)
        results_list.append({
            "title": test_title,
            "text": test_text,
            "predict_label": final_category,
            "true_label": true_label,
            "result": final_result,
            "reason": reason
        })
    out_df = pd.DataFrame(results_list)
    out_csv = "2_1_qwen.csv"
    out_df.to_csv(
        out_csv,
        index=False,
        columns=["title", "text", "predict_label", "true_label", "result", "reason"]
    )
    print(f"[INFO] 完成，共处理 {len(out_df)} 条文本，结果已写入 => {out_csv}")

if __name__ == "__main__":
    main()
