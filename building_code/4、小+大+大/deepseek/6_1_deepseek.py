import os
import re
import json
import copy
import pandas as pd
import random
from openai import OpenAI

client = OpenAI(
    api_key="",  # API密钥
    base_url=""  # DeepSeek服务端地址
)

def call_deepseek(messages, max_tokens=512):
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            max_tokens=max_tokens,
            stream=False
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print("调用 DeepSeek 模型时出错:", e)
        return ""

def initialize_messages(system_message_content, classification_defs, initial_msgs_template):
    messages = []
    messages.append({"role": "system", "content": system_message_content})
    for msg_tmpl in initial_msgs_template:
        content_str = msg_tmpl["content"].replace(
            "{classification_definitions}", json.dumps(classification_defs, ensure_ascii=False)
        )
        messages.append({"role": "user", "content": content_str})
        resp = call_deepseek(messages, max_tokens=1024)
        print("大模型对初始消息的回答：\n", resp)
        print("-" * 30)
        messages.append({"role": "assistant", "content": resp})
    print("-----------初始化完毕，构造原始对话上下文-----------")
    return messages

def main():
    with open("../../prompt_6_1.json", "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    system_message_content = prompt_data["system_message"]
    classification_defs = prompt_data["classification_definitions"]
    initial_msgs_template = prompt_data["initial_messages"]
    instruction = prompt_data["instruction"]

    roberta_file = "../../results/bert_predictions.csv"
    roberta_df = pd.read_csv(roberta_file)
    print(f"[INFO] loaded roberta_df => shape: {roberta_df.shape}")

    label_map_str2num = {
        "direct":    "0",
        "general":   "1",
        "indirect":  "2",
        "method":    "3",
        "others":    "4",
        "reference": "5",
        "term":      "6"
    }
    label_map_num2str = {
        "0": "direct",
        "1": "general",
        "2": "indirect",
        "3": "method",
        "4": "others",
        "5": "reference",
        "6": "term"
    }

    with open("../../Text_Embedding/fusion_alpha_0.50.json", "r", encoding="utf-8") as ff:
        fusion_data = json.load(ff)
    fusion_map = {}
    for item in fusion_data:
        txt = item["test_text"]
        fusion_map[txt] = item["results"]

    base_messages = initialize_messages(system_message_content, classification_defs, initial_msgs_template)

    results_list = []
    for idx, row in roberta_df.iterrows():
        test_text = row["text"]
        predicted_label1 = row["predicted_label"]
        confidence = row["confidence"]
        true_label = row["true_label"]

        skip_num = label_map_str2num.get(predicted_label1, "未知")
        if skip_num == "未知":
            print(f"[WARN] 小模型预测类别 {predicted_label1} 无法映射 => 跳过该行。")
            continue
        if test_text not in fusion_map:
            print(f"[WARN] 测试文本未在 fusion_map 中 => {test_text[:20]}... 跳过该行。")
            continue
        cat_dict = fusion_map[test_text]

        candidate_samples_str = ""
        print(f"\n=== 第 {idx+1}/{len(roberta_df)} 条文本 ===")
        print(f"文本[:20]: {test_text[:20]}")
        print(f"小模型预测类别: {predicted_label1}, skip_num: {skip_num}.")
        for lab_num, sample_list in cat_dict.items():
            if lab_num == skip_num:
                continue
            text_label = label_map_num2str[lab_num]
            candidate_samples_str += f"\n[类别 {text_label} top5样例]:"
            for i, sitem in enumerate(sample_list[:5]):
                cand_text = sitem["train_text"]
                sim_val = sitem["fused_similarity"]
                candidate_samples_str += f"\n{i+1}. {cand_text} (相似度={sim_val:.3f})"
            candidate_samples_str += "\n"
        print("----- 候选样本信息构造完毕 -----\n")

        real_prompt = instruction
        real_prompt = real_prompt.replace("{text}", test_text)
        real_prompt = real_prompt.replace("{predicted_label}", predicted_label1)
        real_prompt = real_prompt.replace("{confidence}", f"{confidence:.4f}")
        real_prompt = real_prompt.replace("{candidate_samples}", candidate_samples_str)

        messages_base = copy.deepcopy(base_messages)
        messages_base.append({"role": "user", "content": real_prompt})

        runs = []
        for _ in range(3):
            reply = call_deepseek(copy.deepcopy(messages_base), max_tokens=512)
            print("[DeepSeek] reply =>\n", reply, "\n")
            match_label = re.search(r"该规范的备选类别为\s*[:：]\s*([^\n]+)", reply)
            predicted_label2 = match_label.group(1).strip() if match_label else "未知"
            match_conf = re.search(r"的混淆点为\s*[:：]\s*([\s\S]+)", reply)
            confusion_point = match_conf.group(1).strip() if match_conf else ""
            runs.append((predicted_label2, confusion_point))

        vote_counts = {}
        for lbl, _ in runs:
            vote_counts[lbl.lower()] = vote_counts.get(lbl.lower(), 0) + 1
        majority_label_lower = sorted(vote_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        majority_label = None
        majority_confusions = []
        for lbl, conf in runs:
            if lbl.lower() == majority_label_lower:
                majority_confusions.append(conf)
                if majority_label is None:
                    majority_label = lbl
        final_predicted_label2 = majority_label
        final_confusion_point = random.choice(majority_confusions) if majority_confusions else ""

        final_result = "错误"
        if true_label in [predicted_label1, final_predicted_label2]:
            final_result = "正确"

        results_list.append({
            "text": test_text,
            "predicted_label1": predicted_label1,
            "predicted_label2": final_predicted_label2,
            "true_label": true_label,
            "result": final_result,
            "confusion_point": final_confusion_point
        })

    out_df = pd.DataFrame(results_list)
    out_csv = "6_1_deepseek.csv"
    out_df.to_csv(
        out_csv,
        index=False,
        columns=["text", "predicted_label1", "predicted_label2", "true_label", "result", "confusion_point"]
    )
    print(f"[INFO] 已处理完全部 {len(out_df)} 条文本，结果写入 => {out_csv}")

if __name__ == "__main__":
    main()
