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

def call_deepseek(messages, max_new_tokens=512):
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            max_tokens=max_new_tokens,
            stream=False
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print("调用 DeepSeek 模型时出错:", e)
        return ""

def initialize_messages(system_message_content, initial_msgs_template, classification_defs):
    messages = []
    messages.append({"role": "system", "content": system_message_content})
    for msg_tmpl in initial_msgs_template:
        content_str = msg_tmpl["content"].replace(
            "{classification_definitions}", json.dumps(classification_defs, ensure_ascii=False)
        )
        messages.append({"role": msg_tmpl["role"], "content": content_str})
        resp = call_deepseek(messages, max_new_tokens=128)
        print("大模型对该条初始消息的回答：\n", resp)
        print("------------------------------")
    print("-----------初始化完毕，开始 2选1 分类-----------")
    return messages

def main():
    with open("../../prompt_2_1.json", "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    system_message_content = prompt_data["system_message"]
    classification_defs = prompt_data["classification_definitions"]
    initial_msgs_template = prompt_data["initial_messages"]
    instruction = prompt_data["instruction"]

    with open("../../Text_Embedding/fusion_alpha_0.50.json", "r", encoding="utf-8") as ff:
        fusion_data = json.load(ff)
    fusion_map = {}
    for item in fusion_data:
        txt = item["test_text"]
        fusion_map[txt] = item["results"]

    df = pd.read_csv("./6_1_deepseek.csv")
    print(f"[INFO] loaded 6_1_deepseek.csv => shape: {df.shape}")

    label_map_str2num = {
        "direct":    "0",
        "general":   "1",
        "indirect":  "2",
        "method":    "3",
        "others":    "4",
        "reference": "5",
        "term":      "6"
    }

    base_messages = initialize_messages(system_message_content, initial_msgs_template, classification_defs)

    results_list = []
    for idx, row in df.iterrows():
        test_text = row["text"]
        predicted_label1 = row["predicted_label1"]
        predicted_label2 = row["predicted_label2"]
        true_label = row["true_label"]
        confusion_point = row["confusion_point"]

        if test_text not in fusion_map:
            print(f"[WARN] 未在 fusion_map 中找到文本 => {test_text[:20]}... 跳过该条。")
            continue
        cat_dict = fusion_map[test_text]

        candidate_samples_str = ""
        for candidate in [predicted_label1, predicted_label2]:
            candidate_num = label_map_str2num.get(candidate)
            if candidate_num is None:
                print(f"[WARN] 候选类别 {candidate} 无法映射到数字标签，跳过。")
                continue
            sample_list = cat_dict.get(candidate_num, [])
            candidate_samples_str += f"\n[类别 {candidate} top5样例]:"
            for i, sitem in enumerate(sample_list[:5]):
                cand_text = sitem["train_text"]
                sim_val = sitem["fused_similarity"]
                candidate_samples_str += f"\n{i+1}. {cand_text} (相似度={sim_val:.3f})"
            candidate_samples_str += "\n"

        real_prompt = instruction
        real_prompt = real_prompt.replace("{text}", test_text)
        real_prompt = real_prompt.replace("{predicted_label1}", predicted_label1)
        real_prompt = real_prompt.replace("{predicted_label2}", predicted_label2)
        real_prompt = real_prompt.replace("{candidate_samples}", candidate_samples_str)
        real_prompt = real_prompt.replace("{confusion_point}", confusion_point)

        base_ctx = copy.deepcopy(base_messages)
        base_ctx.append({"role": "user", "content": real_prompt})

        runs = []
        for _ in range(3):
            reply = call_deepseek(copy.deepcopy(base_ctx), max_new_tokens=512)
            print(f"\n=== 第 {idx+1}/{len(df)} 条文本 运行 ===")
            print("DeepSeek 回复：\n", reply, "\n")
            match_final = re.search(r"最终类别为\s*[:：]\s*([^\n]+)", reply)
            final_category = match_final.group(1).strip() if match_final else "未知"
            match_reason = re.search(r"理由是\s*[:：]\s*([^\n]+)", reply)
            reason = match_reason.group(1).strip() if match_reason else "未匹配到"
            runs.append((final_category, reason))

        vote_counts = {}
        for cat, _ in runs:
            vote_counts[cat.lower()] = vote_counts.get(cat.lower(), 0) + 1
        majority_cat_lower = sorted(vote_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        majority_cat = None
        majority_reasons = []
        for cat, rsn in runs:
            if cat.lower() == majority_cat_lower:
                majority_reasons.append(rsn)
                if majority_cat is None:
                    majority_cat = cat
        final_category = majority_cat
        final_reason = random.choice(majority_reasons) if majority_reasons else "未匹配到"
        final_result = "正确" if final_category.lower() == true_label.lower() else "错误"

        results_list.append({
            "text": test_text,
            "true_label": true_label,
            "predicted_label": final_category,
            "result": final_result,
            "reason": final_reason
        })

    out_df = pd.DataFrame(results_list)
    out_csv = "2_1_deepseek.csv"
    out_df.to_csv(
        out_csv,
        index=False,
        columns=["text", "true_label", "predicted_label", "result", "reason"]
    )
    print(f"[INFO] 已处理完全部 {len(out_df)} 条文本，结果写入 => {out_csv}")

if __name__ == "__main__":
    main()
