import os
import re
import json
import copy
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "" #替换为实际qwen模型路径
print(f"[INFO] Loading Qwen from {model_path}")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

def call_qwen(messages, max_new_tokens=512, temperature=0.7, top_p=0.9):
    try:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        gen_ids = [o[len(inputs.input_ids[0]):] for o in outputs]
        ans = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return ans.split("assistant")[-1].strip()
    except Exception as e:
        print("调用本地 Qwen 模型时出错:", e)
        return ""

def initialize_messages(system_message_content, initial_msgs_template, classification_defs):
    messages = [{"role": "system", "content": system_message_content}]
    for tmpl in initial_msgs_template:
        content = tmpl["content"].replace("{classification_definitions}", json.dumps(classification_defs, ensure_ascii=False))
        messages.append({"role": tmpl["role"], "content": content})
        resp = call_qwen(messages, max_new_tokens=128)
        print("初始消息回复:\n", resp)
        print("------------------------------")
    print("-----------初始化完毕，开始 2选1 分类-----------")
    return messages

def majority_vote(labels):
    freq = {}
    for l in labels:
        freq[l] = freq.get(l, 0) + 1
    return sorted(freq.items(), key=lambda x: (-x[1], x[0]))[0][0]

def main():
    with open("../../prompt_2_1.json", "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    system_message_content = prompt_data["system_message"]
    classification_defs = prompt_data["classification_definitions"]
    initial_msgs_template = prompt_data["initial_messages"]
    chain_of_thought_prompt = prompt_data["chain_of_thought_prompt"]

    with open("../../Text_Embedding/fusion_alpha_0.50.json", "r", encoding="utf-8") as ff:
        fusion_data = json.load(ff)
    fusion_map = {item["test_text"]: item["results"] for item in fusion_data}

    df = pd.read_csv("./6_1_qwen.csv")
    print(f"[INFO] loaded 6_1_qwen.csv => shape: {df.shape}")

    label_map_str2num = {
        "direct":"0","general":"1","indirect":"2","method":"3","others":"4","reference":"5","term":"6"
    }

    base_messages = initialize_messages(system_message_content, initial_msgs_template, classification_defs)

    results_list = []
    for idx, row in df.iterrows():
        test_text = row["text"]
        predicted_label1 = row["predicted_label1"]
        predicted_label2 = row["predicted_label2"]
        true_label = row["true_label"]
        confusion_point = row.get("confusion_point", "")

        if test_text not in fusion_map:
            print(f"[WARN] 未在 fusion_map 中找到文本 => {test_text[:20]}... 跳过")
            continue
        cat_dict = fusion_map[test_text]

        candidate_samples_str = ""
        for candidate in [predicted_label1, predicted_label2]:
            cand_id = label_map_str2num.get(candidate)
            if cand_id is None:
                continue
            sample_list = cat_dict.get(cand_id, [])
            candidate_samples_str += f"\n[类别 {candidate} top5样例]:"
            for i, sitem in enumerate(sample_list[:5]):
                candidate_samples_str += f"\n{i+1}. {sitem['train_text']} (相似度={sitem['fused_similarity']:.3f})"
            candidate_samples_str += "\n"

        real_prompt = (chain_of_thought_prompt
                       .replace("{text}", test_text)
                       .replace("{predicted_label1}", predicted_label1)
                       .replace("{predicted_label2}", predicted_label2)
                       .replace("{candidate_samples}", candidate_samples_str)
                       .replace("{confusion_point}", confusion_point))

        messages_base = copy.deepcopy(base_messages)
        messages_base.append({"role": "user", "content": real_prompt})

        run_final_labels = []
        run_reasons = []
        for run_id in range(3):
            reply = call_qwen(copy.deepcopy(messages_base), max_new_tokens=512)
            print(f"\n=== 第 {idx+1}/{len(df)} 条文本 | 第{run_id+1}次推理 ===")
            print("Qwen 回复：\n", reply, "\n")
            m_cat = re.search(r"最终类别为\s*[:：]\s*([^\n]+)", reply)
            final_category = m_cat.group(1).strip() if m_cat else "未知"
            m_reason = re.search(r"理由是\s*[:：]\s*([^\n]+)", reply)
            reason = m_reason.group(1).strip() if m_reason else "未匹配到"
            run_final_labels.append(final_category)
            run_reasons.append(reason)

        voted_category = majority_vote(run_final_labels)
        chosen_reason = ""
        for c, r in zip(run_final_labels, run_reasons):
            if c == voted_category:
                chosen_reason = r
                break

        final_result = "正确" if voted_category.lower() == true_label.lower() else "错误"

        results_list.append({
            "text": test_text,
            "true_label": true_label,
            "predicted_label": voted_category,
            "result": final_result,
            "reason": chosen_reason
        })

    out_df = pd.DataFrame(results_list)
    out_csv = "2_1_qwen.csv"
    out_df.to_csv(out_csv, index=False, columns=["text","true_label","predicted_label","result","reason"])
    print(f"[INFO] 已处理完全部 {len(out_df)} 条文本，结果写入 => {out_csv}")

if __name__ == "__main__":
    main()
