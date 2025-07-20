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
        generated_ids = [o[len(inputs.input_ids[0]):] for o in outputs]
        ans = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return ans.split("assistant")[-1].strip()
    except Exception as e:
        print("Qwen调用出错:", e)
        return ""

def initialize_messages(system_message_content, initial_msgs_template, classification_defs):
    messages = [{"role": "system", "content": system_message_content}]
    for msg in initial_msgs_template:
        content_str = msg["content"].replace("{classification_definitions}", json.dumps(classification_defs, ensure_ascii=False))
        messages.append({"role": msg["role"], "content": content_str})
        resp = call_qwen(messages, max_new_tokens=256)
        print("初始消息回复:\n", resp)
        print("------------------------------")
    print("-----------初始化完毕，进入 2选1 阶段-----------")
    return messages

def main():
    with open("../prompt_2_1.json", "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    system_message_content = prompt_data["system_message"]
    classification_defs = prompt_data["classification_definitions"]
    initial_msgs_template = prompt_data["initial_messages"]
    instruction = prompt_data["instruction"]

    with open("../Text_Embedding/fusion_alpha_0.50.json", "r", encoding="utf-8") as ff:
        fusion_data = json.load(ff)
    fusion_map = {item["test_text"]: item["results"] for item in fusion_data}

    df = pd.read_csv("./6_1_qwen.csv")
    print(f"[INFO] loaded 6_1_qwen.csv => shape: {df.shape}")

    label_map_str2num = {
        "direct":"0","general":"1","indirect":"2","method":"3",
        "others":"4","reference":"5","term":"6"
    }

    base_messages = initialize_messages(system_message_content, initial_msgs_template, classification_defs)

    results = []
    for idx, row in df.iterrows():
        test_text = row["text"]
        cand1 = row.get("predicted_label1", row.get("predicted_label", ""))
        cand2 = row.get("predicted_label2", "")
        true_label = row.get("true_label", "")
        confusion_point = row.get("confusion_point", "")

        if not cand1 or not cand2:
            print(f"[WARN] 缺少候选类别 => 行{idx}")
            continue
        if test_text not in fusion_map:
            print(f"[WARN] 缺少向量样本: {test_text[:30]}...")
            continue
        cat_dict = fusion_map[test_text]

        candidate_samples_str = ""
        for c in [cand1, cand2]:
            cnum = label_map_str2num.get(c.lower())
            if cnum is None:
                continue
            sample_list = cat_dict.get(cnum, [])
            candidate_samples_str += f"\n[类别 {c} top5样例]:"
            for i, s in enumerate(sample_list[:5]):
                candidate_samples_str += f"\n{i+1}. {s['train_text']} (相似度={s['fused_similarity']:.3f})"
            candidate_samples_str += "\n"

        real_prompt = (instruction
                       .replace("{text}", test_text)
                       .replace("{predicted_label1}", cand1)
                       .replace("{predicted_label2}", cand2)
                       .replace("{candidate_samples}", candidate_samples_str)
                       .replace("{confusion_point}", confusion_point))

        base_ctx = copy.deepcopy(base_messages)
        base_ctx.append({"role": "user", "content": real_prompt})

        run_preds = []
        run_reasons = []
        for run_id in range(3):
            reply = call_qwen(copy.deepcopy(base_ctx), max_new_tokens=512)
            print(f"\n=== 样本 {idx+1}/{len(df)} 第{run_id+1}次 ===")
            print(reply, "\n")
            m_final = re.search(r"最终类别为\s*[:：]\s*([^\n]+)", reply)
            m_reason = re.search(r"理由是\s*[:：]\s*([^\n]+)", reply)
            final_cat = (m_final.group(1).strip() if m_final else "未知").lower()
            reason = m_reason.group(1).strip() if m_reason else "未匹配到"
            run_preds.append(final_cat)
            run_reasons.append(reason)

        vote_count = {}
        for p in run_preds:
            vote_count[p] = vote_count.get(p, 0) + 1
        final_cat_lower = sorted(vote_count.items(), key=lambda x: (-x[1], x[0]))[0][0]
        final_reason = ""
        for p, r in zip(run_preds, run_reasons):
            if p == final_cat_lower:
                final_reason = r
                break
        final_result = "正确" if final_cat_lower == true_label.lower() else "错误"

        results.append({
            "text": test_text,
            "true_label": true_label,
            "predicted_label": final_cat_lower,
            "result": final_result,
            "reason": final_reason
        })

    out_df = pd.DataFrame(results)
    out_csv = "2_1_qwen.csv"
    out_df.to_csv(out_csv, index=False, columns=["text","true_label","predicted_label","result","reason"])
    print(f"[INFO] 已处理完全部 {len(out_df)} 条文本，结果写入 => {out_csv}")

if __name__ == "__main__":
    main()
