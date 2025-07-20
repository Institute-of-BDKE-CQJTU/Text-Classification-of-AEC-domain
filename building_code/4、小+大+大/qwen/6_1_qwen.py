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
        print("调用Qwen出错:", e)
        return ""

def initialize_messages(system_message_content, initial_msgs_template, classification_defs):
    messages = [{"role": "system", "content": system_message_content}]
    for tmpl in initial_msgs_template:
        content = tmpl["content"].replace("{classification_definitions}", json.dumps(classification_defs, ensure_ascii=False))
        messages.append({"role": tmpl["role"], "content": content})
        resp = call_qwen(messages, max_new_tokens=256)
        print("初始消息回复:\n", resp)
        print("------------------------------")
    print("-----------初始化完毕，开始 6选1 分类-----------")
    return messages

def main():
    with open("../../prompt_6_1.json", "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    system_message_content = prompt_data["system_message"]
    classification_defs = prompt_data["classification_definitions"]
    initial_msgs_template = prompt_data["initial_messages"]
    chain_of_thought_prompt = prompt_data["chain_of_thought_prompt"]

    roberta_df = pd.read_csv("./results/bert_predictions.csv")
    print(f"[INFO] loaded roberta_df => shape: {roberta_df.shape}")

    label_map_str2num = {
        "direct":"0","general":"1","indirect":"2","method":"3","others":"4","reference":"5","term":"6"
    }
    label_map_num2str = {v:k for k,v in label_map_str2num.items()}

    with open("./Text_Embedding/fusion_alpha_0.50.json", "r", encoding="utf-8") as ff:
        fusion_data = json.load(ff)
    fusion_map = {item["test_text"]: item["results"] for item in fusion_data}

    base_messages = initialize_messages(system_message_content, initial_msgs_template, classification_defs)

    results = []
    for idx, row in roberta_df.iterrows():
        test_text = row["text"]
        predicted_label1 = row["predicted_label"]
        confidence = row["confidence"]
        true_label = row["true_label"]

        skip_num = label_map_str2num.get(predicted_label1, None)
        if skip_num is None:
            print(f"[WARN] 无法映射类别 {predicted_label1}")
            continue
        if test_text not in fusion_map:
            print(f"[WARN] 缺少样本向量: {test_text[:30]}")
            continue
        cat_dict = fusion_map[test_text]

        candidate_samples_str = ""
        print(f"\n=== 第{idx+1}/{len(roberta_df)}条文本 ===")
        print(f"文本[:20]: {test_text[:20]}")
        print(f"小模型预测={predicted_label1}")

        for lab_num, sample_list in cat_dict.items():
            if lab_num == skip_num:
                continue
            text_label = label_map_num2str[lab_num]
            top5 = sample_list[:5]
            candidate_samples_str += f"\n[类别 {text_label} top5样例]:"
            print(f"--- 类别 {text_label} top5 ---")
            for i, s in enumerate(top5):
                candidate_samples_str += f"\n{i+1}. {s['train_text']} (相似度={s['fused_similarity']:.3f})"
                print(f"{i+1}. {s['train_text'][:40]}... (相似度={s['fused_similarity']:.3f})")
            candidate_samples_str += "\n"
        print("----- 样例构造完成 -----\n")

        real_prompt = (chain_of_thought_prompt
                       .replace("{text}", test_text)
                       .replace("{predicted_label}", predicted_label1)
                       .replace("{confidence}", f"{confidence:.4f}")
                       .replace("{candidate_samples}", candidate_samples_str))

        messages = copy.deepcopy(base_messages)
        messages.append({"role": "user", "content": real_prompt})

        trial_preds = []
        trial_confusions = []
        for run_id in range(3):
            reply = call_qwen(copy.deepcopy(messages), max_new_tokens=512)
            print(f"[QWEN 第{run_id+1}次] reply =>\n{reply}\n")
            m_label = re.search(r"备选类别为\s*[:：]\s*([^\n]+)", reply)
            predicted_label2 = m_label.group(1).strip().lower() if m_label else "未知"
            m_conf = re.search(r"混淆点为\s*[:：]\s*([\s\S]+)", reply)
            confusion_point = m_conf.group(1).strip() if m_conf else ""
            trial_preds.append(predicted_label2)
            trial_confusions.append(confusion_point)

        vote = {}
        for p in trial_preds:
            vote[p] = vote.get(p, 0) + 1
        best_alt = sorted(vote.items(), key=lambda x: (-x[1], x[0]))[0][0]
        chosen_confusion = ""
        for p, c in zip(trial_preds, trial_confusions):
            if p == best_alt:
                chosen_confusion = c
                break

        final_result = "正确" if true_label in [predicted_label1, best_alt] else "错误"

        results.append({
            "text": test_text,
            "predicted_label1": predicted_label1,
            "predicted_label2": best_alt,
            "true_label": true_label,
            "result": final_result,
            "confusion_point": chosen_confusion
        })

    out_df = pd.DataFrame(results)
    out_csv = "6_1_qwen.csv"
    out_df.to_csv(out_csv, index=False, columns=["text","predicted_label1","predicted_label2","true_label","result","confusion_point"])
    print(f"[INFO] 已处理完全部 {len(out_df)} 条文本，结果写入 => {out_csv}")

if __name__ == "__main__":
    main()
