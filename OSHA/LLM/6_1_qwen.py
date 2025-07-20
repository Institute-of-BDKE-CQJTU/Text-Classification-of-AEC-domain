import os
import re
import json
import copy
import pandas as pd
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
        resp = call_qwen(messages, max_new_tokens=512)
        print("大模型对该条初始消息的回答：\n", resp)
        print("------------------------------")
    print("-----------初始化完毕，开始 11类事故备选分类-----------")
    return messages

def main():
    with open("../prompt_6_1.json", "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    system_message_content = prompt_data["system_message"]
    classification_defs = prompt_data["classification_definitions"]
    initial_msgs_template = prompt_data["initial_messages"]
    instruction = prompt_data["instruction"]
    roberta_file = "./false.csv"
    roberta_df = pd.read_csv(roberta_file)
    print(f"[INFO] loaded roberta_df => shape: {roberta_df.shape}")
    with open("../results/results.json", "r", encoding="utf-8") as ff:
        fusion_data = json.load(ff)
    fusion_map = {}
    for item in fusion_data:
        txt = item["test_text"]
        cat_dict = item["results"]
        fusion_map[txt] = cat_dict
    base_messages = initialize_messages(system_message_content, initial_msgs_template, classification_defs)
    results_list = []
    for idx, row in roberta_df.iterrows():
        test_title = row["title"]
        test_text = row["text"]
        predicted_label1 = row["predict_label"]
        confidence = row["confidence"]
        true_label = row.get("true_label", "")
        if test_text not in fusion_map:
            print(f"[WARN] text not in result.json => skip. text[:40]={test_text[:40]}")
            continue
        cat_dict = fusion_map[test_text]
        print(f"\n=== 第{idx+1}/{len(roberta_df)}条 ===")
        print(f"[TITLE] {test_title[:40]}...")
        print(f"[TEXT] {test_text[:40]}...")
        print(f"[Small Model Prediction] {predicted_label1}, conf={confidence:.3f}")
        candidate_samples_str = ""
        for cat_name, sample_list in cat_dict.items():
            top_2 = sample_list[:2]
            candidate_samples_str += f"\n[Category: {cat_name} | top-2 samples]"
            for i, sitem in enumerate(top_2):
                cand_text = sitem["train_text"]
                sim_val = sitem.get("similarity", 0.0)
                candidate_samples_str += f"\n{i+1}. {cand_text} (similarity={sim_val:.3f})"
            candidate_samples_str += "\n"
        real_prompt = instruction
        real_prompt = real_prompt.replace("{title}", test_title)
        real_prompt = real_prompt.replace("{text}", test_text)
        real_prompt = real_prompt.replace("{predicted_label}", predicted_label1)
        real_prompt = real_prompt.replace("{confidence}", f"{confidence:.4f}")
        real_prompt = real_prompt.replace("{candidate_samples}", candidate_samples_str)
        messages_base = copy.deepcopy(base_messages)
        messages_base.append({"role": "user", "content": real_prompt})
        runs = []
        for _ in range(3):
            reply = call_qwen(copy.deepcopy(messages_base), max_new_tokens=512)
            print("[QWEN] reply =>\n", reply, "\n")
            alt_match = re.search(
                r"of this accident is\s*[:：]\s*([^\n]+)",
                reply,
                flags=re.IGNORECASE
            )
            predicted_label2 = alt_match.group(1).strip() if alt_match else "未知"
            conf_match = re.search(
                r"The confusion.*?is\s*[:：]\s*(.*)",
                reply,
                flags=re.IGNORECASE | re.DOTALL
            )
            confusion_point = conf_match.group(1).strip() if conf_match else ""
            runs.append((predicted_label2, confusion_point))
        vote_counts = {}
        for c, r in runs:
            vote_counts[c.lower()] = vote_counts.get(c.lower(), 0) + 1
        majority_label_lower = sorted(vote_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        majority_label = None
        majority_confusions = []
        for c, r in runs:
            if c.lower() == majority_label_lower:
                majority_confusions.append(r)
                if majority_label is None:
                    majority_label = c
        final_predicted_label2 = majority_label
        final_confusion_point = random.choice(majority_confusions) if majority_confusions else ""
        final_result = "正确" if true_label in [predicted_label1, final_predicted_label2] else "错误"
        results_list.append({
            "title": test_title,
            "text": test_text,
            "predicted_label1": predicted_label1,
            "predicted_label2": final_predicted_label2,
            "true_label": true_label,
            "result": final_result,
            "confusion_point": final_confusion_point
        })
    out_df = pd.DataFrame(results_list)
    out_csv = "6_1_qwen.csv"
    out_df.to_csv(
        out_csv,
        index=False,
        columns=["title","text","predicted_label1","predicted_label2","true_label","result","confusion_point"]
    )
    print(f"[INFO] 完成，共处理 {len(out_df)} 条，结果已写入 => {out_csv}")

if __name__ == "__main__":
    main()
