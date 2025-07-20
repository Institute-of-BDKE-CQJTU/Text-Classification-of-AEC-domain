import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import re
import json
import copy
import pandas as pd
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_path = "" #替换为实际llama模型路径
print(f"[INFO] Loading LLaMA from {model_path}")
model = MllamaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

def call_llama(messages, max_new_tokens=512, temperature=0.7, top_p=0.9):
    try:
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.split("assistant")[-1].strip() if "assistant" in answer else answer
    except Exception as e:
        print("调用本地 LLaMA 模型时出错:", e)
        return ""

def majority_vote(items):
    freq = {}
    for x in items:
        freq[x] = freq.get(x, 0) + 1
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

def initialize_messages(system_message_content, initial_msgs_template, classification_defs):
    messages = [{"role": "system", "content": system_message_content}]
    for msg_tmpl in initial_msgs_template:
        content = msg_tmpl["content"].replace("{classification_definitions}", json.dumps(classification_defs, ensure_ascii=False))
        messages.append({"role": msg_tmpl["role"], "content": content})
        resp = call_llama(messages, max_new_tokens=256)
        print("初始消息回复:\n", resp)
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
    fusion_map = {item["test_text"]: item["results"] for item in fusion_data}

    df = pd.read_csv("./6_1_llama.csv")
    print(f"[INFO] loaded 6_1_llama.csv => shape: {df.shape}")

    label_map_str2num = {
        "direct": "0", "general": "1", "indirect": "2", "method": "3",
        "others": "4", "reference": "5", "term": "6"
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
                continue
            sample_list = cat_dict.get(candidate_num, [])
            candidate_samples_str += f"\n[类别 {candidate} top5样例]:"
            for i, sitem in enumerate(sample_list[:5]):
                candidate_samples_str += f"\n{i+1}. {sitem['train_text']} (相似度={sitem['fused_similarity']:.3f})"
            candidate_samples_str += "\n"

        real_prompt = (instruction
                       .replace("{text}", test_text)
                       .replace("{predicted_label1}", predicted_label1)
                       .replace("{predicted_label2}", predicted_label2)
                       .replace("{candidate_samples}", candidate_samples_str)
                       .replace("{confusion_point}", confusion_point))

        messages_base = copy.deepcopy(base_messages)
        messages_base.append({"role": "user", "content": real_prompt})

        final_labels = []
        reasons = []
        for run in range(3):
            reply = call_llama(copy.deepcopy(messages_base), max_new_tokens=512)
            print(f"\n=== 第 {idx+1}/{len(df)} 条文本 第{run+1}次 ===")
            print("LLaMA 回复：\n", reply, "\n")
            match_final = re.search(r"最终类别为\s*[:：]\s*([^\n]+)", reply)
            final_category = match_final.group(1).strip() if match_final else "未知"
            match_reason = re.search(r"理由是\s*[:：]\s*([^\n]+)", reply)
            reason = match_reason.group(1).strip() if match_reason else "未匹配到"
            final_labels.append(final_category)
            reasons.append(reason)

        voted_label = majority_vote(final_labels)
        chosen_reason = ""
        for lab, rs in zip(final_labels, reasons):
            if lab == voted_label:
                chosen_reason = rs
                break

        final_result = "正确" if voted_label.lower() == true_label.lower() else "错误"

        results_list.append({
            "text": test_text,
            "true_label": true_label,
            "predicted_label": voted_label,
            "result": final_result,
            "reason": chosen_reason
        })

    out_df = pd.DataFrame(results_list)
    out_csv = "2_1_llama.csv"
    out_df.to_csv(out_csv, index=False, columns=["text", "true_label", "predicted_label", "result", "reason"])
    print(f"[INFO] 已处理完全部 {len(out_df)} 条文本，结果写入 => {out_csv}")

if __name__ == "__main__":
    main()
