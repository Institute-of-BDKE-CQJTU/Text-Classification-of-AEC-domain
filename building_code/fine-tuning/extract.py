import json
import pandas as pd

jsonl_file = ""
predictions = []
with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:  
            data = json.loads(line)
            predictions.append(data)

df_qwen = pd.read_csv("../data/test.csv")

if len(predictions) != len(df_qwen):
    print("警告：jsonl 文件行数与 CSV 文件行数不一致！")

new_rows = []
for i, row in df_qwen.iterrows():
    text = row["text"]

    pred_info = predictions[i]
    predicted_label = pred_info.get("predict", "")

    true_label = pred_info.get("label", "").strip()

    result = "正确" if predicted_label == true_label else "错误"
    
    new_rows.append({
        "text": text,
        "predicted_label": predicted_label,
        "true_label": true_label,
        "result": result
    })

df_new = pd.DataFrame(new_rows, columns=["text", "predicted_label", "true_label", "result"])

df_new.to_csv("2_1_llama70_tune.csv", index=False)
print("生成的文件已保存为 2_1_qwen_tune.csv")
