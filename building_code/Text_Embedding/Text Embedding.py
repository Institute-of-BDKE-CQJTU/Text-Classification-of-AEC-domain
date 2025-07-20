import os
import json
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

alpha_values = np.arange(0, 1.01, 0.1)  
N = 5 
API_URL = 'https://api.jina.ai/v1/embeddings'
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': ''#替换为实际jina API的授权令牌
}

def get_embeddings_jina(texts, max_length=None, truncate_dim=None):
    data = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "late_chunking": False,
        "dimensions": 1024,
        "embedding_type": "float",
        "input": texts
    }
    if max_length is not None:
        data["max_length"] = max_length
    if truncate_dim is not None:
        data["truncate_dim"] = truncate_dim

    response = requests.post(API_URL, json=data, headers=HEADERS)
    if response.status_code != 200:
        raise ValueError(f"API 调用出错，状态码：{response.status_code}, {response.text}")
    result = response.json()
    if "data" in result:
        return [item["embedding"] for item in result["data"]]
    else:
        raise ValueError("返回数据格式不符合预期")

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist() 


sample_pools = {}
pool_emb_a = {}  
pool_emb_b = {}  
labels = sorted(train_df['label'].unique())

model_a = SentenceTransformer("all-MiniLM-L6-v2")

for label in labels:
    texts = train_df[train_df['label'] == label]['text'].tolist()
    sample_pools[label] = texts

    emb_a = model_a.encode(texts, convert_to_tensor=False)
    pool_emb_a[label] = np.array(emb_a)

    emb_b = get_embeddings_jina(texts)
    pool_emb_b[label] = np.array(emb_b)

test_emb_a = model_a.encode(test_texts, convert_to_tensor=False)
test_emb_a = np.array(test_emb_a)

test_emb_b = get_embeddings_jina(test_texts)
test_emb_b = np.array(test_emb_b)

def cosine_similarity(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b + 1e-8)

def normalize_scores(scores):
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:
        return np.ones_like(scores)
    return (scores - min_val) / (max_val - min_val)

all_alpha_results = {}  
metrics_records = []    

for alpha in alpha_values:
    fusion_results = []  
    correct_metric1 = 0  
    correct_metric2 = 0   
    margin_sum = 0.0    

    for i, test_text in enumerate(test_texts):
        entry = {"test_text": test_text, "results": {}}

        rep_scores_max = {}
        rep_scores_avg = {}
        
        for label in labels:

            sims_a = np.array([cosine_similarity(test_emb_a[i], vec) for vec in pool_emb_a[label]])
            sims_b = np.array([cosine_similarity(test_emb_b[i], vec) for vec in pool_emb_b[label]])

            norm_a = normalize_scores(sims_a)
            norm_b = normalize_scores(sims_b)

            fused_scores = alpha * norm_a + (1 - alpha) * norm_b
            
            if fused_scores.shape[0] <= N:
                top_indices = np.argsort(-fused_scores)
            else:
                top_indices = np.argpartition(-fused_scores, N)[:N]
                top_indices = top_indices[np.argsort(-fused_scores[top_indices])]
            
            candidates = []
            for idx in top_indices:
                candidates.append({
                    "train_text": sample_pools[label][idx],
                    "sim_MiniLM": float(sims_a[idx]),
                    "sim_Jina": float(sims_b[idx]),
                    "fused_similarity": float(fused_scores[idx])
                })
            entry["results"][str(label)] = candidates
            
            rep_scores_max[label] = np.max(fused_scores)
            rep_scores_avg[label] = np.mean(fused_scores)
        
        fusion_results.append(entry)
        
        true_label = test_labels[i]
        
        true_max = rep_scores_max[true_label]
        other_max = [rep_scores_max[l] for l in rep_scores_max if l != true_label]
        if all(true_max > om for om in other_max):
            correct_metric1 += 1
        
        true_avg = rep_scores_avg[true_label]
        other_avg = [rep_scores_avg[l] for l in rep_scores_avg if l != true_label]
        if all(true_avg > oa for oa in other_avg):
            correct_metric2 += 1
        
        margin = true_avg - max(other_avg) if other_avg else 0.0
        margin_sum += margin

    total_tests = len(test_texts)
    metric1_acc = correct_metric1 / total_tests
    metric2_acc = correct_metric2 / total_tests
    metric3_margin = margin_sum

    all_alpha_results[f"{alpha:.2f}"] = fusion_results

    metrics_records.append({
        "alpha": alpha,
        "TopN_Accuracy_Max": metric1_acc,
        "TopN_Accuracy_Avg": metric2_acc,
        "Margin_Sum": metric3_margin
    })
    print(f"Alpha={alpha:.2f}: Metric1 (Max Acc)={metric1_acc:.4f}, Metric2 (Avg Acc)={metric2_acc:.4f}, Margin Sum={metric3_margin:.4f}")

    output_json = f"fusion_alpha_{alpha:.2f}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(fusion_results, f, ensure_ascii=False, indent=4)
    print(f"融合结果（alpha={alpha:.2f}）已保存到 {output_json}")

metrics_df = pd.DataFrame(metrics_records)
metrics_csv = "fusion_metrics.csv"
metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8")
print(f"所有alpha对应的指标记录已保存到 {metrics_csv}")

with open("fusion_all_alpha.json", "w", encoding="utf-8") as f:
    json.dump(all_alpha_results, f, ensure_ascii=False, indent=4)
print("所有alpha对应的候选结果已保存到 fusion_all_alpha.json")
