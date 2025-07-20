import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

N = 5

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
test_texts = test_df['text'].tolist()

sample_pools = {}
pool_emb = {} 
labels = sorted(train_df['label'].unique())

model = SentenceTransformer("all-MiniLM-L6-v2")

for label in labels:
    texts = train_df[train_df['label'] == label]['text'].tolist()
    sample_pools[label] = texts
    emb = model.encode(texts, convert_to_tensor=False)
    pool_emb[label] = np.array(emb)

test_emb = model.encode(test_texts, convert_to_tensor=False)
test_emb = np.array(test_emb)

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

results = []

for i, test_text in enumerate(test_texts):
    entry = {"test_text": test_text, "results": {}}
    for label in labels:

        sims = np.array([cosine_similarity(test_emb[i], vec) for vec in pool_emb[label]])
        norm_sims = normalize_scores(sims)
        
        if norm_sims.shape[0] <= N:
            top_indices = np.argsort(-norm_sims)
        else:
            top_indices = np.argpartition(-norm_sims, N)[:N]
            top_indices = top_indices[np.argsort(-norm_sims[top_indices])]
        
        candidates = []
        for idx in top_indices:
            candidates.append({
                "train_text": sample_pools[label][idx],
                "similarity": float(sims[idx]),
                "normalized_similarity": float(norm_sims[idx])
            })
        entry["results"][str(label)] = candidates
    results.append(entry)

output_json = "results.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print(f"候选结果已保存到 {output_json}")
