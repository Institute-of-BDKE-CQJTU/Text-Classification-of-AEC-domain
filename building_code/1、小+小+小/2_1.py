import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_dir, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def evaluate_with_two_choice(model, data_loader, device, label_mapping, output_file):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['texts']
            true_labels = batch['true_labels']
            candidates = batch['candidates']

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            for i, candidate_labels in enumerate(candidates):
                candidate_indices = [label_mapping[label] for label in candidate_labels]
                restricted_logits = torch.full_like(logits[i], float('-inf'))
                restricted_logits[candidate_indices] = logits[i][candidate_indices]

                predicted_index = torch.argmax(restricted_logits).item()
                predicted_label = candidate_labels[candidate_indices.index(predicted_index)]

                result = {
                    "text": texts[i],
                    "true_label": true_labels[i],
                    "predicted_label": predicted_label,
                    "result": "正确" if true_labels[i] == predicted_label else "错误"
                }
                results.append(result)

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def prepare_data(input_file, tokenizer, max_len):
    input_data = pd.read_csv(input_file)
    texts = input_data['text'].tolist()
    true_labels = input_data['true_label'].tolist()
    candidates = input_data[['predicted_label1', 'predicted_label2']].values.tolist()

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")

    dataset = [
        {
            "input_ids": encodings['input_ids'][i],
            "attention_mask": encodings['attention_mask'][i],
            "texts": texts[i],
            "true_labels": true_labels[i],
            "candidates": candidates[i]
        }
        for i in range(len(texts))
    ]
    return DataLoader(dataset, batch_size=32, collate_fn=lambda x: {
        "input_ids": torch.stack([item['input_ids'] for item in x]),
        "attention_mask": torch.stack([item['attention_mask'] for item in x]),
        "texts": [item['texts'] for item in x],
        "true_labels": [item['true_labels'] for item in x],
        "candidates": [item['candidates'] for item in x]
    })

def main():
    model_dir = '../saved_model/roberta'
    input_file = './result_6_1.csv'
    output_file = './result_2_1.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 128

    label_list = ['direct', 'general', 'indirect', 'method', 'others', 'reference', 'term']
    label_mapping = {label: idx for idx, label in enumerate(label_list)}

    model, tokenizer = load_model(model_dir, device)

    test_loader = prepare_data(input_file, tokenizer, max_len)

    evaluate_with_two_choice(model, test_loader, device, label_mapping, output_file)
if __name__ == "__main__":
    main()
