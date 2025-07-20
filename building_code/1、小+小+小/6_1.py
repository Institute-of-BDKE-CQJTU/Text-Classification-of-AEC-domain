import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_dir, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def evaluate_with_two_predictions(model, data_loader, device, output_file, label_list):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['texts']
            true_labels = batch['true_labels']

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred_labels_1 = torch.argmax(logits, dim=1).cpu().numpy()

            for i, pred_label in enumerate(pred_labels_1):
                logits[i, pred_label] = float('-inf')  
            pred_labels_2 = torch.argmax(logits, dim=1).cpu().numpy()

            for i in range(len(true_labels)):
                true_label_name = true_labels[i]
                predicted_label1_name = label_list[pred_labels_1[i]]
                predicted_label2_name = label_list[pred_labels_2[i]]
                result = {
                    "text": texts[i],
                    "true_label": true_label_name,
                    "predicted_label1": predicted_label1_name,
                    "predicted_label2": predicted_label2_name,
                    "result": "正确" if true_label_name in [predicted_label1_name, predicted_label2_name] else "错误"
                }
                results.append(result)

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def prepare_data(test_file, tokenizer, max_len):
    test_data = pd.read_csv(test_file)
    texts = test_data['text'].tolist()
    true_labels = test_data['true_label'].tolist()

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")

    dataset = [
        {
            "input_ids": encodings['input_ids'][i],
            "attention_mask": encodings['attention_mask'][i],
            "texts": texts[i],
            "true_labels": true_labels[i]
        }
        for i in range(len(texts))
    ]
    return DataLoader(dataset, batch_size=32, collate_fn=lambda x: {
        "input_ids": torch.stack([item['input_ids'] for item in x]),
        "attention_mask": torch.stack([item['attention_mask'] for item in x]),
        "texts": [item['texts'] for item in x],
        "true_labels": [item['true_labels'] for item in x]
    })

def main():
    model_dir = '../saved_model/roberta'
    test_file = './result_7_1.csv'  
    output_file = './result_6_1.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 128

    label_list = ['direct', 'general', 'indirect', 'method', 'others', 'reference', 'term']

    model, tokenizer = load_model(model_dir, device)

    test_loader = prepare_data(test_file, tokenizer, max_len)

    evaluate_with_two_predictions(model, test_loader, device, output_file, label_list)
if __name__ == "__main__":
    main()
