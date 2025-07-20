#test1.py 生成roberta的预测结果
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from config import Config
from dataset import Text_Classification_Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

label_mapping = {
    0: 'direct',
    1: 'general',
    2: 'indirect',
    3: 'method',
    4: 'others',
    5: 'reference',
    6: 'term'
}

def load_model(model_dir, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def evaluate(model, data_loader, device, tokenizer, output_file):
    model.to(device)
    model.eval()
    true_label, pred_label, all_texts, confidences = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["label"].to(device)
            texts = batch["text"]  

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)  
            confidence, predicted_class = torch.max(probs, dim=1)

            true_label += targets.cpu().numpy().tolist()
            pred_label += predicted_class.cpu().numpy().tolist()
            confidences += confidence.cpu().numpy().tolist()
            all_texts += texts

    all_predictions = {
        'text': [],
        'predicted_label': [],
        'confidence': [],
        'true_label': []
    }

    for i in range(len(true_label)):

        true_class_name = label_mapping[true_label[i]]
        pred_class_name = label_mapping[pred_label[i]]

        all_predictions['text'].append(all_texts[i])
        all_predictions['predicted_label'].append(pred_class_name)
        all_predictions['confidence'].append(confidences[i])
        all_predictions['true_label'].append(true_class_name)

    all_df = pd.DataFrame(all_predictions)
    all_df.to_csv(output_file, index=False)
    print(f"All predictions saved to {output_file}")

    f1 = f1_score(true_label, pred_label, average='macro', zero_division=1)
    precision_recall_fscore = classification_report(true_label, pred_label)
    return f1, precision_recall_fscore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a specific model on the test dataset.')
    parser.add_argument('--model', type=str, required=True, help='Specify the model to test: bert, roberta')
    args = parser.parse_args()
    model_name = args.model

    CFG = Config()

    if model_name not in ["bert", "roberta"]:
        raise ValueError("Invalid model name! Choose from: bert, roberta")

    test_data = pd.read_csv('./data/test.csv')
    test_data.columns = ['text', 'label']
    assert test_data['label'].min() >= 0 and test_data['label'].max() < CFG.n_class
    test_data_loader = DataLoader(
        dataset=Text_Classification_Dataset(tokenizer=CFG.tokenizers[model_name], max_len=CFG.max_len, texts=test_data['text'], labels=test_data['label']),
        batch_size=CFG.batch_size, shuffle=False, num_workers=4
    )
    
    model_dir = f'./saved_model/{model_name}'
    print(f'Testing model: {model_name}')
    model, tokenizer = load_model(model_dir, CFG.device)
    
    f1, precision_recall_fscore = evaluate(model, test_data_loader, CFG.device, tokenizer, output_file='./results/bert_predictions.csv')
    
    print(f'F1 score for test set: {f1}')
    print(precision_recall_fscore)
