#test.py 测试各模型结果
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from config import Config
from dataset import Text_Classification_Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
def load_model(model_dir, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)  
    tokenizer = AutoTokenizer.from_pretrained(model_dir)  
    return model, tokenizer
def evaluate(model, data_loader, device):
    model.to(device)
    model.eval()
    true_label, pred_label = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["label"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            true_label += targets.cpu().numpy().tolist()
            pred_label += torch.argmax(logits, dim=1).cpu().numpy().tolist()
    f1 = f1_score(true_label, pred_label, average='macro', zero_division=1)
    precision_recall_fscore = classification_report(true_label, pred_label)
    return f1, precision_recall_fscore, true_label, pred_label
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a specific model on the test dataset.')
    parser.add_argument('--model', type=str, required=True, help='Specify the model to test: "bert", "roberta", "s_s_s", "s_s_b", "s_b_s", "s_b_b", "b_s_s", "b_s_b", "b_b_s", "b_b_b", "deepseek_only","tune_llama", "tune_qwen", "tune_llama70"')
    args = parser.parse_args()
    model_name = args.model
    if model_name not in ["bert", "roberta", "s_s_s", "s_s_b", "s_b_s", "s_b_b", "b_s_s", "b_s_b", "b_b_s", "b_b_b", "deepseek_only", "tune_llama", "tune_qwen", "tune_llama70"]:
        raise ValueError("Invalid model name! Choose from: bert, roberta, s_s_s, s_s_b, s_b_s, s_b_b, b_s_s, b_s_b, b_b_s, b_b_b, deepseek_only, tune_llama, tune_qwen, tune_llama70")
    CFG = Config()
    if model_name in ["s_s_s", "s_s_b", "s_b_s", "s_b_b", "b_s_s", "b_s_b", "b_b_s", "b_b_b", "deepseek_only", "tune_llama", "tune_qwen", "tune_llama70"]:
        if model_name == 's_s_s':
            predictions_df = pd.read_csv('./1、小+小+小/result_2_1.csv')
        elif model_name == 's_s_b':
            predictions_df = pd.read_csv('./2、小+小+大/2_1_qwen.csv')
        elif model_name == 's_b_s':
            predictions_df = pd.read_csv('./3、小+大+小/result_2_1.csv')
        elif model_name == 's_b_b':
            big_version = input("请指定要测试的大模型（可选：deepseek、llama、qwen）：")
            if big_version == "deepseek":
                predictions_df = pd.read_csv('./4、小+大+大/deepseek/2_1_deepseek.csv')
            elif big_version == "llama":
                predictions_df = pd.read_csv('./4、小+大+大/llama/2_1_llama.csv')
            elif big_version == "qwen":
                predictions_df = pd.read_csv('./4、小+大+大/qwen/2_1_qwen.csv')
        elif model_name == 'b_s_s':
            predictions_df = pd.read_csv('./5、大+小+小/result_2_1.csv')
        elif model_name == 'b_s_b':
            predictions_df = pd.read_csv('./6、大+小+大/2_1_qwen.csv')
        elif model_name == 'b_b_s':
            predictions_df = pd.read_csv('./7、大+大+小/result_2_1.csv')
        elif model_name == 'b_b_b':
            predictions_df = pd.read_csv('./8、大+大+大/2_1_qwen.csv')
        elif model_name == 'deepseek_only':
            predictions_df = pd.read_csv('./5、大+小+小/7_1_deepseek.csv')
        elif model_name == 'tune_llama':
            predictions_df = pd.read_csv('./fine-tuning/2_1_llama_tune.csv')
        elif model_name == 'tune_qwen':
            predictions_df = pd.read_csv('./fine-tuning/2_1_qwen_tune.csv')
        elif model_name == 'tune_llama70':
            predictions_df = pd.read_csv('./fine-tuning/2_1_llama70_tune.csv')

        label_list = ['direct', 'general', 'indirect', 'method', 'others', 'reference', 'term']
        label_to_index = {label: idx for idx, label in enumerate(label_list)}
        index_to_label = {idx: label for idx, label in enumerate(label_list)}

        true_labels = predictions_df['true_label'].tolist()
        pred_labels = predictions_df['predicted_label'].tolist()
        true_label_indices = [label_to_index[label] for label in true_labels]
        pred_label_indices = [label_to_index[label] for label in pred_labels]

        f1 = f1_score(true_label_indices, pred_label_indices, average='macro', zero_division=1)
        precision_recall_fscore = classification_report(true_label_indices, pred_label_indices, target_names=label_list, zero_division=1, digits=4)
        print(f'F1 score for test set: {f1}')
        print('The report is:')
        print(precision_recall_fscore)
    else:
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
        print(model)
        f1, precision_recall_fscore, true_label, pred_label = evaluate(model, test_data_loader, CFG.device)
        print(f'F1 score for test set: {f1}')
        print('The report is:')
        print(precision_recall_fscore)
