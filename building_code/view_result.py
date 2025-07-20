#view_result.py
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config
from dataset import Text_Classification_Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_dir, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def evaluate_and_collect_predictions(model, data_loader, device):
    model.to(device)
    model.eval()
    texts = []
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["label"].to(device)
            texts.extend(batch["text"]) 

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            true_labels += targets.cpu().numpy().tolist()
            pred_labels += torch.argmax(logits, dim=1).cpu().numpy().tolist()

    return texts, true_labels, pred_labels

def compare_predictions(texts, true_labels, roberta_preds, bert_preds, deepseek_preds, qwen_preds, llama_preds):
    all_results = []
    for i in range(len(true_labels)):
        all_results.append([
            i + 1,
            texts[i],
            true_labels[i],
            roberta_preds[i],
            bert_preds[i],
            deepseek_preds[i],
            qwen_preds[i],
            llama_preds[i]
        ])

    all_correct = [row for row in all_results if row[2] == row[3] == row[4] == row[5] == row[6] == row[7]]

    some_correct = [row for row in all_results if (row[2] == row[3] or row[2] == row[4] or row[2] == row[5] or row[2] == row[6] or row[2] == row[7])
                    and not (row[2] == row[3] == row[4] == row[5] == row[6] == row[7])]

    all_wrong = [row for row in all_results if row[2] != row[3] and row[2] != row[4] and row[2] != row[5] and row[2] != row[6] and row[2] != row[7]]

    return all_correct, some_correct, all_wrong

if __name__ == "__main__":
    CFG = Config()

    label_list = ['direct', 'general', 'indirect', 'method', 'others', 'reference', 'term']
    label_to_index = {label: idx for idx, label in enumerate(label_list)}
    index_to_label = {idx: label for idx, label in enumerate(label_list)}


    test_data = pd.read_csv('./data/test.csv')
    test_data.columns = ['text', 'label']
    assert test_data['label'].min() >= 0 and test_data['label'].max() < CFG.n_class

    test_data_loader = DataLoader(
        dataset=Text_Classification_Dataset(
            tokenizer=CFG.tokenizers["roberta"],
            max_len=CFG.max_len,
            texts=test_data['text'],
            labels=test_data['label']
        ),
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=4
    )

    roberta_model_dir = './saved_model/roberta'
    print('--------------------------------正在测试 RoBERTa 模型-----------------------------------')
    roberta_model, roberta_tokenizer = load_model(roberta_model_dir, CFG.device)
    texts, true_label_indices, roberta_pred_indices = evaluate_and_collect_predictions(
        roberta_model, test_data_loader, CFG.device)
    true_labels = [index_to_label[idx] for idx in true_label_indices]
    roberta_preds = [index_to_label[idx] for idx in roberta_pred_indices]

    bert_model_dir = './saved_model/bert'
    print('--------------------------------正在测试 BERT 模型-----------------------------------')
    bert_model, bert_tokenizer = load_model(bert_model_dir, CFG.device)
    _, _, bert_pred_indices = evaluate_and_collect_predictions(
        bert_model, test_data_loader, CFG.device)
    bert_preds = [index_to_label[idx] for idx in bert_pred_indices]

    deepseek_predictions_df = pd.read_csv('./4、小+大+大/deepseek/2_1_deepseek.csv')
    qwen_predictions_df = pd.read_csv('./4、小+大+大/qwen/2_1_qwen.csv')
    llama_predictions_df = pd.read_csv('./4、小+大+大/llama/2_1_llama.csv')

    deepseek_pred_dict = dict(zip(deepseek_predictions_df['text'], deepseek_predictions_df['predicted_label']))
    qwen_pred_dict = dict(zip(qwen_predictions_df['text'], qwen_predictions_df['predicted_label']))
    llama_pred_dict = dict(zip(llama_predictions_df['text'], llama_predictions_df['predicted_label']))

    deepseek_preds = []
    qwen_preds = []
    llama_preds = []
    for text in texts:
        deepseek_preds.append(deepseek_pred_dict.get(text, "unknown"))
        qwen_preds.append(qwen_pred_dict.get(text, "unknown"))
        llama_preds.append(llama_pred_dict.get(text, "unknown"))

    all_correct, some_correct, all_wrong = compare_predictions(
        texts, true_labels, roberta_preds, bert_preds, deepseek_preds, qwen_preds, llama_preds)

    columns = ['index', 'text', 'true label', 'RoBERTa Prediction', 'BERT Prediction', 'Deepseek Prediction', 'Qwen Prediction', 'Llama Prediction']
    pd.DataFrame(all_correct, columns=columns).to_csv('./results/all_correct.csv', index=False)
    pd.DataFrame(some_correct, columns=columns).to_csv('./results/some_correct.csv', index=False)
    pd.DataFrame(all_wrong, columns=columns).to_csv('./results/all_wrong.csv', index=False)

    print("结果已保存到 ./results/ 目录下。")
