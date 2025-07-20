# test.py

import os
import torch
import torch.nn.functional as F
import pandas as pd
import sys 
import argparse
from train import Config, build_dataloader, evaluate

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def load_best_model(model_path="best_model.pt"):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found.")
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def inverse_label_map(config):

    return {v: k for k, v in config.label2id.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["instructor", "qwen", "deepseek"],
                        help="选择模型类型：instructor、qwen或deepseek")
    args = parser.parse_args()

    if args.model == "instructor":
        config = Config()

        best_model = load_best_model("best_model.pt")

        test_loader = build_dataloader(config, config.test_path, shuffle=False, sampler=False)

        all_texts = []
        all_preds = []
        all_trues = []
        all_confidences = []

        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            texts = batch['text']

            outputs = best_model(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)
            confidences, preds_batch = torch.max(probs, dim=1)

            all_texts.extend(texts)
            all_preds.extend(preds_batch.detach().cpu().numpy().tolist())
            all_trues.extend(labels.detach().cpu().numpy().tolist())
            all_confidences.extend(confidences.detach().cpu().numpy().tolist())

        _, test_acc, test_f1, report, cm, _, _, _ = evaluate(best_model, test_loader, config, test=True)
        print("===== 复现原测试集结果  =====")
        print("Test Acc:", test_acc)
        print("Test F1:", test_f1)
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", cm)

        choice = input("是否继续生成预测结果? (y/n): ")
        if choice.lower() != 'y':
            print("程序退出。")
            sys.exit(0)

        df_test = pd.read_csv(config.test_path)
        titles = df_test["title"].tolist()

        id2label = inverse_label_map(config)
        pred_labels_str = [id2label[p] for p in all_preds]
        true_labels_str = [id2label[t] for t in all_trues]

        df_out = pd.DataFrame({
            "title": titles,
            "text": all_texts,
            "predict_label": pred_labels_str,
            "true_label": true_labels_str,
            "confidence": all_confidences
        })

        output_path = "./results/predict.csv"
        df_out.to_csv(output_path, index=False, encoding="utf-8")
        print(f"每条测试数据的预测结果（含 title、置信度）已保存在 {output_path} 中。")

    elif args.model == "qwen":

        qwen_csv = "./LLM/2_1_qwen.csv"
        df_qwen = pd.read_csv(qwen_csv)

        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

        y_true = df_qwen["true_label"].tolist()
        y_pred = df_qwen["predict_label"].tolist()

        test_acc = accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, average="weighted")
        report = classification_report(y_true, y_pred, digits=4)
        cm = confusion_matrix(y_true, y_pred)

        print("===== qwen 模型测试集结果  =====")
        print("Test Acc:", test_acc)
        print("Test F1:", test_f1)
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", cm)

    elif args.model == "deepseek":

        qwen_csv = "./LLM/2_1_deepseek.csv"
        df_qwen = pd.read_csv(qwen_csv)

        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

        y_true = df_qwen["true_label"].tolist()
        y_pred = df_qwen["predict_label"].tolist()

        test_acc = accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, average="weighted")
        report = classification_report(y_true, y_pred, digits=4)
        cm = confusion_matrix(y_true, y_pred)

        print("===== qwen 模型测试集结果  =====")
        print("Test Acc:", test_acc)
        print("Test F1:", test_f1)
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", cm)
