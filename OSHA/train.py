# main.py
import time
import json
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch import nn
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from InstructorEmbedding import INSTRUCTOR

from focalloss import FocalLoss
from utils import get_time_dif
from model import ClassificationModel  

class Config(object):
    def __init__(self):

        self.pretrained_path_instructor = "" #instructor-base

        self.train_path = "./data/train.csv"
        self.test_path = "./data/test.csv"

        self.learning_rate = 1e-4
        self.weight_decay = 1e-2
        self.random_seed = 42
        self.dropout_rate = 0.5
        self.batch_size = 4
        self.require_improvement = 600   
        self.num_epochs = 30
        self.print_every = 20      
        self.gamma = 3.0
        self.alpha = None

        self.label2id = {
            'caught in/between objects': 0,
            'collapse of object': 1,
            'electrocution': 2,
            'exposure to chemical substances': 3,
            'exposure to extreme temperatures': 4,
            'falls': 5,
            'fires and explosion': 6,
            'struck by falling object': 7,
            'struck by moving objects': 8,
            'traffic': 9,
            'others': 10
        }
        self.output_dim = len(self.label2id) 

    def to_dict(self):
        return {
            "pretrained_path_instructor": self.pretrained_path_instructor,
            "train_path": self.train_path,
            "test_path": self.test_path,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "random_seed": self.random_seed,
            "dropout_rate": self.dropout_rate,
            "batch_size": self.batch_size,
            "require_improvement": self.require_improvement,
            "num_epochs": self.num_epochs,
            "print_every": self.print_every,
            "FocalLoss gamma": self.gamma,
            "FocalLoss alpha": self.alpha,
            "label2id": self.label2id
        }

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TextDataset(Dataset):

    def __init__(self, df, config):
        self.config = config
        self.tokenizer = INSTRUCTOR(self.config.pretrained_path_instructor).tokenizer

        texts = df['text'].tolist()
        labels = df['label'].tolist()

        if isinstance(labels[0], str):
            label2id = self.config.label2id
            self.labels = [label2id[l] for l in labels]
        else:
            self.labels = labels

        self.texts = texts

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            return_tensors='pt',
            max_length=512,
            padding='max_length',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text 
        }

    def __len__(self):
        return len(self.labels)

def build_dataloader(config, file_path, shuffle=False, sampler=False):
    df = pd.read_csv(file_path)
    dataset = TextDataset(df, config)

    if sampler:
        labels_tensor = torch.tensor(dataset.labels, dtype=torch.long)
        class_sample_count = torch.bincount(labels_tensor)
        weight_per_class = 1.0 / class_sample_count.float()
        samples_weight = weight_per_class[labels_tensor]
        sampler_ = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
        loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler_)
    else:
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)
    return loader

def evaluate(model, data_loader, config, test=False):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0
    preds_list = []
    labels_list = []
    all_texts = [] 

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            texts = batch.get('text', None)

            outputs = model(input_ids, attention_mask)
            loss = FocalLoss(gamma=config.gamma, alpha=config.alpha)(outputs, labels)
            total_loss += loss.item()

            pred = outputs.argmax(dim=1)
            preds_list.extend(pred.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

            if texts is not None:
                all_texts.extend(texts)

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(labels_list, preds_list)
    f1 = f1_score(labels_list, preds_list, average='weighted')

    if test:
        report = classification_report(labels_list, preds_list, digits=6)
        cm = confusion_matrix(labels_list, preds_list)
        return avg_loss, acc, f1, report, cm, preds_list, labels_list, all_texts
    else:
        return avg_loss, acc, f1

if __name__ == "__main__":
    config = Config()
    print("Config:", config.to_dict())
    set_seed(config.random_seed)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    train_loader = build_dataloader(config, config.train_path, shuffle=False, sampler=True)
    test_loader = build_dataloader(config, config.test_path, shuffle=False, sampler=False)

    model = ClassificationModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=100)

    best_acc = 0
    last_improve = 0
    total_batch = 0
    start_time = time.time()

    for epoch in range(config.num_epochs):
        print(f"Epoch [{epoch+1}/{config.num_epochs}]")
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = FocalLoss(gamma=config.gamma, alpha=config.alpha)(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_batch % config.print_every == 0:
                test_loss, test_acc, test_f1 = evaluate(model, test_loader, config, test=False)
                if test_acc > best_acc:
                    best_acc = test_acc
                    last_improve = total_batch

                    torch.save(model, "best_model.pt")
                    improve = "*"
                else:
                    improve = ""

                elapsed = time.time() - start_time
                msg = f"Iter:{total_batch}, Train Loss:{loss.item():.4f}, Test Acc:{test_acc:.4f}, F1:{test_f1:.4f}, Time:{elapsed:.1f}s {improve}"
                print(msg)

                scheduler.step(test_acc)

            total_batch += 1

            if total_batch - last_improve > config.require_improvement:
                print("No improvement for a long time, auto-stopping...")
                break

        if total_batch - last_improve > config.require_improvement:
            break

    print("\n========  Final Test  ========")
    best_model = torch.load("best_model.pt", map_location=device)
    best_model.eval()
    test_loss, test_acc, test_f1, report, cm, preds, trues, texts = evaluate(best_model, test_loader, config, test=True)

    print("Final Test Loss:", test_loss)
    print("Final Test Acc:", test_acc)
    print("Final Test F1:", test_f1)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
