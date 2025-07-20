# train.py
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoConfig, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from config import Config
from dataset import Text_Classification_Dataset, FGM

torch.manual_seed(Config.seed)
np.random.seed(Config.seed)
random.seed(Config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def get_parameters(model, model_init_lr, multiplier, classifier_lr):
    parameters = []
    lr = model_init_lr
    for layer in range(model.config.num_hidden_layers, -1, -1):
        parameters.append({
            'params': [p for n, p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
            'lr': lr
        })
        lr *= multiplier
    parameters.append({
        'params': [p for n, p in model.named_parameters() if 'layer_norm' in n or 'linear' in n or 'pooling' in n],
        'lr': classifier_lr
    })
    return parameters

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def evaluate(model, data_loader, device):
    model.eval()
    true_label, pred_label = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["label"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            true_label += targets.cpu().numpy().tolist()
            pred_label += torch.argmax(logits, dim=1).cpu().numpy().tolist()
    f1 = f1_score(true_label, pred_label, average='macro', zero_division=1)
    precision_recall_fscore = classification_report(true_label, pred_label)
    return f1, precision_recall_fscore

def train_and_evaluate_model(config, train_data, val_data, test_data, model_name):
    train_dataset = Text_Classification_Dataset(
        tokenizer=Config.tokenizers[model_name],
        max_len=Config.max_len,
        texts=train_data['text'].values,
        labels=train_data['label'].values,
        augment=True,
        seed=config['seed']
    )
    val_dataset = Text_Classification_Dataset(
        tokenizer=Config.tokenizers[model_name],
        max_len=Config.max_len,
        texts=val_data['text'].values,
        labels=val_data['label'].values
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, worker_init_fn=seed_worker)
    
    model_config = AutoConfig.from_pretrained(Config.pretrain_model_paths[model_name],
                                                output_hidden_states=True, return_dict=True, num_labels=Config.n_class)
    model = AutoModelForSequenceClassification.from_pretrained(Config.pretrain_model_paths[model_name],
                                                               config=model_config).to(Config.device)
    parameters = get_parameters(model, config['lr'], 0.95, config['lr'])
    optimizer = AdamW(parameters, weight_decay=config['weight_decay'])
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(config['num_warmup_steps'] * total_steps),
                                                num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(Config.device)
    fgm = FGM(model)
    early_stopping = EarlyStopping(patience=5)
    
    best_model = None
    best_f1 = 0
    best_model_instance = None
    
    training_logs = []
    
    for epoch in range(config['epochs']):
        model.train()
        tqdm_bar = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {epoch+1}")
        true_label, pred_label = [], []
        epoch_losses = []
        for batch in tqdm_bar:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            targets = batch["label"].to(Config.device)
            
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, targets)
            loss.backward()
            
            fgm.attack()
            logits_adv = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss_adv = loss_fn(logits_adv, targets)
            loss_adv.backward()
            fgm.restore()
            
            nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            true_label += targets.cpu().numpy().tolist()
            pred_label += torch.argmax(logits, dim=1).cpu().numpy().tolist()
            epoch_losses.append(loss.item())
            f1_metric = f1_score(true_label, pred_label, average='macro', zero_division=1)
            tqdm_bar.set_postfix_str(f'loss: {loss.item():.4f}, f1: {f1_metric:.4f}')
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        training_logs.append({
            'epoch': epoch + 1,
            'avg_loss': avg_epoch_loss,
            'f1': f1_metric
        })
        
        model.eval()
        val_f1, val_report = evaluate(model, val_loader, Config.device)
        print(f"Epoch {epoch+1}, Validation F1: {val_f1}")
        print(val_report)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model.state_dict()
            best_model_instance = model
        
        val_loss = loss.item()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        torch.cuda.empty_cache()
    
    test_dataset = Text_Classification_Dataset(
        tokenizer=Config.tokenizers[model_name],
        max_len=Config.max_len,
        texts=test_data['text'].values,
        labels=test_data['label'].values
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0, worker_init_fn=seed_worker)
    test_f1, test_report = evaluate(best_model_instance, test_loader, Config.device)
    print(f"Test F1 for {model_name} with config {config}: {test_f1}")
    
    logs_df = pd.DataFrame(training_logs)
    logs_csv_path = f'training_logs_{model_name}.csv'
    logs_df.to_csv(logs_csv_path, index=False)
    print(f"Training logs saved to {logs_csv_path}")
    
    return test_f1, best_model, model_config, best_model_instance

if __name__ == "__main__":
    pass
