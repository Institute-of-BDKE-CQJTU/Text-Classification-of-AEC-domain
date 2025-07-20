#test bert and roberta on OSHA dataset
import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR        = 'data'
TEST_FILE       = os.path.join(DATA_DIR, 'test.csv')
CLASS_FILE      = os.path.join(DATA_DIR, 'class.txt')

PRETRAINED_PATH = '' #RoBERTa和BERT的预训练模型路径
MAX_LEN         = 128
BATCH_SIZE      = 64
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

with open(CLASS_FILE, 'r', encoding='utf-8') as f:
    CLASS_LIST = [l.strip() for l in f if l.strip()]

df_test = pd.read_csv(TEST_FILE, quotechar='"')
test_texts = df_test['text'].tolist()
test_labels= df_test['label'].tolist()

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH)
test_ds    = TextDataset(test_texts, test_labels, tokenizer)
test_loader= DataLoader(test_ds, batch_size=BATCH_SIZE)

class PoolingClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_PATH)
        hidden = self.bert.config.hidden_size
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(hidden*2, len(CLASS_LIST))
    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, return_dict=True)
        cls  = out.pooler_output
        seq  = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        mean = (seq * mask).sum(1) / mask.sum(1)
        x    = torch.cat([cls, mean], dim=-1)
        x    = self.dropout(x)
        return self.classifier(x)

model = PoolingClassifier().to(DEVICE)
model.load_state_dict(torch.load('best_roberta.pth', map_location=DEVICE))
model.eval()

all_preds, all_trues = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k!='labels'}
        labels = batch['labels'].to(DEVICE)
        logits = model(**inputs)
        preds  = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_trues.extend(labels.cpu().tolist())

print("Results:")
print(classification_report(all_trues, all_preds, target_names=CLASS_LIST, zero_division=0, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(all_trues, all_preds))
