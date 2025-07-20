#dataset.py
import torch
from torch.utils.data import Dataset
import random

class Text_Classification_Dataset(Dataset):
    def __init__(self, tokenizer, max_len, texts, labels, augment=False, seed=42):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        if self.augment:
            text = self.data_augmentation(text, item)
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[item], dtype=torch.long),
            'text': text
        }

    def data_augmentation(self, text, item):
        random.seed(self.seed + item)
        torch.manual_seed(self.seed + item)
        
        words = text.split()
        if len(words) > 1:

            if random.random() < 0.1:
                del words[random.randint(0, len(words) - 1)]

            if random.random() < 0.1:
                words[random.randint(0, len(words) - 1)] = '[MASK]'

            if random.random() < 0.1:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}