# model.py
import torch
from torch import nn
from InstructorEmbedding import INSTRUCTOR

class ClassificationModel(nn.Module):

    def __init__(self, config):
        super(ClassificationModel, self).__init__()
        self.config = config
        # 加载 INSTRUCTOR 模型
        self.instructor = INSTRUCTOR(config.pretrained_path_instructor)
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.fc = nn.Linear(768, self.config.output_dim)

    def forward(self, input_ids, attention_mask):
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        out = self.instructor(features)
        emb = out["sentence_embedding"]  # shape = (batch_size, 768)
        emb = self.dropout(emb)
        logits = self.fc(emb)
        return logits
