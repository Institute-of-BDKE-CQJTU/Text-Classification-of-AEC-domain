#检查模型完整性
from transformers import AutoModel

model_dirs = ['./saved_model/bert']
model_dirs = ['./saved_model/roberta']

for model_dir in model_dirs:
    model = AutoModel.from_pretrained(model_dir)
    print(model)