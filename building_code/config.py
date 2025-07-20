#config.py
from transformers import AutoTokenizer
import torch

class Config():
    seed = 2024
    pretrain_model_paths = {
        "roberta": "",#替换为实际roberta模型路径
        "bert": ""#替换为实际bert模型路径
    }
    tokenizers = {key: AutoTokenizer.from_pretrained(path) for key, path in pretrain_model_paths.items()}
    n_class = 7  
    max_len = 128
    batch_size = 32
    n_fold = 5
    trn_folds = [0, 1, 2, 3, 4]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = 50
    lr = 5e-5
    eps = 1e-6
    num_warmup_steps = 0.03
    weight_decay = 0.001
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0
    save_prefix = "ensemble"

    lr_choices = [1e-5, 3e-5]
    batch_size_choices = [32, 64]
    weight_decay_choices = [0.001, 0.0001]
    num_warmup_steps_choices = [0.05, 0.1]
