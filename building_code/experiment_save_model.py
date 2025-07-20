# experiment_save_model.py
import argparse
import os
import pandas as pd
import torch
from config import Config
from train import train_and_evaluate_model
from utils import seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--num_warmup_steps", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    seed_everything(args.seed)

    train_data = pd.read_csv('./data/train.csv')
    val_data = pd.read_csv('./data/val.csv')
    test_data = pd.read_csv('./data/test.csv')
    train_data.columns = ['text', 'label']
    val_data.columns = ['text', 'label']
    test_data.columns = ['text', 'label']

    config = {
        'seed': args.seed,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'num_warmup_steps': args.num_warmup_steps,
        'epochs': args.epochs
    }
    print("Saving model with best config:", config, "for model", args.model_name)
    
    test_f1, best_model_state, model_config, best_model_instance = train_and_evaluate_model(
        config, train_data, val_data, test_data, args.model_name
    )
    print("Best model training finished. Test F1:", test_f1)

    model_save_path = f'./saved_model/{args.model_name}'
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(best_model_instance.state_dict(), os.path.join(model_save_path, 'pytorch_model.bin'))
    model_config.save_pretrained(model_save_path)
    Config.tokenizers[args.model_name].save_pretrained(model_save_path)
    print(f"Best model and tokenizer saved in path: {model_save_path}")

if __name__ == "__main__":
    main()
