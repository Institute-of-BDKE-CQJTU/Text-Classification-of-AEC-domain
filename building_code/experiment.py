# experiment.py
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
    print("Running experiment with config:", config, "for model", args.model_name)
    
    test_f1, best_model_state, model_config, best_model_instance = train_and_evaluate_model(
        config, train_data, val_data, test_data, args.model_name
    )
    print("Experiment finished. Test F1:", test_f1)

    result_csv = './results.csv'
    if not os.path.exists(result_csv):
        results_df = pd.DataFrame(columns=[
            'model_name', 'learning_rate', 'batch_size',
            'weight_decay', 'num_warmup_steps', 'epochs', 'seed', 'test_f1'
        ])
        results_df.to_csv(result_csv, index=False)

    new_result = pd.DataFrame([{
        'model_name': args.model_name,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'num_warmup_steps': args.num_warmup_steps,
        'epochs': args.epochs,
        'seed': args.seed,
        'test_f1': test_f1
    }])
    new_result.to_csv(result_csv, mode='a', header=False, index=False)
    print(f"Experiment results saved to {result_csv}")

if __name__ == "__main__":
    main()
