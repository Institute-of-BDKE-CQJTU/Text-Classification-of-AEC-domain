# master.py
import os
import subprocess
import pandas as pd

lr_choices = [1e-4, 3e-5, 5e-5, 7e-5, 9e-5]
batch_size_choices = [64, 128]
weight_decay_choices = [0.01, 0.001]
num_warmup_steps_choices = [0.1]

model_name = "bert"

result_csv = './results.csv'
if os.path.exists(result_csv):
    os.remove(result_csv)
    print("旧的结果文件已删除。")

for lr in lr_choices:
    for bs in batch_size_choices:
        for wd in weight_decay_choices:
            for nws in num_warmup_steps_choices:
                cmd = [
                    "python", "experiment.py",
                    "--model_name", model_name,
                    "--lr", str(lr),
                    "--batch_size", str(bs),
                    "--weight_decay", str(wd),
                    "--num_warmup_steps", str(nws),
                    "--epochs", "50"
                ]
                print("Running command:", " ".join(cmd))
                subprocess.run(cmd, check=True)

if not os.path.exists(result_csv):
    print("没有找到实验结果文件！")
    exit()

results_df = pd.read_csv(result_csv)
print("\n所有实验已完成，最终结果为：")
print(results_df.sort_values(by='test_f1', ascending=False))

best_row = results_df.loc[results_df['test_f1'].idxmax()]
best_lr = best_row['learning_rate']
best_bs = best_row['batch_size']
best_wd = best_row['weight_decay']
best_nws = best_row['num_warmup_steps']
best_seed = best_row['seed']

print("\n最优参数组合为：")
print(best_row)

save_cmd = [
    "python", "experiment_save_model.py",
    "--model_name", model_name,
    "--lr", str(best_lr),
    "--batch_size", str(int(best_bs)),
    "--weight_decay", str(best_wd),
    "--num_warmup_steps", str(best_nws),
    "--epochs", "50",
    "--seed", str(int(best_seed))
]
print("\n将使用最佳参数组合重新训练并保存模型：", save_cmd)
subprocess.run(save_cmd, check=True)

print("完整流程结束。")
