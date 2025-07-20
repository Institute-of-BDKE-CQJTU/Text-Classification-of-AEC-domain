import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.csv")

train_list = []
test_list = []

for label, sub_df in df.groupby("label"):
    train_sub, test_sub = train_test_split(
        sub_df,
        test_size=0.25,
        random_state=42
    )
    train_list.append(train_sub)
    test_list.append(test_sub)

train_df = pd.concat(train_list)
test_df = pd.concat(test_list)

print("训练集样本数：", len(train_df))
print("测试集样本数：", len(test_df))

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)