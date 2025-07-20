#dataset.py 将数据集按7类8:1:1划分成训练集、验证集和测试集。
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_paths):
    texts = []
    labels = []
    label_map = {
        'direct': 0,
        'general': 1,
        'indirect': 2,
        'method': 3,
        'others': 4,
        'reference': 5,
        'term': 6
    }
    for file_path, label_name in file_paths.items():
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                texts.append(line.strip())
                labels.append(label_map[label_name])
    return texts, labels

def save_data(texts, labels, file_path):
    df = pd.DataFrame({'text': texts, 'label': labels})
    df.to_csv(file_path, index=False)

def prepare_data():
    file_paths = {
        '../dataset/direct.txt': 'direct',
        '../dataset/general.txt': 'general',
        '../dataset/indirect.txt': 'indirect',
        '../dataset/method.txt': 'method',
        '../dataset/others.txt': 'others',
        '../dataset/reference.txt': 'reference',
        '../dataset/term.txt': 'term'
    }

    print("Loading datasets...")
    texts, labels = load_data(file_paths)
    print(f"Loaded {len(texts)} samples.")

    data = pd.DataFrame({'text': texts, 'label': labels})

    train_texts = []
    train_labels = []
    val_texts = []
    val_labels = []
    test_texts = []
    test_labels = []

    for label in data['label'].unique():
        subset = data[data['label'] == label]
        train, temp = train_test_split(subset, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        train_texts.extend(train['text'].tolist())
        train_labels.extend(train['label'].tolist())
        val_texts.extend(val['text'].tolist())
        val_labels.extend(val['label'].tolist())
        test_texts.extend(test['text'].tolist())
        test_labels.extend(test['label'].tolist())

    print(f"Training set size: {len(train_texts)}")
    print(f"Validation set size: {len(val_texts)}")
    print(f"Test set size: {len(test_texts)}")

    save_data(train_texts, train_labels, './train.csv')
    save_data(val_texts, val_labels, './val.csv')
    save_data(test_texts, test_labels, './test.csv')

if __name__ == "__main__":
    prepare_data()
