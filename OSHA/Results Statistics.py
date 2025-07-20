import csv
from collections import defaultdict, Counter

input_filename = './LLM/2_1_deepseek.csv'

error_stats = defaultdict(lambda: {"count": 0, "counter": Counter()})
all_true_labels = set() 

with open(input_filename, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        true_label = row["true_label"]
        all_true_labels.add(true_label)
        if row["result"] == "错误":
            error_stats[true_label]["count"] += 1
            error_stats[true_label]["counter"][row["predict_label"]] += 1

print("True Label, Error Count, Most Incorrect Predicted Label")

for label in sorted(all_true_labels):
    count = error_stats[label]["count"]
    if count > 0:
        most_common_predict, _ = error_stats[label]["counter"].most_common(1)[0]
    else:
        most_common_predict = ""
    print(f"{label}, {count}, {most_common_predict}")