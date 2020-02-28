import numpy as np
import collections
import pandas as pd
import re
from argparse import Namespace

args = Namespace(
    raw_file_data='data/surnames/surnames.csv',
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_file_csv="data/surnames/preprocessing_surnames.csv",
    seed=1337
)
raw_data = pd.read_csv(args.raw_file_data)
# print(raw_data.head())

by_nationality = collections.defaultdict(list)
for _, row in raw_data.iterrows():
    by_nationality[row.nationality].append(row.to_dict())

# print(by_nationality)
final_list = []
np.random.seed(args.seed)
for _, item_list in sorted(by_nationality.items()):
    np.random.shuffle(item_list)
    n = len(item_list)
    n_train = int(n * args.train_proportion)
    n_val = int(n * args.val_proportion)
    n_test = int(n * args.test_proportion)

    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train: n_train + n_val]:
        item['split'] = 'val'
    for item in item_list[n_train + n_val:]:
        item['split'] = 'test'

    final_list.extend(item_list)

final_surnames = pd.DataFrame(final_list)
print(final_surnames.head())
print(final_surnames.split.value_counts())

# write the output file csv
final_surnames.to_csv(args.output_file_csv, index=False)
