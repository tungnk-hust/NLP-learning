import numpy as np
import pandas as pd
from argparse import Namespace
import re
import collections

args = Namespace(
    raw_file='data/ag_news/news.csv',
    preprocessing_output='data/ag_news/preprocessing_news.csv',
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    seed=1111
)

np.random.seed(args.seed)

news_df = pd.read_csv(args.raw_file)
# print(news_df.head(10))

by_category = collections.defaultdict(list)
for _, row in news_df.iterrows():
    by_category[row.category].append(row.to_dict())

final_news = []
for _, item_list in sorted(by_category.items()):
    np.random.shuffle(item_list)
    n = len(item_list)
    n_train = int(args.train_split * n)
    n_val = int(args.val_split * n)
    n_test = int(args.test_split * n)

    for item in item_list[:n_train]:
        item['split'] = 'train'

    for item in item_list[n_train: n_train+n_val]:
        item['split'] = 'val'

    for item in item_list[n_train+n_val:]:
        item['split'] = 'test'
    final_news.extend(item_list)

final_news_df = pd.DataFrame(final_news)
print(final_news_df.split.value_counts())
print(final_news_df.head(10))
# Preprocess the reviews
def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text
final_news_df.title = final_news_df.title.apply(preprocess_text)

final_news_df.to_csv(args.preprocessing_output, index=False)