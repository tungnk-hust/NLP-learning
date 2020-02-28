from argparse import Namespace
import collections
import nltk.data
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
from nltk.util import ngrams

args = Namespace(
    raw_dataset_txt='data/books/frankenstein.txt',
    output_preprecessing='data/books/preprocessing_frankenstein.csv',
    window_size=3,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    seed=1337,
    MASK_TOKEN="<MASK>"
)

with open(args.raw_dataset_txt) as fp:
    book = fp.read()

# split to sentences
sentences = sent_tokenize(book)
# print(len(sentences))
# print(sentences[100])

# tokenizer without punctuation
tokenizer = RegexpTokenizer(r'\w+')
cleaned_sentences = [tokenizer.tokenize(sentence.lower()) for sentence in sentences]
# print(cleaned_sentences[100])

# # remove stopword
# stop_words = set(stopwords.words('english'))
#
# print(stop_words)
# for sentence in cleaned_sentences:
#     for word in sentence:
#         if word in stop_words:
#             sentence.remove(word)
#
# print(cleaned_sentences[100])

# Create windows
def extract_ngrams(data, num):
    n_grams = ngrams(tokenizer.tokenize(data), num, pad_left=True, pad_right=True,\
                     left_pad_symbol=args.MASK_TOKEN, right_pad_symbol=args.MASK_TOKEN)
    return [' '.join(grams) for grams in n_grams]

windows = []
for i, sentence in enumerate(sentences):
    grams = extract_ngrams(sentence, 2*args.window_size + 1)
    grams = grams[args.window_size: -args.window_size]
    for j, gram in enumerate(grams):
        tmp = []
        for word in gram.split():
            if word != MASK_TOKEN and word != cleaned_sentences[i][j]:
                tmp.append(word)
        windows.append([' '.join(tmp), cleaned_sentences[i][j]])

# Convert to dataframe
cbow_data = pd.DataFrame(windows, columns=["context", "target"])
# print(cbow_data.head(10))

# Create split data
n = len(cbow_data)
def get_split(row_num):
    if row_num <= n*args.train_proportion:
        return 'train'
    elif (row_num > n*args.train_proportion) and (row_num <= n*args.train_proportion + n*args.val_proportion):
        return 'val'
    else:
        return 'test'
cbow_data['split']= cbow_data.apply(lambda row: get_split(row.name), axis=1)
print(cbow_data.head(10))

# Write split data to file
cbow_data.to_csv(args.output_preprecessing, index=False)

