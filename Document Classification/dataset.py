import numpy as np
import pandas as pd
import string
import json

from collections import Counter
from torch.utils.data import Dataset


class Vocabulary(object):
    def __init__(self, token_to_idx=None):
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx
        self.idx_to_token = {i: w for w, i in token_to_idx.items()}

    def to_serializable(self):
        return {'token_to_idx': self.token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        return self.token_to_idx[token]

    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self.idx_to_token[index]

    def __len__(self):
        return len(self.token_to_idx)

class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):
        super(SequenceVocabulary, self).__init__(token_to_idx)
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.begin_seq_token = begin_seq_token
        self.end_seq_token = end_seq_token

        self.unk_index = self.add_token(unk_token)
        self.mask_index = self.add_token(mask_token)
        self.begin_seq_index = self.add_token(begin_seq_token)
        self.end_seq_index = self.add_token(end_seq_token)

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self.token_to_idx.get(token, self.unk_index)
        else:
            return self.token_to_idx[token]

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({
            'unk_token': self.unk_token,
            'mask_token': self.mask_token,
            'begin_seq_token': self.begin_seq_token,
            'end_seq_token': self.end_seq_token
        })
        return contents

class NewsVectorizer(object):
    def __init__(self, title_vocab, category_vocab):
        self.title_vocab = title_vocab
        self.category_vocab = category_vocab

    def vectorize(self, title, vector_length):
        indices = [self.title_vocab.begin_seq_index]
        token_indices = [self.title_vocab.lookup_token(token) for token in title.split(' ')]
        indices.extend(token_indices)
        indices.append(self.title_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        vector = np.zeros(vector_length, np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = self.title_vocab.mask_index
        return vector

    @classmethod
    def from_dataframe(cls, news_df, cutoff=25):
        category_vocab = Vocabulary()
        for category in sorted(set(news_df.category)):
            category_vocab.add_token(category)

        word_counts = Counter()
        title_vocab = SequenceVocabulary()
        for title in news_df.title:
            for token in title.split(" "):
                if token not in string.punctuation:
                    word_counts[token] += 1

        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                title_vocab.add_token(word)

        return cls(title_vocab, category_vocab)

    @classmethod
    def from_serializable(cls, contents):
        title_vocab = \
            SequenceVocabulary.from_serializable(contents['title_vocab'])
        category_vocab = \
            Vocabulary.from_serializable(contents['category_vocab'])

        return cls(title_vocab=title_vocab, category_vocab=category_vocab)

    def to_serializable(self):
        return {'title_vocab': self.title_vocab.to_serializable(),
                'category_vocab': self.category_vocab.to_serializable()}

class NewsDataset(Dataset):
    def __init__(self, news_df, vectorizer):
        self.new_df = news_df
        self.vectorizer = vectorizer

        self.train_df = news_df[news_df.split == "train"]
        self.train_size = len(self.train_df)
        self.val_df = news_df[news_df.split == "val"]
        self.val_size = len(self.val_df)
        self.test_df = news_df[news_df.split == "test"]
        self.test_size = len(self.test_df)
        self.lookip_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size)
        }

        self.max_length = 0
        for _, row in news_df.iterrows():
            self.max_length = max(self.max_length, len(row.title.split(' ')))
        self.max_length += 2

        self.set_split("train")


    def set_split(self, split):
        self.target_split = split
        self.target_df, self.target_size = self.lookip_dict[split]

    def __len__(self):
        return self.target_size

    def __getitem__(self, index):
        row = self.target_df.iloc[index]
        title_vector = self.vectorizer.vectorize(row.title, self.max_length)

        category_index = self.vectorizer.category_vocab.lookup_token(row.category)

        return {'x_data': title_vector, 'y_target': category_index}

    @classmethod
    def load_data_and_make_vectorizer(cls, news_csv):
        news_df = pd.read_csv(news_csv)
        train_news_df = news_df[news_df.split == 'train']
        return cls(news_df, NewsVectorizer.from_dataframe(train_news_df))

    @classmethod
    def load_data_and_load_vectorizer(cls, news_csv, vectorizer_filepath):
        news_df = pd.read_csv(news_csv)
        with open(vectorizer_filepath) as fp:
            contents = json.load(fp)
        vectorizer = NewsVectorizer.from_serializable(contents)

        return cls(news_df, vectorizer)

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, 'w+') as fp:
            json.dump(self.vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        return self.vectorizer