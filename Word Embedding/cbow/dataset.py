import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import collections
import pandas as pd
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

class Vocabulary(object):
    def __init__(self, token_to_idx=None, mask_token='<MASK>', add_unk=True, unk_token='<UNK>'):
        if token_to_idx is None:
            token_to_idx = {}

        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token
        self._mask_token = mask_token
        self._mask_index = self.add_token(self._mask_token)
        self._unk_index = -1
        if self._add_unk:
            self._unk_index = self.add_token(self._unk_index)

    def add_token(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]

        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        if self._unk_index >= 0:
            return self._token_to_idx.get(token, self._unk_index)

        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        if index not in self._token_to_idx:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def __len__(self):
        return len(self._token_to_idx)

class CbowVectorizer(object):
    def __init__(self, cbow_vocab):
        self._cbow_vocab = cbow_vocab

    def vecterize(self, context, vector_length=-1):
        indices = [self._cbow_vocab.lookup_token(token) for token in tokenizer.tokenize(context)]
        if vector_length < 0:
            vector_length = len(indices)

        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = self._cbow_vocab._mask_index
        return vector

    @classmethod
    def from_dataframe(cls, cbow_df):
        cbow_vocab = Vocabulary()
        for index, row in cbow_df.iterrows():
            for token in tokenizer.tokenize(row.context):
                cbow_vocab.add_token(token)
            cbow_vocab.add_token(row.target)

        return cls(cbow_vocab)

class CbowDataset(Dataset):
    def __init__(self, cbow_df, vecterizer):
        self._vecterizer = vecterizer
        self.cbow_df = cbow_df

        measure_len = lambda context: len(context.split(" "))
        self._max_seq_length = max(map(measure_len, cbow_df.context))

        self.train_df = cbow_df[cbow_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = cbow_df[cbow_df.split == "val"]
        self.val_size = len(self.val_df)

        self.test_df = cbow_df[cbow_df.split == "test"]
        self.test_size = len(self.test_df)

        self._lookup_dir = {"train": self.train_df, "val": self.val_df, "test": self.test_df}

        self.set_split("train")

    def set_split(self, split):
        self._target_split = split
        self._target_df = self._lookup_dir[self._target_split]

    def __len__(self):
        return len(self._target_df)

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        vector = self._vecterizer.vecterize(row.context, self._max_seq_length)
        target_index = self._vecterizer._cbow_vocab.lookup_token(row.target)

        return {'x_train': vector, 'y_target': target_index}

    def get_num_batches(self, batch_size):
        return int(len(self) / batch_size)

    @classmethod
    def load_dataset_and_make_vecterizer(cls, cbow_csv):
        cbow_df = pd.read_csv(cbow_csv)
        train_cbow_df = cbow_df[cbow_df.split == 'train']
        vectorizer = CbowVectorizer.from_dataframe(train_cbow_df)
        return cls(cbow_df, vectorizer)

    def get_vectorizer(self):
        return self._vecterizer

    def get_vocab_size(self):
        return len(self._vecterizer._cbow_vocab)