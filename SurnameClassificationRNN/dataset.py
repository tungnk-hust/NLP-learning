import numpy as np
import pandas as pd
import json

from torch.utils.data import Dataset
class Vocabulary(object):
    def __init__(self, token_to_idx=None):
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_index = token_to_idx
        self.idx_to_token = {i: w for w, i in token_to_idx.items()}

    def __len__(self):
        return len(self.token_to_index)

    def lookup_token(self, token):
        return self.token_to_index[token]

    def lookup_index(self, index):
        if index in self.idx_to_token:
            return self.idx_to_token[index]
        else:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)

    def add_token(self, token):
        if token in self.token_to_index:
            index = self.token_to_index[token]
        else:
            index = len(self.token_to_index)
            self.token_to_index[token] = index
            self.idx_to_token[index] = token

        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def to_serializable(self):
        return {"token_to_idx": self.token_to_index}
class SurnameVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token='<UNK>', mask_token='<MASK>', \
        start_seq_token='<START>', end_seq_token='<END>'):
        super(SurnameVocabulary, self).__init__(token_to_idx)
        self._unk_token = unk_token
        self._mask_token = mask_token
        self._start_seq_token = start_seq_token
        self._end_seq_token = end_seq_token

        self._unk_index = self.add_token(unk_token)
        self._mask_index = self.add_token(mask_token)
        self._start_seq_index = self.add_token(start_seq_token)
        self._end_seq_index = self.add_token(end_seq_token)

    def lookup_token(self, token):
        if self._unk_index >= 0:
            return self.idx_to_token.get(token, self._unk_index)
        else:
            return self.idx_to_token[token]

    def to_serializable(self):
        return {
            "token_to_idx": self.token_to_index,
            "unk_token": self._unk_index,
            "mask_token": self._mask_token,
            "start_seq_token": self._start_seq_token,
            "end_seq_token": self._end_seq_token
        }

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

class Vetorizer(object):
    def __init__(self, surname_vocab, nationality_vocab):
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab

    def vectorize(self, surname, seq_length=-1):
        indices = [self.surname_vocab._start_seq_index]
        indices_token = [self.surname_vocab.lookup_token(token) for token in surname]
        indices.extend(indices_token)
        indices.append(self.surname_vocab._end_seq_index)

        if seq_length < 0:
            seq_length = len(indices)

        vector = np.zeros(seq_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = self.surname_vocab._mask_index

        return vector, len(indices)

    def to_serializable(self):
        return {
            "surname_vocab": self.surname_vocab.to_serializable(),
            "nationality_vocab": self.nationality_vocab.to_serializable()
        }
    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    @classmethod
    def from_dataframe(cls, surname_df):
        nationality_vocab = Vocabulary()
        for x in sorted(set(surname_df.nationality)):
            nationality_vocab.add_token(x)

        surname_vocab = SurnameVocabulary()
        for surname in surname_df.surname:
            for character in surname:
                surname_vocab.add_token(character)

        return cls(surname_vocab, nationality_vocab)

class SurnameDataset(Dataset):
    def __init__(self, surname_df, vectorizer):
        self.surname_df = surname_df
        self.vectorizer = vectorizer

        self.max_len_seq = 0
        for surname in surname_df.surname:
            if len(surname) > self.max_len_seq:
                self.max_len_seq = len(surname)
        self.max_len_seq += 2
        self.train_df = surname_df[surname_df.split == "train"]
        self.train_size = len(self.train_df)
        self.val_df = surname_df[surname_df.split == "val"]
        self.val_size = len(self.val_df)
        self.test_df = surname_df[surname_df.split == "test"]
        self.test_size = len(self.test_df)

        self.lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size)
        }

        self.set_split("train")

    def set_split(self, split):
        self.target_df, self.target_size = self.lookup_dict[split]

    def __len__(self):
        return self.target_size

    def __getitem__(self, index):
        surname_index = self.target_df.iloc[index]
        surname = surname_index.surname
        nationality = surname_index.nationality

        surname_vector, x_length = self.vectorizer.vectorize(surname, self.max_len_seq)
        nationality_index = self.vectorizer.nationality_vocab.lookup_token(nationality)

        return {"x_data": surname_vector, "y_target": nationality_index, "x_length": x_length}

    def save_vetorizer(self, vectorizer_path):
        with open(vectorizer_path, 'w+') as pf:
            json.dump(self.vectorizer.to_serializable(), pf)

    def get_vetorizer(self):
        return self.vectorizer

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv):
        surname_df = pd.read_csv(surname_csv)
        train_df = surname_df[surname_df.split == "train"]
        vectorizer = Vetorizer.from_dataframe(train_df)
        return cls(surname_df, vectorizer)

    @classmethod
    def load_dataset_and_load_vectorizer(cls, surname_csv, vectorizer_path):
        surname_df = pd.read_csv(surname_csv)
        with open(vectorizer_path, 'r') as pf:
            contents = json.load(pf)
        vectorizer = Vetorizer.from_serializable(surname_df)
        return cls(surname_df, vectorizer)

