import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import collections
import pandas as pd

class Vocabulary(object):
    def __init__(self, token_to_idx=None, mask_token='<MASK>', add_unk=True, unk_token='<UNK>'):
        if token_to_idx is None:
            token_to_index = {}

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
            index = self._token_to_idx(token)

        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index