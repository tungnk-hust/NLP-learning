from builtins import print

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from model import CNNSurNameClassifier
from argparse import Namespace
import os

from torch.utils.data import DataLoader, Dataset

class Vocabulary(object):
    def __init__(self, token_to_idx=None, add_unk=True, unk_token='<UNK>'):
        if token_to_idx is None:
            self._token_to_idx = {}
        else:
            self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        self._unk_token = unk_token
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def add_token(self, token):
        try:
           index = self._token_to_idx[token]
        except KeyError:
           index = len(self._token_to_idx)
           self._token_to_idx[token] = index
           self._idx_to_token[index] = token
        return index

    def add_many_tokens(self, tokens):
        return [self.add_token(token) for token in tokens]

    def __len__(self):
        return len(self._token_to_idx)

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        return self._token_to_idx[token]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

class SurnameVectorizer(object):
    def __init__(self, surname_vocab, nationality_vocab, max_surname_length):
        self._surname_vocab = surname_vocab
        self._nationality_vocab = nationality_vocab
        self._max_surname_length = max_surname_length

    def vectorize(self, surname):
        one_hot = np.zeros((len(self._surname_vocab), self._max_surname_length), dtype=np.float32)
        for index, token in enumerate(surname):
            one_hot[self._surname_vocab.lookup_token(token)][index] = 1
        return one_hot

    @classmethod
    def from_dataframe(cls, surname_df):
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)
        max_surname_length = 0

        for index, row in surname_df.iterrows():
            max_surname_length = max(max_surname_length, len(row.surname))

            for letter in row.surname:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)

        return cls(surname_vocab, nationality_vocab, max_surname_length)

class SurnameDataset(Dataset):
    def __init__(self, surname_df, vectorizer):
        self.surname_df = surname_df
        self._vectorizer = vectorizer

        self.train_df = self.surname_df[self.surname_df.split == 'train']
        self.validation_df = self.surname_df[self.surname_df.split == 'val']
        self.test_df = self.surname_df[self.surname_df.split == 'test']

        self._lookup_dict = {
            'train': self.train_df,
            'val': self.validation_df,
            'test': self.test_df
        }

        self.set_split('train')

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split):
        self._target_split = split
        self._target_df = self._lookup_dict[self._target_split]

    def __len__(self):
        return len(self._target_df)

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        surname_vector = self._vectorizer.vectorize(row.surname)
        nationality_target = self._vectorizer._nationality_vocab.lookup_token(row.nationality)
        return {'x_train': surname_vector, 'y_target': nationality_target}

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv):
        surname_df = pd.read_csv(surname_csv)
        train_surname_df = surname_df[surname_df.split == 'train']
        vectorizers = SurnameVectorizer.from_dataframe(train_surname_df)
        return cls(surname_df, vectorizers)

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

def make_state(args):
    return {'stop_early': False,
        'early_stopping_step': 0,
        'early_stopping_best_val': 1e8,
        'learning_rate': args.lr,
        'epoch_index': 0,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': -1,
        'test_acc': -1,
        'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False
    elif train_state['epoch_index'] >= 1:
        loss_t = train_state['val_loss'][-1]

        if loss_t >= train_state['early_stopping_best_val']:
            train_state['early_stopping_step'] += 1

        else:
            torch.save(model.state_dict(), train_state['model_filename'])
            train_state['early_stopping_step'] = 0

        train_state['stop_early'] = train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


args = Namespace(
    # Data and path information
    raw_surnames_csv='data/surnames/preprocessing_surnames.csv',
    model_state_file='model_surnames_clf.pth',
    save_dir="model_storage/ch4/surname_cnn",
    expand_filepaths_to_save_dir=True,

    # Model hyperparameters
    n_layers=2,
    n_hidden=50,
    n_channel=256,

    # Training hyperparameters
    seed=1337,
    lr=0.001,
    n_epochs=20,
    batch_size=128,
    early_stopping_criteria=5,

)

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 1

def set_seed_everywhere(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

if args.expand_filepaths_to_save_dir:
    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)
    print("Expanded filepaths: ")
    print("\t{}".format(args.model_state_file))


# Set seed for reproducibility
set_seed_everywhere(args.seed)

# handle dirs
handle_dirs(args.save_dir)

dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.raw_surnames_csv)

vectorizer = dataset.get_vectorizer()

model = CNNSurNameClassifier(n_input=len(vectorizer._surname_vocab), n_channel=args.n_channel, n_output=len(vectorizer._nationality_vocab))

loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)

train_state = make_state(args)

# training loop
for epoch_index in range(args.n_epochs):
    train_state['epoch_index'] = epoch_index
    dataset.set_split('train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    running_loss = 0.0
    running_acc = 0.0
    model.train()

    for batch_index, batch_dict in enumerate(dataloader):
        # zero the gradients
        optimizer.zero_grad()

        # compute output
        y_pred = model(batch_dict['x_train'])

        # compute loss
        loss = loss_f(y_pred, batch_dict['y_target'])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # use loss to produce gradients
        loss.backward()

        # use optimizer to take gradient step
        optimizer.step()

        # compute accuracy
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)

    model.eval()
    dataset.set_split('val')
    dataloader = DataLoader(dataset)
    running_loss = 0
    running_acc = 0
    for batch_index, batch_dict in enumerate(dataloader):
        y_pred = model(batch_dict['x_train'])

        # compute loss
        loss = loss_f(y_pred, batch_dict['y_target'])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute accuracy
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['val_loss'].append(running_loss)
    train_state['val_acc'].append(running_acc)

    print("Epoch {} val_loss: {} val_acc: {}".format(epoch_index, running_loss, running_acc))
    train_state = update_train_state(args=args, model=model,
                                     train_state=train_state)

    scheduler.step(train_state['val_loss'][-1])

    if train_state['stop_early']:
        break

import matplotlib.pyplot as plt
plt.subplot(121)
plt.plot(np.arange(args.n_epochs), np.array(train_state['val_loss']))
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.subplot(221)
plt.plot(np.arange(args.n_epochs), np.array(train_state['val_acc']))
plt.xlabel('Epoch')
plt.ylabel("Accuracy")
plt.savefig("visualize_val_loss_acc.png")