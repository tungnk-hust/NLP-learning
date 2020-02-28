import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from model import MLPSurNameClassifier

from utils.config import args
from training import SurnameDataset, compute_accuracy
from argparse import Namespace
import os

from torch.utils.data import DataLoader, Dataset
from inference import predict
dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.raw_surnames_csv)
vectorizer = dataset.get_vectorizer()

model = MLPSurNameClassifier(n_input=len(vectorizer._surname_vocab), n_output=len(vectorizer._nationality_vocab),
                             n_hidden=args.n_hidden, n_layers=args.n_layers)

model.load_state_dict(torch.load(args.model_state_file))
loss_f = nn.CrossEntropyLoss()
model.eval()
dataset.set_split('test')
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

print("Testing....!")
print("Loss: ", running_loss)
print("Accuracy: ", running_acc)

for _, row in dataset.surname_df.iterrows():
    pred = predict(model, vectorizer, row.surname)
    print(row.surname, pred, row.nationality)