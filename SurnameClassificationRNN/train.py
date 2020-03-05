import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from torch.utils.data import DataLoader
from dataset import SurnameDataset
from model import SurnameClassifier
from argparse import Namespace

def make_train_state(args):
    return {
        'stop_early': False,
        'early_stop_num_epoch': 0,
        'early_stop_max_epochs': args.early_stop_max_epochs,
        'early_stop_best_val_loss': 1e8,
        'epoch_index': 0,
        'model_filename': args.model_state_file,
        'learning_rate': args.learning_rate,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': 0,
        'test_acc': 0
    }

def update_train_state(model, train_state):
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])

    else:
        loss_t = train_state['val_loss'][-1]
        if loss_t < train_state['early_stop_best_val_loss']:
            torch.save(model.state_dict(), train_state['model_filename'])
            train_state['early_stop_num_epoch'] = 0
            train_state['early_stop_best_val_loss'] = loss_t
        else:
            train_state['early_stop_num_epoch'] += 1

        if train_state['early_stop_num_epoch'] >= train_state['early_stop_max_epochs']:
            train_state['stop_early'] = True

    return train_state

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

args = Namespace(
    # Data and path information
    surname_csv="data/surnames/preprocessing_surnames.csv",
    vectorizer_file="vectorizer.json",
    save_dir="model_storage/surname_classification",
    model_state_file="model.pth",

    # Model hyper parameter
    embedding_size=100,
    rnn_hidden_size=64,

    # Training hyper parameter
    num_epochs=100,
    learning_rate=0.001,
    batch_size=64,
    seed=1337,
    early_stop_max_epochs=5,
    load_vectorizer=False,
    expand_file=True
)
# set seed
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

# handle dir
if os.path.exists(args.save_dir) is False:
    os.makedirs(args.save_dir)
if args.expand_file:
    args.model_state_file = os.path.join(args.save_dir, args.model_state_file)
    args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)

    print("Expand file!")
    print(args.model_state_file)
    print(args.vectorizer_file)

# dataset
if args.load_vectorizer:
    dataset = SurnameDataset.load_dataset_and_load_vectorizer(args.surname_csv, args.vectorizer_file)
else:
    dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv)
    dataset.save_vetorizer(args.vectorizer_file)

# vectorizer
vectorizer = dataset.get_vetorizer()

# model
model = SurnameClassifier(embedding_size=args.embedding_size, num_embeddings=len(vectorizer.surname_vocab),
                          rnn_hidden_size=args.rnn_hidden_size, num_classes=len(vectorizer.nationality_vocab))
# loss function
loss_f = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)
# train_state
train_state = make_train_state(args)

# training
print("Training...")
try:
    for epoch in range(args.num_epochs):
        # train over the training set
        dataset.set_split("train")
        dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
        running_loss = 0
        running_acc = 0
        model.train()
        for batch_index, batch_dict in enumerate(dataloader):
            x_data = batch_dict['x_data']
            y_target = batch_dict['y_target']
            x_length = batch_dict['x_length']

            # zero to gragient
            optimizer.zero_grad()

            # compute the output
            y_pred = model(x_data, x_length)

            # compute loss
            loss = loss_f(y_pred, y_target)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss)/(batch_index + 1)

            # loss backward
            loss.backward()

            # optimize
            optimizer.step()

            # compute the accuracy
            acc_t = compute_accuracy(y_pred, y_target)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)

        # validate over the training set
        dataset.set_split("val")
        dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
        running_loss = 0
        running_acc = 0
        model.eval()
        for batch_index, batch_dict in enumerate(dataloader):
            x_data = batch_dict['x_data']
            y_target = batch_dict['y_target']
            x_length = batch_dict['x_length']

            # compute the output
            y_pred = model(x_data, x_length)

            # compute loss
            loss = loss_f(y_pred, y_target)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(y_pred, y_target)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)
        print("Epoch {} loss: {}  accuracy: {}".format(epoch, running_loss, running_acc))
        train_state = update_train_state(model, train_state)

        scheduler.step(train_state['val_loss'][-1])

        if train_state['stop_early']:
            print()
            print("Stop early!")
            break
except KeyboardInterrupt:
    print("Exits training")
print("Train Done!")


