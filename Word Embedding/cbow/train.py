import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from argparse import Namespace
from dataset import *
from model import CbowClassifier
from torch.utils.data import DataLoader


def make_train_state(args):
    return {
        'stop_early': False,
        'early_stop_step': 0,
        'early_stopping_best_val': 1e8,
        'early_stopping_criteria': args.early_stopping_criteria,
        'learning_rate': args.learning_rate,
        'train_loss': [],
        'val_loss': [],
        'test_loss': -1,
        'train_acc': [],
        'val_acc': [],
        'test_acc': -1,
        'epoch_index': 0,
        'model_filename': args.model_state_file
    }

def update_train_state(model, train_state):
    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    else:
        loss_t = train_state['val_loss'][-1]
        if loss_t < train_state['early_stopping_best_val']:
            train_state['early_stop_step'] = 0
            train_state['early_stopping_best_val'] = loss_t
            torch.save(model.state_dict(), train_state['model_filename'])
        else:
            train_state['early_stop_step'] += 1

        if train_state['early_stop_step'] >= train_state['early_stopping_criteria']:
            train_state['stop_early'] = True

    return train_state

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

def visualize_train(train_state, ):
    plt.figure()
    plt.subplot(121)
    plt.plot(np.arange((train_state['epoch_index'])), train_state['train_loss'])
    plt.plot(np.arange((train_state['epoch_index'])), train_state['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(221)
    plt.plot(np.arange((train_state['epoch_index'])), train_state['train_acc'])
    plt.plot(np.arange((train_state['epoch_index'])), train_state['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.savefig(args.visualize_train_image)


args = Namespace(
    # Data and path information
    cbow_csv='data/books/preprocessing_frankenstein.csv',
    model_state_file="model.pth",
    save_dir='model_storage/ch5/cbow',
    result_dir='results',
    visualize_train_image='visualize_training.png',
    train_state_result='train_state.json',
    # Model hyper parameters
    embedding_size=50,

    # Training hyper parameters
    seed=1337,
    num_epochs=100,
    learning_rate=0.0001,
    batch_size=32,
    early_stopping_criteria=5,

    # Runtime options
    expand_filepaths_to_save_dir=True
)

# Expand file
if args.expand_filepaths_to_save_dir:
    args.model_state_file = os.path.join(args.save_dir, args.model_state_file)
    args.visualize_train_image = os.path.join(args.result_dir, args.visualize_train_image)
    args.train_state_result = os.path.join(args.result_dir, args.train_state_result)
    print("Expanded filepaths: ")
    print("\t{}".format(args.model_state_file))
    print("\t{}".format(args.visualize_train_image))
    print("\t{}".format(args.train_state_result))

# Set seed for random
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Handle dir
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

dataset = CbowDataset.load_dataset_and_make_vecterizer(args.cbow_csv)

model = CbowClassifier(vocab_size=dataset.get_vocab_size(), emb_size=args.embedding_size)

loss_f = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)
train_state = make_train_state(args)

try:
    for epoch_index in range(args.num_epochs):
        # training over the train_df
        running_loss = 0
        running_acc = 0
        dataset.set_split('train')
        model.train()
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

        for batch_index, batch_dict in enumerate(dataloader):
            x = batch_dict['x_train']
            y_target = batch_dict['y_target']

            # zero the gradient
            optimizer.zero_grad()

            # compute output
            y_pred = model(x)

            # compute loss
            loss = loss_f(y_pred, y_target)

            loss_t = loss.item()
            running_loss += (loss_t-running_loss) / (batch_index + 1)

            # use the loss to compute gradient
            loss.backward()

            # optimizer
            optimizer.step()

            # compute accuracy
            acc_t = compute_accuracy(y_pred, y_target)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)

        # validate over the val_df
        running_loss = 0
        running_acc = 0
        dataset.set_split('val')
        model.eval()
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

        for batch_index, batch_dict in enumerate(dataloader):
            x = batch_dict['x_train']
            y_target = batch_dict['y_target']

            # compute output
            y_pred = model(x)

            # compute loss

            loss = loss_f(y_pred, y_target)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute accuracy
            acc_t = compute_accuracy(y_pred, y_target)
            running_acc += (acc_t-running_acc) / (batch_index + 1)

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)
            if batch_index % 100 == 0:
                print("Epoch {} step {} val_loss: {}    val_acc: {}".format(epoch_index, batch_index, running_loss, running_acc))
        train_state['epoch_index'] = epoch_index
        train_state = update_train_state(model, train_state)
        if train_state['stop_early']:
            break

except KeyboardInterrupt:
    print("Exiting loop")

# visualize loss and accuracy on val set and train set
visualize_train(train_state)

# testing

running_loss = 0
running_acc = 0
dataset.set_split('test')
model.eval()
dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

for batch_index, batch_dict in enumerate(dataloader):
    x = batch_dict['x_train']
    y_target = batch_dict['y_target']

    # compute output
    y_pred = model(x)

    # compute loss

    loss = loss_f(y_pred, y_target)
    loss_t = loss.item()
    running_loss += (loss_t - running_loss) / (batch_index + 1)

    # compute accuracy
    acc_t = compute_accuracy(y_pred, y_target)
    running_acc += (acc_t-running_acc) / (batch_index + 1)

train_state['test_loss'] = running_loss
train_state['test_acc'] = running_acc

json = json.dumps(train_state)
f = open(args.train_state_result, "w")
f.write(json)
f.close()