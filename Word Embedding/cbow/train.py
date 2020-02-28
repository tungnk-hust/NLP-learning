import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from argparse import Namespace

args = Namespace(
    # Data and path information
    cbow_csv='data/books/preprocessing_frankenstein.csv',
    model_state_file="model.pth",
    save_dir='model_storage/ch5/cbow',

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
    print("Expanded filepaths: ")
    print("\t{}".format(args.model_state_file))

# Set seed for random
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Handle dir
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
