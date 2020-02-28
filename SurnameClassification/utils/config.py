from argparse import Namespace
import os

args = Namespace(
    # Data and path information
    raw_surnames_csv='data/surnames/preprocessing_surnames.csv',
    model_state_file='model_surnames_clf.pth',
    save_dir="model_storage/ch4/surname_mlp",
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