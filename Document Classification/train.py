import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os

from argparse import Namespace
from dataset import NewsDataset
from model import NewsClassifier
from torch.utils.data import DataLoader
def make_train_state(args):
    return {
        'stop_early': False,
        'early_stop_num_epoch': 0,
        'early_stop_max_epochs': args.early_stop_max_epochs,
        'early_stop_best_val_loss': 1e8,
        'epoch_index': 0,
        'model_filename': args.model_state_file,
        'learning_rate': args.lr,
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

def load_glove_from_file(glove_filepath):
    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r") as fp:
        for index, line in enumerate(fp):
            line = line.split(' ')
            word_to_index[line[0]] = index
            embedding_i = np.array([float(x) for x in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)

def make_embedding_matrix(glove_filepath, words):
    word_to_index, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]
    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_index:
            final_embeddings[i, :] = glove_embeddings[word_to_index[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform(embedding_i)
            final_embeddings[i, :] = embedding_i
    return final_embeddings

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

args = Namespace(
    # Data and path hyper parameters
    news_csv='data/ag_news/preprocessing_news.csv',
    vectorizer_file='vectorizer.json',
    model_state_file='model.pth',
    save_dir='model_storage/document_classification',

    # Model hyper parameters
    glove_filepath='data/glove/glove.6B.100d.txt',
    use_glove=True,
    emb_size=100,
    hidden_dim=100,
    num_channels=100,

    # Training hyper parameters
    early_stop_max_epochs=5,
    seed=1337,
    lr=0.001,
    dropout_p=0.01,
    batch_size=128,
    num_epochs=100,
    expand_file=True,
    reload_from_files=False
)

# expand file
if args.expand_file:
    args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
    args.model_state_file = os.path.join(args.save_dir, args.model_state_file)
    print('Expand file!')
    print("vectorizer_file: {}".format(args.vectorizer_file))
    print("model_state_file: {}".format(args.model_state_file))

# handle path
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# set seed
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

# dataset
args.reload_from_files = True
if args.reload_from_files:
    dataset = NewsDataset.load_data_and_load_vectorizer(args.news_csv, args.vectorizer_file)

else:
    dataset = NewsDataset.load_data_and_make_vectorizer(args.news_csv)
    dataset.save_vectorizer(args.vectorizer_file)

vectorizer = dataset.get_vectorizer()

args.use_glove = True
if args.use_glove:
    words = vectorizer.title_vocab.token_to_idx.keys()
    embeddings = make_embedding_matrix(args.glove_filepath, words)
    print("Using pre-trained embeddings")
else:
    print("Not using pre-trained embeddings")
    embeddings = None

# model
model = NewsClassifier(emb_size=args.emb_size,
                       num_emb=len(vectorizer.title_vocab),
                       num_channels=args.num_channels,
                       num_classes=len(vectorizer.category_vocab),
                       hidden_dim=args.hidden_dim,
                       dropout_p=args.dropout_p,
                       pretrained_embeddings=embeddings)

# loss function
loss_f = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

train_state = make_train_state(args)

try:
    for epoch in range(args.num_epochs):
        train_state['epoch_index'] = epoch
        # train over the training set
        dataset.set_split('train')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        running_loss = 0
        running_acc = 0
        model.train()

        for batch_index, batch_dict in enumerate(dataloader):
            optimizer.zero_grad()

            x_data = batch_dict['x_data']
            y_target = batch_dict['y_target']

            y_pred = model(x_data)

            loss = loss_f(y_pred, y_target)
            loss_t = loss.item()
            running_loss += (loss_t-running_loss)/(batch_index + 1)

            loss.backward()
            optimizer.step()

            acc_t = compute_accuracy(y_pred, y_target)
            running_acc += (acc_t-running_acc)/(batch_index + 1)
        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)

        # validate over the val set
        dataset.set_split('val')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        running_loss = 0
        running_acc = 0
        model.eval()

        for batch_index, batch_dict in enumerate(dataloader):
            x_data = batch_dict['x_data']
            y_target = batch_dict['y_target']

            y_pred = model(x_data)

            loss = loss_f(y_pred, y_target)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            acc_t = compute_accuracy(y_pred, y_target)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)

        print("epoch {} val_loss: {} val_acc: {}".format(epoch, running_loss, running_acc))
        train_state = update_train_state(model, train_state)
        if train_state['stop_early']:
            print("Stop early!")
            break
except KeyboardInterrupt:
    print("Exiting loop")

# compute the loss & accuracy on the test set using the best available model

model.load_state_dict(torch.load(train_state['model_filename']))


dataset.set_split('val')
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

running_loss = 0
running_acc = 0
model.eval()

for batch_index, batch_dict in enumerate(dataloader):
    x_data = batch_dict['x_data']
    y_target = batch_dict['y_target']

    y_pred = model(x_data)

    loss = loss_f(y_pred, y_target)
    loss_t = loss.item()
    running_loss += (loss_t - running_loss) / (batch_index + 1)

    acc_t = compute_accuracy(y_pred, y_target)
    running_acc += (acc_t - running_acc) / (batch_index + 1)

train_state['test_loss'] = running_loss
train_state['test_acc'] = running_acc

print("Test loss: {};".format(train_state['test_loss']))
print("Test Accuracy: {}".format(train_state['test_acc']))
