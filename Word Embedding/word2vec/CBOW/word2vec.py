import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from CBOW.model import CBOWModel
from CBOW.inputdata import *

def make_context_vector(idxs, vocab_size):
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)

def train(model,loss_func, optimizer, sampling_data, epochs=2):
    for epoch in range(epochs):
        total_loss = 0
        for element in sampling_data:
            context = element[0]
            target = element[1]
            context = make_context_vector(context, vocabulary_size)
            print(context, target)
            model.zero_grad()
            log_probs = model(context)
            print(log_probs)
            print(Variable(torch.LongTensor(target)))
            loss = loss_func(log_probs, Variable(torch.LongTensor(target)))
            loss.backward()
            optimizer.step()
            total_loss += loss
        if epoch % 10 == 0:
            print('Epoch {}'.format(epoch), total_loss)

data = "We are about to study the idea of a computational process.\
    Computational processes are abstract beings that inhabit computers.\
    As they evolve, processes manipulate other abstract things called data.\
    The evolution of a process is directed by a pattern of rules\
    called a program. People create programs to direct processes. In effect,\
    we conjure the spirits of the computer with our spells <3"

tokenized_corpus = build_corpus(data)
vocabulary = build_vocab(tokenized_corpus)
word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)

window_size = 2

model = CBOWModel(embedding_size=10, vocabulary_size=vocabulary_size)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
train(model, loss_func, optimizer, sample_training(tokenized_corpus, word2idx), )
torch.save(model, 'w2v_cbow.pt')
