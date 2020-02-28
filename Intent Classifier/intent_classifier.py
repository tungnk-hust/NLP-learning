import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import re
import pandas as pd

data = pd.read_csv("data.csv")
raw_corpus = data['text'].tolist()
intents = data['target'].tolist()

intent_unique = []
for x in intents:
    if x not in intent_unique:
        intent_unique.append(x)
n_output = len(intent_unique)
labels = {x: i for i, x in enumerate(intent_unique)}
def to_one_hot(label, n_output=n_output):
    tmp = torch.zeros(n_output)
    tmp[label] = 1
    return tmp.long()

def build_corpus(raw_corpus):
    corpus = []
    for sentence in raw_corpus:
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        corpus.append(pattern.findall(sentence.lower()))
    return corpus

corpus = build_corpus(raw_corpus)

vocab = []
for sentence in corpus:
    for word in sentence:
        if word not in vocab and word:
            vocab.append(word.lower())
vocab.sort()
word2idx = {word: i for i, word in enumerate(vocab)}

def convert_to_idx(sentence, word2idx):
    sentence_idx = []
    for word in sentence:
        sentence_idx.append(word2idx[word])
    return sentence_idx
corpus_idx = []
for sentence in corpus:
    corpus_idx.append(convert_to_idx(sentence, word2idx))

class IntentClassifier(nn.Module):
    def __init__(self, voab_size, n_ouput, d_model=512, n_hidden1=2058, n_hidden2=2048, n_hidden3=1024):
        super(IntentClassifier, self).__init__()
        self.emb = nn.Embedding(voab_size, d_model)
        self.W_1 = nn.Linear(d_model, n_hidden1)
        self.drop1 = nn.Dropout(0.3)
        self.W_2 = nn.Linear(n_hidden1, n_hidden2)
        self.drop2 = nn.Dropout(0.3)
        self.W_3 = nn.Linear(n_hidden2, n_hidden3)
        self.drop3 = nn.Dropout(0.3)
        self.W_4 = nn.Linear(n_hidden3, n_ouput)

    def forward(self, inputs):
        emb_inputs = self.emb(inputs)
        out = self.drop1(nn.ReLU()(self.W_1(emb_inputs)))
        out = self.drop2(nn.ReLU()(self.W_2(out)))
        out = self.drop3(nn.ReLU()(self.W_3(out)))
        out = nn.Softmax()(self.W_4(out))
        return out

voab_size = len(vocab)

model = IntentClassifier(voab_size, n_output)
loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
n_epochs = 100

for epoch in range(n_epochs):
    optimizer.zero_grad()
    total_loss = 0
    for element in zip(corpus_idx, intents):
        input = Variable(torch.LongTensor(element[0]))
        output = to_one_hot(labels[element[1]])
        pred = model(input)
        loss = loss_f(pred.t(), output.contiguous().view(-1))
        total_loss += loss
        loss.backward()
        optimizer.step()
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_loss))

torch.save(model.state_dict(), 'models/transformer_v1.pkl')