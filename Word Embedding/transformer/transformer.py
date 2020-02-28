import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import re
def build_corpus(raw_corpus):
    corpus = []
    for sentence in raw_corpus:
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        corpus.append(pattern.findall(sentence.lower()))
    return corpus

raw_corpus = ['i dance',
          'she cook for me',
          'he sing with my mon at home']

corpus = build_corpus(raw_corpus)

vocab = []
for sentence in corpus:
    for word in sentence:
        if word not in vocab and word:
            vocab.append(word.lower())
vocab.sort()

# print(vocab)
# print(word2idx)

END_SENT = "E"
START_SENT = "S"
PARSING = "P"

max_len = 0
for sentence in corpus:
    if len(sentence) > max_len:
        max_len = len(sentence)
# print(max_len)

def passing_word(corpus, max_len, START_SENT='S', END_SENT='E', PARSING='P'):
    for i in range(len(corpus)):
        len_sent = len(corpus[i])
        num_parsing = max_len - len_sent
        corpus[i].insert(0, START_SENT)
        for j in range(num_parsing):
            corpus[i].extend(PARSING)
        corpus[i].insert(len(corpus[i]), END_SENT)
    return corpus

corpus = passing_word(corpus, max_len)
# print(corpus)
max_len = max_len + 2
vocab.insert(0, END_SENT)
vocab.insert(0, START_SENT)
vocab.insert(0, PARSING)
word2idx = {word: i for i, word in enumerate(vocab)}
# print(vocab)
# print(word2idx)


def convert_to_idx(sentence, word2idx):
    sentence_idx = []
    for word in sentence:
        sentence_idx.append(word2idx[word])
    return sentence_idx
corpus_idx = []
for sentence in corpus:
    corpus_idx.append(convert_to_idx(sentence, word2idx))
# print(corpus)
# print(corpus_idx)
def outputs_shifted_right(sentence_idx):
    len_sentence = len(sentence_idx)
    outputs = np.zeros((len_sentence, max_len))
    for i in range(len_sentence):
        for j in range(len_sentence):
            if j <= i:
                outputs[i][j] = sentence_idx[j]
    outputs.tolist()
    return torch.LongTensor(outputs)

# print(outputs_shifted_right(corpus_idx[1]))


# build model
vocab_size = len(vocab)
src_len = 5
tgt_len = 5

d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

def poition_encoder(n_position, d_model):
    def compute_angle(pos, i):
        return pos / np.power(10000, 2*i/d_model)
    sinusoid_table = np.zeros((n_position, d_model))
    for pos in range(n_position):
        for i in range(d_model):
            if i % 2 == 0:
                sinusoid_table[pos][i] = np.sin(compute_angle(pos, i))
            else:
                sinusoid_table[pos][i] = np.cos(compute_angle(pos, i))
    return torch.FloatTensor(sinusoid_table)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.t()) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultilHeadAttention(nn.Module):
    def __init__(self):
        super(MultilHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k*n_heads)
        self.W_K = nn.Linear(d_model, d_k*n_heads)
        self.W_V = nn.Linear(d_model, d_v*n_heads)
    def forward(self, Q, K, V):
        residual, batch_size = Q, Q.size(0)
        Q_scores = self.W_Q(Q)
        K_scores = self.W_K(K)
        V_scores = self.W_V(V)
        Qs = Q_scores.view(batch_size, n_heads, d_k)
        Ks = K_scores.view(batch_size, n_heads, d_k)
        Vs = V_scores.view(batch_size, n_heads, d_v)
        contexts = []
        for i in range(n_heads):
            context= ScaledDotProductAttention()(Qs[:, i, :], Ks[:, i, :], Vs[:, i, :])
            contexts.append(context)
        context = torch.cat(contexts, dim=1)
        output = nn.Linear(n_heads*d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual)    # add & norm

class FeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNet, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        residual = inputs
        output = nn.ReLU()(self.w_1(inputs))
        output = self.w_2(output)
        return nn.LayerNorm(d_model)(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultilHeadAttention()
        self.enc_ff = FeedForwardNet(d_model, d_ff)
    def forward(self, enc_inputs):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = self.enc_ff(enc_outputs)
        return enc_outputs

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultilHeadAttention()
        self.dec_enc_attn = MultilHeadAttention()
        self.dec_ff = FeedForwardNet(d_model, d_ff)
    def forward(self, dec_inputs, enc_outputs):
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs)
        dec_outputs = self.dec_enc_attn(enc_outputs, enc_outputs, dec_outputs)
        dec_outputs = self.dec_ff(dec_outputs)
        return dec_outputs

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(poition_encoder(max_len, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    def forward(self, enc_inputs):
        enc_outputs = self.emb(enc_inputs)
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)
        return enc_outputs

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(poition_encoder(max_len, d_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs):
        dec_outputs = self.emb(dec_inputs)
        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs)
        return dec_outputs

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return F.log_softmax(dec_logits.view(-1, dec_logits.size(-1)), dim=-1)

model = Transformer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+raw_corpus[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+raw_corpus[2].split(), fontdict={'fontsize': 14})
    plt.show()

n_epochs = 100

for epoch in range(n_epochs):
    optimizer.zero_grad()
    total_loss = 0
    for sentence in corpus_idx:
        enc_inputs = Variable(torch.LongTensor(sentence))
        outputs_sr = outputs_shifted_right(sentence)
        for i in range(len(sentence)-1):
            dec_inputs, target = outputs_sr[i], outputs_sr[i+1]
            dec_inputs = Variable(dec_inputs)
            outputs = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, target.contiguous().view(-1))

            total_loss += loss
            loss.backward()
            optimizer.step()
    # print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_loss))

torch.save(model.state_dict(), 'models/transformer_v1.pkl')





