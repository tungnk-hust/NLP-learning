import torch
import torch.nn as nn
import torch.nn.functional as F

class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(ElmanRNN, self).__init__()
        self.rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
        self.batch_firts = batch_first
        self.hidden_size = hidden_size

    def _init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))

    def forward(self, x_in, init_hidden=None):
        if self.batch_firts:
            batch_size, seq_size, feat_size = x_in.size()
            x_in = x_in.permute(1, 0, 2)
        else:
            seq_size, batch_size, feat_size = x_in.size()

        hiddens = []

        if init_hidden is None:
            init_hidden = self._init_hidden(batch_size)

        hidden_t = init_hidden

        for t in range(seq_size):
            hidden_t = self.rnn_cell(x_in[t], hidden_t)
            hiddens.append(hidden_t)

        hiddens = torch.stack(hiddens)

        if self.batch_firts:
            hiddens = hiddens.permute(1, 0, 2)

        return hiddens

def column_gather(y_out, x_lengths):
    x_lengths = x_lengths.long().detach().cpu().numpy() - 1
    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(y_out[batch_index, column_index])

    return torch.stack(out)

class SurnameClassifier(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_classes, rnn_hidden_size, batch_first=True, padding_idx=0):
        super(SurnameClassifier, self).__init__()
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_size, padding_idx=padding_idx)
        self.rnn = ElmanRNN(input_size=embedding_size, hidden_size=rnn_hidden_size, batch_first=batch_first)

        self.fc1 = nn.Linear(in_features=rnn_hidden_size, out_features=rnn_hidden_size)
        self.fc2 = nn.Linear(in_features=rnn_hidden_size, out_features=num_classes)

    def forward(self, x_in, x_lengths=None, apply_softmax=False):  # dim(x_in) = (bs, ss, fs)
        x_embedded = self.emb(x_in)
        out = self.rnn(x_embedded)

        if x_lengths is not None:
            out = column_gather(out, x_lengths)
        else:
            out = out[:, -1, :]

        out = F.dropout(out, p=0.3)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.3)
        out = self.fc2(out)

        if apply_softmax:
            out = F.softmax(out, dim=1)

        return out



