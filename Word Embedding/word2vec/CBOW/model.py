import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

class CBOWModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_size=100):
        super(CBOWModel, self).__init__()
        self.U_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.V_embedding = nn.Linear(embedding_size, vocabulary_size)

    def forward(self, inputs):
        print(inputs)
        lookup_embs = self.U_embedding(inputs)
        sums = lookup_embs.sum(dim=0)
        out = self.V_embedding(sums)
        out = F.log_softmax(out)
        return out
