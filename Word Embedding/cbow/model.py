import torch
import torch.nn as nn
import torch.nn.functional as F

class CbowClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx=0):
        super(CbowClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size, padding_idx=padding_idx)
        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, x, apply_softmax=False):
        embed_sum = F.dropout(self.embedding(x).sum(dim=1), 0.3)
        out = self.fc(embed_sum)
        if apply_softmax:
            out = F.softmax(out, dim=1)
        return out
