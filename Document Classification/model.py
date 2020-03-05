import torch
import torch.nn as nn
import torch.nn.functional as F

class NewsClassifier(nn.Module):
    def __init__(self, emb_size, num_emb, num_channels, hidden_dim, num_classes, dropout_p=0.3,
                 pretrained_embeddings=None, padding_idx=0):
        super(NewsClassifier, self).__init__()
        if pretrained_embeddings is None:
            self.emb = nn.Embedding(embedding_dim=emb_size, num_embeddings=num_emb, padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim=emb_size, num_embeddings=num_emb,
                                    padding_idx=padding_idx, _weight=pretrained_embeddings)

        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=emb_size, out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3),
            nn.ELU()
        )
        self.dropout_p = dropout_p
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_in, apply_softmax=False):
        x_emb = self.emb(x_in).permute(0, 2, 1)
        features = self.convnet(x_emb)

        remaining_size = features.size(dim=2)
        features = F.avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p=self.dropout_p)

        out = F.relu(F.dropout(self.fc1(features), p=self.dropout_p))
        out = self.fc2(out)

        if apply_softmax:
            out = F.softmax(out, dim=1)
        return out