import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPSurNameClassifier(nn.Module):
    def __init__(self, n_input, n_output, n_layers=2, n_hidden=100, hidden_activation=nn.ReLU):
        super(MLPSurNameClassifier, self).__init__()
        self.layers = nn.ModuleList()
        input_size = n_input
        output_size = n_hidden
        for i in range(n_layers):
            self.layers.append(nn.Linear(input_size, output_size))
            self.layers.append(hidden_activation())
            input_size = output_size

        self.final_fc = nn.Linear(input_size, n_output)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.final_fc(x)
        return out

class CNNSurNameClassifier(nn.Module):
    def __init__(self, n_input, n_output, n_channel):
        super(CNNSurNameClassifier, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=n_input, out_channels=n_channel, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=n_channel, out_channels=n_channel, kernel_size=3, stride=2),
            nn.ELU()
        )
        self.fc = nn.Linear(n_channel, n_output)

    def forward(self, x):
        feature = self.convnet(x).squeeze(dim=2)
        out = self.fc(feature)
        return out

