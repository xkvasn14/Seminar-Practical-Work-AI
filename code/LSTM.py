import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, fc_dim: int = 32, output_dim: int = 11,
                 dropout: float = 0.3):

        """
            output_dim must equal the number of clusters used in k-means algorithm
        """
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm1(x)  # out: (batch, seq_len, hidden_dim*2)
        out = self.dropout(out)

        out, _ = self.lstm2(out)  # out: (batch, seq_len, hidden_dim)
        out = self.dropout(out)

        out = out[:, -1, :]  # out: (batch, hidden_dim)

        out = self.fc1(out)  # (batch, fc_dim)
        out = self.bn(out)
        out = torch.relu(out)

        logits = self.fc2(out)  # (batch, output_dim)
        probs = self.softmax(logits)
        return probs
