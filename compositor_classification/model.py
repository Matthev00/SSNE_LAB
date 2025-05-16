import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        bidirectional=False,
        dropout=0.5,
    ):
        super(LSTMClassifier, self).__init__()
        self.proj_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for LSTMClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_size).
        """
        _, (hidden, _) = self.lstm(x)

        if self.bidirectional:
            hidden = hidden.view(self.num_layers, 2, x.size(0), self.hidden_size)
            hidden = torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=1)
        else:
            hidden = hidden[-1]

        out = self.fc(hidden)
        return out
