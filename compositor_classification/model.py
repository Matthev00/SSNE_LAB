import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.proj_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, state

    def forward(self, x, hidden):
        lstm_out, (hidden, state) = self.lstm(x, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        out = self.fc(lstm_out[:, -1, :])
        return out, (hidden, state)
