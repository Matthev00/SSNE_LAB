import torch
import torch.nn as nn


class HouseNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.5,
    ):
        super(HouseNet, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_dim)

        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(hidden_dim / 2**i), int(hidden_dim / 2 ** (i + 1))),
                    nn.BatchNorm1d(int(hidden_dim / 2 ** (i + 1))),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                )
                for i in range(num_layers)
            ]
        )

        self.output_layer = nn.Linear(int(hidden_dim / 2**num_layers), output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
