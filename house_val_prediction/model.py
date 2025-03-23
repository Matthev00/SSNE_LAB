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
        batch_norm: bool = True,
    ):
        super(HouseNet, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_dim)

        if batch_norm:
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
        else:
            self.hidden_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(int(hidden_dim / 2**i), int(hidden_dim / 2 ** (i + 1))),
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


class HybridNet(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, num_layers: int, dropout: float = 0.5, batch_norm: bool = True,):
        super(HybridNet, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_dim)
        if batch_norm:
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
        else:
            self.hidden_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(int(hidden_dim / 2**i), int(hidden_dim / 2 ** (i + 1))),
                        nn.LeakyReLU(),
                        nn.Dropout(dropout),
                    )
                    for i in range(num_layers)
                ]
            )

        self.regression_head = nn.Linear(int(hidden_dim / 2**num_layers), 1)

        self.classification_head = nn.Linear(int(hidden_dim / 2**num_layers), 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)

        regression_output = self.regression_head(x)
        classification_output = self.classification_head(x)

        return regression_output, classification_output
