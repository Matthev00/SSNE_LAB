import torch
import torch.nn as nn


class LeNetPlus(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        hidden_size: int = 32,
        num_classes: int = 50,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_size, 2 * hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * hidden_size * 16 * 16, 8 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8 * hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class TinyVGG(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        hidden_size: int = 32,
        num_classes: int = 50,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_size, 2 * hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * hidden_size, 2 * hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(2 * hidden_size, 4 * hidden_size, 3, padding=1), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * hidden_size * 8 * 8, 8 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8 * hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.classifier(x)

        return x


class CNN_BN(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        hidden_size: int = 32,
        num_classes: int = 50,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_size, 2 * hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * hidden_size, 4 * hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * hidden_size * 8 * 8, 8 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8 * hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class TinyResNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        hidden_size: int = 32,
        num_classes: int = 50,
    ):
        super().__init__()

        self.prep = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )

        self.layer1 = ResidualBlock(hidden_size, 2 * hidden_size, stride=2)
        self.layer2 = ResidualBlock(2 * hidden_size, 4 * hidden_size, stride=2)
        self.layer3 = ResidualBlock(4 * hidden_size, 8 * hidden_size, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(8 * hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def main():
    model = CNN_BN(num_classes=50, hidden_size=8)
    print(model)
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    main()
