import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Args:
        nz (int): latent vector size.
        ngf (int): Number of filters in first layer of Generator.
        num_classes (int): Number of classes.
    """

    def __init__(self, nz: int, ngf: int, num_classes: int):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.nz = nz
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz + num_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_embed = self.label_emb(labels)
        x = torch.cat([noise, label_embed], dim=1)
        x = x.unsqueeze(2).unsqueeze(3)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, ndf: int, num_classes: int):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 32 * 32)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3 + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.real_fake_head = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
        self.class_head = nn.Conv2d(ndf * 4, num_classes, 4, 1, 0, bias=False)

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        label_map = self.label_emb(labels).view(-1, 1, 32, 32)
        x = torch.cat([img, label_map], dim=1)
        features = self.feature_extractor(x)
        real_fake = self.real_fake_head(features).view(-1)
        class_logits = self.class_head(features).squeeze()
        return real_fake, class_logits

