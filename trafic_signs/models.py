import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Args:
        nz (int): latent vector size.
        ngf (int): Number of filters in first layer of Generator.
        num_classes (int): Number of classes.
        embedding_dim (int): Dimension of the label embedding.
    """

    def __init__(self, nz: int, ngf: int, num_classes: int, embedding_dim: int = 32):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.nz = nz
        self.embedding_dim = embedding_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz + embedding_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_embed = self.label_emb(labels)
        x = torch.cat([noise, label_embed], dim=1)
        x = x.view(-1, self.nz + self.embedding_dim, 1, 1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, ndf: int, num_classes: int, embedding_dim: int = 32):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)

        self.model = nn.Sequential(
            nn.Conv2d(3 + embedding_dim, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        label_embeddings = self.label_emb(labels).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, label_embeddings], dim=1)
        return self.model(x).view(-1, 1)
