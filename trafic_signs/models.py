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

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3 + embedding_dim, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.adv_head = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)  
        self.cls_head = nn.Conv2d(ndf * 4, num_classes, 4, 1, 0, bias=False)

    def forward(self, img: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_emb = self.label_emb(y).unsqueeze(-1).unsqueeze(-1)
        y_emb = y_emb.expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, y_emb], dim=1)

        features = self.feature_extractor(x)
        real_fake_logits = self.adv_head(features).view(-1, 1)
        class_logits = self.cls_head(features).view(-1, self.label_emb.num_embeddings)
        return real_fake_logits, class_logits
