"""Model definitions for all four tasks."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


def down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 4, stride=2, padding=1), nn.BatchNorm2d(cout),
                         nn.LeakyReLU(0.2, inplace=True), ResBlock(cout))


def up(cin, cout, dropout=0.0):
    layers = [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
              nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True)]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    layers.append(ResBlock(cout))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Task 1 / Task 2 specialists: convolutional autoencoder with a spatial bottleneck
# --------------------------------------------------------------------------- #
class ConvAutoencoder(nn.Module):
    """Encoder 128->64->32->16->8 (channels base*[1,2,4,8]) -> 1x1 conv to a
    `latent_ch` x 8 x 8 code (the bottleneck) -> mirrored decoder.

    The latent has latent_ch*64 values versus 3*128*128 = 49,152 input values,
    e.g. 64 channels -> 4,096 dims (12x compression). There are NO skip
    connections, so all information must pass through the bottleneck.
    """

    def __init__(self, base: int = 32, latent_ch: int = 64, dropout: float = 0.1):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c[0] // 2, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            down(c[0] // 2, c[0]), down(c[0], c[1]), down(c[1], c[2]), down(c[2], c[3]),
            nn.Conv2d(c[3], latent_ch, 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, c[3], 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(dropout),
            up(c[3], c[2], dropout), up(c[2], c[1], dropout), up(c[1], c[0]), up(c[0], c[0] // 2),
            nn.Conv2d(c[0] // 2, 3, 3, padding=1), nn.Sigmoid())
        self.latent_dim = latent_ch * 8 * 8

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# --------------------------------------------------------------------------- #
# Task 2 classifier / Task 3 gate
# --------------------------------------------------------------------------- #
class CorruptionClassifier(nn.Module):
    def __init__(self, channels: Sequence[int] = (32, 64, 128, 256), dropout: float = 0.3, n_classes: int = 4):
        super().__init__()
        layers, cin = [], 3
        for i, ch in enumerate(channels):
            layers += [nn.Conv2d(cin, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                       nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            cin = ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin * 2, n_classes))

    def forward(self, x):
        f = self.features(x)
        # concat avg + max pooling: max pooling helps detect sparse impulse noise / masks
        f = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        return self.head(f)


# --------------------------------------------------------------------------- #
# Task 3: soft mixture of experts
# --------------------------------------------------------------------------- #
class SoftMoE(nn.Module):
    """x_hat = w0*x + w1*A_salt(x) + w2*A_blur(x) + w3*A_occ(x),  w = softmax(G(x)/tau)."""

    def __init__(self, gate: CorruptionClassifier, experts: Sequence[nn.Module], tau: float = 1.0):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, x):
        logits = self.gate(x)
        w = torch.softmax(logits / self.tau, dim=1)
        outs = torch.stack([x] + [e(x) for e in self.experts], 1)  # B,4,3,H,W
        xhat = (w[:, :, None, None, None] * outs).sum(1)
        return xhat, w, logits


# --------------------------------------------------------------------------- #
# Task 4: style-conditioned U-Net generator and PatchGAN discriminator
# --------------------------------------------------------------------------- #
class UNetGenerator(nn.Module):
    """pix2pix-style U-Net (128 -> 1x1 after 7 downsamplings) with a learned style
    embedding injected (a) as extra input planes and (b) at the bottleneck."""

    def __init__(self, base: int = 48, emb_dim: int = 16, n_styles: int = 3, dropout: float = 0.5, out_ch: int = 1):
        super().__init__()
        self.emb = nn.Embedding(n_styles, emb_dim)
        c = [base, base * 2, base * 4, base * 8, base * 8, base * 8, base * 8]
        self.downs = nn.ModuleList()
        cin = 3 + emb_dim
        for i, ch in enumerate(c):
            layers = [nn.Conv2d(cin, ch, 4, 2, 1, bias=i == 0)]
            if 0 < i < len(c) - 1:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.downs.append(nn.Sequential(*layers))
            cin = ch
        self.ups = nn.ModuleList()
        rev = c[::-1]  # bottleneck first
        cin = rev[0] + emb_dim
        for i in range(len(c) - 1):
            cout = rev[i + 1]
            layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            if i < 3 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.ups.append(nn.Sequential(*layers))
            cin = cout * 2
        self.final = nn.Sequential(nn.ConvTranspose2d(cin, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x, style):
        e = self.emb(style)  # B,E
        h = torch.cat([x, e[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])], 1)
        skips = []
        for d in self.downs:
            h = d(h)
            skips.append(h)
        h = torch.cat([h, e[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])], 1)
        skips = skips[:-1][::-1]
        for u, s in zip(self.ups, skips):
            h = torch.cat([u(h), s], 1)
        return self.final(h)


class PatchDiscriminator(nn.Module):
    """70x70-style PatchGAN on (photo, sketch, style planes); outputs a 14x14 logit map at 128px."""

    def __init__(self, base: int = 64, emb_dim: int = 16, n_styles: int = 3, sketch_ch: int = 1):
        super().__init__()
        self.emb = nn.Embedding(n_styles, emb_dim)
        cin = 3 + sketch_ch + emb_dim

        def blk(i, o, s, norm=True):
            L = [nn.Conv2d(i, o, 4, s, 1, bias=not norm)]
            if norm:
                L.append(nn.BatchNorm2d(o))
            L.append(nn.LeakyReLU(0.2, inplace=True))
            return L

        self.net = nn.Sequential(*blk(cin, base, 2, False), *blk(base, base * 2, 2), *blk(base * 2, base * 4, 2),
                                 *blk(base * 4, base * 8, 1), nn.Conv2d(base * 8, 1, 4, 1, 1))

    def forward(self, photo, sketch, style):
        e = self.emb(style)[:, :, None, None].expand(-1, -1, photo.shape[2], photo.shape[3])
        return self.net(torch.cat([photo, sketch, e], 1))


def init_gan_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
