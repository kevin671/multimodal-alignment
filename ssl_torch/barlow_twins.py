from typing import Optional

import torch
import torch.nn as nn
import torchvision


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, backbone, args):
        super(BarlowTwins, self).__init__()
        self.args = args
        self.backbone = backbone
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048, 8192, 8192, 8192]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))  # N x 2048
        z2 = self.projector(self.backbone(y2))  # N x 2048

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)  # 2048 x 2048

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        lmbda = 5e-3
        loss = on_diag + lmbda * off_diag
        return loss
