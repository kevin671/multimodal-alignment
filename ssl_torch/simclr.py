# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from tqdm import tqdm


class SimCLR(nn.Module):
    def __init__(self, backbone, args):
        super(SimCLR, self).__init__()
        self.args = args
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        dim_mlp = self.backbone.fc.in_features

        self.projector = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), args.out_dim)

        # normalization layer for the representations z1 and z2
        # self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def info_nce_loss(self, features):
        pass

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # pairwise similarity matrix
        z = torch.cat((z1, z2))
        z = F.normalize(z, dim=1)
        similarity_matrix = torch.matmul(z, z.T)

        # softmax_cross_entropy
