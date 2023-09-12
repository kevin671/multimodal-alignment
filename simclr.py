# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from tqdm import tqdm

BATCH_SIZE = 256


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class SimCLR(object):
    def __init__(self, model, loader, device):
        self.model = model
        self.device = device
        self.criteria = nn.CrossEntropyLoss().to(device)
        self.args = {
            "batch_size": BATCH_SIZE,
            "epochs": 1000,
            "learning_rate": 1e-5,
            "fp16_precision": True,
            "weight_decay": 1e-6,
        }
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(loader), eta_min=0, last_epoch=-1
        )

    def info_nce(self, features):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        """
        WIP
        """

    def train(self):
        scaler = GradScaler(enabled=self.args["fp16_precision"])

        n_iter = 0
        for epoch in range(self.args["epochs"]):
            for images, _ in tqdm(self.loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.args["fp16_precision"]):
                    features = self.model(images)
                    logits, labels = self.info_nce(features)
                    loss = self.criteria(logits, torch.arange(logits.shape[0]).to(self.device))
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

            if n_iter % 100 == 0:
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                print(
                    f"Epoch: {epoch}, Loss: {loss.item()}, Top1 accuracy: {top1.item()}, Top5 accuracy: {top5.item()}"
                )
            n_iter += 1


# %%
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset

dataset = ContrastiveLearningDataset(root_folder="../data/cifar10").get_dataset("cifar10", 2)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# %%
from timm import create_model

model = create_model("resnet50", pretrained=False, num_classes=0)
# %%
simclr = SimCLR(model, loader, "cuda")
simclr.train()
