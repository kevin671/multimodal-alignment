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
    def __init__(self, model, loader, device="cuda:0"):
        self.args = {
            "device": device,
            "batch_size": BATCH_SIZE,
            "n_views": 2,
            "temperature": 0.07,
            "epochs": 1000,
            "learning_rate": 3e-4,
            "fp16_precision": True,
            "weight_decay": 1e-4,
        }
        self.device = device
        self.model = model.to(self.device)
        self.loader = loader
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(loader), eta_min=0, last_epoch=-1
        )

    def info_nce_loss(self, features):
        labels = torch.cat(
            [torch.arange(self.args["batch_size"]) for i in range(self.args["n_views"])], dim=0
        )  # (2 * batch_size, )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (2 * batch_size, 2 * batch_size)
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.args["temperature"]
        return logits, labels

    def train(self):
        scaler = GradScaler(enabled=self.args["fp16_precision"])

        n_iter = 0
        for epoch in range(self.args["epochs"]):
            for images, _ in tqdm(self.loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.device)

                with torch.cuda.amp.autocast(enabled=self.args["fp16_precision"]):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % 20 == 0:
                    print(f"Iteration: {n_iter}, Loss: {loss.item()}")
                    """
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    print(
                        f"Epoch: {epoch}, Loss: {loss.item()}, Top1 accuracy: {top1.item()}, Top5 accuracy: {top5.item()}"
                    )
                    """
                n_iter += 1


# %%
import sys

sys.path.append("../")
from datasets.contrastive_learning_dataset import ContrastiveLearningDataset

# dataset_names = ["cifar10", "stl10"]
# dataset = ContrastiveLearningDataset(root_folder="../data/cifar10").get_dataset("cifar10", 2)
dataset = ContrastiveLearningDataset(root_folder="../data/stl10").get_dataset("stl10", 2)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# %%
import torchvision

model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Identity()
# %%
simclr = SimCLR(model, loader, "cuda:0")
simclr.train()

# %%
