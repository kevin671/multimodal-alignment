import argparse
import importlib
import os

import torch
import torchvision
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from datasets.contrastive_learning_dataset import ContrastiveLearningDataset


def main(args, backbone):
    model_name = {
        "barlow_twins": "BarlowTwins",
        "vicreg": "VICReg",
        "simclr": "SimCLR",
    }
    model = getattr(importlib.import_module("ssl_torch." + args.ssl_method), model_name[args.ssl_method])(
        backbone, args
    )
    model = model.to(args.device)
    dataset = ContrastiveLearningDataset(root_folder=os.path.join(args.data_dir, args.dataset_name)).get_dataset(
        args.dataset_name, 2
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    for epoch in range(10):
        for i, (images, _) in tqdm(enumerate(loader)):
            x1, x2 = images
            x1, x2 = x1.to(args.device), x2.to(args.device)  #  torch.Size([bs, 3, 32, 32])
            # with autocast(enabled=args.fp16_precision):
            # features = model(x1, x2)
            # loss = model.compute_loss(features)
            loss = model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 20 == 0:
                print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl_method", type=str, default="simclr", choices=["barlow_twins", "vicreg", "simclr"])
    parser.add_argument("--dataset_name", type=str, default="cifar10", choices=["cifar10", "stl10"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", default=0.07, type=float, help="softmax temperature (default: 0.07)")
    parser.add_argument("--out_dim", default=128, type=int, help="feature dimension (default: 128)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    backbone = torchvision.models.resnet50(pretrained=False)

    main(args, backbone)
