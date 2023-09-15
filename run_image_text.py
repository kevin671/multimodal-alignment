import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import clip

Image.MAX_IMAGE_PIXELS = 1000000000


class ConceptualDataset(Dataset):
    def __init__(self, data_dir: str, suffix="train", preprocess=None):
        super().__init__()
        self.data_dir = data_dir
        self.suffix = suffix
        self.preprocess = preprocess
        self.image_dir = os.path.join(data_dir, suffix)
        self.captions = self.load_captions()
        self.image_ids = list(self.captions.keys())
        self.image_ids.sort()
        self.image_id2idx = {image_id: idx for idx, image_id in enumerate(self.image_ids)}
        self.idx2image_id = {idx: image_id for idx, image_id in enumerate(self.image_ids)}

    def load_captions(self):
        captions = {}
        if self.suffix == "train":
            tsv_path = f"{self.data_dir}/Train_GCC-training.tsv"
        else:
            tsv_path = f"{self.data_dir}/Validation_GCC-1.1.0-Validation.tsv"

        with open(tsv_path, "r") as f:
            read_tsv = csv.reader(f, delimiter="\t")
            for i, row in enumerate(read_tsv):
                caption, _ = row
                image_id = f"{i:08d}"
                captions[image_id] = caption
        return captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_id = self.idx2image_id[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = self.load_image(image_path)
        caption = self.captions[image_id]
        return image, caption

    def load_image(self, image_path: str):
        if not os.path.exists(image_path):
            image = torch.zeros(3, 224, 224)

        try:
            image = Image.open(image_path)
            image = self.preprocess(image)
        except:  # PIL.UnidentifiedImageError: cannot identify image file '../data/conceptual/train/01424292.jpg'
            image = torch.zeros(3, 224, 224)

        return image


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def train(model, dataloader, device, batch_size):
    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
    )  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    EPOCH = 10
    for epoch in range(EPOCH):
        for i, (images, captions) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            texts = clip.tokenize(captions).to(device)

            logits_per_image, logits_per_text = model(
                images, texts
            )  # (batch_size, batch_size),  logits_per_image == logits_per_text.T

            """
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            i_e = image_features / image_features.norm(dim=-1, keepdim=True)  # i_e.shape = (batch_size, 512)
            t_e = text_features / text_features.norm(dim=-1, keepdim=True)  # t_e.shape = (batch_size, 512)
            logits = (100.0 * i_e @ t_e.T).float()  # logits.shape = (batch_size, batch_size)
            assert torch.allclose(logits, logits_per_image)
            """

            labels = torch.arange(batch_size).long().to(device)
            loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Iter {i+1} loss: {loss.item()}")

        print(f"Epoch {epoch} loss: {loss.item()}")


def main():
    BATCH_SIZE = 256
    device = "cuda:0"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    dataset = ConceptualDataset("../data/conceptual-3m", suffix="train", preprocess=preprocess)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    train(model, train_dataloader, device, BATCH_SIZE)


if __name__ == "__main__":
    main()
