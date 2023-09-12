# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from gaussian_blur import GaussianBlur
from torchvision import datasets, transforms
from torchvision.transforms import transforms

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


# %%
class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * size)),
                transforms.ToTensor(),
            ]
        )
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {
            "cifar10": lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(32), n_views),
                download=True,
            ),
            "stl10": lambda: datasets.STL10(
                self.root_folder,
                split="unlabeled",
                transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(96), n_views),
                download=True,
            ),
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise ValueError(f"Invalid dataset {name}.")
        else:
            return dataset_fn()


# %%
cifar10_dataset = ContrastiveLearningDataset(root_folder="../data/cifar10").get_dataset("cifar10", 2)
print(cifar10_dataset.data.shape)  # (50000, 32, 32, 3)
loader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=64, shuffle=False, num_workers=2)
for i, data in enumerate(loader):
    print(len(data))  # 2
    print(len(data[0]))  # 2
    print(data[0][0].shape)  # torch.Size([64, 3, 32, 32])
    # show the first 24 images
    fig, axs = plt.subplots(4, 6)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(data[0][0][i].permute(1, 2, 0))
        ax.axis("off")

    plt.show()

    fig, axs = plt.subplots(4, 6)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(data[0][1][i].permute(1, 2, 0))
        ax.axis("off")

    plt.show()
    break

# show the first 24 images
fig, axs = plt.subplots(4, 6)
for i, ax in enumerate(axs.flatten()):
    ax.imshow(cifar10_dataset.data[i])
    ax.axis("off")

plt.show()


# %%
