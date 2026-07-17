import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import CIFAR10, MNIST, STL10


torch.manual_seed(0)


class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ["glioma", "meningioma", "pituitary", "notumor"]
        self.images = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_path = self.root_dir / class_name
            if not class_path.exists():
                raise ValueError(f"Class directory {class_name} not found in {root_dir}")

            for img_path in class_path.glob("*.jpg"):
                self.images.append(img_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("L")
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class FeatureWhitenTorch:
    def __init__(self, eps=1e-5):
        self.eps = eps
        self.mean = None
        self.scale = None

    @staticmethod
    def _per_sample_rms_normalize(x):
        x = x.float()
        x = x.reshape(x.shape[0], -1)
        x = x - x.mean(dim=1, keepdim=True)
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        return x / rms

    def fit(self, dataloader):
        n = 0
        sum_x = None
        sum_x2 = None

        for batch in dataloader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = self._per_sample_rms_normalize(x)

            if sum_x is None:
                sum_x = x.sum(dim=0, keepdim=True)
                sum_x2 = (x ** 2).sum(dim=0, keepdim=True)
            else:
                sum_x += x.sum(dim=0, keepdim=True)
                sum_x2 += (x ** 2).sum(dim=0, keepdim=True)

            n += x.shape[0]

        self.mean = sum_x / n
        var = (sum_x2 / n) - self.mean ** 2
        self.scale = 1.0 / torch.sqrt(var + self.eps)

        return self

    def transform(self, x):
        original_shape = x.shape
        x = self._per_sample_rms_normalize(x)
        x = (x - self.mean.to(x.device)) * self.scale.to(x.device)
        return x.reshape(original_shape)


def load_dataset(dataset, root):
    if dataset == "cifar10":
        img_size = 32
        transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
        train_set = CIFAR10(os.path.join(root, dataset), train=True, download=True, transform=transform)
        test_set = CIFAR10(os.path.join(root, dataset), train=False, download=True, transform=transform)

    elif dataset == "stl10":
        img_size = 96
        transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
        train_set = STL10(os.path.join(root, dataset), split="train", download=True, transform=transform)
        test_set = STL10(os.path.join(root, dataset), split="test", download=True, transform=transform)

    elif dataset == "mnist":
        img_size = 28
        transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
        train_set = MNIST(os.path.join(root, dataset), train=True, download=True, transform=transform)
        test_set = MNIST(os.path.join(root, dataset), train=False, download=True, transform=transform)

    elif dataset == "brain_tumor":
        img_size = 256
        transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
        train_set = BrainTumorDataset(os.path.join(root, "brain_tumor", "train"), transform=transform)
        test_set = BrainTumorDataset(os.path.join(root, "brain_tumor", "test"), transform=transform)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_set, test_set


def image_for_plot(x):
    is_grayscale = x.shape[0] == 1

    if is_grayscale:
        x = x.squeeze(0)
    else:
        x = x.permute(1, 2, 0)

    x = x.detach().cpu().numpy()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    return x, is_grayscale


def save_pre_post_whitening_plot(original, whitened, out_path, num_samples=5):
    num_samples = min(num_samples, original.shape[0])

    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))
    axes = np.asarray(axes).reshape(2, num_samples)

    for i in range(num_samples):
        orig, is_grayscale = image_for_plot(original[i])
        whit, _ = image_for_plot(whitened[i])

        axes[0, i].imshow(orig, cmap="gray" if is_grayscale else None)
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

        axes[1, i].imshow(whit, cmap="gray" if is_grayscale else None)
        axes[1, i].axis("off")
        axes[1, i].set_title("Whitened")

    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="brain_tumor", choices=["cifar10", "stl10", "mnist", "brain_tumor"])
    parser.add_argument("--root", default="datasets")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--out-dir", default="graphics")
    args = parser.parse_args()

    train_set, test_set = load_dataset(args.dataset, args.root)

    fit_loader = DataLoader(
        ConcatDataset([train_set, test_set]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    whitener = FeatureWhitenTorch(eps=args.eps)
    whitener.fit(fit_loader)

    plot_loader = DataLoader(
        train_set,
        batch_size=max(args.batch_size, args.num_samples),
        shuffle=False,
        num_workers=args.num_workers,
    )

    original_batch, _ = next(iter(plot_loader))

    with torch.no_grad():
        whitened_batch = whitener.transform(original_batch)

    print(f"Original batch shape: {original_batch.shape}")
    print(f"Whitened batch shape: {whitened_batch.shape}")

    out_path = Path(args.out_dir) / f"{args.dataset}_pre_post_whitening"
    save_pre_post_whitening_plot(
        original_batch,
        whitened_batch,
        out_path,
        num_samples=args.num_samples,
    )

    print(f"Saved: {out_path.with_suffix('.png')}")
    print(f"Saved: {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()