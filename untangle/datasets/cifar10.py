"""Wrapper dataset for CIFAR-10 to match repo dataset API."""
from pathlib import Path
from typing import Callable

from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset


class HardCIFAR10(Dataset):
    """Light wrapper around torchvision CIFAR10 to match project's API.

    Args:
        root: Root directory for datasets.
        split: 'train' or 'test' or 'val'
        transform: Transform applied to images.
        target_transform: Transform applied to labels.
        download: Whether to download the dataset.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        train = split == "train"
        # Map 'val' to test split (no separate validation in CIFAR-10)
        if split == "val":
            train = False

        self.ds = CIFAR10(root=str(self.root), train=train, transform=None, target_transform=None, download=download)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int):
        img, target = self.ds[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def set_ood(self) -> None:
        # No-op for CIFAR-10 wrapper, kept for API compatibility
        return
