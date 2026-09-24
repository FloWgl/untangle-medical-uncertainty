"""Smoke test: instantiate dataset, model, wrapper, and run one forward batch."""
import torch
from pathlib import Path
import sys
import os

# Ensure repo root is on sys.path so `untangle` package can be imported when running
# this script directly from tools/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchvision import transforms

from untangle.datasets.pathmnist import PathMNIST
from untangle.models.resnet_cifar_preact import resnet_c_preact_26
from torch.utils.data import DataLoader


def main():
    data_root = Path("c:/Users/flori/OneDrive/Desktop/BA/untangle/tmp_pathmnist_export")
    # Create a simple transform for PathMNIST exported ImageFolder
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])),
    ])

    dataset = PathMNIST(root=str(data_root), split="train", transform=tf, target_transform=None)

    print("Dataset len:", len(dataset))

    model = resnet_c_preact_26(num_classes=9, in_chans=3)
    print("Model ready.")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    print("Starting one forward pass...")
    it = iter(loader)
    x, y = next(it)
    with torch.no_grad():
        out = model(x)
    print("Forward pass output shape:", out.shape)


if __name__ == "__main__":
    main()
