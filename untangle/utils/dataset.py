"""Dataset utilities."""

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from untangle.datasets import (
    DATASET_NAME_TO_PATH,
    MHIST,
    PathMNIST,
    PathMNISTC,
    SoftDataset,
    Subset,
)
from untangle.datasets.cifar10 import HardCIFAR10
from untangle.utils.transform import create_transform, hard_target_transform


class LabelNoiseDataset(Dataset):
    """Dataset wrapper that applies deterministic label noise.

    The wrapper changes labels only at selected indices. Images and transforms stay
    untouched, so validation and test sets can remain clean while training receives
    a controlled aleatoric-label-noise intervention.
    """

    def __init__(
        self,
        dataset: Dataset,
        fraction: float,
        num_classes: int,
        seed: int,
        mode: str = "symmetric",
    ) -> None:
        if not 0.0 <= fraction < 1.0:
            msg = "--label-noise-fraction must be in [0, 1)."
            raise ValueError(msg)
        if num_classes < 2:
            msg = "--label-noise-num-classes must be at least 2."
            raise ValueError(msg)
        if mode != "symmetric":
            msg = f"Unsupported label-noise mode: {mode}"
            raise ValueError(msg)

        self.dataset = dataset
        self.fraction = fraction
        self.num_classes = num_classes
        self.seed = seed
        self.mode = mode
        self.label_noise_indices: set[int] = set()
        self.noisy_labels: dict[int, int] = {}
        self.clean_labels: dict[int, int] = {}

        num_samples = len(dataset)
        num_noisy = int(round(fraction * num_samples))
        if num_noisy == 0:
            return

        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(num_samples, generator=generator)[:num_noisy].tolist()
        self.label_noise_indices = set(int(index) for index in selected)

        for index in selected:
            target = self._get_clean_label(int(index))
            # Draw uniformly from all wrong classes.
            offset = int(torch.randint(1, num_classes, (1,), generator=generator).item())
            noisy_target = (target + offset) % num_classes
            self.clean_labels[int(index)] = target
            self.noisy_labels[int(index)] = noisy_target

    def _get_clean_label(self, index: int) -> int:
        parent = self.dataset
        if hasattr(parent, "indices") and hasattr(parent, "dataset"):
            source_index = int(parent.indices[index])
            source = parent.dataset
            if hasattr(source, "targets"):
                return self._target_to_int(source.targets[source_index])
            if hasattr(source, "_labels"):
                return self._target_to_int(source._labels[source_index])

        if hasattr(parent, "targets"):
            return self._target_to_int(parent.targets[index])
        if hasattr(parent, "_labels"):
            return self._target_to_int(parent._labels[index])

        return self._target_to_int(parent[index][1])

    @staticmethod
    def _target_to_int(target) -> int:
        return int(torch.as_tensor(target).reshape(-1)[-1].item())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, target = self.dataset[idx]
        noisy_target = self.noisy_labels.get(int(idx))
        if noisy_target is not None:
            target = noisy_target
        return image, target

    def __getitems__(self, indices: Sequence[int]) -> list:
        return [self[index] for index in indices]

    def __getattr__(self, name: str):
        if name in {
            "dataset",
            "fraction",
            "num_classes",
            "seed",
            "mode",
            "label_noise_indices",
            "noisy_labels",
            "clean_labels",
        }:
            return object.__getattribute__(self, name)
        return getattr(self.dataset, name)


def _normalize_dataset_name(name: str) -> str:
    """Maps legacy PathMNIST names to the hard-label namespace."""
    name = name.lower()
    if name in {"pathmnist", "pathmnist-c"}:
        return f"hard/{name}"

    return name


def create_dataset(
    name: str,
    root: str,
    label_root: str,
    split: str,
    download: bool,
    seed: int,
    subset: float,
    input_size: int,
    padding: int,
    is_training_dataset: bool,
    use_prefetcher: bool,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    hflip: float,
    color_jitter: float,
    interpolation: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    crop_pct: float,
    ood_transform_type: str | None,
    severity: int,
    convert_soft_labels_to_hard: bool,
    label_noise_fraction: float = 0.0,
    label_noise_num_classes: int | None = None,
    label_noise_seed: int | None = None,
    label_noise_mode: str = "symmetric",
)-> SoftDataset | Subset | PathMNIST | PathMNISTC | MHIST:
    """Creates and returns a dataset based on the given parameters.

    This function creates a dataset with the specified configuration, applying
    transformations and subset selection as needed.

    Args:
        name: Name of the dataset.
        root: Root directory of the dataset.
        label_root: Root directory for labels (used for soft datasets).
        split: Data split to use ('train' or 'val').
        download: Whether to download the dataset if not present.
        seed: Random seed for subset selection.
        subset: Fraction of the dataset to use (1.0 means use all data).
        input_size: Size of the input images.
        padding: Padding to apply to the images.
        is_training_dataset: Whether this is a training dataset.
        use_prefetcher: Whether to use a prefetcher.
        scale: Scale range for transforms.
        ratio: Aspect ratio range for transforms.
        hflip: Horizontal flip probability.
        color_jitter: Color jitter factor.
        interpolation: Interpolation method for resizing.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
        crop_pct: Crop percentage for transforms.
        ood_transform_type: Type of out-of-distribution transform to apply.
        severity: Severity of the OOD transform.
        convert_soft_labels_to_hard: Whether to convert soft labels to hard labels.
        label_noise_fraction: Fraction of labels to flip after subset selection.
        label_noise_num_classes: Number of classes used for wrong-label sampling.
        label_noise_seed: Seed for deterministic label flips.
        label_noise_mode: Label-noise mode. Currently only symmetric noise is supported.

    Returns:
        The created dataset.

    Raises:
        ValueError: If an unsupported dataset or configuration is specified.
    """
    dataset_name = _normalize_dataset_name(name)
    pathmnist_dataset = "pathmnist" in dataset_name
    transform_ood_transform_type = None if pathmnist_dataset else ood_transform_type

    transform = create_transform(
        input_size=input_size,
        dataset_name=dataset_name,
        padding=padding,
        is_training_dataset=is_training_dataset,
        use_prefetcher=use_prefetcher,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        color_jitter=color_jitter,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        ood_transform_type=transform_ood_transform_type,
        severity=severity,
    )

    target_transform = None
    if convert_soft_labels_to_hard:
        target_transform = hard_target_transform

    name = dataset_name
    if name.startswith("hard/"):
        name = name.split("/", 2)[-1]

        if name == "pathmnist":
            dataset = PathMNIST(
                root=root,
                split=split,
                transform=transform,
                target_transform=target_transform,
                download=download,
                corruption_type=ood_transform_type,
                corruption_severity=severity,
            )
        elif name == "pathmnist-c":
            dataset = PathMNISTC(
                root=root,
                split=split,
                transform=transform,
                target_transform=target_transform,
                download=download,
                corruption_type=ood_transform_type,
                corruption_severity=severity,
            )
        elif name == "cifar10":
            dataset = HardCIFAR10(
                root=root,
                split=split,
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
        elif name == "mhist":
            dataset = MHIST(
                root=root,
                split=split,
                transform=transform,
                target_transform=target_transform,
                download=download,
                soft_labels=False,
            )
        else:
            msg = "Unsupported dataset"
            raise ValueError(msg)
    elif name.startswith("soft/"):
        name = name.split("/", 2)[-1]

        if name == "mhist":
            dataset = MHIST(
                root=label_root,
                split=split,
                transform=transform,
                target_transform=target_transform,
                download=download,
                soft_labels=True,
            )
        elif name in DATASET_NAME_TO_PATH:
            dataset = SoftDataset(
                name=name,
                root=root,
                split=split,
                transform=transform,
                target_transform=target_transform,
            )
        else:
            msg = "Unsupported soft dataset"
            raise ValueError(msg)
    else:
        msg = "Unsupported dataset type"
        raise ValueError(msg)

    if hasattr(dataset, "set_ood"):
        dataset.set_ood()

    if subset < 1.0:
        num_samples = len(dataset)
        indices = torch.randperm(
            num_samples, generator=torch.Generator().manual_seed(seed)
        )
        subset_size = int(subset * num_samples)
        subset_indices = indices[:subset_size]
        dataset = Subset(dataset, subset_indices)

    if label_noise_fraction > 0.0:
        if label_noise_num_classes is None:
            msg = "label_noise_num_classes must be provided when label noise is enabled"
            raise ValueError(msg)
        dataset = LabelNoiseDataset(
            dataset=dataset,
            fraction=label_noise_fraction,
            num_classes=label_noise_num_classes,
            seed=seed if label_noise_seed is None else label_noise_seed,
            mode=label_noise_mode,
        )

    return dataset
