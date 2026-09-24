"""PathMNIST dataset wrappers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

try:
    from medmnist import PathMNIST as MedPathMNIST
except Exception:
    MedPathMNIST = None


def _coerce_label(label: object) -> int:
    array = np.asarray(label).reshape(-1)
    if array.size == 0:
        msg = "Empty label returned by medmnist PathMNIST"
        raise ValueError(msg)

    return int(array[-1])


def _to_pil_image(img: object) -> Image.Image:
    if isinstance(img, np.ndarray):
        return Image.fromarray(np.asarray(img, dtype=np.uint8))

    return img


def _brightness_down(x: Image.Image, severity: int = 1, rng=None) -> Image.Image:
    del rng
    factors = (0.90, 0.80, 0.70, 0.60, 0.50)
    factor = factors[min(max(severity, 1), len(factors)) - 1]
    array = np.asarray(x, dtype=np.float32) * factor
    return Image.fromarray(np.uint8(np.clip(array, 0, 255)))


def _contrast_up(x: Image.Image, severity: int = 1, rng=None) -> Image.Image:
    del rng
    factors = (1.20, 1.40, 1.60, 1.80, 2.00)
    factor = factors[min(max(severity, 1), len(factors)) - 1]
    array = np.asarray(x, dtype=np.float32)
    mean = np.mean(array, axis=(0, 1), keepdims=True)
    array = (array - mean) * factor + mean
    return Image.fromarray(np.uint8(np.clip(array, 0, 255)))


def _bubble(x: Image.Image, severity: int = 1, rng=None) -> Image.Image:
    if rng is None:
        rng = np.random.default_rng()

    array = np.asarray(x, dtype=np.float32).copy()
    height, width = array.shape[:2]
    count = (1, 2, 3, 4, 5)[min(max(severity, 1), 5) - 1]
    radius = max(2, round(min(height, width) * (0.08 + 0.02 * severity)))
    yy, xx = np.ogrid[:height, :width]

    for _ in range(count):
        center_y = int(rng.integers(0, height))
        center_x = int(rng.integers(0, width))
        mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
        array[mask] = 255

    return Image.fromarray(np.uint8(np.clip(array, 0, 255)))


def _stain_deposit(x: Image.Image, severity: int = 1, rng=None) -> Image.Image:
    if rng is None:
        rng = np.random.default_rng()

    array = np.asarray(x, dtype=np.float32).copy()
    height, width = array.shape[:2]
    count = (2, 4, 6, 8, 10)[min(max(severity, 1), 5) - 1]
    radius = max(1, round(min(height, width) * (0.04 + 0.01 * severity)))
    stain = np.array([115, 55, 120], dtype=np.float32)
    yy, xx = np.ogrid[:height, :width]

    for _ in range(count):
        center_y = int(rng.integers(0, height))
        center_x = int(rng.integers(0, width))
        mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
        array[mask] = 0.65 * array[mask] + 0.35 * stain

    return Image.fromarray(np.uint8(np.clip(array, 0, 255)))


def _build_single_corruption(corruption_type: str, severity: int):
    from untangle.transforms.ood_transforms_cifar import OOD_TRANSFORM_DICT_CIFAR

    pathmnist_aliases = {
        "brightness_down": _brightness_down,
        "contrast_up": _contrast_up,
        "jpeg_compression": OOD_TRANSFORM_DICT_CIFAR["jpeg"],
        "bubble": _bubble,
        "stain_deposit": _stain_deposit,
    }

    if corruption_type in pathmnist_aliases:
        return pathmnist_aliases[corruption_type]

    if corruption_type not in OOD_TRANSFORM_DICT_CIFAR:
        msg = f"Unsupported PathMNIST-C corruption: {corruption_type}"
        raise ValueError(msg)

    return OOD_TRANSFORM_DICT_CIFAR[corruption_type]


def _build_corruption(corruption_type: str | Sequence[str] | None, severity: int):
    if corruption_type is None or severity <= 0:
        return None

    if isinstance(corruption_type, str):
        corruption = _build_single_corruption(corruption_type, severity)

        def apply_corruption(img: Image.Image, idx: int) -> Image.Image:
            rng = np.random.default_rng(idx)
            return corruption(img, severity, rng)

        return apply_corruption

    corruptions = tuple(_build_single_corruption(name, severity) for name in corruption_type)
    if not corruptions:
        return None

    def apply_varied_corruption(img: Image.Image, idx: int) -> Image.Image:
        corruption_index = idx % len(corruptions)
        rng = np.random.default_rng(idx)
        return corruptions[corruption_index](img, severity, rng)

    return apply_varied_corruption


class _NPZPathMNISTC(Dataset):
    """Read PathMNIST-C samples directly from Zenodo NPZ archives."""

    def __init__(self, npz_path: Path, split: str, transform=None, target_transform=None):
        self._npz_path = npz_path
        self._split = split
        self.transform = transform
        self.target_transform = target_transform

        archive_split = "test" if split in {"val", "validation"} else split
        image_key = f"{archive_split}_images"
        label_key = f"{archive_split}_labels"

        image_sidecar = npz_path.with_name(f"{npz_path.stem}_{image_key}.npy")
        label_sidecar = npz_path.with_name(f"{npz_path.stem}_{label_key}.npy")
        if image_sidecar.exists() and label_sidecar.exists():
            self._npz = None
            self._images = np.load(image_sidecar, mmap_mode="r")
            self._labels = np.load(label_sidecar, mmap_mode="r")
            return

        self._npz = np.load(npz_path, mmap_mode="r")
        if image_key not in self._npz or label_key not in self._npz:
            msg = (
                f"NPZ file {npz_path} does not contain keys "
                f"'{image_key}' and '{label_key}'"
            )
            raise ValueError(msg)

        self._images = self._npz[image_key]
        self._labels = self._npz[label_key]

    def __len__(self):
        return int(self._images.shape[0])

    def __getitem__(self, idx):
        image = Image.fromarray(np.asarray(self._images[idx], dtype=np.uint8))
        label = _coerce_label(self._labels[idx])

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


class _VariedNPZPathMNISTC(Dataset):
    """Deterministically mix several PathMNIST-C NPZ corruption archives."""

    def __init__(
        self,
        npz_paths: Sequence[Path],
        split: str,
        transform=None,
        target_transform=None,
    ):
        if not npz_paths:
            msg = "At least one PathMNIST-C NPZ archive is required"
            raise ValueError(msg)

        self._datasets = tuple(
            _NPZPathMNISTC(
                npz_path=npz_path,
                split=split,
                transform=transform,
                target_transform=target_transform,
            )
            for npz_path in npz_paths
        )
        lengths = {len(dataset) for dataset in self._datasets}
        if len(lengths) != 1:
            msg = "All PathMNIST-C NPZ archives must contain the same number of samples"
            raise ValueError(msg)

    def __len__(self):
        return len(self._datasets[0])

    def __getitem__(self, idx):
        dataset = self._datasets[idx % len(self._datasets)]
        return dataset[idx]


class _MedMNISTWrapper(Dataset):
    """Wrap a medmnist dataset so it behaves like a regular torch dataset."""

    def __init__(
        self,
        med_dataset,
        transform=None,
        target_transform=None,
        corruption_type: str | Sequence[str] | None = None,
        corruption_severity: int = 0,
    ) -> None:
        self.med = med_dataset
        self.transform = transform
        self.target_transform = target_transform
        self.corruption = _build_corruption(corruption_type, corruption_severity)

    def __len__(self):
        return len(self.med)

    def __getitem__(self, idx):
        item = self.med[idx]

        if isinstance(item, (tuple, list)):
            img, label = item
        else:
            try:
                img = item[0]
                label = item[1]
            except Exception as exc:
                raise RuntimeError("Unsupported medmnist item format") from exc

        pil = _to_pil_image(img)
        if self.corruption is not None:
            pil = self.corruption(pil, idx)

        if self.transform is not None:
            pil = self.transform(pil)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return pil, _coerce_label(label)


class PathMNIST(ImageFolder):
    """ImageFolder-like loader for PathMNIST with optional MedMNIST fallback."""

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        download=False,
        prefer_medmnist: bool | None = None,
        corruption_type: str | Sequence[str] | None = None,
        corruption_severity: int = 0,
    ):
        root = Path(root)
        split = str(split)
        dataset_root = root / split

        env_pref = os.environ.get("PATHMNIST_PREFER_MEDMNIST")
        if prefer_medmnist is None:
            prefer_medmnist = False if env_pref is None else env_pref.lower() not in {"0", "false", "no"}

        use_imagefolder = dataset_root.exists() and not prefer_medmnist and corruption_type is None
        if use_imagefolder:
            super().__init__(str(dataset_root), transform=transform, target_transform=target_transform)
            return

        if MedPathMNIST is None:
            msg = (
                f"PathMNIST expected dataset directory {dataset_root} to exist, "
                "and medmnist package is not installed to download/load it."
            )
            raise ValueError(msg)

        try:
            med = MedPathMNIST(split=split, download=download)
        except TypeError:
            med = MedPathMNIST(download=download, split=split)

        self._wrapped = _MedMNISTWrapper(
            med,
            transform=transform,
            target_transform=target_transform,
            corruption_type=corruption_type,
            corruption_severity=corruption_severity,
        )

    def __len__(self):
        if hasattr(self, "_wrapped"):
            return len(self._wrapped)

        return super().__len__()

    def __getitem__(self, idx):
        if hasattr(self, "_wrapped"):
            return self._wrapped[idx]

        return super().__getitem__(idx)


class PathMNISTC(PathMNIST):
    """MedMNIST-C style PathMNIST wrapper with corruption applied on load."""

    def __init__(
        self,
        root,
        split="test",
        transform=None,
        target_transform=None,
        download=False,
        corruption_type: str | Sequence[str] | None = None,
        corruption_severity: int = 1,
    ):
        root_path = Path(root)
        if corruption_type is not None:
            corruption_types = (
                (corruption_type,) if isinstance(corruption_type, str) else tuple(corruption_type)
            )
            npz_paths = []
            for current_type in corruption_types:
                candidates = (
                    root_path / f"{current_type}.npz",
                    root_path / "pathmnist" / f"{current_type}.npz",
                )
                npz_paths.extend(npz_path for npz_path in candidates if npz_path.exists())

            if npz_paths:
                if len(npz_paths) != len(corruption_types):
                    found = {npz_path.stem for npz_path in npz_paths}
                    missing = sorted(set(corruption_types) - found)
                    msg = f"Missing PathMNIST-C NPZ archives for: {missing}"
                    raise ValueError(msg)

                if corruption_severity != 1:
                    msg = (
                        "Zenodo PathMNIST-C NPZ archives do not encode multiple severities; "
                        "use --severities 1 with this dataset source."
                    )
                    raise ValueError(msg)

                if len(npz_paths) == 1:
                    self._wrapped = _NPZPathMNISTC(
                        npz_path=npz_paths[0],
                        split=split,
                        transform=transform,
                        target_transform=target_transform,
                    )
                else:
                    self._wrapped = _VariedNPZPathMNISTC(
                        npz_paths=npz_paths,
                        split=split,
                        transform=transform,
                        target_transform=target_transform,
                    )
                return

            msg = (
                "Official PathMNIST-C NPZ archives are required for hard/pathmnist-c. "
                f"Expected files such as {root_path / (corruption_types[0] + '.npz')} "
                f"or {root_path / 'pathmnist' / (corruption_types[0] + '.npz')}. "
                "Download pathmnist.zip from the MedMNIST-C Zenodo record and extract "
                "it below the configured --data-dir-id."
            )
            raise FileNotFoundError(msg)

        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
            prefer_medmnist=True,
            corruption_type=corruption_type,
            corruption_severity=corruption_severity,
        )
