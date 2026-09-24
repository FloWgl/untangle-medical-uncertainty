import argparse
from pathlib import Path
from typing import Iterable
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from untangle.datasets.pathmnist import PathMNIST


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_image_array(arr: np.ndarray, path: Path) -> None:
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    Image.fromarray(arr).save(path)


def _coerce_label(label: object) -> int:
    try:
        return int(np.asarray(label).squeeze())
    except Exception:
        return int(label)


def _iter_samples(dataset, max_samples: int):
    total = len(dataset)
    if max_samples > 0:
        total = min(total, max_samples)

    for index in range(total):
        yield index, dataset[index]


def export_dataset(dataset, output_dir: Path, split: str, max_samples: int) -> int:
    count = 0
    out_split_dir = output_dir / split
    ensure_dir(out_split_dir)

    for index, item in _iter_samples(dataset, max_samples):
        if isinstance(item, (tuple, list)):
            image, label = item
        elif isinstance(item, dict):
            image = item.get("image", None) or item.get("img", None) or item.get("x", None)
            label = item.get("label", None) or item.get("y", None)
        else:
            raise RuntimeError("Unsupported dataset item format")

        if image is None or label is None:
            raise RuntimeError("Could not extract image/label from dataset item")

        label_dir = out_split_dir / str(_coerce_label(label))
        ensure_dir(label_dir)
        save_image_array(np.asarray(image), label_dir / f"{index}.png")
        count += 1

    return count


def load_pathmnist(split: str, download: bool):
    return PathMNIST(
        root="unused",
        split=split,
        transform=None,
        target_transform=None,
        download=download,
        prefer_medmnist=True,
    )


def export_clean_pathmnist(output_dir: Path, splits: Iterable[str], download: bool, max_samples: int) -> int:
    total = 0
    for split in splits:
        print(f"Loading PathMNIST split={split} (download={download})...")
        dataset = load_pathmnist(split, download)
        count = export_dataset(dataset, output_dir / "pathmnist", split, max_samples)
        print(f"Exported {count} samples for PathMNIST split={split}")
        total += count

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default="pathmnist",
        choices=["pathmnist"],
        help="Which dataset export to run.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits for PathMNIST.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples per split (0 = all).",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    pathmnist_splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    total = export_clean_pathmnist(output_dir, pathmnist_splits, args.download, args.max_samples)

    print(f"Total exported: {total}")


if __name__ == "__main__":
    main()
