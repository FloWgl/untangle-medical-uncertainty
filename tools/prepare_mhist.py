#!/usr/bin/env python3
"""Prepare MHIST for the untangle training pipeline.

The official MHIST page sends expiring download links by email after accepting
the dataset use agreement. This script accepts those URLs, downloads the files,
extracts ``images.zip``, and checks that the expected train/test split is
available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


ACCESS_URL = "https://bmirds.github.io/MHIST/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare MHIST.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/MHIST"),
        help="Destination MHIST root. The final layout is root/images and root/annotations.csv.",
    )
    parser.add_argument("--images-url", type=str, default="", help="Emailed images.zip URL.")
    parser.add_argument(
        "--annotations-url",
        type=str,
        default="",
        help="Emailed annotations.csv URL.",
    )
    parser.add_argument("--md5-url", type=str, default="", help="Optional emailed MD5SUMs URL.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only validate/extract files that already exist under root.",
    )
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    if not url:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name} ...")
    urlretrieve(url, destination)


def extract_images(root: Path) -> None:
    images_dir = root / "images"
    if images_dir.exists() and any(images_dir.iterdir()):
        return

    archive = root / "images.zip"
    if not archive.exists():
        msg = f"Missing {archive}. Provide --images-url or place images.zip there."
        raise FileNotFoundError(msg)

    print("Extracting images.zip ...")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)

    nested = root / "images"
    if nested.exists():
        return

    pngs = list(root.glob("*.png"))
    if pngs:
        images_dir.mkdir(exist_ok=True)
        for path in pngs:
            shutil.move(str(path), images_dir / path.name)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_md5s(root: Path) -> None:
    md5_path = root / "MD5SUMs.txt"
    if not md5_path.exists():
        return

    print("Checking MD5 sums ...")
    for line in md5_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        expected, filename = parts[0], parts[-1].lstrip("*")
        candidate = root / filename
        if not candidate.exists():
            continue
        actual = file_md5(candidate)
        if actual != expected:
            msg = f"MD5 mismatch for {candidate}: expected {expected}, got {actual}"
            raise RuntimeError(msg)


def validate_annotations(root: Path) -> None:
    annotations = root / "annotations.csv"
    images = root / "images"
    if not annotations.exists():
        msg = f"Missing {annotations}. Provide --annotations-url or place it there."
        raise FileNotFoundError(msg)
    if not images.exists():
        msg = f"Missing {images}. Extract images.zip first."
        raise FileNotFoundError(msg)

    with annotations.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("annotations.csv has no header")

        partition_col = next(
            (name for name in reader.fieldnames if name.lower() == "partition"),
            None,
        )
        image_col = next(
            (
                name
                for name in reader.fieldnames
                if name.lower() in {"image name", "image", "filename", "file_name"}
            ),
            None,
        )
        if partition_col is None or image_col is None:
            msg = "annotations.csv must contain image-name and Partition columns"
            raise RuntimeError(msg)

        counts = {"train": 0, "test": 0}
        missing_images = []
        for row in reader:
            partition = row[partition_col].strip().lower()
            if partition in counts:
                counts[partition] += 1
            image_name = row[image_col].strip()
            if image_name and not (images / image_name).exists():
                missing_images.append(image_name)

    if missing_images:
        msg = f"{len(missing_images)} annotation rows reference missing images"
        raise RuntimeError(msg)

    print(f"MHIST ready at {root}")
    print(f"train={counts['train']} test={counts['test']}")


def main() -> int:
    args = parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)

    has_local_files = (root / "images.zip").exists() and (root / "annotations.csv").exists()

    if not args.skip_download:
        if not args.images_url or not args.annotations_url:
            print(
                "MHIST does not provide stable public file URLs. Request access at "
                f"{ACCESS_URL}, then rerun this script with --images-url and "
                "--annotations-url. If you already downloaded the files, place "
                "images.zip and annotations.csv under the root and use --skip-download.",
                file=sys.stderr,
            )
            if not has_local_files:
                return 2
        download(args.images_url, root / "images.zip")
        download(args.annotations_url, root / "annotations.csv")
        download(args.md5_url, root / "MD5SUMs.txt")

    extract_images(root)
    check_md5s(root)
    validate_annotations(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
