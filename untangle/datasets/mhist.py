"""MHIST dataset wrapper.

MHIST is distributed as ``annotations.csv`` plus an ``images`` directory.
The annotations contain a majority-vote label and the number of pathologists
who selected SSA out of seven raters.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader


CLASS_NAME_TO_INDEX = {
    "HP": 0,
    "SSA": 1,
}


@dataclass(frozen=True)
class MHISTRecord:
    image_name: str
    label: int
    num_ssa_votes: int
    partition: str


def _normalize_column_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    normalized = {_normalize_column_name(name): name for name in fieldnames}
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]

    msg = (
        "MHIST annotations.csv is missing one of the expected columns: "
        f"{', '.join(candidates)}"
    )
    raise ValueError(msg)


def _read_records(annotations_path: Path, split: str) -> list[MHISTRecord]:
    split = "test" if split == "val" else split
    if split not in {"train", "test", "all"}:
        msg = f"Unsupported MHIST split: {split}"
        raise ValueError(msg)

    with annotations_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            msg = f"MHIST annotations file is empty: {annotations_path}"
            raise ValueError(msg)

        fieldnames = list(reader.fieldnames)
        image_col = _find_column(
            fieldnames,
            (
                "Image Name",
                "Image",
                "image",
                "filename",
                "file_name",
            ),
        )
        label_col = _find_column(
            fieldnames,
            (
                "Majority Vote Label",
                "Majority Vote",
                "majority_vote_label",
                "label",
            ),
        )
        votes_col = _find_column(
            fieldnames,
            (
                "Number of Annotators who Selected SSA",
                "Number of Annotators who Selected SSA (Out of 7)",
                "Number of Annotators Who Selected SSA",
                "Number of Annotators Who Selected SSA (Out of 7)",
                "num_ssa",
                "ssa_votes",
            ),
        )
        partition_col = _find_column(
            fieldnames,
            (
                "Partition",
                "split",
            ),
        )

        records = []
        for row in reader:
            partition = row[partition_col].strip().lower()
            if split != "all" and partition != split:
                continue

            label_name = row[label_col].strip().upper()
            if label_name not in CLASS_NAME_TO_INDEX:
                msg = f"Unsupported MHIST label {label_name!r}"
                raise ValueError(msg)

            records.append(
                MHISTRecord(
                    image_name=row[image_col].strip(),
                    label=CLASS_NAME_TO_INDEX[label_name],
                    num_ssa_votes=int(row[votes_col]),
                    partition=partition,
                )
            )

    if len(records) == 0:
        msg = f"Found 0 MHIST samples for split {split!r} in {annotations_path}"
        raise RuntimeError(msg)

    return records


class MHIST(Dataset):
    """Hard- or soft-label MHIST loader.

    For hard labels, targets are integer majority-vote labels. For soft labels,
    targets contain ``[HP vote count, SSA vote count, majority label]`` so the
    last value remains compatible with ``hard_target_transform``.
    """

    num_raters = 7

    def __init__(
        self,
        root,
        split: str = "train",
        transform=None,
        target_transform=None,
        soft_labels: bool = False,
        download: bool = False,
    ) -> None:
        if download:
            msg = (
                "MHIST requires accepting the dataset use agreement on the BMIRDS "
                "access page. Use tools/prepare_mhist.py with the emailed URLs, "
                "or place annotations.csv and images/ under the MHIST root."
            )
            raise RuntimeError(msg)

        self.root = Path(root)
        self.dataset_root = self._resolve_dataset_root(self.root)
        self.images_root = self.dataset_root / "images"
        self.annotations_path = self.dataset_root / "annotations.csv"
        self.transform = transform
        self.target_transform = target_transform
        self.soft_labels = soft_labels

        if not self.annotations_path.exists():
            msg = f"Missing MHIST annotations file: {self.annotations_path}"
            raise FileNotFoundError(msg)
        if not self.images_root.exists():
            msg = f"Missing MHIST images directory: {self.images_root}"
            raise FileNotFoundError(msg)

        self.records = _read_records(self.annotations_path, split)

    @staticmethod
    def _resolve_dataset_root(root: Path) -> Path:
        if (root / "annotations.csv").exists():
            return root
        if (root / "MHIST" / "annotations.csv").exists():
            return root / "MHIST"
        if (root / "mhist" / "annotations.csv").exists():
            return root / "mhist"
        return root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, int | Tensor]:
        record = self.records[index]
        image = pil_loader(str(self.images_root / record.image_name))

        if self.transform is not None:
            image = self.transform(image)

        target: int | Tensor
        if self.soft_labels:
            target = torch.tensor(
                [
                    self.num_raters - record.num_ssa_votes,
                    record.num_ssa_votes,
                    record.label,
                ],
                dtype=torch.long,
            )
        else:
            target = record.label

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
