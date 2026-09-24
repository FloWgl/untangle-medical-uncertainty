"""Smoke tests for MHIST loading."""

from pathlib import Path

from PIL import Image
from torchvision import transforms

from untangle.datasets.mhist import MHIST
from untangle.utils.dataset import create_dataset


def _write_tiny_mhist(root: Path) -> None:
    images = root / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (224, 224), color=(200, 80, 120)).save(images / "MHIST_aaa.png")
    Image.new("RGB", (224, 224), color=(80, 160, 200)).save(images / "MHIST_aab.png")
    (root / "annotations.csv").write_text(
        "\n".join(
            [
                "Image Name,Majority Vote Label,Number of Annotators who Selected SSA (Out of 7),Partition",
                "MHIST_aaa.png,HP,2,train",
                "MHIST_aab.png,SSA,6,test",
            ]
        )
        + "\n"
    )


def test_mhist_hard_and_soft_loading(tmp_path: Path) -> None:
    _write_tiny_mhist(tmp_path)
    transform = transforms.ToTensor()

    hard = MHIST(tmp_path, split="train", transform=transform)
    image, label = hard[0]
    assert image.shape == (3, 224, 224)
    assert label == 0

    soft = MHIST(tmp_path, split="val", transform=transform, soft_labels=True)
    _, target = soft[0]
    assert target.tolist() == [1, 6, 1]


def test_mhist_create_dataset_aliases(tmp_path: Path) -> None:
    _write_tiny_mhist(tmp_path)

    hard = create_dataset(
        name="hard/mhist",
        root=tmp_path,
        label_root=tmp_path,
        split="train",
        download=False,
        seed=0,
        subset=1.0,
        input_size=(3, 224, 224),
        padding=0,
        is_training_dataset=True,
        use_prefetcher=False,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        hflip=0.0,
        color_jitter=0.0,
        interpolation="bilinear",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        crop_pct=1.0,
        ood_transform_type=None,
        severity=0,
        convert_soft_labels_to_hard=False,
    )
    assert len(hard) == 1

    soft = create_dataset(
        name="soft/mhist",
        root=tmp_path,
        label_root=tmp_path,
        split="test",
        download=False,
        seed=0,
        subset=1.0,
        input_size=(3, 224, 224),
        padding=0,
        is_training_dataset=False,
        use_prefetcher=False,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        hflip=0.0,
        color_jitter=0.0,
        interpolation="bilinear",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        crop_pct=1.0,
        ood_transform_type=None,
        severity=0,
        convert_soft_labels_to_hard=True,
    )
    _, target = soft[0]
    assert int(target) == 1
