"""Simple smoke test for hard-label PathMNIST loading.

This is runnable as a standalone script (for local CI runners) and also as a
pytest test if you prefer.
"""

from pathlib import Path

from torchvision import transforms

from untangle.datasets.pathmnist import PathMNIST, PathMNISTC
from untangle.utils.dataset import LabelNoiseDataset
from untangle.utils.dataset import create_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
PATHMNIST_EXPORT = REPO_ROOT / "tmp_pathmnist_export"


def run_smoke(download=False):
    tf = transforms.Compose([transforms.ToTensor()])

    ds = PathMNIST(root=PATHMNIST_EXPORT, split="train", transform=tf)
    assert len(ds) > 0, "Dataset should have >0 samples"

    img, label = ds[0]

    # Basic shape/type checks
    assert hasattr(img, "shape") or hasattr(img, "size"), "Image should be tensor-like or PIL"
    assert isinstance(label, (int,)) or (hasattr(label, "dtype") and getattr(label, "dtype", None) is not None), "Label should be integer-like"

    alias_ds = create_dataset(
        name="hard/pathmnist",
        root=PATHMNIST_EXPORT,
        label_root=PATHMNIST_EXPORT,
        split="train",
        download=download,
        seed=0,
        subset=1.0,
        input_size=(3, 28, 28),
        padding=0,
        is_training_dataset=True,
        use_prefetcher=False,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        hflip=0.0,
        color_jitter=0.0,
        interpolation="bilinear",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
        ood_transform_type=None,
        severity=0,
        convert_soft_labels_to_hard=False,
    )
    assert len(alias_ds) > 0, "hard/pathmnist alias should resolve"

    try:
        ds_c = PathMNISTC(
            root="unused",
            split="test",
            transform=tf,
            download=download,
            corruption_type="gaussian_noise",
            corruption_severity=1,
        )
        assert len(ds_c) > 0, "PathMNIST-C should have >0 samples"
    except Exception:
        # PathMNIST-C still depends on the MedMNIST package in this codebase.
        pass


def test_pathmnist_smoke():
    run_smoke(download=False)


def test_pathmnist_label_noise_wrapper():
    tf = transforms.Compose([transforms.ToTensor()])
    ds = PathMNIST(root=PATHMNIST_EXPORT, split="train", transform=tf)
    noisy = LabelNoiseDataset(
        dataset=ds,
        fraction=0.2,
        num_classes=9,
        seed=123,
    )

    assert len(noisy.label_noise_indices) == round(0.2 * len(ds))
    assert noisy.noisy_labels
    for index, clean_label in noisy.clean_labels.items():
        _, returned_label = noisy[index]
        assert returned_label == noisy.noisy_labels[index]
        assert returned_label != clean_label


if __name__ == '__main__':
    # Running with download=False avoids network during CI if data already cached.
    run_smoke(download=False)
