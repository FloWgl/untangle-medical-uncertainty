# MHIST Integration

MHIST is now available in the dataset namespace as:

- `hard/mhist`: majority-vote hard labels for training.
- `soft/mhist`: vote-count soft labels for evaluation. Targets have the form `[HP_votes, SSA_votes, majority_label]`.

The expected data layout is:

```text
data/MHIST/
  annotations.csv
  images/
    MHIST_*.png
```

The prepared dataset in this repo currently has:

| Split | Samples | HP | SSA |
|---|---:|---:|---:|
| train | 2,175 | 1,545 | 630 |
| test | 977 | 617 | 360 |
| all | 3,152 | 2,162 | 990 |

The SSA vote count spans the full range from 0 to 7 in both train and test.

The official dataset page requires accepting the research use agreement and sends expiring download links by email. After receiving those links, prepare the data with:

```bash
python tools/prepare_mhist.py \
  --root data/MHIST \
  --images-url '<emailed images.zip URL>' \
  --annotations-url '<emailed annotations.csv URL>' \
  --md5-url '<optional MD5SUMs.txt URL>'
```

If the files were downloaded manually, place `images.zip` and `annotations.csv` under `data/MHIST` and run:

```bash
python tools/prepare_mhist.py --root data/MHIST --skip-download
```

MHIST has no separate validation split. The loader maps `split="val"` to the official `test` partition, matching common MHIST usage. For final reporting, document that validation and test use the same official partition unless an internal train split is added.

Example smoke command:

```bash
python train.py \
  --data-dir data/MHIST \
  --soft-label-root data/MHIST \
  --dataset hard/mhist \
  --dataset-id soft/mhist \
  --num-classes 2 \
  --img-size 224 \
  --epochs 0 \
  --discard-ood-test-sets \
  --storage-device cpu
```
