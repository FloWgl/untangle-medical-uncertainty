#!/usr/bin/env bash
# Helper script to download CIFAR-10 to a target directory for experiments.
# Usage: ./tools/prepare_cifar10.sh /path/to/data_dir
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/data_dir"
  exit 1
fi
DATA_DIR="$1"
python3 - <<PY
from pathlib import Path
from torchvision.datasets import CIFAR10
p=Path("$DATA_DIR")
print('Downloading CIFAR-10 to',p)
CIFAR10(root=str(p), train=True, download=True)
CIFAR10(root=str(p), train=False, download=True)
print('Done')
PY
