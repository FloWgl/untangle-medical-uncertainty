#!/usr/bin/env bash
# Download official MedMNIST-C PathMNIST-C NPZ files and submit corruption evals.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset root}"
PATHMNIST_C_ROOT="${PATHMNIST_C_ROOT:-$DATA_ROOT/pathmnist_c}"
PATHMNIST_C_DIR="${PATHMNIST_C_DIR:-$PATHMNIST_C_ROOT/pathmnist}"
ARCHIVE="${ARCHIVE:-$PATHMNIST_C_ROOT/pathmnist.zip}"
URL="${URL:-https://zenodo.org/records/11471504/files/pathmnist.zip?download=1}"
EXPECTED_MD5="${EXPECTED_MD5:-bf62498906ec0383c3ec5ff12ac70c00}"
CMD_FILE="${CMD_FILE:-$REPO_ROOT/logs/pathmnist_hpc_20260730-fix3-pathmnist-corruption/commands.txt}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-pathmnist-c-official}"

required=(
  brightness_down
  contrast_up
  defocus_blur
  motion_blur
  jpeg_compression
  pixelate
  bubble
  stain_deposit
)

mkdir -p "$PATHMNIST_C_ROOT" "$PATHMNIST_C_DIR" logs/slurm "logs/pathmnist_c_official_$RUN_ID"

if [[ ! -s "$ARCHIVE" ]]; then
  echo "Downloading official PathMNIST-C archive from Zenodo:"
  echo "  $URL"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --continue-at - "$URL" -o "$ARCHIVE"
  elif command -v wget >/dev/null 2>&1; then
    wget -c "$URL" -O "$ARCHIVE"
  else
    echo "Neither curl nor wget is available." >&2
    exit 2
  fi
else
  echo "Archive already exists: $ARCHIVE"
fi

actual_md5="$(md5sum "$ARCHIVE" | awk '{print $1}')"
if [[ "$actual_md5" != "$EXPECTED_MD5" ]]; then
  echo "MD5 mismatch for $ARCHIVE" >&2
  echo "Expected: $EXPECTED_MD5" >&2
  echo "Actual:   $actual_md5" >&2
  exit 3
fi
echo "MD5 verified: $actual_md5"

if [[ ! -f "$PATHMNIST_C_DIR/brightness_down.npz" ]]; then
  tmp_extract="$PATHMNIST_C_ROOT/extract_tmp"
  rm -rf "$tmp_extract"
  mkdir -p "$tmp_extract"
  unzip -q "$ARCHIVE" -d "$tmp_extract"

  for name in "${required[@]}"; do
    npz_path="$(find "$tmp_extract" -type f -name "$name.npz" | head -n 1)"
    if [[ -z "$npz_path" ]]; then
      echo "Missing $name.npz inside $ARCHIVE" >&2
      exit 4
    fi
    cp "$npz_path" "$PATHMNIST_C_DIR/$name.npz"
  done
  rm -rf "$tmp_extract"
fi

for name in "${required[@]}"; do
  test -f "$PATHMNIST_C_DIR/$name.npz"
done
echo "Official PathMNIST-C files ready in $PATHMNIST_C_DIR"

if [[ ! -f "$CMD_FILE" ]]; then
  echo "Command file not found: $CMD_FILE" >&2
  exit 5
fi

num_tasks="$(wc -l < "$CMD_FILE")"
if [[ "$num_tasks" -lt 1 ]]; then
  echo "Command file has no tasks: $CMD_FILE" >&2
  exit 6
fi

echo "Submitting official PathMNIST-C corruption evaluation: $num_tasks tasks"
sbatch \
  --job-name pathmnist-corruption-official \
  --array "1-$num_tasks" \
  --gres gpu:1 \
  --time 1-00:00:00 \
  --export "ALL,CMD_FILE=$CMD_FILE,WORK_DIR=$REPO_ROOT,ENV_COMMAND=module load python/pytorch2.6py3.12" \
  tools/hpc/untangle_job_array.sbatch
