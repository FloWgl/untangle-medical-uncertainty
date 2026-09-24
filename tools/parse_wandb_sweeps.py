#!/usr/bin/env python3
import csv
from pathlib import Path

SWEEPS_DIR = Path(__file__).resolve().parent.parent / "docs" / "wandb_sweeps"
OUT_CSV = Path(__file__).resolve().parent.parent / "docs" / "cifar10_rerun_plan.csv"

def parse_yaml_simple(p: Path):
    data = {
        "sweep": p.stem,
        "method_name": "",
        "dataset": "",
        "dataset_id": "",
        "epochs": "",
        "weight_paths_count": 0,
    }
    lines = p.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # handle both `key: value` and `key:\n  value: ...` patterns
        if line.startswith('method-name:'):
            maybe = line.split(':',1)[1].strip()
            if maybe:
                data['method_name'] = maybe
            else:
                # look ahead for 'value:'
                j = i+1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1
                if j < len(lines) and lines[j].strip().startswith('value:'):
                    data['method_name'] = lines[j].split(':',1)[1].strip()
        elif line.startswith('dataset:'):
            maybe = line.split(':',1)[1].strip()
            if maybe:
                data['dataset'] = maybe
            else:
                j = i+1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1
                if j < len(lines) and lines[j].strip().startswith('value:'):
                    data['dataset'] = lines[j].split(':',1)[1].strip()
        elif line.startswith('dataset-id:'):
            maybe = line.split(':',1)[1].strip()
            if maybe:
                data['dataset_id'] = maybe
            else:
                j = i+1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1
                if j < len(lines) and lines[j].strip().startswith('value:'):
                    data['dataset_id'] = lines[j].split(':',1)[1].strip()
        elif line.startswith('epochs:'):
            maybe = line.split(':',1)[1].strip()
            if maybe:
                data['epochs'] = maybe
            else:
                j = i+1
                while j < len(lines) and lines[j].strip() == '':
                    j += 1
                if j < len(lines) and lines[j].strip().startswith('value:'):
                    data['epochs'] = lines[j].split(':',1)[1].strip()
        elif line.startswith('weight-paths:'):
            # read following lines for list entries
            j = i+1
            cnt = 0
            while j < len(lines) and (lines[j].lstrip().startswith('-') or lines[j].strip().startswith('values:')):
                if lines[j].lstrip().startswith('-'):
                    cnt += 1
                j += 1
            data['weight_paths_count'] = cnt
        i += 1
    return data

def main():
    rows = []
    for p in sorted(SWEEPS_DIR.glob('*.yaml')):
        rows.append(parse_yaml_simple(p))

    # write CSV
    with OUT_CSV.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'sweep',
                'method_name',
                'dataset',
                'dataset_id',
                'epochs',
                'weight_paths_count',
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

if __name__ == '__main__':
    main()
