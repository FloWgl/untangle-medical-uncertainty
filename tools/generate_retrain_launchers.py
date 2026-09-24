#!/usr/bin/env python3
"""Generate retrain launchers for missing checkpoints.

This script reads `docs/cifar10_rerun_plan.csv` and `docs/cifar10_artifact_map.csv`,
finds mapping entries that are not downloaded, and generates per-sweep retrain
launcher scripts under `tools/launchers/retrain_<sweep>.sh`. It attempts to
reuse an existing training launcher that matches the method; if none found,
it generates a simple fallback command.
"""
import csv
import os
from pathlib import Path


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def find_existing_launcher_for_method(method_name):
    launchers = Path('tools/launchers').glob('run_*.sh')
    for p in launchers:
        try:
            text = p.read_text()
        except Exception:
            continue
        if f'--method-name {method_name}' in text or f'--method-name {method_name.replace("-","_")}' in text:
            return p, text
    return None, None


def make_retrain_script(outpath: Path, base_cmd: str | None, sweep: str, method: str, dataset: str, dataset_id: str, epochs: int):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if base_cmd:
        # replace epochs in base_cmd if present
        import re
        cmd = re.sub(r'--epochs\s+\d+', f'--epochs {epochs}', base_cmd)
        header = f"#!/usr/bin/env bash\n# Retrain for sweep {sweep} (method={method})\n"
        outpath.write_text(header + cmd + "\n")
    else:
        header = f"#!/usr/bin/env bash\n# Fallback retrain for sweep {sweep} (method={method})\n"
        cmd = f"python3 train.py --method-name {method} --dataset {dataset} --dataset-id {dataset_id} --epochs {epochs} --batch-size 128 --data-dir ./data --storage-device cuda"
        outpath.write_text(header + cmd + "\n")
    outpath.chmod(0o755)


def main():
    plan = read_csv('docs/cifar10_rerun_plan.csv')
    mapping = read_csv('docs/cifar10_artifact_map.csv')

    # find missing mapping entries
    missing = [m for m in mapping if m.get('status') != 'downloaded']
    if not missing:
        print('No missing artifacts to retrain.')
        return

    # index plan by method
    plan_by_method = {}
    for row in plan:
        name = (row.get('method_name') or '').strip()
        plan_by_method.setdefault(name, []).append(row)

    retrain_scripts = []
    for m in missing:
        method = (m.get('method') or '').strip()
        sweep = m.get('sweep') or 'unknown'
        # find training epochs from plan (pick first with epochs>0)
        epochs = 200
        for r in plan_by_method.get(method, []):
            try:
                e = int(r.get('epochs') or 0)
            except Exception:
                e = 0
            if e > 0:
                epochs = e
                break

        launcher_src, text = find_existing_launcher_for_method(method)
        outpath = Path('tools/launchers') / f'retrain_{sweep}.sh'
        make_retrain_script(outpath, text, sweep, method, (plan_by_method.get(method,[{}])[0].get('dataset') or 'hard/cifar10'), (plan_by_method.get(method,[{}])[0].get('dataset_id') or 'soft/cifar10'), epochs)
        retrain_scripts.append(outpath)

    # write master queue
    master = Path('tools/launchers/run_retrain_missing.sh')
    lines = ['#!/usr/bin/env bash', '# Master queue to run retrain scripts for missing artifacts']
    for s in retrain_scripts:
        lines.append(f'bash {s} &')
    lines.append('wait')
    master.write_text('\n'.join(lines) + '\n')
    master.chmod(0o755)

    print(f'Generated {len(retrain_scripts)} retrain scripts and {master}')


if __name__ == '__main__':
    main()
