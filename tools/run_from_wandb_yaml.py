#!/usr/bin/env python3
"""Construct and optionally run train.py using a saved W&B sweep YAML.

Usage:
  python tools/run_from_wandb_yaml.py docs/wandb_sweeps/<ID>.yaml [--execute]

The script reproduces the sweep's command-line invocation as closely as possible.
"""
import shlex
import subprocess
import sys
from pathlib import Path
import yaml


def param_to_arg(key: str, entry) -> list[str]:
    # key is like 'batch-size' -> CLI flag --batch-size
    flag = f"--{key}"

    # entry may contain 'value' or 'values'
    if isinstance(entry, dict):
        if 'value' in entry:
            val = entry['value']
        elif 'values' in entry:
            vals = entry['values']
            # join lists into comma-separated string
            val = ','.join(map(str, vals))
        else:
            return []
    else:
        val = entry

    # Booleans: if True include flag as action, if False omit
    if isinstance(val, bool):
        return [flag] if val else []

    # None: skip
    if val is None:
        return []

    return [flag, str(val)]


def build_command(yaml_path: Path) -> list[str]:
    data = yaml.safe_load(yaml_path.read_text())
    cfg = data.get('config', {})

    program = cfg.get('program', 'train.py')
    params = cfg.get('parameters', {})

    cmd = [sys.executable, program]

    # Iterate in deterministic order
    for key in sorted(params.keys()):
        entry = params[key]
        args = param_to_arg(key, entry)
        if args:
            cmd.extend(args)

    # W&B uses ${program} ${args_no_boolean_flags} etc.; keep program args from sweep
    return cmd


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('yaml', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--cwd', type=Path, default=Path('.'))
    args = p.parse_args()

    cmd = build_command(args.yaml)

    print('Constructed command:')
    print(' '.join(shlex.quote(x) for x in cmd))

    if args.execute:
        print('\nRunning... (press Ctrl-C to stop)')
        subprocess.run(cmd, cwd=str(args.cwd))


if __name__ == '__main__':
    main()
