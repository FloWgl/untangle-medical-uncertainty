#!/usr/bin/env python3
"""Fetch W&B sweep configurations for given sweep IDs and save as YAML.

Usage:
  export WANDB_API_KEY=...   # required
  python tools/fetch_wandb_sweeps.py adp0fyi8 xibtpo9s ...

The script saves files to `docs/wandb_sweeps/<SWEEP_ID>.yaml`.
"""
import sys
import os
import yaml

def main():
    try:
        import wandb
    except Exception as e:
        print("wandb not available:", e)
        sys.exit(2)

    if 'WANDB_API_KEY' not in os.environ:
        print("WANDB_API_KEY not set in environment. Set it and re-run.")
        sys.exit(3)

    api = wandb.Api()
    sweep_ids = sys.argv[1:]
    if not sweep_ids:
        print("Provide one or more sweep IDs as arguments.")
        sys.exit(1)

    out_dir = os.path.join('docs', 'wandb_sweeps')
    os.makedirs(out_dir, exist_ok=True)

    for sid in sweep_ids:
        # try common patterns: entity/project/sweep_id and project/sweep_id
        tried = []
        candidates = [f"bmucsanyi/untangle/{sid}", f"bmucsanyi/untangle/sweeps/{sid}", sid]
        sweep = None
        for cand in candidates:
            try:
                sweep = api.sweep(cand)
                break
            except Exception:
                tried.append(cand)
                continue

        if sweep is None:
            print(f"Failed to find sweep for {sid}. Tried: {tried}")
            continue

        config = None
        try:
            config = sweep.config
        except Exception:
            # fallback: inspect attributes
            try:
                config = sweep._attrs.get('config', None)
            except Exception:
                config = None

        out_path = os.path.join(out_dir, f"{sid}.yaml")
        with open(out_path, 'w') as f:
            yaml.safe_dump(dict(config=config), f)
        print(f"Saved sweep config for {sid} -> {out_path}")

if __name__ == '__main__':
    main()
