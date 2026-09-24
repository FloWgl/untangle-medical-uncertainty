#!/usr/bin/env python3
"""Fetch candidate W&B artifacts referenced in the CIFAR-10 rerun plan.

Usage:
  # login first (recommended):
  wandb login

  # quick map generation (no W&B API calls):
  python3 tools/fetch_wandb_artifacts.py --plan docs/cifar10_rerun_plan.csv --out docs/cifar10_artifact_map.csv

  # attempt to locate and download matching checkpoint files from the W&B project
  python3 tools/fetch_wandb_artifacts.py --plan docs/cifar10_rerun_plan.csv --out docs/cifar10_artifact_map.csv --download --project bmucsanyi/untangle

The script will NOT accept or print API keys; authenticate locally using `wandb login` or
`export WANDB_API_KEY=...` before running with --download.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict


def parse_plan(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def split_paths(s):
    if not s:
        return []
    # weight_paths_sample may contain ';' separators or commas
    parts = []
    for semi in s.split(';'):
        for comma in semi.split(','):
            p = comma.strip().strip('"')
            if p:
                parts.append(p)
    return parts


def build_map(rows):
    mapping = []
    seen = set()
    for r in rows:
        sweep = r.get('sweep','')
        method = r.get('method_name','')
        sample = r.get('weight_paths_sample','')
        for p in split_paths(sample):
            key = (p,)
            if key in seen:
                continue
            seen.add(key)
            mapping.append({
                'sweep': sweep,
                'method': method,
                'remote_path': p,
                'basename': os.path.basename(p),
                'wandb_run': '',
                'local_path': '',
                'status': 'pending',
            })
    return mapping


def write_map(mapping, outpath):
    fieldnames = ['sweep','method','remote_path','basename','wandb_run','local_path','status']
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in mapping:
            writer.writerow(m)


def attempt_wandb_download(mapping, project, dest_dir='artifacts', timeout=29, max_runs=None):
    try:
        import wandb
    except Exception as e:
        print('wandb SDK not available; install with `pip install wandb` to enable --download', file=sys.stderr)
        return mapping
    api = wandb.Api(timeout=timeout)
    print(f'Listing runs in project {project} (this may be slow)...')
    try:
        # optionally limit the number of runs to inspect to avoid long queries
        if max_runs is None:
            runs = list(api.runs(project))
        else:
            runs = []
            for i, run in enumerate(api.runs(project)):
                runs.append(run)
                if i+1 >= max_runs:
                    break
    except Exception as e:
        print('Failed to list runs from W&B. Ensure you are logged in (wandb login) and the project exists.', file=sys.stderr)
        return mapping

    basename_to_runs = defaultdict(list)
    for run in runs:
        try:
            files = run.files()
        except Exception:
            files = []
        for f in files:
            basename_to_runs[os.path.basename(f.name)].append((run, f))

    os.makedirs(dest_dir, exist_ok=True)

    for m in mapping:
        base = m['basename']
        candidates = basename_to_runs.get(base, [])
        if not candidates:
            m['status'] = 'not-found'
            continue
        # pick first candidate and download
        run, f = candidates[0]
        try:
            print(f'Downloading {f.name} from run {run.path}...')
            dl_path = f.download(root=dest_dir)
            m['wandb_run'] = run.path
            m['local_path'] = dl_path
            m['status'] = 'downloaded'
        except Exception as e:
            m['status'] = 'download-failed'

    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, help='Path to docs/cifar10_rerun_plan.csv')
    parser.add_argument('--out', required=True, help='Output mapping CSV')
    parser.add_argument('--download', action='store_true', help='Attempt to locate and download matching files from W&B')
    parser.add_argument('--project', default=None, help='W&B project (owner/project) to search (deprecated in favor of --projects)')
    parser.add_argument('--projects', default=None, help='Comma-separated list of W&B projects (owner/project) to search')
    parser.add_argument('--timeout', type=int, default=29, help='W&B API timeout in seconds (default 29)')
    parser.add_argument('--max-runs', type=int, default=None, help='Maximum number of runs to inspect (helps avoid long GraphQL queries)')
    args = parser.parse_args()

    rows = parse_plan(args.plan)
    mapping = build_map(rows)
    if args.download:
        projects_to_try = []
        if args.projects:
            projects_to_try = [p.strip() for p in args.projects.split(',') if p.strip()]
        if args.project:
            projects_to_try.append(args.project)
        if not projects_to_try:
            projects_to_try = ['bmucsanyi/untangle']
        # try each project until mapping entries are downloaded or all tried
        for proj in projects_to_try:
            print(f'--- Searching project: {proj} ---')
            mapping = attempt_wandb_download(mapping, proj, timeout=args.timeout, max_runs=args.max_runs)
    write_map(mapping, args.out)
    print(f'Wrote mapping to {args.out} ({len(mapping)} entries)')


if __name__ == '__main__':
    main()
