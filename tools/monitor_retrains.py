#!/usr/bin/env python3
"""Monitor retrain jobs and update docs/paper_methods_status.csv

Usage: python3 tools/monitor_retrains.py
This scans pids/, docs/retrain_reports/summary.csv, and logs/retrain_all_sequential.out
and updates the status/pid/log fields in the CSV.
"""
import csv
from pathlib import Path
import re

ROOT = Path('.').resolve()
STATUS_CSV = ROOT / 'docs' / 'paper_methods_status.csv'
SUMMARY = ROOT / 'docs' / 'retrain_reports' / 'summary.csv'
PIDS_DIR = ROOT / 'pids'
MASTER_LOG = ROOT / 'logs' / 'retrain_all_sequential.out'

def load_summary():
    s = {}
    if SUMMARY.exists():
        with SUMMARY.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                s[r['script']] = r
    return s

def master_started_scripts():
    started = set()
    if MASTER_LOG.exists():
        text = MASTER_LOG.read_text(errors='ignore')
        # lines like '=== Starting retrain_42thx27s at ...'
        for m in re.finditer(r"=== Starting ([^ ]+) at", text):
            started.add(m.group(1))
    return started

def pid_is_running(pid):
    try:
        pid = int(pid)
    except Exception:
        return False
    import os
    return Path(f'/proc/{pid}').exists()

def update():
    summary = load_summary()
    started = master_started_scripts()
    rows = []
    with STATUS_CSV.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            launcher = r['launcher']
            # try to find matching script in summary (script names may have .sh)
            script = launcher if launcher.endswith('.sh') else launcher
            # check summary
            sumr = summary.get(script) or summary.get(launcher + '.sh') or None
            status = r.get('status','not-started')
            pid = r.get('pid','')
            log = r.get('log','')
            notes = r.get('notes','')
            # update from summary
            if sumr:
                status = sumr.get('status', status)
                pid = sumr.get('pid', pid)
                # keep log path
            else:
                # if master log shows starting this launcher, mark queued->running
                name_noext = launcher.replace('.sh','')
                if name_noext in started:
                    status = 'running'
                    # pid likely master pid; leave pid blank
            # check pid liveness if pid present
            if pid and pid_is_running(pid):
                status = 'running'
            elif pid and not pid_is_running(pid) and status in ('running','started'):
                status = 'exited'

            r['status'] = status
            r['pid'] = pid
            r['log'] = log
            r['notes'] = notes
            rows.append(r)

    # write back
    with STATUS_CSV.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':
    update()
    print('Updated', STATUS_CSV)
