#!/usr/bin/env python3
"""Normalize `docs/paper_results.csv` into `docs/paper_results_normalized.csv`.
Creates canonical metric names and units.
"""
import csv
from pathlib import Path

IN = Path('docs/paper_results.csv')
OUT = Path('docs/paper_results_normalized.csv')

def clean(x):
    return x.strip() if x is not None else ''

rows = []
with IN.open() as f:
    reader = csv.reader(f)
    header = next(reader)
    for r in reader:
        # pad
        r = r + ['']*(6-len(r))
        table, metric, method, mean, mn, mx = [clean(x) for x in r[:6]]
        if table.startswith('I.') and 'forward_time' in metric:
            rows.append([table, method, 'forward_time_s', '', mean, 's'])
        elif table.startswith('I.') and 'forward_time' in metric.lower():
            rows.append([table, method, 'forward_time_s', '', mean, 's'])
        elif table.startswith('H.') and metric.startswith('IT_decomp'):
            # mean holds fraction like '10%'
            fraction = mean
            value = mn or mx
            # metric name from metric field (e.g., IT_decomp/Deep)
            sub = metric.split('/')[-1] if '/' in metric else metric
            # denote as it_decomp_<sub>
            rows.append([table, method, f'it_decomp_{sub.lower()}', fraction, value, 'unitless'])
        elif table.startswith('H.') and metric=='rank_corr':
            # method holds metric name, value in mx or mn
            value = mx or mn or mean
            rows.append([table, method, 'rank_correlation', '', value, 'unitless'])
        else:
            # fallback: record mean if numeric
            val = mean or mn or mx
            if val:
                rows.append([table, method or metric, metric, '', val, ''])

with OUT.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['table','method','metric','fraction','value','unit'])
    for row in rows:
        w.writerow(row)

print('Wrote', OUT)
