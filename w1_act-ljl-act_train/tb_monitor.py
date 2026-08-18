#!/usr/bin/env python3
import argparse, re, sys, time, os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def parse_step(line, step_interval):
    progress = re.findall(r'\|\s*([0-9,]+)/[0-9,]+\s*\[', line)
    if progress:
        value = int(progress[-1].replace(',', ''))
        return ((value + step_interval - 1) // step_interval) * step_interval
    m = re.search(r'\bstep:([0-9,.]+)([KMBTQ]?)\b', line)
    if not m:
        return None
    multipliers = {'': 1, 'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000,
                   'T': 1_000_000_000_000, 'Q': 1_000_000_000_000_000}
    value = float(m.group(1).replace(',', ''))
    return int(value * multipliers[m.group(2)])

def parse_metrics(line, step_interval=100):
    step = parse_step(line, step_interval)
    if step is None:
        return None, {}
    metrics = {}
    for m in re.finditer(r'(\w+):([0-9,.eE+-]+)', line):
        key = m.group(1)
        val_str = m.group(2).replace(',', '')
        if key in ('step','smpl','ep'): continue
        try:
            metrics[key] = float(val_str) if '.' in val_str or 'e' in val_str.lower() else int(val_str)
        except ValueError: continue
    return step, metrics

def reader_follow(path, poll_sec=2.0):
    while not os.path.exists(path):
        print('[TB] waiting for log file: ' + path, file=sys.stderr, flush=True)
        time.sleep(3)
    with open(path, 'r') as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line: yield line
            else: time.sleep(poll_sec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir', default='tb_logs')
    ap.add_argument('--flush-secs', type=int, default=30)
    ap.add_argument('--follow', default=None)
    ap.add_argument('--tags', default=None,
        help='white-list comma-separated metric names (e.g. loss,grad_norm)')
    ap.add_argument('--step-interval', type=int, default=100)
    args = ap.parse_args()
    if args.step_interval <= 0:
        raise ValueError('--step-interval must be greater than zero')
    allowed = set(args.tags.split(',')) if args.tags else None

    logdir = Path(args.logdir); logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logdir))
    last_flush, count = time.time(), 0
    reader = reader_follow(args.follow) if args.follow else None
    print('[TB] dir=' + str(logdir.resolve()) + ' follow=' + str(args.follow) + ' tags=' + str(allowed), file=sys.stderr, flush=True)
    if reader:
        for line in reader:
            step, metrics = parse_metrics(line, args.step_interval)
            if step is None or not metrics: continue
            for k, v in metrics.items():
                if allowed and k not in allowed:
                    continue
                writer.add_scalar('train/' + k, v, step)
            count += 1
            if time.time() - last_flush > args.flush_secs:
                writer.flush(); last_flush = time.time()
                print('[TB] ' + str(count) + ' lines, step=' + str(step), file=sys.stderr, flush=True)
    writer.close()

if __name__ == '__main__': main()
