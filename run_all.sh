#!/bin/sh
# Re-trains the attention model with epoch snapshots, then the no-attention baseline (same seed / settings).
# Safe to re-run: finished runs (those with metrics.json) are skipped.
cd /home/claude/nmt
[ -f outputs/attn_run/metrics.json ] || /tmp/env/bin/python train.py --epochs 7 --out-dir outputs/attn_run --snapshot-epochs 1 2 3 5 7 > outputs/attn_run.log 2>&1
[ -f outputs/baseline_run/metrics.json ] || /tmp/env/bin/python train.py --epochs 7 --out-dir outputs/baseline_run --no-attention > outputs/baseline_run.log 2>&1
