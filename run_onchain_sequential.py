#!/usr/bin/env python3
"""Serialize the on-chain backfills so they don't OOM (one chain at a time).
The earlier parallel launch (5 concurrent ~2GB procs + poller + BTC) caused
the 2026-07-16 10:08 OOM that killed the poller. Running sequentially keeps
peak memory bounded. Each chain writes its full history to
data/onchain/<CHAIN>_features_daily.csv (resumable via done-set in main())."""
import subprocess, sys, time

REPO = "/home/lars/trading-bot"
CHAINS = ["btc", "eth", "base", "arbitrum", "aptos", "provenance", "xrp"]

for c in CHAINS:
    print(f"\n=== on-chain backfill: {c} ({time.strftime('%H:%M:%S')}) ===", flush=True)
    # run to completion (resumes if partial), then free memory before next
    subprocess.run(
        [f"{REPO}/.venv/bin/python", "-u", f"{REPO}/backfill_onchain.py",
         "--chain", c],
        cwd=REPO,
    )
    print(f"=== {c} done ({time.strftime('%H:%M:%S')}) ===", flush=True)
print("\nALL ON-CHAIN BACKFILLS SEQUENTIAL DONE", flush=True)
