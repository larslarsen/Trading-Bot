#!/usr/bin/env python3
"""
Edge-case watchdog for the trading-bot logs.

Scans new log content since the last run (byte-offset state file), filters out
KNOWN-BENIGN noise (Binance 429 throttle, DexScreener 401 pagination cap, dead
Bybit process, DEX snapshot skips, informational "insufficient symbols"), and
reports anything that looks like a real new edge case (tracebacks, worker deaths,
parse errors, OOM/guard trips, file/permission errors, uncaught worker errors).

Stdlib-only so it runs in the cron session without venv assumptions.
Output: a concise report. Exit 0 always (watchdog pattern — never alarms the
scheduler; the report text is the signal).
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

REPO = "/home/lars/trading-bot"
LOG_DIR = os.path.join(REPO, "logs")
STATE = os.path.join(LOG_DIR, ".edgecase_state.json")

LOGS = [
    "data_poller.log",
    "cex_ml_xgb_5m.log",
    "cex_ml_xgb_1d.log",
    "dex_ml_xgb_1d.log",
    "cex_multi_screen_1d.log",
    "dex_screen_1d.log",
    "bulk_backfill_5m.log",
    "collector_daemon.log",
    "micro_poller.log",
]

# Lines matching these are KNOWN-BENIGN -> always skipped.
BENIGN = [
    re.compile(r"HTTP 429"),
    re.compile(r"backoff"),
    re.compile(r"page\d+\s+ERR:\s*<HTTPError 401", re.I),
    re.compile(r"Too many visits"),
    re.compile(r"retCode.{0,12}10006"),
    re.compile(r"DEX 5m snapshot.*skipped"),
    re.compile(r"insufficient symbols found"),
    re.compile(r"^\s*$"),
    # Python 3.14 GC cleanup noise on closed HTTP responses (confirmed harmless)
    re.compile(r"Exception ignored while finalizing"),
    re.compile(r"ValueError: I/O operation on closed file"),
    re.compile(r"Traceback \(most recent call last\):\s*$"),
    re.compile(r"File \".*http/client\.py\""),
    re.compile(r"http\.client\."),
]

# Lines matching these are REAL edge cases -> always reported.
ALERT = [
    re.compile(r"Traceback"),
    re.compile(r"\bException\b"),
    re.compile(r"worker died"),
    re.compile(r"doesn't match format"),
    re.compile(r"\bOOM\b|Killed|out of memory", re.I),
    re.compile(r"Permission denied"),
    re.compile(r"FileNotFoundError"),
    re.compile(r"No space left"),
    re.compile(r"MEMORY GUARD"),
    re.compile(r"KeyError|ValueError|TypeError|IndexError"),
]

# Poller worker caught-errors -> reported (tagged transient) so new ones surface.
WORKER_ERR = re.compile(r"\[\w+\]\s+(?:error|history error):", re.I)


def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    try:
        with open(STATE, "w") as f:
            json.dump(st, f)
    except Exception:
        pass


def scan():
    st = load_state()
    now = datetime.now(timezone.utc)
    report = []
    for name in LOGS:
        path = os.path.join(LOG_DIR, name)
        if not os.path.exists(path):
            continue
        off = int(st.get(name, 0))
        try:
            size = os.path.getsize(path)
        except Exception:
            continue
        if off > size:  # log rotated/truncated
            off = 0
        with open(path, "r", errors="replace") as f:
            f.seek(off)
            new = f.read()
        st[name] = size
        if not new.strip():
            continue
        for line in new.splitlines():
            if not line.strip():
                continue
            if any(b.search(line) for b in BENIGN):
                continue
            sev = None
            if any(a.search(line) for a in ALERT):
                sev = "ALERT"
            elif WORKER_ERR.search(line):
                sev = "worker(transient?)"
            elif re.search(r"error|fail|exception|traceback|timeout|refused|denied", line, re.I):
                sev = "suspect"
            if sev:
                report.append(f"  [{sev}] {name}: {line.strip()}")
    save_state(st)
    print(f"edge-case scan @ {now:%Y-%m-%d %H:%M:%S} UTC")
    if report:
        print(f"NEW ISSUES ({len(report)}):")
        for r in report[:50]:
            print(r)
        if len(report) > 50:
            print(f"  ... and {len(report) - 50} more")
    else:
        print("CLEAN — no new edge cases since last scan.")


if __name__ == "__main__":
    scan()
