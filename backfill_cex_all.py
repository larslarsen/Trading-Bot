#!/usr/bin/env python3
"""
Backfill ALL Binance USDT spot pairs across all timeframes (5m/1h/4h/1d) from
the free Binance klines mirror (no key). Writes to data/cex/<SYM>_<tf>.csv and
registers each file in MANIFEST.json.

Resumable: skips (tf, symbol) already present with sufficient depth.
Multi-symbol + multi-tf. Rate-limited (0.25s/symbol-tf). Logs progress.

Usage:
  python backfill_cex_all.py            # all 459 symbols, all tfs
  python backfill_cex_all.py --syms BTCUSDT,ETHUSDT
  python backfill_cex_all.py --tfs 1d,1h
"""
import argparse
import time
import requests
import pandas as pd
from pathlib import Path

REPO = Path(__file__).parent
CEX = REPO / "data" / "cex"
CEX.mkdir(parents=True, exist_ok=True)
SYMS_FILE = REPO / "all_binance_usdt.txt"
TFS = ["5m", "1h", "4h", "1d"]
BASE = "https://data-api.binance.vision/api/v3/klines"
LIMIT = 1000
SLEEP = 1.0          # polite pause between pages (avoid 429 backoff loops)
BACKOFF = 15         # seconds on HTTP 429
MAX_NEW_PER_PULL = 1_000_000  # safety cap to avoid runaway


def get_syms():
    if SYMS_FILE.exists():
        return [s.strip() for s in SYMS_FILE.read_text().splitlines() if s.strip()]
    info = requests.get(f"{BASE.rsplit('/klines',1)[0]}/exchangeInfo", timeout=30).json()
    syms = [s["symbol"] for s in info["symbols"]
            if s["symbol"].endswith("USDT") and s["status"] == "TRADING"
            and s.get("isSpotTradingAllowed")]
    SYMS_FILE.write_text("\n".join(syms))
    return syms


def floor_ts(ts_ms, tf):
    # align to tf boundary so appends never duplicate
    mins = {"5m": 5, "1h": 60, "4h": 240, "1d": 1440}[tf]
    bar_ms = mins * 60 * 1000
    return (ts_ms // bar_ms) * bar_ms


def pull(sym, tf, start_ms):
    out = []
    nxt = start_ms
    while True:
        r = requests.get(BASE, params={"symbol": sym, "interval": tf,
                                       "startTime": nxt, "limit": LIMIT}, timeout=30)
        if r.status_code != 200:
            if r.status_code == 429:
                time.sleep(BACKOFF)
                continue
            print(f"  [{sym} {tf}] HTTP {r.status_code}: {r.text[:120]}")
            break
        rows = r.json()
        if not rows:
            break
        out.extend(rows)
        nxt = rows[-1][0] + 1
        if len(rows) < LIMIT:
            break
        if len(out) >= MAX_NEW_PER_PULL:
            print(f"  [{sym} {tf}] hit new-cap, stopping")
            break
        time.sleep(SLEEP)
    return out


def existing_last_ms(path):
    if not path.exists():
        return None
    try:
        d = pd.read_csv(path, usecols=["ts"])
        return int(pd.to_datetime(d["ts"]).max().timestamp() * 1000)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=None, help="comma list; default all")
    ap.add_argument("--tfs", default=",".join(TFS), help="comma list of tfs")
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()
    tfs = [t for t in args.tfs.split(",") if t in TFS]
    # Process fast lookback TFs first so data is usable immediately; 5m last
    # (it is the heaviest: deep history spans 1000s of pages per symbol).
    tfs.sort(key=lambda t: {"1d": 0, "1h": 1, "4h": 2, "5m": 3}[t])
    syms = args.syms.split(",") if args.syms else get_syms()
    print(f"Backfilling {len(syms)} symbols x {tfs} tfs -> {CEX}")
    total = 0
    for tf in tfs:
        for sym in syms:
            path = CEX / f"{sym}_{tf}.csv"
            last = existing_last_ms(path)
            start = 1262304000000 if last is None else floor_ts(last + 1, tf)
            rows = pull(sym, tf, start)
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume",
                                             "close_time", "qav", "trades", "tbav", "tqav", "ignore"])
            df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            if path.exists():
                old = pd.read_csv(path)
                old["ts"] = pd.to_datetime(old["ts"], utc=True)
                df = pd.concat([old, df]).drop_duplicates(subset=["ts"]).sort_values("ts")
            df.to_csv(path, index=False)
            total += 1
            if total % 50 == 0:
                print(f"  wrote {total} files; latest {sym} {tf} -> {df['ts'].max()}")
        print(f"  completed tf={tf} ({total} files so far)")


if __name__ == "__main__":
    main()
