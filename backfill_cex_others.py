#!/usr/bin/env python3
"""
Bybit + OKX 5m historical backfill (FREE, NO API KEY).

Two top-tier CEX venues for cross-venue ML (we already have Binance +
Kraken-pending, plus Gate/KuCoin/Bitget/Coinbase via backfill_cex_all.py).
Both serve deep 5m klines via public REST, no key:
  - Bybit: /v5/market/kline  (walk backward via `end` param)
  - OKX:   /api/v5/market/history-candles  (walk backward via `after` param)

Writes SUFFIXED files data/<SYM>_5m_<venue>_max.csv (never clobber Binance).
Appends PER PAGE to the target so progress is visible + resumable (a crash
mid-symbol loses at most one page, not the whole symbol).

Rate-limit handling: OKX is strict (403 on rapid) -> 0.5s pace + curl retry.

Usage: python backfill_cex_others.py --venue bybit|okx [--symbols BTCUSDT,...] [--start 2021-01-01]
"""
import argparse
import json
import time
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent
OUT = REPO / "data"
UA = {"User-Agent": "Mozilla/5.0 (research backfill)"}


def fetch_json(url, timeout=30, tries=4):
    for i in range(tries):
        try:
            out = subprocess.run(["curl", "-s", "--max-time", str(timeout),
                                  "-A", UA["User-Agent"], url],
                                 capture_output=True, text=True, timeout=timeout + 5)
            if out.returncode != 0:
                time.sleep(min(2 ** i * 2, 30)); continue
            if not out.stdout.strip():
                time.sleep(min(2 ** i * 2, 30)); continue
            return json.loads(out.stdout), None
        except subprocess.TimeoutExpired:
            time.sleep(min(2 ** i * 5, 30)); continue
        except Exception as e:
            return None, str(e)[:120]
    return None, "exhausted retries"


def okx_klines(symbol, after_ms, tgt, limit=100):
    """Walk BACKWARD from now to after_ms, appending each page to tgt
    immediately (resumable + visible progress). OKX `after` = bars ts < after."""
    cursor = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    seen = set()
    if tgt.exists():
        try:
            seen = set(pd.read_csv(tgt, parse_dates=["ts"])["ts"].view("int64") // 10**9)
        except Exception:
            seen = set()
    while True:
        u = (f"https://www.okx.com/api/v5/market/history-candles?instId={symbol.replace('USDT','-USDT')}"
             f"&bar=5m&limit={limit}&after={cursor}")
        d, err = fetch_json(u)
        if err:
            return None, err
        if d.get("code") != "0":
            return None, str(d.get("msg"))
        page = d["data"]
        if not page:
            break
        rows = []
        oldest = int(page[-1][0])
        for r in page:
            ts = int(r[0]) // 1000
            if ts in seen:
                continue
            seen.add(ts)
            rows.append([pd.to_datetime(ts, unit="s", utc=True),
                         float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
        if rows:
            pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).to_csv(
                tgt, mode="a", header=not tgt.exists(), index=False)
        if oldest <= after_ms:
            break
        cursor = oldest
        time.sleep(0.5)
    return True, None


def bybit_klines(symbol, start_ms, tgt, limit=1000):
    """Bybit /v5/market/kline: walk backward from now via `end` param,
    appending each page to tgt immediately (resumable + visible progress)."""
    end = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    seen = set()
    if tgt.exists():
        try:
            seen = set(pd.read_csv(tgt, parse_dates=["ts"])["ts"].view("int64") // 10**9)
        except Exception:
            seen = set()
    pages = 0
    while pages < 5000:
        u = (f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}"
             f"&interval=5&limit={limit}&end={end}")
        d, err = fetch_json(u)
        if err:
            return None, err
        if d.get("retCode") != 0:
            return None, d.get("retMsg")
        page = d["result"]["list"]
        if not page:
            break
        rows = []
        oldest = int(page[-1][0])
        for row in page:
            ts = int(row[0]) // 1000
            if ts in seen:
                continue
            seen.add(ts)
            rows.append([pd.to_datetime(ts, unit="s", utc=True),
                         float(row[1]), float(row[2]), float(row[3]),
                         float(row[4]), float(row[5])])
        if rows:
            pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).to_csv(
                tgt, mode="a", header=not tgt.exists(), index=False)
        pages += 1
        if oldest <= start_ms or len(rows) == 0:
            break
        end = oldest - 1
        time.sleep(0.3)
    return True, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True, choices=["bybit", "okx"])
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,DOGEUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,LINKUSDT")
    ap.add_argument("--start", default="2021-01-01")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)

    for sym in syms:
        tgt = OUT / f"{sym}_5m_{args.venue}_max.csv"
        print(f"[{args.venue}] {sym}: from {pd.Timestamp(start_ms, unit='ms', tz='UTC')} (append per page)", flush=True)
        if args.venue == "bybit":
            ok, err = bybit_klines(sym, start_ms, tgt)
        else:
            ok, err = okx_klines(sym, start_ms, tgt)
        if err:
            print(f"  ERR {err}", flush=True)
            continue
        # sort ascending + dedupe (API returns newest-first per page)
        if tgt.exists():
            d = pd.read_csv(tgt, parse_dates=["ts"]).sort_values("ts").drop_duplicates("ts")
            d.to_csv(tgt, index=False)
        n = 0
        if tgt.exists():
            n = sum(1 for _ in open(tgt)) - 1
        print(f"  wrote {n} rows -> {tgt.name}", flush=True)
    print(f"{args.venue} backfill done.", flush=True)


if __name__ == "__main__":
    main()
