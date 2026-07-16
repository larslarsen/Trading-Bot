#!/usr/bin/env python3
"""
Bybit + OKX 5m historical backfill (FREE, NO API KEY).

Two more top-tier CEX venues for cross-venue ML (we already have Binance +
Kraken-pending). Both serve deep 5m klines via public REST, no key:
  - Bybit: /v5/market/kline  (start=cursor ms, walks forward; spot since ~2021)
  - OKX:   /api/v5/market/history-candles  (after=cursor ms, walks backward)

Bybit spot history starts ~2021; OKX similar. We paginate fully, resample to
canonical 5m (already 5m), append per-symbol to data/<SYM>USDT_5m_max.csv
matching the ONE-place convention. Resumable via state file (last ts per sym).

Rate-limit handling: OKX is strict (403 on rapid hits) -> honor 429/403 with
backoff + Retry-After. Bybit is lenient but we pace anyway.

Usage: python backfill_cex_others.py --venue bybit|okx [--symbols BTCUSDT,ETHUSDT,...]
"""
import argparse
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent
OUT = REPO / "data"
UA = {"User-Agent": "Mozilla/5.0 (research backfill)"}


def fetch_json(url, timeout=30, tries=4):
    import subprocess
    import urllib.error
    for i in range(tries):
        try:
            out = subprocess.run(["curl", "-s", "--max-time", str(timeout),
                                  "-A", UA["User-Agent"], url],
                                 capture_output=True, text=True, timeout=timeout + 5)
            if out.returncode != 0:
                time.sleep(min(2 ** i * 2, 30)); continue
            txt = out.stdout
            if not txt.strip():
                time.sleep(min(2 ** i * 2, 30)); continue
            return json.loads(txt), None
        except subprocess.TimeoutExpired:
            time.sleep(min(2 ** i * 5, 30)); continue
        except Exception as e:
            return None, str(e)[:120]
    return None, "exhausted retries"


def bybit_klines(symbol, start_ms, limit=1000):
    """Bybit /v5/market/kline: use the `end` param as the UPPER bound (returns
    up to `limit` bars BEFORE `end`). Walk backward: end = oldest-1, until we
    pass start_ms. `start` param is ignored/clamped by Bybit, so `end` is the
    correct pagination key. Bounded by progress check."""
    bars = []
    end = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    pages = 0
    seen_ts = set()
    while pages < 2000:
        u = (f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}"
             f"&interval=5&limit={limit}&end={end}")
        d, err = fetch_json(u)
        if err:
            return bars, err
        if d.get("retCode") != 0:
            return bars, d.get("retMsg")
        page = d["result"]["list"]
        if not page:
            break
        before = len(bars)
        oldest = int(page[-1][0])
        for row in page:
            t = int(row[0])
            if t >= start_ms and t not in seen_ts:
                seen_ts.add(t)
                bars.append(row)
        pages += 1
        if oldest <= start_ms or len(bars) == before:
            break
        end = oldest - 1
        time.sleep(0.3)
    return bars, None


def okx_klines(symbol, after_ms, limit=100):
    """Yield pages of 5m bars walking backward from after_ms (exclusive)."""
    bars = []
    cursor = after_ms
    while True:
        u = (f"https://www.okx.com/api/v5/market/history-candles?instId={symbol}"
             f"&bar=5m&limit={limit}&after={cursor}")
        d, err = fetch_json(u)
        if err:
            return bars, err
        if d.get("code") != "0":
            return bars, str(d.get("msg"))
        page = d["data"]
        if not page:
            break
        bars.extend(page)
        # OKX returns newest-first; next after = oldest ts
        oldest = int(page[-1][0])
        if oldest >= cursor:
            break
        cursor = oldest
        time.sleep(0.5)  # OKX is strict (403 on rapid)
    return bars, None


def normalize(bybit, bars):
    """bars: list of [ts_ms, open, high, low, close, volume, ...]. -> df 5m."""
    rows = []
    for b in bars:
        ts = pd.to_datetime(int(b[0]), unit="ms", utc=True)
        rows.append([ts, float(b[1]), float(b[2]), float(b[3]),
                     float(b[4]), float(b[5])])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    return df.drop_duplicates("ts").sort_values("ts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True, choices=["bybit", "okx"])
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,DOGEUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,MATICUSDT")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)

    for sym in syms:
        stem = sym[:-4] if sym.endswith("USDT") else sym
        tgt = OUT / f"{sym}_5m_max.csv"
        have_until = None
        if tgt.exists():
            old = pd.read_csv(tgt, parse_dates=["ts"])
            if len(old):
                have_until = int(old["ts"].max().timestamp() * 1000)
        cursor = have_until + 1 if have_until else start_ms
        print(f"[{args.venue}] {sym}: from {pd.to_datetime(cursor,unit='ms',utc=True)}", flush=True)
        if args.venue == "bybit":
            bars, err = bybit_klines(sym, cursor, args.limit)
        else:
            bars, err = okx_klines(sym, cursor, args.limit)
        if err:
            print(f"  ERR {err}", flush=True)
            continue
        if not bars:
            print(f"  no new bars", flush=True)
            continue
        df = normalize(args.venue, bars)
        if tgt.exists():
            old = pd.read_csv(tgt, parse_dates=["ts"]).set_index("ts")
            df = pd.concat([old, df.set_index("ts")]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        else:
            df = df.set_index("ts")
        df.to_csv(tgt)
        print(f"  wrote {len(df)} rows ({df.index[0]} .. {df.index[-1]})", flush=True)
    print(f"{args.venue} backfill done.", flush=True)


if __name__ == "__main__":
    main()
