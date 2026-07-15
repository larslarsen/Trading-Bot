#!/usr/bin/env python3
"""Backfill DEX token daily OHLCV from GeckoTerminal's FREE public API (no key).

GeckoTerminal (CoinGecko's DEX terminal) serves DEX pool OHLC history for free:
  GET /networks/{chain}/tokens/{contract}/pools  -> token's pools (highest vol first)
  GET /networks/{chain}/pools/{pool}/ohlcv/day?limit=N -> daily OHLC [ts,o,h,l,c,vol]
No API key. Needs a browser User-Agent (default urllib UA gets 403).

For each token in dex_universe.csv (symbol, network, pool_address):
  - resolve its top pool on `chain` (highest volume)
  - fetch daily OHLC (limit=1000 ~ 3yr; pagination via limit/before)
  - normalize to ts,open,high,low,close,volume -> dex_data/<SYM>_1d_max.csv
Resume: skip tokens already <= 2023-01-01. Rate-limit: 1s between calls.

Usage:
    python backfill_dex_history.py [--sleep 1.0] [--chain eth] [--limit 1000]
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import urllib.request

DEX = Path("dex_data")
DEX.mkdir(exist_ok=True)
API = "https://api.geckoterminal.com/api/v2"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
DEEP_ENOUGH = pd.Timestamp("2023-01-01")


def _get(url, tries=5):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                wait = 5 * (2 ** i)  # 5,10,20,40,80s backoff
                print(f"    HTTP {e.code} -> backoff {wait}s")
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(5)
    raise last if last else RuntimeError("get failed")


def fetch_ohlcv(chain, pool, limit):
    url = f"{API}/networks/{chain}/pools/{pool}/ohlcv/day?limit={limit}"
    d = _get(url)
    rows = d.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
    if not rows:
        return None
    # GeckoTerminal: [timestamp, open, high, low, close, volume]
    out = [{"ts": pd.to_datetime(r[0], unit="s").strftime("%Y-%m-%d"),
            "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
           for r in rows]
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--chain", default="eth")
    ap.add_argument("--limit", type=int, default=1000, help="daily bars per token (~3yr)")
    ap.add_argument("--universe", default="dex_universe.csv")
    args = ap.parse_args()
    uni = Path(args.universe)
    if not uni.exists():
        print(f"ERROR: {uni} not found. Run build_dex_universe.py first.")
        return
    udf = pd.read_csv(uni)
    print(f"DEX history backfill (GeckoTerminal, free): {len(udf)} tokens from {uni} -> dex_data/")
    done = skipped = 0
    for _, row in udf.iterrows():
        sym = str(row["symbol"]).upper()
        net = str(row["network"])
        pool = str(row["pool_address"])
        out = DEX / f"{sym}_1d_max.csv"
        if out.exists():
            try:
                if pd.Timestamp(pd.read_csv(out)["ts"].min()) <= DEEP_ENOUGH:
                    skipped += 1
                    continue
            except Exception:
                pass
        df = fetch_ohlcv(net, pool, args.limit)
        if df is None or len(df) == 0:
            print(f"  {sym}: no OHLC")
            time.sleep(args.sleep)
            continue
        df.to_csv(out, index=False)
        done += 1
        print(f"  {sym}: {len(df)} bars -> {out.name} earliest={df['ts'].min()}")
        time.sleep(args.sleep)
    print(f"\nDEX backfill complete: {done} tokens, {skipped} already deep (skipped).")


if __name__ == "__main__":
    main()
