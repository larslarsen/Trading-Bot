#!/usr/bin/env python3
"""Backfill daily OHLCV bars for coins missing from local data, using FREE
public exchange APIs (no API key) via ccxt.

Purpose: close the survivorship gap the PIT backtest fix exposed. Coins that
existed on a past screen date but delisted before/within our data window have
no MEXC/BloFin bars. Other exchanges (KuCoin, Bitget, HTX, Bybit) often still
list them and serve deep daily history on their PUBLIC endpoints for $0.

For each symbol appearing in any dated PIT screen (backtest_output/screen_liqu_idio_*.csv)
that lacks data/<SYM>USDT_1d_max.csv, fetch daily OHLCV (limit=450 ~ 2025-03..now)
from the first free exchange that has it, and write the file in the exact format
the screen pipeline expects (ts,open,high,low,close,volume; ts = ISO date).

Free-tier safe: 1.2s between fetches, 429 backoff, resume (skip existing files).
No API key required for public fetch_ohlcv on these exchanges.

Usage:
    python backfill_delisted_bars.py [--limit 450] [--sleep 1.2]
"""
import argparse
import time
from pathlib import Path

import ccxt
import pandas as pd

ROOT = Path("data")
SCR = Path("backtest_output")
# Free exchanges with public OHLCV (no key). Order = preference.
FREE_EXCHANGES = ["kucoin", "bitget", "htx", "bybit", "mexc"]


def needed_symbols():
    """All base symbols across every dated PIT screen that lack local bars under
    ANY quote (USDT/USDC/USD/USDE...). Match by file stem starting with the
    base symbol (quote suffix varies: MEXC=USDT, BloFin=USD, etc.)."""
    have_bases = set()
    for p in ROOT.glob("*_1d_max.csv"):
        stem = p.name.replace("_1d_max.csv", "").upper()
        # strip a trailing known quote suffix to recover the base
        for q in ("USDT", "USDC", "USD", "USDE", "USDX"):
            if stem.endswith(q) and len(stem) > len(q):
                have_bases.add(stem[: -len(q)])
                break
        else:
            have_bases.add(stem)  # no recognised quote -> treat whole stem as base
    need = set()
    for f in SCR.glob("screen_liqu_idio_*.csv"):
        df = pd.read_csv(f)
        if "symbol" not in df:
            continue
        for s in df["symbol"].dropna().astype(str).str.strip().str.upper():
            if not s:
                continue
            if s in have_bases:
                continue
            need.add(s)
    return sorted(need)


def fetch_one(sym, limit, sleep):
    """Try free exchanges for SYM/USDT daily bars; write file; return path or None."""
    for exname in FREE_EXCHANGES:
        try:
            ex = getattr(ccxt, exname)()
        except Exception:
            continue
        if not ex.has.get("fetchOHLCV"):
            continue
        pair = f"{sym}/USDT"
        try:
            ohlcv = ex.fetch_ohlcv(pair, "1d", limit=limit)
        except Exception as e:
            # symbol not on this exchange, or rate-limited -> try next
            time.sleep(sleep)
            continue
        if not ohlcv:
            time.sleep(sleep)
            continue
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms").dt.tz_localize(None).dt.strftime("%Y-%m-%d")
        out = ROOT / f"{sym}USDT_1d_max.csv"
        df.to_csv(out, index=False)
        print(f"  {exname} {pair}: {len(df)} bars -> {out.name}")
        return out
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=450, help="daily bars to fetch (~2025-03..now)")
    ap.add_argument("--sleep", type=float, default=1.2, help="seconds between fetches")
    args = ap.parse_args()

    syms = needed_symbols()
    print(f"Backfilling bars for {len(syms)} symbols missing local bars (free exchanges)")
    done = 0
    for i, sym in enumerate(syms, 1):
        out = ROOT / f"{sym}USDT_1d_max.csv"
        if out.exists():
            continue  # resume
        print(f"[{i}/{len(syms)}] {sym}", flush=True, end="")
        res = fetch_one(sym, args.limit, args.sleep)
        if res:
            done += 1
        else:
            print("  (no free exchange has it)")
        time.sleep(args.sleep)
    print(f"\nBackfill complete: {done} new bar files written to {ROOT}")


if __name__ == "__main__":
    main()
