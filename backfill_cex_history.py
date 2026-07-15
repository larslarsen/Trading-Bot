#!/usr/bin/env python3
"""Expand CEX daily-OHLCV history for FREE, prioritizing exchanges by how far
back data is retrievable. This compounds: deeper history is gathered now (no
paid back-data later) and feeds both the PIT backtests and the ML project.

Why: our MEXC/BloFin forward collection starts ~2025-03. Free public exchanges
reach much further (measured): bitfinex ~2019, kucoin ~2022, binance/bybit/htx
~2023. Many (bitget/okx/coinbase/gemini/upbit/poloniex) only reach ~2025-03 or
later -> NO better than what we have, so they are excluded (stop where marginal
exchanges add nothing).

For each target symbol (broad universe + all PIT screens):
  - fetch S/USDT daily from every PRIORITY exchange that lists it (union bars
    across exchanges fills gaps), limit=2000 (max free depth per call).
  - merge with the existing local <S>USDT_1d_max.csv (if any) by ts, dedup, sort.
  - rewrite -> extends history BACKWARD in place; forward bars still come from
    the collector, so no data is lost.
Resume: skip symbols whose file already starts <= 2023-01-01 (deep enough).
Rate-limit: 1.2s between fetches; 429/error -> skip that exchange, continue.
No API key required (public endpoints).

Usage:
    python backfill_cex_history.py [--sleep 1.2]
"""
import argparse
import time
from pathlib import Path

import ccxt
import pandas as pd

ROOT = Path("data")
SCR = Path("backtest_output")
# All reachable free public exchanges (no API key). Ordered deepest-first so the
# union fills gaps with the longest history first, but EVERY exchange contributes
# its bars -- breadth matters: a coin listed only on a shallow exchange (e.g.
# bitget/okx/coinbase) is unreachable from the deep ones, so exclude none.
# (gateio is not a ccxt class name; mexc is our live forward source but included
# here for backfill completeness.)
PRIORITY = ["bitfinex", "kucoin", "binance", "bybit", "htx", "hitbtc", "bitstamp",
            "kraken", "okx", "bitget", "coinbase", "gemini", "upbit", "poloniex",
            "bitmart", "bingx", "mexc"]
DEEP_ENOUGH = pd.Timestamp("2023-01-01")  # skip symbols already this deep


def target_symbols():
    """Broad universe + every PIT screen symbol."""
    syms = set()
    # broad universe
    bu = ROOT / "universe_broad.csv"
    if bu.exists():
        df = pd.read_csv(bu)
        if "symbol" in df:
            syms |= set(df["symbol"].dropna().astype(str).str.strip().str.upper())
    for f in SCR.glob("screen_liqu_idio_*.csv"):
        df = pd.read_csv(f)
        if "symbol" in df:
            syms |= set(df["symbol"].dropna().astype(str).str.strip().str.upper())
    return sorted(s for s in syms if s)


def fetch_exchange(sym, exname, sleep):
    try:
        ex = getattr(ccxt, exname)()
    except Exception:
        return None
    if not ex.has.get("fetchOHLCV"):
        return None
    try:
        o = ex.fetch_ohlcv(f"{sym}/USDT", "1d", limit=2000)
    except Exception:
        time.sleep(sleep)
        return None
    if not o:
        return None
    df = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms").dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    return df


def merge_symbol(sym, sleep):
    out = ROOT / f"{sym}USDT_1d_max.csv"
    # existing local bars (any quote variant) to union with
    existing = None
    for q in ("USDT", "USD", "USDC"):
        p = ROOT / f"{sym}{q}_1d_max.csv"
        if p.exists():
            existing = pd.read_csv(p)
            break
    frames = [existing] if existing is not None else []
    earliest = pd.Timestamp("2100-01-01")
    for exname in PRIORITY:
        df = fetch_exchange(sym, exname, sleep)
        if df is not None and len(df):
            frames.append(df)
            earliest = min(earliest, pd.Timestamp(df["ts"].min()))
        time.sleep(sleep)
    if len(frames) == 0:
        return None, None
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset="ts").sort_values("ts")
    merged.to_csv(out, index=False)
    return out, earliest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=1.2)
    args = ap.parse_args()
    syms = target_symbols()
    print(f"CEX history backfill: {len(syms)} target symbols, priority={PRIORITY}")
    done = skipped = 0
    for i, sym in enumerate(syms, 1):
        out = ROOT / f"{sym}USDT_1d_max.csv"
        # resume: already deep enough?
        if out.exists():
            try:
                cur = pd.read_csv(out)["ts"].min()
                if pd.Timestamp(cur) <= DEEP_ENOUGH:
                    skipped += 1
                    continue
            except Exception:
                pass
        res, earliest = merge_symbol(sym, args.sleep)
        if res:
            done += 1
            print(f"[{i}/{len(syms)}] {sym}: -> {res.name} earliest={earliest.date()}")
        else:
            print(f"[{i}/{len(syms)}] {sym}: no free exchange has it")
    print(f"\nBackfill complete: {done} extended, {skipped} already deep enough (skipped).")


if __name__ == "__main__":
    main()
