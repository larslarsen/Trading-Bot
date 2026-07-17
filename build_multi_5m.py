#!/usr/bin/env python3
"""Rebuild multi_5m.csv from ALL CEX assets (minimizing stablecoins) so
single-pair ML models train on EVERY other asset's 5m data as cross-asset
features. No narrowing to a top-K basket — the user requires ALL data.

Reads data/cex/<SYM>_5m.csv for each symbol (produced by backfill_cex_all.py).
BTC is seeded from btc_5m.csv (deepest). The target's own symbol is included
so cross features (e.g. DOGE_btc_ratio) are computable. XGBoost handles the
resulting wide feature matrix (hundreds of assets x cross-features) fine.

Usage:
  python build_multi_5m.py            # all non-stable USDT pairs in data/cex + BTC
  python build_multi_5m.py --syms BTCUSDT,ETHUSDT,DOGEUSDT,SOLUSDT
"""
import argparse
import pandas as pd
from pathlib import Path

REPO = Path(__file__).parent
CEX = REPO / "data" / "cex"
DATADIR = REPO / "data"
OUT = REPO / "multi_5m.csv"
STABLES = {"USDCUSDT", "FDUSDUSDT", "RLUSDUSDT", "USD1USDT", "EURUSDT",
           "UUSDT", "TUSDUSDT", "DAIUSDT", "BUSDUSDT", "USDPUSDT", "AEURUSDT"}


def all_syms():
    # BTC is seeded from the deepest dedicated root file; every other USDT pair
    # lives at data/<SYM>USDT_5m_max.csv (the canonical path the poller +
    # pipeline.fetch_data() use). Scan that dir, not the legacy data/cex tree
    # (which no longer holds the screener symbols -> multi_5m.csv ended with
    # only BTCUSDT and every model ranked FLAT).
    syms = ["BTCUSDT"]
    for p in DATADIR.glob("*_5m_max.csv"):
        s = p.stem.replace("_5m_max", "").replace("USDT", "") + "USDT"
        if s not in STABLES:
            syms.append(s)
    return syms


def load_5m(sym):
    # BTC has the deepest dedicated file; others from the canonical data/ path.
    if sym == "BTCUSDT":
        p = REPO / "btc_5m.csv"
    else:
        stem = sym.replace("USDT", "")
        p = DATADIR / f"{stem}USDT_5m_max.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, parse_dates=["ts"])
    if d["ts"].dt.tz is None:
        d["ts"] = d["ts"].dt.tz_localize("UTC")
    d["symbol"] = sym
    return d[["ts", "symbol", "open", "high", "low", "close", "volume"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=None)
    args = ap.parse_args()
    syms = args.syms.split(",") if args.syms else all_syms()
    frames = []
    for sym in syms:
        d = load_5m(sym)
        if d is None:
            print(f"[skip] {sym}: no 5m file yet")
            continue
        frames.append(d)
        print(f"[ok] {sym}: {len(d)} bars ({d['ts'].min().date()} -> {d['ts'].max().date()})")
    if not frames:
        print("No data.")
        return
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["ts", "symbol"]).sort_values(["symbol", "ts"]).reset_index(drop=True)
    merged = merged.dropna(subset=["ts", "open", "high", "low", "close"])
    cols = ["ts", "symbol", "open", "high", "low", "close", "volume"]
    merged = merged[cols]
    merged.to_csv(OUT, index=False)
    print(f"Wrote {len(merged)} rows -> {OUT}")
    print(f"Symbols ({len(merged['symbol'].unique())}): {sorted(merged['symbol'].unique())}")


if __name__ == "__main__":
    main()
