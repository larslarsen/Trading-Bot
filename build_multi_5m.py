#!/usr/bin/env python3
"""Rebuild multi_5m.csv from the top-K liquid CEX assets (minimizing
stablecoins) so single-pair ML models train on a broad cross-asset basket.

Reads data/cex/<SYM>_5m.csv for each basket symbol (produced by
backfill_cex_all.py). BTC is seeded from btc_5m.csv (deepest). The target's
own symbol is included so cross features (e.g. DOGE_btc_ratio) are computable.
Basket = top liquid USDT pairs minus stablecoins (pegged = no signal).

Usage:
  python build_multi_5m.py            # use top50_liquid.txt minus stables
  python build_multi_5m.py --syms BTCUSDT,ETHUSDT,DOGEUSDT,SOLUSDT
"""
import argparse
import pandas as pd
from pathlib import Path

REPO = Path(__file__).parent
CEX = REPO / "data" / "cex"
OUT = REPO / "multi_5m.csv"
STABLES = {"USDCUSDT", "FDUSDUSDT", "RLUSDUSDT", "USD1USDT", "EURUSDT",
           "UUSDT", "TUSDUSDT", "DAIUSDT", "BUSDUSDT", "USDPUSDT", "AEURUSDT"}


def basket_from_file():
    p = REPO / "top50_liquid.txt"
    if not p.exists():
        return ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT"]
    syms = [s.strip() for s in p.read_text().splitlines() if s.strip()]
    return [s for s in syms if s not in STABLES]


def load_5m(sym):
    # BTC has the deepest dedicated file; others from the CEX sweep tree.
    if sym == "BTCUSDT":
        p = REPO / "btc_5m.csv"
    else:
        p = CEX / f"{sym}_5m.csv"
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
    syms = args.syms.split(",") if args.syms else basket_from_file()
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
    merged = merged.dropna(subset=["open", "high", "low", "close"])
    cols = ["ts", "symbol", "open", "high", "low", "close", "volume"]
    merged = merged[cols]
    merged.to_csv(OUT, index=False)
    print(f"Wrote {len(merged)} rows -> {OUT}")
    print(f"Symbols ({len(merged['symbol'].unique())}): {sorted(merged['symbol'].unique())}")


if __name__ == "__main__":
    main()
