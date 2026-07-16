#!/usr/bin/env python3
"""Rebuild multi_5m.csv from deep 5m history so single-pair models actually
get cross-asset features for their trading era.

Universal schema: ts, symbol, open, high, low, close, volume
Sources:
  - btc_5m.csv            (symbol BTCUSDT) 2012->2026-07   [benchmark base]
  - data/DOGEUSDT_5m_max.csv (symbol DOGEUSDT) 2020->2026-07
BTCUSDT must be present (make_cross_features requires it as base).
"""
from pathlib import Path
import pandas as pd

ROOT = Path("/home/lars/trading-bot")
out = ROOT / "multi_5m.csv"
backup = Path("/tmp/multi_5m_stale_2018.csv")

# reversible: keep the old file under /tmp
if out.exists() and not backup.exists():
    out.replace(backup)
    print(f"backed up stale multi_5m.csv -> {backup}")

frames = []
# BTC
btc = pd.read_csv(ROOT / "btc_5m.csv", parse_dates=["ts"])
btc["symbol"] = "BTCUSDT"
frames.append(btc[["ts", "symbol", "open", "high", "low", "close", "volume"]])
n_btc = len(btc)
# DOGE
doge = pd.read_csv(ROOT / "data" / "DOGEUSDT_5m_max.csv", parse_dates=["ts"])
doge["symbol"] = "DOGEUSDT"
frames.append(doge[["ts", "symbol", "open", "high", "low", "close", "volume"]])
n_doge = len(doge)

multi = pd.concat(frames, ignore_index=True)
multi = multi.sort_values(["symbol", "ts"]).reset_index(drop=True)
multi.to_csv(out, index=False)
print(f"wrote {out}: {len(multi)} rows")
print(f"  BTCUSDT: {n_btc} | DOGEUSDT: {n_doge}")
print(f"  span: {multi['ts'].min()} -> {multi['ts'].max()}")
print(f"  symbols: {sorted(multi['symbol'].unique())}")
