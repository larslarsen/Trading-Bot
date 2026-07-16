#!/usr/bin/env python3
"""Derive CEX 1h/4h/1d OHLCV from local 5m data -- ZERO API calls.

If we hold FULL-DEPTH 5m for a symbol, the higher TFs are exact resamples
(OHLCV aggregation is lossless), so there is no need to pull them from the
API. This cuts the backfill to a single TF (5m) -- 1/4 the API load -- and
avoids rate-limit throttling on the lookback TFs entirely.

Reads data/cex/<SYM>_5m.csv, writes data/cex/<SYM>_{1h,4h,1d}.csv. Derived
files fully overwrite any partial direct-pull higher-TF files, because the
5m span is the complete source of truth.

Usage:
    python derive_cex_tf.py                 # all symbols with 5m data
    python derive_cex_tf.py --syms BTCUSDT,DOGEUSDT
    python derive_cex_tf.py --tfs 1h 1d
"""
import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
RULE = {"1h": "1h", "4h": "4h", "1d": "1d"}
FMT = {"1h": "%Y-%m-%d %H:%M:%S+0000", "4h": "%Y-%m-%d %H:%M:%S+0000", "1d": "%Y-%m-%d"}


def derive_sym(sym, tfs):
    # Single source of truth: same canonical 5m path the poller writes and the
    # bots/trainer read. BTC -> btc_5m.csv; others -> data/<SYM>USDT_5m_max.csv.
    if sym == "BTCUSDT":
        src = REPO / "btc_5m.csv"
    else:
        stem = sym.replace("USDT", "")
        src = REPO / "data" / f"{stem}USDT_5m_max.csv"
    if not src.exists():
        return {tf: 0 for tf in tfs}
    df = pd.read_csv(src)
    if df.empty or len(df) < 2:
        return {tf: 0 for tf in tfs}
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d = d.set_index("ts").sort_index()
    out = {}
    for tf in tfs:
        r = d.resample(RULE[tf]).agg(AGG).dropna().reset_index()
        r["ts"] = r["ts"].dt.strftime(FMT[tf])
        # canonical path: data/<SYM>USDT_<tf>_max.csv (same tree as 5m + DEX)
        stem = sym.replace("USDT", "")
        out_path = REPO / "data" / f"{stem}USDT_{tf}_max.csv"
        r.to_csv(out_path, index=False)
        out[tf] = len(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=None, help="comma list; default all with 5m")
    ap.add_argument("--tfs", nargs="+", choices=list(RULE), default=list(RULE))
    args = ap.parse_args()
    if args.syms:
        syms = [s.strip().upper() for s in args.syms.split(",") if s.strip()]
    else:
        # canonical 5m files: btc_5m.csv + data/<SYM>USDT_5m_max.csv
        syms = []
        if (REPO / "btc_5m.csv").exists():
            syms.append("BTCUSDT")
        syms += sorted({p.name.replace("USDT_5m_max.csv", "") + "USDT"
                        for p in (REPO / "data").glob("*USDT_5m_max.csv")})
    print(f"Derive CEX TFs from local 5m (no API): {len(syms)} symbols, TFs={args.tfs}")
    tot = {tf: 0 for tf in args.tfs}
    for sym in syms:
        added = derive_sym(sym, args.tfs)
        for tf in args.tfs:
            tot[tf] += added.get(tf, 0)
        if any(added.values()):
            print(f"  {sym}: +{added}")
    print(f"Derive complete. Bars written: {tot}")


if __name__ == "__main__":
    main()
