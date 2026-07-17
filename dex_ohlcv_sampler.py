#!/usr/bin/env python3
"""
Free going-forward 1m (or 5m) DEX OHLCV sampler.

WHY: free GeckoTerminal serves `minute` OHLCV but only ~5 bars of HISTORY
(no deep 1m backfill free). So we SAMPLE LIVE: every `interval` we pull the
newest 1m (or 5m) candle per token and append to data/dex/<TOK>_<tf>_max.csv.
Forward bars are real 1m; history before "now" is whatever GeckoTerminal
returns (negligible). We then derive 1h/4h/1d locally from the 1m base --
matching the CEX pattern (collect fine TF, derive up).

RATE LIMIT: free GeckoTerminal ~10-30 calls/min. Top-N tokens at 1m must
stay under that. Default N=20 @ 1m = 20 calls/min (feasible). Larger N or
finer TF needs a paid CoinGecko /onchain key (wire GT_API_KEY later).

Reuses backfill_dex_history_gt.resolve_top_pool + gt_ohlcv (DexScreener ->
GeckoTerminal net/pool resolution).

Usage:
  python dex_ohlcv_sampler.py                 # loop, top-20 @ 1m
  python dex_ohlcv_sampler.py --top 50 --tf 5m
  python dex_ohlcv_sampler.py --once         # one cycle (test)
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).parent
DEX_DIR = REPO / "dex_data"
MICRO_DIR = REPO / "data" / "dex_micro"
OUT_DIR = REPO / "data" / "dex"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from dex_resolve import real_top_pool, gt_pool_ohlcv, safe_name, NETMAP

SLEEP_PER_CALL = 3.0  # ~20 calls/min -> within free 10-30 limit


def rank_tokens(top_n):
    """Rank DEX universe tokens by live liquidity_usd (from micro poller)."""
    rows = []
    for p in sorted(DEX_DIR.glob("*_1d_max.csv")):
        tok = p.stem.replace("_1d_max", "").replace("_1d", "")
        liq = 0.0
        mp = MICRO_DIR / f"{tok}.csv"
        if mp.exists():
            try:
                d = pd.read_csv(mp)
                if "liquidity_usd" in d.columns and len(d):
                    liq = float(pd.to_numeric(d["liquidity_usd"], errors="coerce").max())
            except Exception:
                pass
        rows.append((tok, liq))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in rows[:top_n]]


def append_bar(tok, tf, bar):
    """bar = [ts_sec, o, h, l, c, v]. Append if ts newer than last saved."""
    tgt = OUT_DIR / f"{tok}_{tf}_max.csv"
    ts = int(bar[0])
    o, h, l, c, v = [float(x) for x in bar[1:6]]
    new = pd.DataFrame([{
        "ts": pd.to_datetime(ts, unit="s", utc=True),
        "open": o, "high": h, "low": l, "close": c, "volume": v,
    }])
    if tgt.exists():
        old = pd.read_csv(tgt, parse_dates=["ts"])
        last = old["ts"].max()  # tz-aware datetime (pandas 3.0 = [us, UTC])
        if pd.to_datetime(ts, unit="s", utc=True) <= last:
            return 0  # already have this bar
        out = pd.concat([old, new]).drop_duplicates(subset=["ts"]).sort_values("ts")
    else:
        out = new
    out.to_csv(tgt, index=False)
    return 1


def derive_tfs(tok, base_tf="1m"):
    """Derive higher TFs from the base (1m or 5m) by resampling.

    1m base -> derive 5m, 1h, 4h, 1d. 5m base -> derive 1h, 4h, 1d (skip 5m).
    """
    base = OUT_DIR / f"{tok}_{base_tf}_max.csv"
    if not base.exists():
        return
    d = pd.read_csv(base, parse_dates=["ts"]).set_index("ts").sort_index()
    # (filename_suffix, resample_alias) -- pandas 3.0 needs '5min' not '5m'
    if base_tf == "1m":
        tfs = (("5m", "5min"), ("1h", "1h"), ("4h", "4h"), ("1d", "1d"))
    else:
        tfs = (("1h", "1h"), ("4h", "4h"), ("1d", "1d"))
    for suffix, alias in tfs:
        r = d["close"].resample(alias).last()
        o = d["open"].resample(alias).first()
        h = d["high"].resample(alias).max()
        l = d["low"].resample(alias).min()
        v = d["volume"].resample(alias).sum()
        out = pd.DataFrame({"open": o, "high": h, "low": l, "close": r, "volume": v}).dropna()
        out.index.name = "ts"
        out.reset_index().to_csv(OUT_DIR / f"{tok}_{suffix}_max.csv", index=False)


def cycle(top_n, tf, once):
    toks = rank_tokens(top_n)
    print(f"[{pd.Timestamp.now('UTC'):%H:%M:%S}] cycle: {len(toks)} tokens @ {tf}", flush=True)
    added = 0
    for tok in toks:
        try:
            res, err = real_top_pool(tok)
            if not res:
                continue
            net, pool, liq = res
            # GeckoTerminal timeframe names: 'minute'/'hour'/'day'. 1m -> minute
            # agg 5; 5m -> minute agg 5. Derive 1h/4h/1d locally from base.
            gt_tf = "minute"
            agg = 5 if tf == "5m" else 1
            # fetch newest 2 candles of the base tf (free POOL-level endpoint)
            bars, e = gt_pool_ohlcv(net, pool, gt_tf, 1, 2, aggregate=agg)
            if bars:
                for bar in bars:
                    added += append_bar(tok, tf, bar)
                derive_tfs(tok, tf)
        except Exception as ex:
            print(f"  {tok} err {ex!r}"[:120], flush=True)
        time.sleep(SLEEP_PER_CALL)
    print(f"  +{added} new {tf} bars", flush=True)
    if once:
        return
    time.sleep(55)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--tf", default="1m", choices=["1m", "5m"])
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    print(f"DEX OHLCV sampler: top={args.top} tf={args.tf} (free GeckoTerminal)")
    while True:
        try:
            cycle(args.top, args.tf, args.once)
        except Exception as e:
            print(f"cycle error: {e!r}", flush=True)
        if args.once:
            break


if __name__ == "__main__":
    main()
