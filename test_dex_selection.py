#!/usr/bin/env python3
"""DEX per-coin SELECTION, parallelized across all CPU cores.

Scores each DEX coin on the LIVE strategy (rei-trend/cci-chop + d40 fill) over
its OWN ~181-bar history, ranks by OOS return, writes a shortlist for the DEX
paper trader (B). Parallelized with multiprocessing.Pool (each coin is an
independent PortfolioEngine replay -> trivially parallel).

Reuses test_rule_scorecard.run_strategy / sign_test_p verbatim (same engine).
"""
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from test_rule_scorecard import run_strategy, TRAIN, STEP, OOS
import config as cfg  # LOGICAL_CORES for pure-CPU work, PHYSICAL_CORES for BW-bound

DEX = Path("dex_data")
MIN_BARS = TRAIN + OOS
LIVE_CHOP = "cci"
COMBO_FILL = "ma30_ema"


def load_all(min_bars=MIN_BARS):
    data = {}
    for p in sorted(DEX.glob("*_1d_max.csv")):
        stem = p.name.replace("_1d_max.csv", "").upper()
        try:
            df = pd.read_csv(p, parse_dates=["ts"]).dropna(subset=["close", "high", "low", "volume"])
        except Exception:
            continue
        df = df.sort_values("ts").reset_index(drop=True)
        if len(df) >= min_bars:
            data[stem] = df.set_index("ts").sort_index()
    return data


def _score(args):
    """Worker: score one coin. Top-level for pickling."""
    stem, df = args
    dates = pd.DatetimeIndex(df.index)
    out = []
    i = TRAIN
    while i + OOS <= len(dates):
        seg = dates[i:i + OOS]; i += STEP
        r = run_strategy({stem: df}, seg, LIVE_CHOP, combo=True, fill_rule=COMBO_FILL)
        out.append(r["ret"])
    r = np.array(out, float)
    if len(r) == 0:
        return None
    return (stem, float(r.mean()), float(np.median(r)), float((r > 0).mean() * 100),
            float(r.min()), len(r))


def main():
    coins = load_all()
    print(f"DEX per-coin SELECTION (parallel, {cfg.N_WORKERS_CPU} workers = logical-1): {len(coins)} coins "
          f">= {MIN_BARS} bars (live rule: rei-trend/cci-chop + d40 fill)\n")
    if len(coins) < 5:
        print("Too few coins. Aborting."); return
    items = list(coins.items())
    # Pure-CPU replay (no memory-bandwidth contention) -> use LOGICAL cores
    # minus one for system/control headroom.
    with Pool(cfg.N_WORKERS_CPU) as pool:
        results = pool.map(_score, items)
    rows = [r for r in results if r is not None]
    rows.sort(key=lambda x: -x[1])
    print(f"{'coin':14s} {'meanRet':>8} {'medRet':>7} {'pos%':>6} {'worst':>7} {'slices':>6}")
    for stem, m, md, pos, w, n in rows:
        print(f"{stem:14s} {m:+8.2f} {md:+7.2f} {pos:>5.1f}% {w:+7.1f} {n:>6}")
    good = [r for r in rows if r[1] > 0]
    print(f"\nShortlist (mean OOS > 0): {len(good)} / {len(rows)} coins")
    for stem, m, md, pos, w, n in good[:25]:
        print(f"   {stem:14s} {m:+8.2f}%  pos%={pos:.0f}  worst={w:+.1f}")
    short = pd.DataFrame([(s, m) for s, m, *_ in good], columns=["symbol", "mean_ret"])
    short.to_csv("dex_selection_shortlist.csv", index=False)
    print(f"\nWrote dex_selection_shortlist.csv ({len(short)} symbols)")


if __name__ == "__main__":
    main()
