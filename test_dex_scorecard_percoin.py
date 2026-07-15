#!/usr/bin/env python3
"""DEX rule scorecard, PER-COIN windows (no screen, no shared date axis).

The shared-axis loader collapsed 400+ DEX coins to 18 (few share the same 72
calendar days). This runs each coin on ITS OWN timeline instead: for every coin
with enough bars, walk-forward single-asset slices, then POOL all (coin x slice)
OOS returns per rule and paired-sign-test each rule vs donchian40 across the
pool. That turns a 1-slice dead end into a few-hundred-slice descriptive sample.

Reuses test_rule_scorecard.run_strategy / sign_test_p verbatim (same engine,
metrics, test) -- only the data feed is per-coin instead of a shared universe.

CAVEAT: single-asset (not the live 5-position portfolio) and still free 6mo
history. This is a DESCRIPTIVE cross-coin read of rule behavior, not a
portfolio backtest and not proof of edge.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from test_rule_scorecard import run_strategy, sign_test_p, CANDIDATES, TRAIN, STEP, OOS
import config as _cfg  # N_WORKERS_CPU = logical cores - 1 (headroom)

DEX = Path("dex_data")
MIN_BARS = TRAIN + OOS  # need at least one full walk-forward window


def load_all(min_bars=MIN_BARS):
    """Every dex_data coin with >= min_bars, on its own timeline (no alignment)."""
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


def coin_slices(dates):
    out, i = [], TRAIN
    while i + OOS <= len(dates):
        out.append(dates[i:i + OOS]); i += STEP
    return out


def _dex_percoin_task(task):
    """Top-level worker for the parallel per-coin DEX pool (must be picklable).

    task = (stem, df, cname, kind, seg); returns (cname, OOS return).
    """
    stem, df, cname, kind, seg = task
    combo = (kind == "combo")
    chop_rule = "donchian40" if combo else cname
    return cname, run_strategy({stem: df}, seg, chop_rule, combo=combo)["ret"]


def main():
    coins = load_all()
    print(f"DEX per-coin scorecard: {len(coins)} coins with >={MIN_BARS} bars "
          f"(TRAIN={TRAIN}/OOS={OOS}/STEP={STEP})")
    print("CAVEAT: single-asset, free 6mo history -> descriptive cross-coin read, not edge.\n")
    if len(coins) < 5:
        print("Too few coins. Aborting.")
        return

    # Pool (coin x slice) OOS returns per rule.
    res = {c[0]: [] for c in CANDIDATES}
    n_slices = 0
    from multiprocessing import Pool

    tasks = []
    for stem, df in coins.items():
        dates = pd.DatetimeIndex(df.index)
        slices = coin_slices(dates)
        if not slices:
            continue
        n_slices += len(slices)
        for cname, kind in CANDIDATES:
            for seg in slices:
                tasks.append((stem, df, cname, kind, seg))
    with Pool(_cfg.N_WORKERS_CPU) as pool:
        for cname, ret in pool.map(_dex_percoin_task, tasks):
            res[cname].append(ret)

    print(f"pooled slices per rule: {len(res['donchian40'])} "
          f"({len(coins)} coins x their windows)\n")
    print(f"{'candidate':16s} {'meanRet':>8} {'medRet':>7} {'pos%':>6} {'worst':>7}")
    agg = {}
    for cname, _ in CANDIDATES:
        r = np.array(res[cname], dtype=float)
        agg[cname] = (r.mean(), np.median(r), (r > 0).mean() * 100, r.min())
    for cname in sorted(agg, key=lambda x: -agg[x][0]):
        m, md, pos, w = agg[cname]
        print(f"{cname:16s} {m:+8.2f} {md:+7.2f} {pos:>5.1f}% {w:+7.1f}")

    print(f"\n=== PAIRED vs donchian40 (exact sign test, pooled coin x slice) ===")
    live = np.array(res["donchian40"], dtype=float)
    for cname, _ in CANDIDATES:
        if cname == "donchian40":
            continue
        other = np.array(res[cname], dtype=float)
        p = sign_test_p(other, live)
        delta = other.mean() - live.mean()
        better = int((other > live).sum())
        print(f"   {cname:16s} Δmean={delta:+6.2f}  beats in {better}/{len(live)}  "
              f"p={p:.4f}{' *' if p < 0.05 else ''}")
    print(f"\n* p<0.05. With pooled n={len(live)} a star is worth a look, but coins "
          f"overlap in time -> slices NOT independent; treat as suggestive only.")


if __name__ == "__main__":
    main()
