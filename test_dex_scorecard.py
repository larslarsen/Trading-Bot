#!/usr/bin/env python3
"""Run the rule scorecard on the DEX universe (dex_data/ + latest screen_dex_idio_*).

Reuses test_rule_scorecard.run_strategy / sign_test_p verbatim -- identical
engine, metrics, and paired sign test as the CEX scorecard. Only the data
loader differs: DEX bars live in dex_data/ and the screen is screen_dex_idio_*.

CAVEAT: free GeckoTerminal DEX history caps at ~181 daily bars (~6mo). With
TRAIN=60/OOS=12/STEP=12 that yields only a few walk-forward slices, so NOTHING
here can be statistically significant -- this is descriptive, not falsifiable.
Stated up front so the output isn't over-read.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import test_rule_scorecard as sc
from test_rule_scorecard import run_strategy, sign_test_p, CANDIDATES, TRAIN, STEP, OOS
import config as _cfg  # N_WORKERS_CPU = logical cores - 1 (headroom)

DEX = Path("dex_data")
OUT = Path("backtest_output")


def load_dex(min_bars=120):
    """Load DEX screen + dex_data bars into {stem: df} (same shape as CEX loader)."""
    screens = sorted(OUT.glob("screen_dex_idio_*.csv"))
    if not screens:
        raise FileNotFoundError("no screen_dex_idio_*.csv -- run screen_dex_idio.py first")
    screen = pd.read_csv(screens[-1])
    screen = screen[screen.tier.isin(["large", "mid", "tail"])]
    data = {}
    for _, row in screen.iterrows():
        stem = str(row.get("stem", row["symbol"])).strip().upper()
        p = DEX / f"{stem}_1d_max.csv"
        if not p.exists() or stem in data:
            continue
        df = pd.read_csv(p, parse_dates=["ts"]).dropna(subset=["close", "high", "low", "volume"])
        df = df.sort_values("ts").reset_index(drop=True)
        if len(df) < min_bars:
            continue
        data[stem] = df
    return data


def common_axis(data):
    """Intersect dates so all coins share one axis (engine expects aligned bars)."""
    idx = None
    aligned = {}
    for s, d in data.items():
        di = d.set_index("ts").sort_index()
        idx = di.index if idx is None else idx.intersection(di.index)
        aligned[s] = di
    aligned = {s: d.loc[idx] for s, d in aligned.items() if len(d.loc[idx]) > 0}
    return aligned, pd.DatetimeIndex(sorted(idx))


def _dex_scorecard_task(task):
    """Top-level worker for the parallel DEX WF pool (must be picklable).

    task = (cname, kind, seg, data); returns (cname, run_strategy result).
    """
    cname, kind, seg, data = task
    combo = (kind == "combo")
    chop_rule = "donchian40" if combo else cname
    return cname, run_strategy(data, seg, chop_rule, combo=combo)


def main():
    data = load_dex(min_bars=TRAIN + OOS)  # need at least one full slice
    if len(data) < 5:
        print(f"Only {len(data)} DEX coins with >={TRAIN+OOS} shared bars -- too few. "
              f"DEX history is thin (free 6mo cap). Aborting.")
        return
    data, dates = common_axis(data)
    slices = []
    i = TRAIN
    while i + OOS <= len(dates):
        slices.append(dates[i:i + OOS]); i += STEP
    print(f"DEX rule scorecard: {len(data)} coins, {len(dates)} shared bars, "
          f"{len(slices)} WF slices (TRAIN={TRAIN}/OOS={OOS}/STEP={STEP})")
    print("CAVEAT: thin/short DEX history -> descriptive only, NOT significant.\n")
    if not slices:
        print("Zero WF slices -- shared history shorter than one TRAIN+OOS window. Aborting.")
        return

    res = {c[0]: [] for c in CANDIDATES}
    from multiprocessing import Pool

    # task = (cname, kind, seg, data) — data precomputed, picklable.
    tasks = [(c[0], c[1], seg, data) for c in CANDIDATES for seg in slices]
    with Pool(_cfg.N_WORKERS_CPU) as pool:
        for cname, out in pool.map(_dex_scorecard_task, tasks):
            res[cname].append(out)

    print(f"{'candidate':14s} {'meanRet':>8} {'worst':>7} {'pos/n':>6} {'meanEffSR':>9} {'meanDD':>7} {'meanWin%':>9}")
    agg = {}
    for cname, _ in CANDIDATES:
        rs = res[cname]
        agg[cname] = {"ret": np.mean([r["ret"] for r in rs]),
                      "worst": min(r["ret"] for r in rs),
                      "pos": sum(1 for r in rs if r["ret"] > 0),
                      "effSR": np.mean([r["effSR"] for r in rs]),
                      "DD": np.mean([r["maxDD"] for r in rs]),
                      "win": np.mean([r["win%"] for r in rs])}
    for cname in sorted(agg, key=lambda x: -agg[x]["ret"]):
        a = agg[cname]
        print(f"{cname:14s} {a['ret']:+8.1f} {a['worst']:+7.1f} {a['pos']}/{len(slices):<3} "
              f"{a['effSR']:+9.2f} {a['DD']:>6.1f}% {a['win']:>8.1f}%")

    print(f"\n=== PAIRED vs donchian40 (exact sign test) ===")
    live = [r["ret"] for r in res["donchian40"]]
    for cname, _ in CANDIDATES:
        if cname == "donchian40":
            continue
        other = [r["ret"] for r in res[cname]]
        p = sign_test_p(other, live)
        delta = np.mean(other) - np.mean(live)
        better = sum(1 for o, l in zip(other, live) if o > l)
        print(f"   {cname:14s} Δmean={delta:+6.1f}  beats in {better}/{len(slices)}  p={p:.3f}{' *' if p < 0.05 else ''}")
    print(f"\n* p<0.05 (but n={len(slices)} slices -> treat any star as noise, not edge)")


if __name__ == "__main__":
    main()
