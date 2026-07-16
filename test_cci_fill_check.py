"""
Does the live chop fill-in (ma30_ema on silent-cci chop days) help or hurt?

Live semantics (paper_trader_multi.py):
  chop day -> compute cci entries; if cci produces ZERO entries, fall back to
  ma30_ema. This is NOT a blended signal; it's a silent-day safety net.

We test:
  A) cci ALONE          (combo=False)
  B) cci + ma30_ema FILL (combo=True, fill_rule="ma30_ema")   == live config

If A == B, the fill-in never triggers (cci always fires) -> it's inert safety net.
If B < A, the fill-in is actively hurting on silent days -> candidate for removal.
If B > A, it adds value -> keep.

8 WF slices, 68 coins, via the scorecard's proven run_strategy (single source).
"""
import numpy as np
import pandas as pd
from scipy import stats as _stats
import test_rule_scorecard as sc

TRAIN, STEP, OOS = 60, 12, 12


def main():
    data, dates = sc.load_common(n_min=150)
    slices = []
    i = TRAIN
    while i + OOS <= len(dates):
        slices.append(dates[i:i + OOS]); i += STEP

    a_alone = [sc.run_strategy(data, seg, "cci", combo=False) for seg in slices]
    b_fill  = [sc.run_strategy(data, seg, "cci", combo=True, fill_rule="ma30_ema") for seg in slices]

    def agg(rs):
        return {
            "ret": np.mean([r["ret"] for r in rs]),
            "effSR": np.mean([r["effSR"] for r in rs]),
            "DD": np.mean([r["maxDD"] for r in rs]),
            "calmar": np.mean([r["calmar"] for r in rs]),
            "win": np.mean([r["win%"] for r in rs]),
            "trades": np.mean([r["trades"] for r in rs]),
        }

    print(f"cci vs cci+ma30_ema-fill (silent-day safety net): {len(slices)} WF slices, {len(data)} coins\n")
    print(f"{'variant':32s} {'meanRet':>8} {'effSR':>7} {'meanDD':>7} {'calmar':>7} {'win%':>6} {'trades':>7}")
    aa, ab = agg(a_alone), agg(b_fill)
    print(f"{'A) cci ALONE':32s} {aa['ret']:+8.1f} {aa['effSR']:+7.2f} {aa['DD']:>6.1f}% {aa['calmar']:>6.2f} {aa['win']:>5.1f}% {aa['trades']:>7.0f}")
    print(f"{'B) cci + ma30_ema FILL':32s} {ab['ret']:+8.1f} {ab['effSR']:+7.2f} {ab['DD']:>6.1f}% {ab['calmar']:>6.2f} {ab['win']:>5.1f}% {ab['trades']:>7.0f}")

    # per-slice difference (does fill ever change the outcome?)
    diffs = np.array([b_fill[s]["ret"] - a_alone[s]["ret"] for s in range(len(slices))])
    n_changed = int(np.sum(diffs != 0))
    print(f"\nSlices where fill-in changed the result: {n_changed}/{len(slices)}")
    print(f"Mean ret delta (B - A): {diffs.mean():+.2f}%")

    # paired sign test on slices where fill triggered
    if n_changed:
        k = int(np.sum(diffs > 0)); n = n_changed
        p = float(_stats.binomtest(k, n).pvalue)
        print(f"On triggered slices: fill beats alone in {k}/{n}  p={p:.3f}")
    else:
        print("Fill-in never triggered -> inert safety net (identical to cci alone).")

    # trade-count evidence: if fill triggers, B should have >= trades than A
    ta = [r["trades"] for r in a_alone]; tb = [r["trades"] for r in b_fill]
    print(f"Total trades  A={sum(ta)}  B={sum(tb)}  (B>A means fill added entries somewhere)")


if __name__ == "__main__":
    main()
