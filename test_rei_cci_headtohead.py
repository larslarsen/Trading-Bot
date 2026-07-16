"""
Head-to-head: live chop primary = donchian40 (current) vs cci (candidate).

Isolates ONE variable: the chop-regime PRIMARY entry rule, while keeping the
ma30_ema fill-in identical (used on silent-chop days) and the trend entry
fixed to rei in both. Any difference is attributable to the chop primary.

Uses the scorecard's own run_strategy (single source of truth) via the new
fill_rule parameter, so execution logic is identical to the full scorecard.
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

    variants = {
        "LIVE rei+d40(+ma30 fill)": ("donchian40", "ma30_ema", True),
        "CAND rei+cci(+ma30 fill)": ("cci", "ma30_ema", True),
        "CAND rei+cci(no fill)":     ("cci", None, False),
    }
    print(f"rei-chop-primary head-to-head: {len(slices)} WF slices, {len(data)} coins\n")
    results = {}
    for name, (chop, fill, combo) in variants.items():
        rs = [sc.run_strategy(data, seg, chop, combo=combo, fill_rule=fill) for seg in slices]
        results[name] = rs

    print(f"{'variant':26s} {'meanRet':>8} {'effSR':>7} {'meanDD':>7} {'calmar':>7} {'win%':>6} {'trades':>7}")
    agg = {}
    for name, rs in results.items():
        a = {
            "ret": np.mean([r["ret"] for r in rs]),
            "effSR": np.mean([r["effSR"] for r in rs]),
            "DD": np.mean([r["maxDD"] for r in rs]),
            "calmar": np.mean([r["calmar"] for r in rs]),
            "win": np.mean([r["win%"] for r in rs]),
            "trades": np.mean([r["trades"] for r in rs]),
        }
        agg[name] = a
        print(f"{name:26s} {a['ret']:+8.1f} {a['effSR']:+7.2f} {a['DD']:>6.1f}% {a['calmar']:>6.2f} {a['win']:>5.1f}% {a['trades']:>7.0f}")

    live = [r["ret"] for r in results["LIVE rei+d40(+ma30 fill)"]]
    cand = [r["ret"] for r in results["CAND rei+cci(+ma30 fill)"]]
    d = np.array(cand) - np.array(live)
    n = int(np.sum(d != 0)); k = int(np.sum(d > 0))
    p = float(_stats.binomtest(k, n).pvalue) if n else 1.0
    print(f"\nrei+cci vs rei+d40 (paired, slice returns): Δmean={np.mean(cand)-np.mean(live):+.1f}% "
          f"beats live in {k}/{len(slices)} slices  p={p:.3f}{' *' if p<0.05 else ''}")
    print("* = significant at p<0.05 (exact binomial sign test)")


if __name__ == "__main__":
    main()
