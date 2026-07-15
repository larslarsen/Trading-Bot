#!/usr/bin/env python3
"""Rigorous significance test of trading rules on the PIT (survivorship-corrected)
universe.

Why this exists: the scorecard's "sign test on 8 slice-means vs donchian40" is
low-power and doesn't correct for (a) autocorrelation in daily returns,
(b) fat tails, or (c) testing 16 candidates (multiple-comparison inflation).
This harness applies the standard quant panel (Bailey & Lopez de Prado 2014;
Hansen/Hsu SPA; Newey-West 1987; Politis-White stationary bootstrap):

  For each rule:
    1. Run run_strategy on every PIT WF slice (as-of each slice start),
       concatenate daily returns -> pooled series (large N).
    2. Newey-West HAC t-test: mean return != 0 (autocorrelation-robust).
    3. Exact sign test: P(#positive >= observed) under p=0.5 (non-parametric).
    4. Stationary bootstrap: empirical p-value that SR <= 0 (no normality assumpt).
    5. Deflated Sharpe Ratio (DSR): P(SR beats min-profitable-SR after correcting
       for n_trials multiple tests).
  Then Holm step-down correction across all candidates on the NW p-value.
  A rule is "significant" only if it survives on ALL of {NW, sign, DSR} after Holm.

Usage: python test_significance_pit.py
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import sys
sys.path.insert(0, ".")
from engine import load_screened_universe, deflated_sharpe_ratio
import test_rule_scorecard as sc


def pooled_returns(rule, kind):
    """Run the rule over all PIT WF slices, concat daily returns. Mirrors the
    scorecard's PIT mode (as-of each slice start). Returns concatenated rets."""
    data0, dates = sc.load_common(n_min=150)
    slices = []
    i = sc.TRAIN
    while i + sc.OOS <= len(dates):
        slices.append(dates[i:i + sc.OOS]); i += sc.STEP
    combo = (kind == "combo")
    chop = "donchian40" if combo else rule
    all_rets = []
    for seg in slices:
        data, _ = sc.load_common(n_min=150, as_of=str(seg[0].date()))
        out = sc.run_strategy(data, seg, chop, combo=combo)
        r = out.get("rets")
        if r is not None and len(r):
            all_rets.append(np.asarray(r, dtype=float))
    if not all_rets:
        return np.array([])
    return np.concatenate(all_rets)


def newey_west_p(rets, lags=5):
    """HAC t-test: H0 mean=0. Returns p-value (two-sided)."""
    y = rets - rets.mean()  # center; test mean==0 equivalently
    X = add_constant(np.ones_like(y))
    model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    # coef on const == 0 under H0; t-stat + p from the const term
    p = model.pvalues[0]
    t = model.tvalues[0]
    return float(p), float(t)


def sign_test_p(rets):
    """Exact binomial sign test: P(#pos >= k) under p=0.5."""
    n = len(rets)
    k = int((rets > 0).sum())
    # two-sided: use binomtest with alternative='two-sided'
    res = stats.binomtest(k, n, 0.5, alternative="two-sided")
    return float(res.pvalue), k, n


def stationary_bootstrap_p(rets, B=2000, block=5, seed=0):
    """Politis-White stationary bootstrap. Empirical p-value that bootstrapped
    SR (annualized) <= 0. Block bootstrap preserves autocorrelation."""
    rng = np.random.default_rng(seed)
    n = len(rets)
    sr0 = rets.mean() / rets.std(ddof=1) * np.sqrt(252) if rets.std(ddof=1) > 0 else 0.0
    srs = np.empty(B)
    for b in range(B):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            if rng.random() < 1.0 / block:
                i = rng.integers(n)
            else:
                i = (i + 1) % n
            idx[t] = i
        sample = rets[idx]
        s = sample.mean() / sample.std(ddof=1) * np.sqrt(252) if sample.std(ddof=1) > 0 else 0.0
        srs[b] = s
    # P(SR_boot <= 0): fraction of bootstrap SRs at or below zero
    p = float((srs <= 0).mean())
    return p, float(sr0)


def annualized_sr(rets):
    sd = rets.std(ddof=1)
    return float(rets.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def main():
    n_trials = len(sc.CANDIDATES)
    rows = []
    print(f"RIGOROUS PIT SIGNIFICANCE TEST: {n_trials} candidates, HAC t + sign + stationary-bootstrap + DSR\n")
    pooled_cache = {}
    for cname, kind in sc.CANDIDATES:
        key = (cname, kind)
        if key not in pooled_cache:
            pooled_cache[key] = pooled_returns(cname, kind)
        rets = pooled_cache[key]
        n = len(rets)
        if n < 30:
            rows.append((cname, n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        p_nw, t_nw = newey_west_p(rets)
        p_sign, kpos, _ = sign_test_p(rets)
        p_boot, sr0 = stationary_bootstrap_p(rets)
        sr_ann = annualized_sr(rets)
        # DSR: correct for multiple testing (n_trials candidates)
        dsr = deflated_sharpe_ratio(sr_ann, n_trials=n_trials, T=n)
        rows.append((cname, n, t_nw, p_nw, p_sign, p_boot, sr_ann, dsr))
        print(f"{cname:16} n={n:4}  tNW={t_nw:7.2f} pNW={p_nw:.4f}  pSign={p_sign:.4f}  "
              f"pBoot={p_boot:.4f}  SRann={sr_ann:6.2f}  DSR={dsr:.4f}")

    df = pd.DataFrame(rows, columns=["rule", "n", "tNW", "pNW", "pSign", "pBoot", "SRann", "DSR"])

    # Holm correction on the primary (NW) p-values across candidates
    pv = df["pNW"].dropna().values
    order = np.argsort(pv)
    m = len(pv)
    holm = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, max(prev, (m - rank) * pv[idx]))
        holm[idx] = val
        prev = val
    df.loc[df["pNW"].notna(), "pNW_holm"] = holm

    # Significance: survive ALL of {NW(holm)<.05, sign<.05, DSR>.95 (i.e. P(SR>min)>0.95)}
    sig = (df["pNW_holm"] < 0.05) & (df["pSign"] < 0.05) & (df["DSR"] > 0.95)
    df["significant"] = sig.fillna(False)

    print("\n=== HOLM-CORRECTED + MULTI-TEST RESULTS ===")
    print(df[["rule", "n", "pNW_holm", "pSign", "pBoot", "SRann", "DSR", "significant"]]
          .sort_values("pNW_holm").to_string(index=False))
    n_sig = int(sig.sum())
    print(f"\nRules surviving ALL tests (NW-Holm & sign & DSR, p<0.05): {n_sig}/{n_trials}")
    if n_sig == 0:
        print("=> NO rule shows a statistically significant edge on PIT-corrected data.")


if __name__ == "__main__":
    main()
