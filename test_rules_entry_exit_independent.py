"""
Independent entry/exit rule test.

The scorecard (test_rule_scorecard.py) scores each rule as a FULL strategy
(own entry + own exit) vs LIVE donchian40. That answers "which combined
strategy is best" but NOT "is the edge coming from entry timing or exit
timing?" This harness decouples them.

Method (faithful to the scorecard's live gating: rei on trend days, rule R on
chop days, market-wide regime via improved_compute_live_regime):

  For each rule R build CAUSAL panels ENTRY_R / EXIT_R (0/1 per day per symbol),
  gated exactly as the live system does. Then run three simulate_portfolio
  variants per rule, per walk-forward slice:

    FULL(R)        = entry ENTRY_R   + exit EXIT_R        (own strategy)
    ENTRY_iso(R)   = entry ENTRY_R   + exit EXIT_d40      (exit held fixed)
    EXIT_iso(R)    = entry ENTRY_d40 + exit EXIT_R        (entry held fixed)

  Attribution:
    exit contribution  ~= FULL(R) - ENTRY_iso(R)   (same entries, different exits)
    entry contribution ~= FULL(R) - EXIT_iso(R)    (same exits, different entries)

Panel metrics per slice (simulate_portfolio native, sqrt(365) basis):
  return_pct, effective_sharpe, max_dd_pct, trades, dsr.
Paired exact sign test vs LIVE (donchian40) on slice returns.

This is a research/ad-hoc script (not part of the committed unit suite).
"""
import numpy as np
import pandas as pd
from scipy import stats as _stats

from engine import (
    load_screened_universe, get_regime_signals, improved_compute_live_regime,
    simulate_portfolio,
)

COST_BPS = 8
SLIP_BPS = 5
MAX_POS, POS_PCT = 5, 0.20
INITIAL = 10000.0

CORE = ["donchian40", "williams_r", "williams_r_buggy", "rei", "bbwp",
        "stochastic", "mfi", "ift_rsi",
        "ma30_ema", "cci", "tsi", "bop", "mtf"]
TRAIN, STEP, OOS = 60, 12, 12


# ---- universe + slice helpers (mirror test_rule_scorecard) -----------------
def load_common(n_min=150):
    cd = load_screened_universe(min_bars=n_min)
    data = {s: d.set_index("ts").sort_index() for s, d in cd.items()}
    idx = None
    for d in data.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    data = {s: d.loc[idx] for s, d in data.items() if len(d.loc[idx]) > 0}
    return data, pd.DatetimeIndex(sorted(idx))


def prefix(data, day):
    return {s: d.loc[:day] for s, d in data.items() if len(d.loc[:day]) > 0}


def compute_regimes(data, dates):
    """Market-wide regime per date (expanding), exactly as the scorecard."""
    out = {}
    for day in dates:
        pre = prefix(data, day)
        try:
            out[day] = improved_compute_live_regime(pre)
        except Exception:
            out[day] = "trend"
    return out


def build_panels(rule, data, dates, regimes):
    """Return (entry_df, exit_df) 0/1 panels, regime-gated (trend->rei, chop->rule)."""
    syms = list(data.keys())
    entry = pd.DataFrame(0, index=dates, columns=syms, dtype=int)
    exit_ = pd.DataFrame(0, index=dates, columns=syms, dtype=int)
    for day in dates:
        reg = regimes[day]
        gated = "rei" if reg == "trend" else rule
        pre = prefix(data, day)
        for s in syms:
            dfp = pre.get(s)
            if dfp is None or len(dfp) < 2:
                continue
            try:
                ent, ex = get_regime_signals(gated, dfp.reset_index())
            except Exception:
                ent, ex = pd.Series([0]), pd.Series([0])
            if len(ent) and int(ent.iloc[-1]):
                entry.at[day, s] = 1
            if len(ex) and int(ex.iloc[-1]):
                exit_.at[day, s] = 1
    return entry, exit_


def price_panel(data, dates):
    """Close-price panel indexed by dates, columns=symbols (aligned to `dates`)."""
    syms = list(data.keys())
    out = pd.DataFrame(index=dates, columns=syms, dtype=float)
    for s in syms:
        out[s] = data[s]["close"].reindex(dates)
    return out


def run_variant(entry_df, exit_df, price_df):
    res = simulate_portfolio(
        price_df, entry_df, exit_signal_df=exit_df,
        initial=INITIAL, max_positions=MAX_POS, max_position_pct=POS_PCT,
        cost_bps=COST_BPS, slippage_bps=SLIP_BPS, min_equity=100.0,
    )
    return res


def sign_test_p(a, b):
    d = np.array(a) - np.array(b)
    n = int(np.sum(d != 0)); k = int(np.sum(d > 0))
    if n == 0:
        return 1.0
    try:
        return float(_stats.binomtest(k, n).pvalue)
    except Exception:
        from math import comb
        p = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
        return float(min(1.0, 2 * p))


def main():
    data, dates = load_common(n_min=150)
    slices = []
    i = TRAIN
    while i + OOS <= len(dates):
        slices.append(dates[i:i + OOS]); i += STEP
    print(f"INDEPENDENT entry/exit test: {len(CORE)} rules, {len(slices)} WF slices, {len(data)} coins\n")

    regimes = compute_regimes(data, dates)
    price = price_panel(data, dates)

    # reference panels (donchian40) built once
    ent_d40, ex_d40 = build_panels("donchian40", data, dates, regimes)

    results = {}  # rule -> {variant: [per-slice metric dicts]}
    for rule in CORE:
        ent_r, ex_r = build_panels(rule, data, dates, regimes)
        per = {"FULL": [], "ENTRY_iso": [], "ENTRY_iso_ret": [], "EXIT_iso": [], "EXIT_iso_ret": []}
        for seg in slices:
            p = price.loc[seg]
            r_full = run_variant(ent_r.loc[seg], ex_r.loc[seg], p)
            r_eiso = run_variant(ent_r.loc[seg], ex_d40.loc[seg], p)
            r_xiso = run_variant(ent_d40.loc[seg], ex_r.loc[seg], p)
            per["FULL"].append(r_full)
            per["ENTRY_iso"].append(r_eiso)
            per["EXIT_iso"].append(r_xiso)
            per["ENTRY_iso_ret"].append(r_eiso["return_pct"])
            per["EXIT_iso_ret"].append(r_xiso["return_pct"])
        results[rule] = per

    def agg(rs):
        return {
            "ret": np.mean([r["return_pct"] for r in rs]),
            "effSR": np.mean([r["effective_sharpe"] for r in rs]),
            "DD": np.mean([r["max_dd_pct"] for r in rs]),
            "trades": np.mean([r["trades"] for r in rs]),
            "dsr": np.mean([r["dsr"] for r in rs]),
        }

    # ---- attribution table ----
    print(f"{'rule':11s} | {'FULL ret':>8} {'eSR':>6} {'DD%':>6} | {'ENTRYiso ret':>12} {'EXITiso ret':>12} | {'entry contr':>11} {'exit contr':>10}")
    for rule in CORE:
        a = agg(results[rule]["FULL"])
        ei = agg(results[rule]["ENTRY_iso"])
        xi = agg(results[rule]["EXIT_iso"])
        entry_contr = a["ret"] - xi["ret"]   # same exits (d40) -> difference is entry
        exit_contr = a["ret"] - ei["ret"]    # same entries (R) -> difference is exit
        print(f"{rule:11s} | {a['ret']:+8.1f} {a['effSR']:+6.2f} {a['DD']:>5.1f}% | "
              f"{ei['ret']:+12.1f} {xi['ret']:+12.1f} | {entry_contr:+11.1f} {exit_contr:+10.1f}")

    # ---- paired vs LIVE donchian40 (FULL + isolated) ----
    print(f"\n=== PAIRED vs LIVE donchian40 (exact sign test on slice returns) ===")
    live_full = [r["return_pct"] for r in results["donchian40"]["FULL"]]
    for rule in CORE:
        for variant, key in [("FULL", "FULL"), ("ENTRY_iso", "ENTRY_iso"), ("EXIT_iso", "EXIT_iso")]:
            other = [r["return_pct"] for r in results[rule][key]]
            p = sign_test_p(other, live_full)
            delta = np.mean(other) - np.mean(live_full)
            better = sum(1 for o, l in zip(other, live_full) if o > l)
            print(f"   {rule:11s} {variant:10s} Δmean={delta:+6.1f} beats live {better}/{len(slices)} p={p:.3f}{' *' if p < 0.05 else ''}")
    print("\n* = significant at p<0.05 (exact binomial sign test, paired by slice)")


if __name__ == "__main__":
    main()
