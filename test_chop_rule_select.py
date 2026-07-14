"""
Chop-regime rule selection (literature-correct Williams + ATR trailing vs others).

Question: which EXIT (and rule) should drive CHOP regime? We now have:
  - williams_r        : literature-correct entry (cross UP -80) + recovery exit (cross UP -20) [UNVALIDATED]
  - williams_r_buggy  : legacy buggy exit (sells at oversold bottom) [UNVALIDATED, pathological]
  - rei               : trend rule (momentum) — our validated TREND rule
  - atr_trailing      : not a signal rule; an EXIT overlay. We test "williams-entry + ATR(14,2.0) trailing
                        exit, gated to trend" which is what live already uses in trend.

Method: causal day-by-day, shared PortfolioEngine, walk-forward across 3 windows.
For the ATR-overlay variant we use williams entry + ATR trailing stop as the exit
(trend-gated per live config). This lets us compare:
  A) williams (recov exit)      B) williams_buggy exit
  C) williams + ATR trailing     D) rei (as a chop proxy / control)
Reporting ret%, effSR, mdd%, trades for each, so we can actually SELECT a chop rule.
"""

import numpy as np
import pandas as pd
from portfolio_engine import PortfolioEngine, EngineConfig

import engine
from engine import load_screened_universe, get_regime_signals, atr_trailing_exit

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS, POS_PCT = 5, 0.20
ATR_P, ATR_M = 14, 2.0


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


def run_variant(data, dates, rule, use_atr=False):
    cfg = EngineConfig(
        initial_capital=10000.0, max_daily_loss_pct=0.03, max_drawdown_pct=0.20,
        max_positions=MAX_POS, max_position_pct=POS_PCT, min_equity_to_trade=100.0,
        flash_crash_bars=5, flash_crash_pct=0.50, extreme_move_pct=0.90,
        cost_bps=COST_BPS, slippage_bps=SLIP_BPS, enable_vol_target=False,
    )
    eng = PortfolioEngine(cfg)
    eq = []
    for day in dates:
        pre = prefix(data, day)
        if not pre:
            eq.append(eng.equity); continue
        # forced chop: active rule is the one under test
        active = [s for s in pre if int(get_regime_signals(rule, pre[s].reset_index())[0].iloc[-1])]
        prices = {s: float(pre[s]["close"].iloc[-1]) for s in pre}
        eng.start_daily_bar(next(iter(prices.values()), None))
        ok, _ = eng.check_circuit_breakers()
        if not ok:
            eng.flatten_all(prices); eq.append(eng.equity); continue
        to_close = set()
        for s in list(eng.positions.keys()):
            dfp = pre.get(s)
            if dfp is None: continue
            ent, ex = get_regime_signals(rule, dfp.reset_index())
            if len(ex) and int(ex.iloc[-1]): to_close.add(s)
            if use_atr and len(dfp) >= ATR_P * 2:
                try:
                    if int(atr_trailing_exit(dfp, ATR_P, ATR_M).iloc[-1]) == 1:
                        to_close.add(s)
                except Exception:
                    pass
        for s in to_close:
            px = prices.get(s)
            if px and px > 0: eng.close_position(s, px)
        for s in active:
            if s in eng.positions or len(eng.positions) >= MAX_POS: continue
            px = prices.get(s)
            if not px or px <= 0: continue
            ok, _ = eng.check_circuit_breakers()
            if not ok: break
            eng.open_position(s, px, eng.equity * POS_PCT)
        eq.append(eng.mark_to_market(prices))
    eq = pd.Series(eq)
    final = eq.iloc[-1]
    ret = (final / 10000.0 - 1) * 100
    rets = eq.pct_change().dropna()
    sr = (rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    peak = 10000.0; mdd = 0.0
    for e in eq:
        peak = max(peak, e); mdd = max(mdd, (peak - e) / peak)
    return {"ret": round(ret, 1), "sr": round(sr, 2), "mdd": round(mdd*100, 1), "trades": len(eng.trades)}


def main():
    data, dates = load_common(n_min=150)
    windows = {
        "last 90d": dates[-90:],
        "last 60d": dates[-60:],
        "mid 60d ": dates[30:90],
    }
    variants = [
        ("williams (recov)", "williams_r", False),
        ("williams_buggy",   "williams_r_buggy", False),
        ("williams+ATR",     "williams_r", True),
        ("rei (control)",    "rei", False),
    ]
    print(f"Forced-CHOP comparison, {len(data)} coins, shared engine\n")
    results = {v[0]: [] for v in variants}
    for wlabel, w in windows.items():
        if len(w) < 30: continue
        print(f"[{wlabel}]")
        for vlabel, rule, use_atr in variants:
            r = run_variant(data, w, rule, use_atr)
            results[vlabel].append(r["ret"])
            print(f"   {vlabel:16s} ret={r['ret']:>6}%  sr={r['sr']:>5}  mdd={r['mdd']:>4}%  tr={r['trades']}")
        print()
    print("=== MEAN ret% across windows ===")
    for vlabel, _, _ in variants:
        m = float(np.mean(results[vlabel]))
        print(f"   {vlabel:16s} {m:+.1f}%")
    best = max(results, key=lambda k: float(np.mean(results[k])))
    print(f"\n   BEST chop variant by mean ret: {best}")


if __name__ == "__main__":
    main()
