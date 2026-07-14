"""
Regime-switching COMBO comparison (non-overlapping walk-forward).

Fixes the flaw in test_rule_headtohead.py (overlapping windows double-counted).
Each strategy is a FULL switched system: pick rule by regime each day
(trend -> trend_rule, chop -> chop_rule), run its own entry+exit on the
shared PortfolioEngine (identical to live math).

Combos tested (trend rule fixed = rei; vary the CHOP rule, the weak link):
  rei + williams_r        (CURRENT LIVE config)
  rei + williams_r_buggy  (legacy, pathological)
  rei + cci
  rei + tsi
  rei + rsi

Walk-forward: train 60d (IS, not reported) then OOS 30d, step 30d, NON-OVERLAPPING
OOS windows. Reports ret%, effSR, mdd%, trades per OOS window + MEAN +
#positive windows, so we pick the best chop rule to pair with REI.
"""

import numpy as np
import pandas as pd
from portfolio_engine import PortfolioEngine, EngineConfig

import engine
from engine import load_screened_universe, get_regime_signals, improved_compute_live_regime

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS, POS_PCT = 5, 0.20

COMBOS = [
    ("rei+williams_r", "rei", "williams_r"),
    ("rei+williams_bug", "rei", "williams_r_buggy"),
    ("rei+cci", "rei", "cci"),
    ("rei+tsi", "rei", "tsi"),
    ("rei+rsi", "rei", "rsi"),
]


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


def run_combo(data, dates, trend_rule, chop_rule):
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
        try:
            reg = improved_compute_live_regime(pre)
        except Exception:
            reg = "trend"
        rule = trend_rule if reg == "trend" else chop_rule
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
    # Find regime per day so we can build OOS windows with REAL chop exposure.
    # (The earlier naive windows were ~all-trend, so the chop rule never fired
    #  and all combos degenerated to pure REI — a false "tie".)
    day_regime = []
    for day in dates:
        pre = prefix(data, day)
        if not pre:
            day_regime.append("trend"); continue
        try:
            day_regime.append(improved_compute_live_regime(pre))
        except Exception:
            day_regime.append("trend")
    # Build ~30d OOS windows stepping 30d, but report chop-day count per window.
    oos_len = 30
    oos_windows = []
    i = 60
    while i + oos_len <= len(dates):
        seg = dates[i:i + oos_len]
        chop_n = sum(1 for d in seg if day_regime[list(dates).index(d)] == "chop")
        oos_windows.append((dates[i], dates[i + oos_len - 1], chop_n))
        i += oos_len
    print(f"Regime-switching combo comparison, {len(data)} coins, shared engine")
    print(f"{len(oos_windows)} x {oos_len}d OOS windows (train 60d each); chop days shown per window\n")
    res = {c[0]: [] for c in COMBOS}
    for wi, (s, e, cn) in enumerate(oos_windows, 1):
        w = dates[(dates >= s) & (dates <= e)]
        print(f"[OOS W{wi}: {str(s.date())}..{str(e.date())}] chop_days={cn}")
        for name, tr, cr in COMBOS:
            out = run_combo(data, w, tr, cr)
            res[name].append(out["ret"])
            print(f"   {name:18s} ret={out['ret']:>7}%  sr={out['sr']:>5}  mdd={out['mdd']:>4}%  tr={out['trades']}")
        print()
    print("=== MEAN ret% across OOS windows ===")
    means = {c[0]: float(np.mean(res[c[0]])) for c in COMBOS}
    for name in sorted(means, key=means.get, reverse=True):
        pos = sum(1 for x in res[name] if x > 0)
        print(f"   {name:18s} {means[name]:+7.1f}%   positive OOS {pos}/{len(res[name])}")
    best = max(means, key=means.get)
    print(f"\n   BEST chop rule paired with REI: {best} ({means[best]:+.1f}%)")


if __name__ == "__main__":
    main()
