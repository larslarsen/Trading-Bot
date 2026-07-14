"""
Chop EXIT screen: Williams %R ENTRY fixed; vary ONLY the EXIT.

Goal: find a robust chop exit (not the pathological buggy one, not the
underperforming recovery exit). Williams entry = cross UP through -80.
Exits tested:
  1. recovery     : Williams cross UP -20 (lit-correct, current)
  2. time5        : fixed 5-day hold then exit (pure mean-reversion horizon)
  3. regime_flip  : exit when regime leaves chop (trend takes over)
  4. cci_100      : exit when CCI crosses above +100 (overbought)
  5. rsi_50       : exit when RSI crosses above 50 (recovered to midline)

Method: causal day-by-day, shared PortfolioEngine, 3 walk-forward windows.
Reports ret%, effSR, mdd%, trades per window + MEAN — so we can select a
robust exit, not a lucky-window one.
"""

import numpy as np
import pandas as pd
from portfolio_engine import PortfolioEngine, EngineConfig

import engine
from engine import (
    load_screened_universe, get_regime_signals, atr_trailing_exit,
    improved_compute_live_regime,
)

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS, POS_PCT = 5, 0.20
HOLD_DAYS = 5


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


def entry_signal(sym_df):
    ent, _ = get_regime_signals("williams_r", sym_df.reset_index())
    return int(ent.iloc[-1]) if len(ent) else 0


def run_exit(data, dates, exit_kind):
    cfg = EngineConfig(
        initial_capital=10000.0, max_daily_loss_pct=0.03, max_drawdown_pct=0.20,
        max_positions=MAX_POS, max_position_pct=POS_PCT, min_equity_to_trade=100.0,
        flash_crash_bars=5, flash_crash_pct=0.50, extreme_move_pct=0.90,
        cost_bps=COST_BPS, slippage_bps=SLIP_BPS, enable_vol_target=False,
    )
    eng = PortfolioEngine(cfg)
    # track entry day per symbol for time-hold
    entry_day = {}
    eq = []
    for i, day in enumerate(dates):
        pre = prefix(data, day)
        if not pre:
            eq.append(eng.equity); continue
        prices = {s: float(pre[s]["close"].iloc[-1]) for s in pre}
        eng.start_daily_bar(next(iter(prices.values()), None))
        ok, _ = eng.check_circuit_breakers()
        if not ok:
            eng.flatten_all(prices); eq.append(eng.equity); continue

        # regime for regime_flip exit
        try:
            reg = improved_compute_live_regime(pre)
        except Exception:
            reg = "trend"

        to_close = set()
        for s in list(eng.positions.keys()):
            dfp = pre.get(s)
            if dfp is None:
                continue
            if exit_kind == "recovery":
                _, ex = get_regime_signals("williams_r", dfp.reset_index())
                if len(ex) and int(ex.iloc[-1]): to_close.add(s)
            elif exit_kind == "time5":
                if i - entry_day.get(s, i) >= HOLD_DAYS: to_close.add(s)
            elif exit_kind == "regime_flip":
                if reg != "chop": to_close.add(s)
            elif exit_kind == "cci_100":
                _, ex = get_regime_signals("cci", dfp.reset_index())
                if len(ex) and int(ex.iloc[-1]): to_close.add(s)
            elif exit_kind == "rsi_50":
                _, ex = get_regime_signals("rsi", dfp.reset_index())
                if len(ex) and int(ex.iloc[-1]): to_close.add(s)
        for s in to_close:
            px = prices.get(s)
            if px and px > 0: eng.close_position(s, px)

        # entries (Williams entry)
        for s in pre:
            if s in eng.positions or len(eng.positions) >= MAX_POS: continue
            px = prices.get(s)
            if not px or px <= 0: continue
            if entry_signal(pre[s]):
                ok, _ = eng.check_circuit_breakers()
                if not ok: break
                eng.open_position(s, px, eng.equity * POS_PCT)
                entry_day[s] = i
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
    windows = {"last 90d": dates[-90:], "last 60d": dates[-60:], "mid 60d": dates[30:90]}
    exits = ["recovery", "time5", "regime_flip", "cci_100", "rsi_50"]
    print(f"Chop EXIT screen (Williams entry fixed), {len(data)} coins, shared engine\n")
    results = {e: [] for e in exits}
    for wlabel, w in windows.items():
        if len(w) < 30: continue
        print(f"[{wlabel}]")
        for e in exits:
            r = run_exit(data, w, e)
            results[e].append(r["ret"])
            print(f"   {e:12s} ret={r['ret']:>6}%  sr={r['sr']:>5}  mdd={r['mdd']:>4}%  tr={r['trades']}")
        print()
    print("=== MEAN ret% across windows (robustness) ===")
    means = {e: float(np.mean(results[e])) for e in exits}
    for e in sorted(means, key=means.get, reverse=True):
        print(f"   {e:12s} {means[e]:+.1f}%")
    best = max(means, key=means.get)
    print(f"\n   BEST chop exit by mean ret: {best} ({means[best]:+.1f}%)")
    # robustness: how many windows positive?
    pos = {e: sum(1 for x in results[e] if x > 0) for e in exits}
    print("   windows positive (of 3):", {e: pos[e] for e in exits})


if __name__ == "__main__":
    main()
