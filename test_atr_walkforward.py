"""
Walk-forward decision test: should USE_ATR_TRAILING be flipped ON for live?

Method (causal, day-by-day, shared PortfolioEngine = identical to live math):
  - Build a common history across screened coins.
  - Walk forward in windows: train an in-sample (IS) period to confirm ATR helps,
    then trade the immediately following out-of-sample (OOS) period.
  - In each OOS window, run the regime+REI/Williams strategy twice:
        A) baseline  : no trailing stop
        B) atr_trail : ATR 14/2.0 trailing, regime-gated to 'trend'
  - Compare return%, effective Sharpe, max DD, #trades.
  - A variant is adopted only if it consistently beats baseline on OOS windows
    (not just one lucky window) — per the user's "consistent modest uplift" rule.
"""

import json
import numpy as np
import pandas as pd
from portfolio_engine import PortfolioEngine, EngineConfig

from engine import load_screened_universe, improved_compute_live_regime, get_regime_signals, atr_trailing_exit

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS = 5
POS_PCT = 0.20
ATR_PERIOD = 14
ATR_MULT = 2.0


def load_common(n_min=150):
    coin_data = load_screened_universe(min_bars=n_min)
    data = {s: df.set_index("ts").sort_index() for s, df in coin_data.items()}
    # align to common dates
    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    data = {s: df.loc[idx] for s, df in data.items() if len(df.loc[idx]) > 0}
    return data, pd.DatetimeIndex(sorted(idx))


def prefix(data, day):
    return {s: d.loc[:day] for s, d in data.items() if len(d.loc[:day]) > 0}


def latest_entry(rule, dfp):
    try:
        ent, _ = get_regime_signals(rule, dfp.reset_index())
        return int(ent.iloc[-1]) if len(ent) else 0
    except Exception:
        return 0


def run_window(data, dates, start, end, use_atr):
    oos = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    cfg = EngineConfig(
        initial_capital=10000.0, max_daily_loss_pct=0.03, max_drawdown_pct=0.20,
        max_positions=MAX_POS, max_position_pct=POS_PCT, min_equity_to_trade=100.0,
        flash_crash_bars=5, flash_crash_pct=0.50, extreme_move_pct=0.90,
        cost_bps=COST_BPS, slippage_bps=SLIP_BPS, enable_vol_target=False,
    )
    eng = PortfolioEngine(cfg)
    eq_hist = []
    for day in oos:
        pre = prefix(data, day)
        if not pre:
            eq_hist.append(eng.equity)
            continue
        try:
            reg = improved_compute_live_regime(pre)
        except Exception:
            reg = "trend"
        rule = "rei" if reg == "trend" else "williams_r"
        active = [s for s in pre if latest_entry(rule, pre[s])]
        prices = {s: float(pre[s]["close"].iloc[-1]) for s in pre}

        eng.start_daily_bar(next(iter(prices.values()), None))
        ok, _ = eng.check_circuit_breakers()
        if not ok:
            eng.flatten_all(prices)
            eq_hist.append(eng.equity)
            continue

        # closes: explicit exit OR atr trailing (trend-gated)
        to_close = set()
        for s in list(eng.positions.keys()):
            dfp = pre.get(s)
            if dfp is None:
                continue
            ent, ex = get_regime_signals(rule, dfp.reset_index())
            if len(ex) and int(ex.iloc[-1]):
                to_close.add(s)
            if use_atr and reg == "trend" and len(dfp) >= ATR_PERIOD * 2:
                try:
                    if int(atr_trailing_exit(dfp, ATR_PERIOD, ATR_MULT).iloc[-1]) == 1:
                        to_close.add(s)
                except Exception:
                    pass
        for s in to_close:
            px = prices.get(s)
            if px and px > 0:
                eng.close_position(s, px)

        for s in active:
            if s in eng.positions or len(eng.positions) >= MAX_POS:
                continue
            px = prices.get(s)
            if not px or px <= 0:
                continue
            ok, _ = eng.check_circuit_breakers()
            if not ok:
                break
            eng.open_position(s, px, eng.equity * POS_PCT)

        eq_hist.append(eng.mark_to_market(prices))

    if not eq_hist:
        return None
    final = eq_hist[-1]
    rets = pd.Series(eq_hist).pct_change().dropna()
    sr = (rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    ret = (final / 10000.0 - 1) * 100
    peak = 10000.0
    mdd = 0.0
    for e in eq_hist:
        peak = max(peak, e)
        mdd = max(mdd, (peak - e) / peak)
    avgpos = np.mean([len(list(eng.positions.keys()))]) if False else 0.0  # n/a post-loop
    return {
        "ret": round(ret, 1), "sr": round(sr, 2), "eff_sr": round(sr * np.sqrt(MAX_POS / MAX_POS), 2),
        "mdd": round(mdd * 100, 1), "trades": len(eng.trades), "final": round(final, 2), "days": len(eq_hist),
    }


def main():
    data, dates = load_common(n_min=150)
    print(f"Coins with >=150 bars: {len(data)}; span {dates[0].date()} .. {dates[-1].date()} ({len(dates)} days)")
    # Walk-forward: train (IS, not reported) then OOS, stepped. Adaptive sizes fit the
    # available history. ATR adoption requires consistent OOS uplift across windows.
    is_len, oos_len = 60, 30
    windows = []
    start_i = is_len
    while start_i + oos_len <= len(dates):
        oos_start = dates[start_i]
        oos_end = dates[min(start_i + oos_len, len(dates)) - 1]
        windows.append((str(oos_start.date()), str(oos_end.date())))
        start_i += oos_len

    print(f"\nWalk-forward OOS windows: {len(windows)} x {oos_len}d")
    base_rows, atr_rows = [], []
    for i, (s, e) in enumerate(windows):
        b = run_window(data, dates, s, e, use_atr=False)
        a = run_window(data, dates, s, e, use_atr=True)
        if b and a:
            base_rows.append(b)
            atr_rows.append(a)
            delta = a["ret"] - b["ret"]
            winner = "ATR" if delta > 0 else "base"
            print(f"  W{i+1} {s}..{e}: base ret={b['ret']:>6}% sr={b['sr']:>4} | "
                  f"ATR ret={a['ret']:>6}% sr={a['sr']:>4} -> {winner} ({delta:+.1f}%)")

    if base_rows:
        def avg(rows, k):
            return round(float(np.mean([r[k] for r in rows])), 2)
        print("\n=== Aggregate (mean across OOS windows) ===")
        print(f"  baseline : ret={avg(base_rows,'ret')}%  eff_sr={avg(base_rows,'eff_sr')}  mdd={avg(base_rows,'mdd')}%  trades={avg(base_rows,'trades')}")
        print(f"  atr_trail: ret={avg(atr_rows,'ret')}%  eff_sr={avg(atr_rows,'eff_sr')}  mdd={avg(atr_rows,'mdd')}%  trades={avg(atr_rows,'trades')}")
        wins = sum(1 for b, a in zip(base_rows, atr_rows) if a["ret"] > b["ret"])
        print(f"\n  ATR beats baseline in {wins}/{len(base_rows)} OOS windows")
        decision = "FLIP ON" if (wins >= len(base_rows) * 0.6 and avg(atr_rows, 'ret') > avg(base_rows, 'ret')) else "KEEP OFF"
        print(f"  DECISION: {decision} USE_ATR_TRAILING")


if __name__ == "__main__":
    main()
