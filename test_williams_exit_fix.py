"""
Chop-regime replay: buggy Williams exit vs fixed Williams exit.

Causal, day-by-day, shared PortfolioEngine (identical to live math).
Forces CHOP regime so Williams %R is the active rule for BOTH variants.
The ONLY difference is the exit condition inside williams_r_signals:

  buggy: exit = (wr < -20) & falling        -> sells at oversold bottom
  fixed: exit = (prev >= -20) & (wr < -20)  -> sells only after recovery

We monkeypatch engine.williams_r_signals to the buggy form for one variant
and use the (now fixed) module default for the other, so the comparison is
isolated to the exit logic alone.
"""

import copy
import numpy as np
import pandas as pd
from portfolio_engine import PortfolioEngine, EngineConfig

import engine
from engine import load_screened_universe

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS, POS_PCT = 5, 0.20


def williams_buggy(df, period=14):
    # SAME entry as fixed (cross-up through -80) so we ISOLATE the exit.
    # Only the exit differs: buggy = sell whenever oversold & falling.
    high = pd.Series(df["high"].values, index=df.index)
    low = pd.Series(df["low"].values, index=df.index)
    close = pd.Series(df["close"].values, index=df.index)
    hh = high.rolling(period, min_periods=1).max()
    ll = low.rolling(period, min_periods=1).min()
    wr = -100 * (hh - close) / (hh - ll + 1e-12)
    prev = wr.shift(1)
    entry = ((prev <= -80) & (wr > -80)).astype(int)
    exit_sig = ((wr < -20) & (wr.diff() < 0)).astype(int)
    return entry, exit_sig


def load_common(n_min=150):
    cd = load_screened_universe(min_bars=n_min)
    data = {s: d.set_index("ts").sort_index() for s, d in cd.items()}
    idx = None
    for d in data.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    data = {s: d.loc[idx] for s, d in data.items() if len(d.loc[idx]) > 0}
    return data, pd.DatetimeIndex(sorted(idx))


def run(data, dates, williams_fn):
    cfg = EngineConfig(
        initial_capital=10000.0, max_daily_loss_pct=0.03, max_drawdown_pct=0.20,
        max_positions=MAX_POS, max_position_pct=POS_PCT, min_equity_to_trade=100.0,
        flash_crash_bars=5, flash_crash_pct=0.50, extreme_move_pct=0.90,
        cost_bps=COST_BPS, slippage_bps=SLIP_BPS, enable_vol_target=False,
    )
    saved = engine.williams_r_signals
    engine.williams_r_signals = williams_fn
    try:
        eng = PortfolioEngine(cfg)
        eq = []
        for day in dates:
            pre = {s: d.loc[:day] for s, d in data.items() if len(d.loc[:day]) > 0}
            if not pre:
                eq.append(eng.equity); continue
            # force CHOP: Williams is the rule
            active = [s for s in pre if int(engine.get_regime_signals("williams_r", pre[s].reset_index())[0].iloc[-1])]
            prices = {s: float(pre[s]["close"].iloc[-1]) for s in pre}
            eng.start_daily_bar(next(iter(prices.values()), None))
            ok, _ = eng.check_circuit_breakers()
            if not ok:
                eng.flatten_all(prices); eq.append(eng.equity); continue
            to_close = set()
            for s in list(eng.positions.keys()):
                dfp = pre.get(s)
                if dfp is None: continue
                ent, ex = engine.get_regime_signals("williams_r", dfp.reset_index())
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
        return {"ret": round(ret, 1), "sr": round(sr, 2), "mdd": round(mdd*100, 1), "trades": len(eng.trades), "final": round(final, 2)}
    finally:
        engine.williams_r_signals = saved


def main():
    data, dates = load_common(n_min=150)
    windows = {
        "last 90d": dates[-90:],
        "last 60d": dates[-60:],
        "mid 60d ": dates[30:90],
    }
    print(f"Forced-CHOP Williams %R replay, {len(data)} coins\n")
    agg_bug, agg_fix = [], []
    for label, w in windows.items():
        if len(w) < 30:
            continue
        bug = run(data, w, williams_buggy)
        fix = run(data, w, engine.williams_r_signals)
        agg_bug.append(bug["ret"]); agg_fix.append(fix["ret"])
        delta = fix["ret"] - bug["ret"]
        verdict = "FIXED better" if delta > 0 else "BUGGY better"
        print(f"  [{label}] BUGGY ret={bug['ret']:>6}% mdd={bug['mdd']:>4}% tr={bug['trades']} | "
              f"FIXED ret={fix['ret']:>6}% mdd={fix['mdd']:>4}% tr={fix['trades']} -> {verdict} ({delta:+.1f}%)")
    mb, mf = float(np.mean(agg_bug)), float(np.mean(agg_fix))
    print(f"\n  MEAN across windows: BUGGY {mb:.1f}%  FIXED {mf:.1f}%  delta {mf-mb:+.1f}%")
    print(f"  VERDICT: literature-correct exit {'IMPROVES' if mf > mb else 'DOES NOT IMPROVE'} chop results on this data")


if __name__ == "__main__":
    main()
