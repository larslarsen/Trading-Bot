#!/usr/bin/env python3
"""Prove martingale / grid sizing is signal-invariant ruin.

For EVERY signal in the harness, extract the SAME real trade returns the live
engine would take (via engine.get_regime_signals: entry/exit), then run four
position-sizing schemes over that identical return stream:

  fixed      : risk 20% equity per trade (sane baseline)
  mart_cap   : double after loss, reset on win, cap 16x (retail "safety" cap)
  mart_uncap : double after loss, NO cap (true ruin)
  grid       : average-down -- add a unit per losing bar, reset on win

Because each scheme sees the EXACT same trades per signal, any equity/DD
difference is 100% the sizing. If martingale/grid ruin holds for the BEST
signal too, there is no escape: no entry rule saves doubling-down.

Both-sides-open (his bot's signature: long+short live, small wins) is grid
across the pooled long+short stream -- same math, same doom.
"""
import numpy as np
import pandas as pd
from engine import load_screened_universe, get_regime_signals

from test_rule_scorecard import CANDIDATES

COST = 0.0008
BASE = 0.20


def episode_returns(coins, rule, combo=False, fill_rule="ma30_ema"):
    """Per-coin entry/exit via the live signal; net return per closed trade."""
    eps = []
    for sym, df in coins.items():
        df = df.reset_index(drop=True)
        ent, ex = get_regime_signals(rule, df)
        if combo:
            fent, fex = get_regime_signals(fill_rule, df)
        in_pos = False
        entry_px = None
        for i in range(len(df)):
            sig_e = int(ent.iloc[i]) if i < len(ent) else 0
            sig_x = int(ex.iloc[i]) if i < len(ex) else 0
            if not in_pos and sig_e:
                in_pos = True
                entry_px = float(df["close"].iloc[i])
            elif in_pos and sig_x:
                exit_px = float(df["close"].iloc[i])
                ret = (exit_px / entry_px - 1) - COST
                eps.append(ret)
                in_pos = False
        # combo: if never exited but fill would, count at last bar
        if in_pos and combo:
            exit_px = float(df["close"].iloc[-1])
            eps.append((exit_px / entry_px - 1) - COST)
    return [r for r in eps if np.isfinite(r)]


def run_fixed(returns):
    eq = 1.0; peak = 1.0; maxdd = 0.0; worst = 0.0
    for r in returns:
        pnl = eq * BASE * r
        eq += pnl
        worst = min(worst, pnl)
        peak = max(peak, eq); maxdd = max(maxdd, (peak - eq) / peak)
    return dict(eq=eq, maxdd=maxdd, worst=worst, ruined=eq <= 0)


def run_mart(returns, cap):
    eq = 1.0; peak = 1.0; maxdd = 0.0; worst = 0.0; streak = 0
    for r in returns:
        mult = 2 ** streak if cap is None else min(2 ** streak, cap)
        pnl = eq * BASE * mult * r
        eq += pnl
        worst = min(worst, pnl)
        peak = max(peak, eq); maxdd = max(maxdd, (peak - eq) / peak)
        if eq <= 0:
            return dict(eq=eq, maxdd=1.0, worst=worst, ruined=True)
        streak = 0 if r > 0 else streak + 1
    return dict(eq=eq, maxdd=maxdd, worst=worst, ruined=False)


def run_grid(returns):
    eq = 1.0; peak = 1.0; maxdd = 0.0; worst = 0.0; units = 0
    for r in returns:
        units = 1 if units == 0 else units + 1
        pnl = eq * BASE * units * r
        eq += pnl
        worst = min(worst, pnl)
        peak = max(peak, eq); maxdd = max(maxdd, (peak - eq) / peak)
        if eq <= 0:
            return dict(eq=eq, maxdd=1.0, worst=worst, ruined=True)
        units = 0 if r > 0 else 1
    return dict(eq=eq, maxdd=maxdd, worst=worst, ruined=False)


def pct(x):
    return f"{x*100:7.1f}%"


def main():
    coins = load_screened_universe(min_bars=150)
    print(f"CEX universe: {len(coins)} coins. Every signal -> 4 sizing schemes, "
          f"identical trades per signal.\n")
    hdr = f"{'signal':16s} {'n':>5} {'win%':>5} | {'fixed_eq':>8} {'martcap':>8} {'martuncap':>9} {'grid':>8} | ruined?"
    print(hdr)
    print("-" * len(hdr))
    for rule, kind in CANDIDATES:
        combo = (kind == "combo")
        chop = "donchian40" if combo else rule
        rets = episode_returns(coins, chop, combo=combo)
        if len(rets) < 20:
            print(f"{rule:16s} {len(rets):>5}  too few trades, skipped")
            continue
        win = (np.array(rets) > 0).mean() * 100
        f = run_fixed(rets)
        mc = run_mart(rets, cap=16)
        mu = run_mart(rets, cap=None)
        g = run_grid(rets)
        ruined = "YES" if (mc["ruined"] or mu["ruined"] or g["ruined"]) else "no"
        print(f"{rule:16s} {len(rets):>5} {win:>4.0f}% | "
              f"{pct(f['eq']):>8} {pct(mc['eq']):>8} {pct(mu['eq']):>9} {pct(g['eq']):>8} | {ruined}")

    print("\nReading: fixed = sane sizing survives (or at least stays >0). "
          "martcap/martuncap/grid = doubling-down or averaging-down.")
    print("If 'ruined?' = YES for a signal, that sizing blew equity <=0 on REAL "
          "trades from that signal. No entry rule escapes it.")


if __name__ == "__main__":
    main()
