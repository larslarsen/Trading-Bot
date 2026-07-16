#!/usr/bin/env python3
"""Futures grid on BloFin: how many average-downs before FORCED liquidation.

Your friend: "average down into infinity, 2-3x levered, long win streaks."
Reality on futures: the exchange liquidates you. Liquidation price of a
blended (averaged) long is

    liq_price = Pav * (1 - 1/L)

where Pav = volume-weighted average entry, L = leverage. This is EXACT for
isolated margin, equal leverage per add, ignoring fees/maintenance (so it is
GENEROUS to him -- real maintenance margin liquidates sooner).

We simulate his grid on REAL CEX 1d price paths:
  - enter long 1 unit at a dip (RSI<35)
  - every `step` adverse move, add 1 unit (average down)
  - liquidate the moment price <= liq_price
  - also cap total margin at account balance B (finite capital)
  - if price recovers to Pav first, close at profit (the "win streak")
Report: adds-to-liquidation distribution, and total % Adverse move at liq.
"""
import numpy as np
import pandas as pd
from engine import load_screened_universe


def rsi(s, n=14):
    d = s.diff()
    rs = d.clip(lower=0).rolling(n).mean() / (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + rs)


def sim_grid(price, L, step, B):
    """Walk one price path. Return list of outcomes: ('liq', adds, adv_pct)
    or ('tp', adds)."""
    out = []
    r = rsi(price)
    sig = (r < 35).astype(int)
    for e in range(1, len(price) - 1):
        if sig.iloc[e] == 0:
            continue
        # build a grid starting at e
        units = [1.0]
        pents = [price.iloc[e]]
        Pav = price.iloc[e]
        margin_used = pents[0] / L
        liq = Pav * (1 - 1.0 / L)
        last_add = price.iloc[e]
        adds = 0
        liquidated = False
        for i in range(e + 1, len(price)):
            p = price.iloc[i]
            # average down
            if p <= last_add * (1 - step) and margin_used < B:
                units.append(1.0)
                pents.append(p)
                U = sum(units)
                Pav = sum(u * pp for u, pp in zip(units, pents)) / U
                margin_used = sum(pp / L for pp in pents)
                last_add = p
                adds += 1
                liq = Pav * (1 - 1.0 / L)
            # liquidation check
            if p <= liq:
                adv = (Pav - p) / Pav
                out.append(("liq", adds, adv))
                liquidated = True
                break
            # take profit: price recovered to avg entry
            if p >= Pav:
                out.append(("tp", adds))
                break
        if not liquidated and not out:
            pass
    return out


def main():
    coins = load_screened_universe(min_bars=200)
    # normalize each to a starting price of 100 for readability of % moves
    paths = []
    for sym, df in coins.items():
        p = df["close"].astype(float).reset_index(drop=True)
        paths.append(p / p.iloc[0] * 100)

    for L in (2, 3):
        for step in (0.05, 0.10):
            B = 1e9  # effectively infinite account (tests pure liq-price math)
            res = []
            for p in paths:
                res.extend(sim_grid(p, L, step, B))
            liqs = [x for x in res if x[0] == "liq"]
            tps = [x for x in res if x[0] == "tp"]
            if liqs:
                adds = np.array([x[1] for x in liqs])
                adv = np.array([x[2] for x in liqs]) * 100
                print(f"L={L}x step={int(step*100)}% | grids={len(res)} "
                      f"liq={len(liqs)} tp={len(tps)} | "
                      f"adds-to-liq median={np.median(adds):.0f} max={adds.max()} | "
                      f"adverse%@liq median={np.median(adv):.1f}% max={adv.max():.1f}%")
            else:
                print(f"L={L}x step={int(step*100)}% | grids={len(res)} "
                      f"liq=0 tp={len(tps)} (no liquidations in sample)")
    print("\nKey: at Lx, liquidation band is 1/L adverse (2x=50%, 3x=33%). "
          "Averaging down does NOT widen it -- liq moves with avg entry. "
          "Finite account only tightens it further.")


if __name__ == "__main__":
    main()
