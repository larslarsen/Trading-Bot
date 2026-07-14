"""
Chop-leg deep test: tsi, bop, ma30_50 as the chop rule (trend=REI), vs the LIVE
combo (donchian40 + ma30_ema fill). These three led the finer-slice scorecard
(+26.5 / +36.1 / +32.0% mean). Test whether any is a VALIDATED improvement over
live, using the same full-strategy panel + paired exact sign test, 8 WF slices.

Each candidate runs as: chop rule = candidate (donchian40 fill NOT used, to isolate
the candidate's own chop behavior); plus a 'combo' variant = donchian40 + candidate
fill, to see if the candidate also helps as a fill-in.
"""

import numpy as np
import pandas as pd
from portfolio_engine import PortfolioEngine, EngineConfig

import engine
from engine import load_screened_universe, get_regime_signals, improved_compute_live_regime

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS, POS_PCT = 5, 0.20
CANDIDATES = ["tsi", "bop", "ma30_50"]


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


def run(data, dates, chop_rule, combo=False):
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
        rule = "rei" if reg == "trend" else chop_rule
        active = []
        for s in pre:
            ent, _ = get_regime_signals(rule, pre[s].reset_index())
            if len(ent) and int(ent.iloc[-1]): active.append(s)
        if combo and reg == "chop" and len(active) == 0:
            for s in pre:
                ent, _ = get_regime_signals(chop_rule, pre[s].reset_index())
                if len(ent) and int(ent.iloc[-1]): active.append(s)
        prices = {s: float(pre[s]["close"].iloc[-1]) for s in pre}
        eng.start_daily_bar(next(iter(prices.values()), None))
        ok, _ = eng.check_circuit_breakers()
        if not ok:
            eng.flatten_all(prices); eq.append(eng.equity); continue
        to_close = set()
        for s in list(eng.positions.keys()):
            dfp = pre.get(s)
            if dfp is None: continue
            _, ex = get_regime_signals(rule, dfp.reset_index())
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
    rets = eq.pct_change().dropna()
    ret = (eq.iloc[-1] / 10000.0 - 1) * 100
    sr = (rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    in_mkt = (eq.diff() != 0).astype(int)
    exp = in_mkt.mean() if len(in_mkt) else 0.0
    eff_sr = sr * (exp ** 0.5) if exp > 0 else 0.0
    peak = 10000.0; mdd = 0.0
    for e in eq:
        peak = max(peak, e); mdd = max(mdd, (peak - e) / peak)
    calmar = (ret / 100.0) / mdd if mdd > 0 else 0.0
    return ret, eff_sr, mdd * 100, calmar


def main():
    data, dates = load_common(n_min=150)
    slices = []
    i = 60
    while i + 12 <= len(dates):
        slices.append(dates[i:i + 12]); i += 12
    print(f"Chop-leg deep test: tsi/bop/ma30_50 (+combo fill) vs LIVE, {len(slices)} WF slices, {len(data)} coins\n")
    res = {"LIVE(d40+ma30_ema)": []}
    for c in CANDIDATES:
        res[f"{c}"] = []
        res[f"{c}+fill"] = []
    # LIVE
    for seg in slices:
        res["LIVE(d40+ma30_ema)"].append(run(data, seg, "donchian40", combo=True))
    for c in CANDIDATES:
        res[c] = []
        res[f"{c}+fill"] = []
        for seg in slices:
            # candidate as SOLE chop rule (no donchian fill)
            res[c].append(run(data, seg, c, combo=False))
            # candidate as donchian40 fill-in (primary d40, candidate fills silent days)
            res[f"{c}+fill"].append(run(data, seg, c, combo=True))
    print(f"{'config':22s} {'meanRet':>8} {'effSR':>7} {'meanDD':>7} {'Calmar':>7}")
    agg = {}
    for k in res:
        rs = res[k]
        agg[k] = (np.mean([r[0] for r in rs]), np.mean([r[1] for r in rs]),
                  np.mean([r[2] for r in rs]), np.mean([r[3] for r in rs]))
    for k in sorted(agg, key=lambda x: -agg[x][0]):
        a = agg[k]
        print(f"{k:22s} {a[0]:+8.1f} {a[1]:+7.2f} {a[2]:>6.1f}% {a[3]:>7.2f}")
    # paired vs LIVE
    base = [r[0] for r in res["LIVE(d40+ma30_ema)"]]
    print("\nPaired vs LIVE (exact sign test on slice returns):")
    for k in res:
        if k == "LIVE(d40+ma30_ema)": continue
        other = [r[0] for r in res[k]]
        d = np.array(other) - np.array(base)
        n = np.sum(d != 0); kk = np.sum(d > 0)
        from math import comb
        p = min(1.0, 2 * sum(comb(n, i) for i in range(min(kk, n - kk) + 1)) / 2 ** n) if n else 1.0
        print(f"   {k:22s} Δmean={np.mean(other)-np.mean(base):+6.1f}  beats LIVE in {kk}/{n}  p={p:.3f}")


if __name__ == "__main__":
    main()
