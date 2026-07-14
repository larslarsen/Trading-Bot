"""
Walk-forward: settle the CHOP-LEG decision — donchian40 (live now) vs tsi.

Both paired with REI trend, full switched strategy (own entry+exit),
shared PortfolioEngine. 6 non-overlapping 20d OOS slices over the whole
history, with chop-day counts per slice so we see where each rule actually
fires. Reports mean ret%, #positive slices, and chop-heavy mean for each.
"""

import numpy as np
import pandas as pd
from portfolio_engine import PortfolioEngine, EngineConfig

import engine
from engine import load_screened_universe, get_regime_signals, improved_compute_live_regime

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS, POS_PCT = 5, 0.20
CANDIDATES = ["donchian40", "tsi"]


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


def run_strategy(data, dates, chop_rule):
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
    train, step, oos = 40, 20, 20
    slices = []
    i = train
    while i + oos <= len(dates):
        slices.append(dates[i:i + oos]); i += step
    print(f"Chop-leg walk-forward: donchian40 (LIVE) vs tsi, {len(slices)} non-overlapping {oos}d slices, {len(data)} coins\n")
    res = {c: [] for c in CANDIDATES}
    chop_res = {c: [] for c in CANDIDATES}
    for si, seg in enumerate(slices, 1):
        chop_n = 0
        for day in seg:
            pre = prefix(data, day)
            if not pre: continue
            try:
                if improved_compute_live_regime(pre) == "chop": chop_n += 1
            except Exception: pass
        row = []
        for c in CANDIDATES:
            out = run_strategy(data, seg, c)
            res[c].append(out["ret"])
            if chop_n >= oos * 0.5: chop_res[c].append(out["ret"])
            row.append(f"{out['ret']:+.1f}")
        print(f"  slice{si:2d} ({str(seg[0].date())}..{str(seg[-1].date())}) chop={chop_n:2d}/20  " +
              "  ".join(f"{c[:4]}={r}" for c, r in zip(CANDIDATES, row)))
    print("\n=== MEAN ret% ===")
    for c in sorted(CANDIDATES, key=lambda x: -np.mean(res[x])):
        pos = sum(1 for x in res[c] if x > 0)
        ch = np.mean(chop_res[c]) if chop_res[c] else float('nan')
        print(f"   rei+{c:11s} all={np.mean(res[c]):+6.1f}%  pos={pos}/{len(res[c])}  chopHeavyMean={ch:+.1f}%")
    best = max(CANDIDATES, key=lambda x: np.mean(res[x]))
    print(f"\n   BEST chop leg: rei+{best} ({np.mean(res[best]):+.1f}%)")
    print(f"   LIVE donchian40 mean: {np.mean(res['donchian40']):+.1f}%  vs tsi {np.mean(res['tsi']):+.1f}%")


if __name__ == "__main__":
    main()
