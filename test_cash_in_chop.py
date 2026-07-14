"""
Baseline (c): CASH-IN-CHOP. Run the SAME system but sit flat during chop regime
(only trade trend = REI + ATR). Answers: is chop trading even adding value vs
just skipping it?

Compare vs the LIVE full system (donchian40 + ma30_ema fill in chop) on the
same panel + paired slices. If cash-in-chop matches/exceeds live, chop trading
is net-negative and we should disable it.

Finer slicing (12d step, 12d OOS) for more WF windows -> paired stat power.
"""

import numpy as np
import pandas as pd
from portfolio_engine import PortfolioEngine, EngineConfig

import engine
from engine import load_screened_universe, get_regime_signals, improved_compute_live_regime

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS, POS_PCT = 5, 0.20
OOS_LEN = 12
STEP = 12


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


def run(data, dates, cash_in_chop):
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
        if reg == "chop" and cash_in_chop:
            # sit flat: close everything, take no new entries
            prices = {s: float(pre[s]["close"].iloc[-1]) for s in pre}
            eng.start_daily_bar(next(iter(prices.values()), None))
            for s in list(eng.positions.keys()):
                px = prices.get(s)
                if px and px > 0: eng.close_position(s, px)
            eq.append(eng.mark_to_market(prices)); continue
        rule = "rei" if reg == "trend" else "donchian40"
        active = []
        for s in pre:
            ent, _ = get_regime_signals(rule, pre[s].reset_index())
            if len(ent) and int(ent.iloc[-1]): active.append(s)
        if reg == "chop" and len(active) == 0:
            for s in pre:
                ent, _ = get_regime_signals("ma30_ema", pre[s].reset_index())
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
    return ret, eff_sr, mdd * 100


def main():
    data, dates = load_common(n_min=150)
    slices = []
    i = 60
    while i + OOS_LEN <= len(dates):
        slices.append(dates[i:i + OOS_LEN]); i += STEP
    print(f"(c) CASH-IN-CHOP baseline: {len(slices)} finer WF slices ({OOS_LEN}d step {STEP}d), {len(data)} coins\n")
    full, cash = [], []
    for si, seg in enumerate(slices, 1):
        rf = run(data, seg, cash_in_chop=False)
        rc = run(data, seg, cash_in_chop=True)
        full.append(rf); cash.append(rc)
        print(f"  slice{si:2d} ({str(seg[0].date())}..{str(seg[-1].date())})  full={rf[0]:+6.1f}%  cashChop={rc[0]:+6.1f}%")
    mf, mc = np.mean([r[0] for r in full]), np.mean([r[0] for r in cash])
    ef, ec = np.mean([r[1] for r in full]), np.mean([r[1] for r in cash])
    df, dc = np.mean([r[2] for r in full]), np.mean([r[2] for r in cash])
    print(f"\n  FULL system   : meanRet={mf:+6.1f}%  effSR={ef:+5.2f}  meanDD={df:5.1f}%")
    print(f"  CASH-in-CHOP : meanRet={mc:+6.1f}%  effSR={ec:+5.2f}  meanDD={dc:5.1f}%")
    # paired sign test on returns
    d = np.array([r[0] for r in full]) - np.array([r[0] for r in cash])
    n = np.sum(d != 0); k = np.sum(d > 0)
    from math import comb
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n) if n else 1.0
    print(f"\n  FULL beats CASH-in-chop in {k}/{n} slices; paired p={p:.3f}")
    print("  -> if p>0.05, chop trading adds NO significant value; consider disabling.")


if __name__ == "__main__":
    main()
