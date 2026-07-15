"""Expanded significance check: 10/30 vs 50/200 (the live pair).

The first sweep favored 10/30 on every risk-adjusted metric but the exact
sign test was inconclusive (p=1.000, only 3/8 slices). This expands the
sample: more WF slices (STEP=6) + a bootstrap paired test to get real
significance on mean slice-return and effSR differences.

Literature framing (Shu et al 2024): pick the persistence width by CV that
maximizes OOS Sharpe. Here: is faster MA (10/30) significantly better?
"""
import numpy as np
import pandas as pd

from test_rule_scorecard import load_common, prefix, sign_test_p, TRAIN, OOS
from portfolio_engine import PortfolioEngine, EngineConfig
from engine import get_regime_signals, improved_compute_live_regime

COST_BPS, SLIP_BPS, MAX_POS, POS_PCT = 8.0/10000.0, 5.0/10000.0, 5, 0.20
FAST, LIVE = (10, 30), (50, 200)
STEP = 6  # denser walk-forward -> more slices


def run_ma_pair(data, dates, chop_rule, short, long, combo=True, fill_rule="ma30_ema"):
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
            reg = improved_compute_live_regime(pre, method="ma", ma_short=short, ma_long=long)
        except Exception:
            reg = "trend"
        rule = "rei" if reg == "trend" else chop_rule
        active = []
        for s in pre:
            ent, _ = get_regime_signals(rule, pre[s].reset_index())
            if len(ent) and int(ent.iloc[-1]): active.append(s)
        if combo and reg == "chop" and len(active) == 0:
            for s in pre:
                ent, _ = get_regime_signals(fill_rule, pre[s].reset_index())
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
    return {"ret": ret, "effSR": eff_sr}


def bootstrap_p(a_vals, b_vals, n=10000, seed=0):
    """Paired bootstrap: P(median(A-B) > 0)."""
    rng = np.random.default_rng(seed)
    a = np.array(a_vals, float); b = np.array(b_vals, float)
    diffs = a - b
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    return float(np.mean(boot > 0)), float(np.mean(boot < 0))


def main():
    data0, dates = load_common(n_min=150)
    slices = []
    i = TRAIN
    while i + OOS <= len(dates):
        slices.append(dates[i:i + OOS]); i += STEP
    print(f"Expanded check: 10/30 vs 50/200 (LIVE), method=ma, {len(slices)} PIT WF slices (STEP={STEP})\n")
    fast, live = [], []
    for seg in slices:
        data, _ = load_common(n_min=150, as_of=str(seg[0].date()))
        fast.append(run_ma_pair(data, seg, "cci", *FAST))
        live.append(run_ma_pair(data, seg, "cci", *LIVE))
    fr = [r["ret"] for r in fast]; lr = [r["ret"] for r in live]
    fe = [r["effSR"] for r in fast]; le = [r["effSR"] for r in live]

    print(f"10/30   meanRet={np.mean(fr):+6.2f}  meanEffSR={np.mean(fe):+6.2f}")
    print(f"50/200  meanRet={np.mean(lr):+6.2f}  meanEffSR={np.mean(le):+6.2f}")
    print(f"Δret={np.mean(fr)-np.mean(lr):+6.2f}  ΔeffSR={np.mean(fe)-np.mean(le):+6.2f}")

    p_exact = sign_test_p(fr, lr)
    p_hi, p_lo = bootstrap_p(fr, lr)
    better = sum(1 for a, b in zip(fr, lr) if a > b)
    print(f"\nExact sign test (ret):  beats in {better}/{len(slices)} slices  p={p_exact:.3f}")
    print(f"Bootstrap (ret):  P(10/30>50/200)={p_hi:.3f}  P(50/200>10/30)={p_lo:.3f}")

    p_exact_e, p_hi_e, p_lo_e = None, None, None
    p_exact_e = sign_test_p(fe, le)
    p_hi_e, p_lo_e = bootstrap_p(fe, le)
    better_e = sum(1 for a, b in zip(fe, le) if a > b)
    print(f"\nExact sign test (effSR): beats in {better_e}/{len(slices)} slices  p={p_exact_e:.3f}")
    print(f"Bootstrap (effSR): P(10/30>50/200)={p_hi_e:.3f}  P(50/200>10/30)={p_lo_e:.3f}")

    sig = "SIGNIFICANT" if (p_hi > 0.95 or p_exact < 0.05) else "inconclusive"
    print(f"\nVerdict on 10/30 vs 50/200: {sig}")


if __name__ == "__main__":
    main()
