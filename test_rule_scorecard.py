"""
Upgraded rule comparison methodology.

Earlier we compared rules on ENTRY quality only (signal separation) -> wrong.
Then full entry+exit walk-forward -> better but single-metric (return%) and no
statistical teeth. This harness fixes BOTH:

1. FULL STRATEGY (own entry+exit) via shared PortfolioEngine, identical to live.
2. PANEL of metrics per walk-forward slice (not just return):
     ret%        total return
     effSR       exposure-adjusted Sharpe (ret / (ret_std * sqrt(252)) * vol_scale)
     maxDD%      max drawdown
     calmar      ret / maxDD
     win%        trade win rate
     trades      count
3. PAIRED statistics across slices: each slice is the SAME market period, so
   differences between rules are paired. Report mean diff + sign-test p-value
   (exact binomial) so we can say "A beats B at p<0.05", not just "higher mean".
4. STABILITY: worst-slice return + positive fraction + std of slice returns.

Chop-leg candidates (trend fixed = rei), plus the LIVE combined (d40 + ma30_ema fill).
"""

import numpy as np
import pandas as pd
from scipy import stats as _stats  # optional; fall back if absent
from portfolio_engine import PortfolioEngine, EngineConfig

import engine
from engine import load_screened_universe, get_regime_signals, improved_compute_live_regime
import config as _cfg  # N_WORKERS_CPU = logical cores - 1 (headroom)

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS, POS_PCT = 5, 0.20

CANDIDATES = [
    ("donchian40", "chop"), ("ma30_ema", "chop"), ("cci", "chop"), ("tsi", "chop"),
    ("rsi", "chop"), ("bop", "chop"), ("mtf", "chop"), ("ma30", "chop"),
    ("williams_r", "chop"), ("williams_r_buggy", "chop"),
    ("rei", "chop"), ("bbwp", "chop"),
    ("stochastic", "chop"), ("mfi", "chop"), ("ift_rsi", "chop"),
    ("d40+ma30_ema", "combo"),
]


def _scorecard_task(task):
    """Top-level worker for the parallel WF pool (must be picklable).

    task = (cname, kind, seg, data); returns (cname, run_strategy result).
    """
    cname, kind, seg, data = task
    combo = (kind == "combo")
    chop_rule = "donchian40" if combo else cname
    return cname, run_strategy(data, seg, chop_rule, combo=combo)



# Walk-forward slice geometry (shared with test_significance_pit)
TRAIN, STEP, OOS = 60, 12, 12


def load_common(n_min=150, as_of=None):
    cd = load_screened_universe(min_bars=n_min, as_of=as_of)
    data = {s: d.set_index("ts").sort_index() for s, d in cd.items()}
    idx = None
    for d in data.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    data = {s: d.loc[idx] for s, d in data.items() if len(d.loc[idx]) > 0}
    return data, pd.DatetimeIndex(sorted(idx))


def prefix(data, day):
    return {s: d.loc[:day] for s, d in data.items() if len(d.loc[:day]) > 0}


def run_strategy(data, dates, chop_rule, combo=False, fill_rule="ma30_ema"):
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
    # exposure-adjusted (eff) Sharpe: deflate by fraction of time in market
    in_mkt = (eq.diff() != 0).astype(int)
    exp = in_mkt.mean() if len(in_mkt) else 0.0
    eff_sr = sr * (exp ** 0.5) if exp > 0 else 0.0
    peak = 10000.0; mdd = 0.0
    for e in eq:
        peak = max(peak, e); mdd = max(mdd, (peak - e) / peak)
    calmar = (ret / 100.0) / mdd if mdd > 0 else 0.0
    wins = sum(1 for t in eng.trades if t.get("pnl", 0) > 0)
    winrate = (wins / len(eng.trades) * 100) if eng.trades else 0.0
    return {"ret": ret, "effSR": eff_sr, "maxDD": mdd * 100, "calmar": calmar,
            "win%": winrate, "trades": len(eng.trades), "rets": rets}


def sign_test_p(a_vals, b_vals):
    """Exact binomial sign test: P(A>B) under H0=median diff=0."""
    d = np.array(a_vals) - np.array(b_vals)
    n = np.sum(d != 0)
    k = np.sum(d > 0)
    if n == 0:
        return 1.0
    try:
        p = _stats.binomtest(k, n).pvalue
    except Exception:
        # fallback manual two-sided
        from math import comb
        p = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
        p = min(1.0, 2 * p)
    return p


def main():
    # PIT (point-in-time) mode: TRUE removes survivorship bias by building each
    # WF slice's universe from the screen as-of that slice's START date, not
    # today's survivor list. This is the literature-correct methodology.
    PIT = True
    data0, dates = load_common(n_min=150)   # full survivor set, only for the date axis
    slices = []
    i = TRAIN
    while i + OOS <= len(dates):
        slices.append(dates[i:i + OOS]); i += STEP
    mode = "PIT (point-in-time, survivorship-corrected)" if PIT else "survivor (today's list)"
    print(f"UPGRADED rule scorecard: {len(CANDIDATES)} candidates, {len(slices)} WF slices, mode={mode}\n")
    # Precompute the PIT universe per slice ONCE (not per candidate inside the
    # pool) so each worker gets ready data instead of reloading the universe.
    # seg is a DatetimeIndex (unhashable) -> key by tuple(seg). load_common
    # returns (data, dates); we only need data for the worker.
    slice_data = {}
    for seg in slices:
        key = tuple(seg)
        slice_data[key] = load_common(n_min=150, as_of=str(seg[0].date()))[0] if PIT else data0
    res = {c[0]: [] for c in CANDIDATES}

    from multiprocessing import Pool

    # task = (cname, kind, seg, data) — data precomputed above, picklable.
    tasks = [(c[0], c[1], seg, slice_data[tuple(seg)]) for c in CANDIDATES for seg in slices]
    with Pool(_cfg.N_WORKERS_CPU) as pool:
        for cname, out in pool.map(_scorecard_task, tasks):
            res[cname].append(out)
    # Per-candidate aggregate table
    print(f"{'candidate':14s} {'meanRet':>8} {'worst':>7} {'pos/n':>6} {'meanEffSR':>9} {'meanDD':>7} {'meanCalmar':>11} {'meanWin%':>9}")
    agg = {}
    for cname, _ in CANDIDATES:
        rs = res[cname]
        agg[cname] = {
            "ret": np.mean([r["ret"] for r in rs]),
            "worst": min(r["ret"] for r in rs),
            "pos": sum(1 for r in rs if r["ret"] > 0),
            "effSR": np.mean([r["effSR"] for r in rs]),
            "DD": np.mean([r["maxDD"] for r in rs]),
            "calmar": np.mean([r["calmar"] for r in rs]),
            "win": np.mean([r["win%"] for r in rs]),
        }
    for cname in sorted(agg, key=lambda x: -agg[x]["ret"]):
        a = agg[cname]
        print(f"{cname:14s} {a['ret']:+8.1f} {a['worst']:+7.1f} {a['pos']}/{len(slices):<3} {a['effSR']:+9.2f} {a['DD']:>6.1f}% {a['calmar']:>10.2f} {a['win']:>8.1f}%")
    # Paired comparison vs LIVE (donchian40)
    print(f"\n=== PAIRED vs LIVE donchian40 (exact sign test on slice returns) ===")
    live = [r["ret"] for r in res["donchian40"]]
    for cname, _ in CANDIDATES:
        if cname == "donchian40": continue
        other = [r["ret"] for r in res[cname]]
        p = sign_test_p(other, live)
        delta = np.mean(other) - np.mean(live)
        better = sum(1 for o, l in zip(other, live) if o > l)
        print(f"   {cname:14s} Δmean={delta:+6.1f}  beats live in {better}/{len(slices)} slices  p={p:.3f}{' *' if p<0.05 else ''}")
    print("\n* = statistically significant at p<0.05 (exact binomial sign test, paired by slice)")


if __name__ == "__main__":
    main()
