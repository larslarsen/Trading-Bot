"""
Volatility TARGETING test (Hurst/AQR "A Century of Evidence on Trend-Following"):
scaling positions to a target annualized vol improves return-to-drawdown (Calmar) of
trend systems across a century. Our live config has enable_vol_target=False. Test it
on the deep 66-slice universe: fixed (current) vs VT@10% vs VT@15% vs VT@20%.

Panel + paired exact sign test vs current fixed, reporting Calmar prominently.
This directly targets the 'best Calmar' property of the live rule.
"""

import numpy as np
import pandas as pd
from portfolio_engine import PortfolioEngine, EngineConfig

import engine
from engine import get_regime_signals, improved_compute_live_regime

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS, POS_PCT = 5, 0.20
MIN_BARS = 800
TARGETS = [("fixed", None), ("VT@10%", 0.10), ("VT@15%", 0.15), ("VT@20%", 0.20)]


def load_deep(n_min=MIN_BARS):
    from pathlib import Path
    files = sorted(Path('data').glob('*_1d_max.csv'))
    data = {}
    for p in files:
        try:
            df = pd.read_csv(p)
            if len(df) < n_min:
                continue
            df['ts'] = pd.to_datetime(df['ts'], utc=True).dt.tz_localize(None)
            df = df.sort_values('ts').reset_index(drop=True)
            data[p.name.replace('_1d_max.csv', '')] = df.set_index('ts')
        except Exception:
            continue
    cut = pd.Timestamp('2021-01-01')
    data = {s: d for s, d in data.items() if d.index.min() < cut}
    idx = None
    for d in data.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    data = {s: d.loc[idx] for s in data if len(d.loc[idx]) > 0}
    return data, pd.DatetimeIndex(sorted(idx))


def prefix(data, day):
    return {s: d.loc[:day] for s, d in data.items() if len(d.loc[:day]) > 0}


def run(data, dates, target):
    vt = target is not None
    cfg = EngineConfig(
        initial_capital=10000.0, max_daily_loss_pct=0.03, max_drawdown_pct=0.20,
        max_positions=MAX_POS, max_position_pct=POS_PCT, min_equity_to_trade=100.0,
        flash_crash_bars=5, flash_crash_pct=0.50, extreme_move_pct=0.90,
        cost_bps=COST_BPS, slippage_bps=SLIP_BPS,
        enable_vol_target=False,  # we do per-asset VT manually in-harness (AQR style)
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
            # per-asset volatility targeting (AQR): scale size by target_vol / asset_vol
            size = eng.equity * POS_PCT
            if vt:
                closes = pre[s]["close"].iloc[-21:]
                if len(closes) > 5:
                    avol = float(closes.pct_change().std() * np.sqrt(365))
                    if avol > 1e-6:
                        scale = float(np.clip(target / avol, 0.25, 1.5))
                        size = eng.equity * POS_PCT * scale
            eng.open_position(s, px, size)
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
    data, dates = load_deep()
    oos, step, warmup = 20, 20, 120
    slices = []
    i = warmup
    while i + oos <= len(dates):
        slices.append(dates[i:i + oos]); i += step
    print(f"VOL-TARGETING test (Hurst/AQR): {len(TARGETS)} configs, {len(slices)} WF slices, {len(data)} deep majors\n")
    res = {name: [run(data, seg, tgt) for seg in slices] for name, tgt in TARGETS}
    print(f"{'config':12s} {'meanRet':>8} {'effSR':>7} {'meanDD':>7} {'Calmar':>7}")
    agg = {}
    for name in res:
        rs = res[name]
        agg[name] = (np.mean([r[0] for r in rs]), np.mean([r[1] for r in rs]),
                     np.mean([r[2] for r in rs]), np.mean([r[3] for r in rs]))
    for name in sorted(agg, key=lambda x: -agg[x][3]):
        a = agg[name]
        print(f"{name:12s} {a[0]:+8.1f} {a[1]:+7.2f} {a[2]:>6.1f}% {a[3]:>7.2f}")
    print(f"\nPaired vs current 'fixed' (exact sign test, {len(slices)} slices):")
    base = [r[0] for r in res["fixed"]]
    for name in res:
        if name == "fixed": continue
        other = [r[0] for r in res[name]]
        d = np.array(other) - np.array(base)
        n = np.sum(d != 0); k = np.sum(d > 0)
        from math import comb
        p = min(1.0, 2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n) if n else 1.0
        print(f"   {name:12s} Δmean={np.mean(other)-np.mean(base):+6.1f}  beats in {k}/{n}  p={p:.3f}{' *' if p<0.05 else ''}")
    print("\n* = significant at p<0.05. Calmar is the key metric (return/DD).")


if __name__ == "__main__":
    main()
