"""
SCREENING test on Kraken universe (473 coins, 14 WF slices). Isolates the SCREENING
decision (which coins to trade) from the rule. Two pluggable levers:

 (A) UNIVERSE SCREEN: how to pick the candidate pool from all coins.
     - broad:   all coins with >=min_bars (no screen)
     - adv:     top-N by avg daily volume (liquidity)
     - idio:    top-N by idiosyncratic vol (current live screen philosophy)
 (B) STRENGTH RANK: how to pick top-5 among signaled coins.
     - ma_dist: % distance above MA (current live)
     - mom:     20d momentum
     - idio:    idiosyncratic vol (volatile = tradeable)

Same strategy (REI trend + ATR; donchian40 + ma30_ema chop). Panel + paired.
Goal: find if the live screen (idio universe + ma_dist rank) is actually best, or a
simpler screen beats it.
"""

import numpy as np
import pandas as pd
from portfolio_engine import PortfolioEngine, EngineConfig

import engine
from engine import get_regime_signals, improved_compute_live_regime

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS, POS_PCT = 5, 0.20
MIN_BARS = 400
TOPN = 80  # pool size for screened variants


def load_kraken(n_min=MIN_BARS):
    from pathlib import Path
    kp = set(pd.read_csv('data/kraken_pairs.csv')['pair_id'].dropna().astype(str))
    files = sorted(Path('data').glob('*_1d_max.csv'))
    data = {}
    for p in files:
        if p.name.replace('_1d_max.csv', '') not in kp:
            continue
        try:
            df = pd.read_csv(p)
            if len(df) < n_min:
                continue
            df['ts'] = pd.to_datetime(df['ts'], utc=True).dt.tz_localize(None)
            df = df.sort_values('ts').reset_index(drop=True)
            data[p.name.replace('_1d_max.csv', '')] = df.set_index('ts')
        except Exception:
            continue
    idx = None
    for d in data.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    data = {s: d.loc[idx] for s in data if len(d.loc[idx]) > 0}
    return data, pd.DatetimeIndex(sorted(idx))


def prefix(data, day):
    return {s: d.loc[:day] for s, d in data.items() if len(d.loc[:day]) > 0}


def adv_of(df):
    return float(df['close'].iloc[-1] * df['volume'].iloc[-20:].mean())


def idio_of(df):
    r = df['close'].pct_change().rolling(20).std()
    return float(r.iloc[-1]) if not np.isnan(r.iloc[-1]) else 0.0


def mom_of(df):
    c = df['close']
    return float((c.iloc[-1] / c.iloc[-20] - 1) if len(c) > 20 else 0.0)


def build_pool(data, universe):
    if universe == "broad":
        return list(data.keys())
    scored = []
    for s, d in data.items():
        if universe == "adv":
            scored.append((s, adv_of(d)))
        elif universe == "idio":
            scored.append((s, idio_of(d)))
    scored.sort(key=lambda x: -x[1])
    return [s for s, _ in scored[:TOPN]]


def strength_fn(df, kind):
    close = df["close"]
    ma = close.ewm(span=30, adjust=False).mean()
    if kind == "ma_dist":
        return float(((close.iloc[-1] - ma.iloc[-1]) / (ma.iloc[-1] + 1e-12)) * 100)
    if kind == "mom":
        return mom_of(df)
    if kind == "idio":
        return idio_of(df)
    return 0.0


def run(data, dates, universe, rank):
    pool = build_pool(data, universe)
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
        pre = {s: d for s, d in pre.items() if s in pool}
        if not pre:
            eq.append(eng.equity); continue
        try:
            reg = improved_compute_live_regime(pre)
        except Exception:
            reg = "trend"
        rule = "rei" if reg == "trend" else "donchian40"
        raw = {}
        for s in pre:
            ent, ex = get_regime_signals(rule, pre[s].reset_index())
            last_e = int(ent.iloc[-1]) if len(ent) else 0
            last_x = int(ex.iloc[-1]) if len(ex) else 0
            raw[s] = dict(entry=last_e, exit=last_x, strength=strength_fn(pre[s], rank))
        active = [s for s, v in raw.items() if v['entry']]
        if reg == "chop" and not active:
            # LIVE config: donchian40 primary; ma30_ema recapture as FILL-IN
            # only when donchian is silent (no entries this bar).
            for s in pre:
                ent, ex = get_regime_signals("donchian40", pre[s].reset_index())
                last_e = int(ent.iloc[-1]) if len(ent) else 0
                last_x = int(ex.iloc[-1]) if len(ex) else 0
                raw[s] = dict(entry=last_e, exit=last_x,
                              strength=strength_fn(pre[s], rank))
            active = [s for s, v in raw.items() if v['entry']]
        if reg == "chop" and not active:
            for s in pre:
                ent, ex = get_regime_signals("ma30_ema", pre[s].reset_index())
                last_e = int(ent.iloc[-1]) if len(ent) else 0
                raw[s] = dict(entry=last_e, exit=int(ex.iloc[-1]) if len(ex) else 0,
                              strength=strength_fn(pre[s], rank))
            active = [s for s, v in raw.items() if v['entry']]
        active = sorted(active, key=lambda s: -raw[s]['strength'])[:MAX_POS]
        prices = {s: float(pre[s]["close"].iloc[-1]) for s in pre}
        eng.start_daily_bar(next(iter(prices.values()), None))
        ok, _ = eng.check_circuit_breakers()
        if not ok:
            eng.flatten_all(prices); eq.append(eng.equity); continue
        for s in list(eng.positions.keys()):
            if raw.get(s, {}).get('exit'):
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
    return ret, eff_sr, mdd * 100, calmar, getattr(eng, 'trade_count', 0)


def main():
    data, dates = load_kraken()
    oos, step, warmup = 20, 20, 120
    slices = []
    i = warmup
    while i + oos <= len(dates):
        slices.append(dates[i:i + oos]); i += step
    UNIV = ["broad", "adv", "idio"]
    RANK = ["ma_dist", "mom", "idio"]
    print(f"SCREENING test: {len(UNIV)}x{len(RANK)} = {len(UNIV)*len(RANK)} configs, {len(slices)} WF slices, {len(data)} Kraken coins\n")
    res = {}
    for u in UNIV:
        for r in RANK:
            key = f"{u}|{r}"
            res[key] = [run(data, seg, u, r) for seg in slices]
    print(f"{'config':16s} {'meanRet':>8} {'effSR':>7} {'meanDD':>7} {'Calmar':>7} {'trades':>7}")
    agg = {}
    for key in res:
        rs = res[key]
        agg[key] = (np.mean([r[0] for r in rs]), np.mean([r[1] for r in rs]),
                    np.mean([r[2] for r in rs]), np.mean([r[3] for r in rs]),
                    np.mean([r[4] for r in rs]))
    for key in sorted(agg, key=lambda x: -agg[x][0]):
        a = agg[key]
        print(f"{key:16s} {a[0]:+8.1f} {a[1]:+7.2f} {a[2]:>6.1f}% {a[3]:>7.2f} {a[4]:>7.0f}")
    print("NOTE: if avg trades/config is near 0, the daily signal fires too rarely "
          "to discriminate screen methods -- report as 'insufficient trades', not a ranking.")
    # paired vs live-equivalent (idio|ma_dist)
    base_key = "idio|ma_dist"
    print(f"\nPaired vs LIVE screen ({base_key}):")
    base = [r[0] for r in res[base_key]]
    for key in res:
        if key == base_key: continue
        other = [r[0] for r in res[key]]
        d = np.array(other) - np.array(base)
        n = np.sum(d != 0); k = np.sum(d > 0)
        from math import comb
        p = min(1.0, 2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n) if n else 1.0
        print(f"   {key:16s} Δmean={np.mean(other)-np.mean(base):+6.1f}  beats in {k}/{n}  p={p:.3f}{' *' if p<0.05 else ''}")
    print("\n* = significant at p<0.05")


if __name__ == "__main__":
    main()
