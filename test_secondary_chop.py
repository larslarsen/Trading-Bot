"""
Secondary chop-rule test: donchian40 (LIVE) is sparse - it only fires on
40d breakouts, so on most chop days it's silent. Add a SECONDARY rule that
engages ONLY when donchian40 is silent (0 entries that day), to fill the
idle chop days. Test which MA-recapture variant is the best fill-in.

Candidates (secondary, used only when donchian40 silent):
  ma30_ema     (your friend's MA30 EMA recapture)
  ma30         (SMA30 recapture)
  ma30_rising  (SMA30 recapture + rising-MA filter)
  ma30_50      (MA30 recapture while price > MA50)

Also report:
  donchian40-only (current live)
  pure <variant>  (the MA rule as the sole chop rule, for reference)

Walk-forward: 6 non-overlapping 20d OOS slices, REI trend, shared engine.
"""

import numpy as np
import pandas as pd
from portfolio_engine import PortfolioEngine, EngineConfig

import engine
from engine import load_screened_universe, get_regime_signals, improved_compute_live_regime

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS, POS_PCT = 5, 0.20
SECONDARIES = ["ma30_ema", "ma30", "ma30_rising", "ma30_50"]


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


def run_strategy(data, dates, mode, secondary=None):
    """mode: 'd40' (donchian only), 'pure' (secondary as sole chop rule),
    or 'fill' (d40 primary, secondary only when d40 silent)."""
    chop_rule = secondary if mode == "pure" else "donchian40"
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
        # primary active set
        active = []
        for s in pre:
            ent, _ = get_regime_signals(rule, pre[s].reset_index())
            if len(ent) and int(ent.iloc[-1]): active.append(s)
        # secondary fill-in when d40 silent
        if mode == "fill" and reg == "chop" and len(active) == 0:
            for s in pre:
                ent, _ = get_regime_signals(secondary, pre[s].reset_index())
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
    ret = (eq.iloc[-1] / 10000.0 - 1) * 100
    rets = eq.pct_change().dropna()
    sr = (rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    peak = 10000.0; mdd = 0.0
    for e in eq:
        peak = max(peak, e); mdd = max(mdd, (peak - e) / peak)
    return {"ret": round(ret, 1), "sr": round(sr, 2), "mdd": round(mdd*100, 1), "trades": len(eng.trades)}


def run_all(data, slices):
    res = {}
    # baselines
    res["d40_only"] = [run_strategy(data, seg, "d40")["ret"] for seg in slices]
    for c in SECONDARIES:
        res[f"pure_{c}"] = [run_strategy(data, seg, "pure", c)["ret"] for seg in slices]
        res[f"fill_{c}"] = [run_strategy(data, seg, "fill", c)["ret"] for seg in slices]
    return res


def main():
    data, dates = load_common(n_min=150)
    train, step, oos = 40, 20, 20
    slices = []
    i = train
    while i + oos <= len(dates):
        slices.append(dates[i:i + oos]); i += step
    print(f"Secondary chop-rule (fill-in) test, {len(slices)} non-overlapping {oos}d slices, {len(data)} coins\n")
    res = run_all(data, slices)
    # per-slice table
    print("slice        " + "  ".join(f"{k:>9}" for k in res))
    for si, seg in enumerate(slices):
        print(f"  slice{si+1:<7}" + "  ".join(f"{res[k][si]:+9.1f}" for k in res))
    print("\n=== MEAN ret% across slices ===")
    for k in sorted(res, key=lambda x: -np.mean(res[x])):
        pos = sum(1 for x in res[k] if x > 0)
        print(f"   {k:14s} {np.mean(res[k]):+7.1f}%  pos={pos}/{len(res[k])}")
    best = max(res, key=lambda x: np.mean(res[x]))
    print(f"\n   BEST config: {best} ({np.mean(res[best]):+.1f}%)")


if __name__ == "__main__":
    main()
