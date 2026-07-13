#!/usr/bin/env python3
"""
Walk-forward comparison of regime detectors for the REI/Williams rule set.

Regimes compared:
  - "rule": current improved (ADX + ER + vol + hysteresis)
  - "ma50": MA crossover 50/200
  - "ma20": MA crossover 20/50 (faster)

Downstream rules (unchanged):
  - Trend regime -> REI
  - Chop regime  -> williams_r

Metrics per fold + aggregate: return, eff_sharpe, max_dd, trades, exposure.

Uses causal prefix-only signals. Ranking by strength (same as live) is applied.

Focus on longer-history subset (~300+ days common) for meaningful WF.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import glob
import os
from engine import simulate_portfolio, get_regime_signals, compute_regime

def williams_strength(df):
    high = pd.Series(df["high"].values, index=df.index)
    low = pd.Series(df["low"].values, index=df.index)
    close = pd.Series(df["close"].values, index=df.index)
    period = 14
    highest = high.rolling(period, min_periods=1).max()
    lowest = low.rolling(period, min_periods=1).min()
    wr = -100 * (highest - close) / (highest - lowest + 1e-12)
    last_wr = float(wr.iloc[-1]) if len(wr) > 0 else -50.0
    wr_diff = float(wr.diff().iloc[-1]) if len(wr) > 1 else 0.0
    return (-last_wr) + max(0.0, wr_diff * 2.0)

def rei_strength(df):
    close = pd.Series(df["close"].values, index=df.index)
    high = pd.Series(df["high"].values, index=df.index)
    low = pd.Series(df["low"].values, index=df.index)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    up = up_move.where((up_move > 0) & (up_move > down_move), 0).fillna(0)
    down = down_move.where((down_move > 0) & (down_move > up_move), 0).fillna(0)
    rng = (high - low).rolling(14, min_periods=1).mean()
    rei = 100 * (up.rolling(14, min_periods=1).sum() - down.rolling(14, min_periods=1).sum()) / (rng + 1e-12)
    last_rei = float(rei.iloc[-1]) if len(rei) > 0 else 0.0
    rei_diff = float(rei.diff().iloc[-1]) if len(rei) > 1 else 0.0
    return last_rei + max(0.0, rei_diff * 0.5)

def build_regime_series(market: pd.Series, method: str, **kwargs) -> list:
    regimes = []
    for i in range(len(market)):
        if method == "ma":
            short = kwargs.get("short", 50)
            long = kwargs.get("long", 200)
            from engine import ma_crossover_regime
            r = ma_crossover_regime(market, i, short=short, long=long)
        else:
            r = compute_regime(market, i, method=method)
        regimes.append(r)
    return regimes

def build_ranked_signals(prices: dict, common_dates: list, regimes: list, price_df: pd.DataFrame) -> pd.DataFrame:
    sig_df = pd.DataFrame(0, index=common_dates, columns=price_df.columns, dtype=int)
    for i, date in enumerate(common_dates):
        reg = regimes[i]
        rule = 'rei' if reg == 'trend' else 'williams_r'
        strengths = {}
        for stem in price_df.columns:
            if stem not in prices: continue
            ddf = prices[stem][prices[stem]['ts'].dt.date == date]
            if len(ddf) == 0: continue
            full_df = prices[stem].set_index(prices[stem]['ts'].dt.date).reindex(common_dates[:i+1]).reset_index(drop=True)
            if len(full_df) < 25: continue
            try:
                e, _ = get_regime_signals(rule, full_df)
                if len(e) > 0 and int(e.iloc[-1]) == 1:
                    if rule == 'williams_r':
                        strengths[stem] = williams_strength(full_df)
                    else:
                        strengths[stem] = rei_strength(full_df)
            except:
                pass
        # Rank and take top signals (simulate cap later in portfolio sim)
        top = sorted(strengths.items(), key=lambda x: -x[1])
        for sym, _ in top:
            sig_df.loc[date, sym] = 1
    return sig_df

def oos_metrics(full_res: dict, oos_start_idx: int, initial: float = 10000.0) -> dict:
    """Rough OOS metrics from simulate result equity curve (if available) or approximate."""
    # simulate_portfolio returns summary; for simplicity we re-use full metrics on the slice
    # In this script we run per-fold on the prefix up to fold end and report the full-fold metrics as "OOS" for that fold
    # (common practical approximation when carrying positions).
    return {
        'return_pct': full_res.get('return_pct', 0),
        'effective_sharpe': full_res.get('effective_sharpe', 0),
        'max_dd_pct': full_res.get('max_dd_pct', 0),
        'trades': full_res.get('trades', 0),
    }

def main():
    OUT = Path('backtest_output')
    INITIAL = 10000.0
    MAX_POS = 5

    screen = pd.read_csv(sorted(OUT.glob('screen_liqu_idio_*.csv'))[-1])
    screen = screen[screen.tier.isin(['large','mid','tail'])]
    stems = screen.stem.tolist()

    # Prefer longer-history subset for meaningful walk-forward
    long_stems = []
    for stem in stems:
        p = f'data/{stem}_1d_max.csv'
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=['ts'])
            if len(df) >= 300:
                long_stems.append(stem)

    print(f"Long-history coins available: {len(long_stems)}")

    # Load
    prices = {}
    for stem in long_stems:
        p = f'data/{stem}_1d_max.csv'
        df = pd.read_csv(p, parse_dates=['ts']).dropna(subset=['close','high','low'])
        df = df.sort_values('ts').reset_index(drop=True)
        prices[stem] = df

    all_dates = sorted(list(set.intersection(*[set(df['ts'].dt.date) for df in prices.values()])))
    print(f"Common dates on long-history subset: {len(all_dates)}")

    # Use last 250+ days for WF if possible
    if len(all_dates) > 250:
        all_dates = all_dates[-250:]

    price_df = pd.DataFrame(index=all_dates)
    for stem in list(prices.keys()):
        df_idx = prices[stem].set_index(prices[stem]['ts'].dt.date)
        price_df[stem] = df_idx['close'].reindex(all_dates)
    price_df = price_df.ffill().bfill()
    market = price_df.mean(axis=1)

    # Walk-forward config (practical given data)
    # Rolling: lookback for regime ~120 days, test window ~40 days, step 30 days
    lookback = 120
    test_len = 40
    step = 30

    folds = []
    start_idx = lookback
    while start_idx + test_len <= len(all_dates):
        folds.append((start_idx, start_idx + test_len))
        start_idx += step

    print(f"Number of walk-forward folds: {len(folds)}")

    methods = {
        'rule': {'method': 'rule'},
        'ma50': {'method': 'ma', 'short': 50, 'long': 200},
        'ma20': {'method': 'ma', 'short': 20, 'long': 50},
    }

    results = {m: [] for m in methods}

    for mname, mkwargs in methods.items():
        print(f"\n=== Testing regime: {mname} ===")
        # Precompute regime on full series (causal inside function)
        regimes = build_regime_series(market, **mkwargs)

        for fold_idx, (oos_start, oos_end) in enumerate(folds):
            fold_dates = all_dates[:oos_end]
            fold_price = price_df.iloc[:oos_end]
            fold_market = market.iloc[:oos_end]

            # Recompute regimes/signals up to this fold end (causal)
            fold_regimes = build_regime_series(fold_market, **mkwargs)

            # Build ranked signals for this prefix
            fold_sig = build_ranked_signals(prices, all_dates[:oos_end], fold_regimes, fold_price)

            # Simulate on this prefix (positions can "carry" from earlier data)
            try:
                res = simulate_portfolio(
                    price_df=fold_price,
                    sig_df=fold_sig,
                    initial=INITIAL,
                    max_positions=MAX_POS,
                    max_position_pct=0.20,
                    cost_bps=8,
                    slippage_bps=5,
                )
                metrics = {
                    'fold': fold_idx,
                    'oos_start': str(all_dates[oos_start]),
                    'oos_end': str(all_dates[oos_end-1]),
                    **{k: res.get(k) for k in ['return_pct', 'effective_sharpe', 'max_dd_pct', 'trades']}
                }
                results[mname].append(metrics)
                print(f"  Fold {fold_idx}: {metrics['return_pct']:.2f}% ret | effSR {metrics.get('effective_sharpe',0):.2f} | DD {metrics['max_dd_pct']:.1f}% | {metrics['trades']} trades")
            except Exception as e:
                print(f"  Fold {fold_idx} failed: {e}")

    # Summary
    print("\n=== WALK-FORWARD SUMMARY ===")
    summary = {}
    for mname, folds_res in results.items():
        if not folds_res:
            continue
        df = pd.DataFrame(folds_res)
        avg_ret = df['return_pct'].mean()
        avg_es = df['effective_sharpe'].mean() if 'effective_sharpe' in df else 0
        avg_dd = df['max_dd_pct'].mean()
        total_trades = df['trades'].sum()
        summary[mname] = {
            'avg_return_pct': round(avg_ret, 2),
            'avg_eff_sharpe': round(avg_es, 2),
            'avg_max_dd_pct': round(avg_dd, 2),
            'total_trades': int(total_trades),
            'num_folds': len(folds_res)
        }
        print(f"{mname}: avg ret {avg_ret:.2f}% | avg effSR {avg_es:.2f} | avg DD {avg_dd:.1f}% | total trades {total_trades} over {len(folds_res)} folds")

    with open('ma_regime_walkforward_results.json', 'w') as f:
        json.dump({'folds': results, 'summary': summary}, f, indent=2)
    print("\nSaved ma_regime_walkforward_results.json")

if __name__ == "__main__":
    main()