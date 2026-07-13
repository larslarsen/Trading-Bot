#!/usr/bin/env python3
"""Test new momentum variants on existing deep-history pairs."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / 'backtest_output'
OUT.mkdir(exist_ok=True)
DATA = ROOT / 'data'
FEE_BP = 0.8

PAIRS = [
    ('ARB/USDT', 'binance', 1207),
    ('ADA/USDT', 'binance', 3008),
    ('DOGE/USDT', 'binance', 2564),
    ('APE/USDT', 'binance', 1578),
    ('ENS/USDT', 'binance', 1705),
    ('CRV/USDT', 'binance', 2157),
    ('EDEN/USDT', 'mexc', 285),
    ('ALGO/USDT', 'binance', 2577),
]

def load_pair(symbol, ex):
    fname = DATA / f'{symbol.replace("/","_")}_{ex}_1d_max.csv'
    if not fname.exists(): return pd.DataFrame()
    df = pd.read_csv(fname)
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    return df.dropna(subset=['close']).sort_values('ts').reset_index(drop=True)

# ---- Existing baseline ----
def rule_momentum_baseline(df, lookback=14):
    if len(df) < lookback+1: return pd.Series(0,index=df.index), {}
    mom = df['close'].pct_change(lookback)
    sig = pd.Series(0, index=df.index)
    sig.loc[mom > 0] = 1
    sig.loc[mom < 0] = -1
    return sig.ffill().fillna(0), {}

# ---- New momentum variants ----
def rule_momentum_vol_scaled(df, lookback=14, vol_lookback=20):
    """Momentum scaled by recent volatility: only take strong trends."""
    if len(df) < max(lookback, vol_lookback)+1: return pd.Series(0,index=df.index), {}
    mom = df['close'].pct_change(lookback)
    vol = df['close'].pct_change().rolling(vol_lookback).std()
    sig = pd.Series(0, index=df.index)
    sig.loc[mom > 0] = 1
    sig.loc[mom < 0] = -1
    # hold flat if regime is extremely choppy
    choppy = vol > 2.5 * vol.rolling(100).mean()
    sig = sig.mask(choppy, 0)
    return sig.ffill().fillna(0), {}

def rule_momentum_rsi_filter(df, mom_lookback=14, rsi_lookback=14):
    """Momentum direction only taken if RSI confirms."""
    if len(df) < max(mom_lookback, rsi_lookback)+1: return pd.Series(0,index=df.index), {}
    mom = df['close'].pct_change(mom_lookback)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/rsi_lookback, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/rsi_lookback, adjust=False).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
    sig = pd.Series(0, index=df.index)
    sig.loc[(mom > 0) & (rsi > 50)] = 1
    sig.loc[(mom < 0) & (rsi < 50)] = -1
    return sig.ffill().fillna(0), {}

def rule_momentum_double(df, l1=10, l2=30):
    """Dual lookback: only long if both short and long momentum agree."""
    if len(df) < l2+1: return pd.Series(0,index=df.index), {}
    m1 = df['close'].pct_change(l1)
    m2 = df['close'].pct_change(l2)
    sig = pd.Series(0, index=df.index)
    sig.loc[(m1 > 0) & (m2 > 0)] = 1
    sig.loc[(m1 < 0) & (m2 < 0)] = -1
    return sig.ffill().fillna(0), {}

def rule_momentum_volume_confirm(df, lookback=14, vol_mult=1.2):
    """Momentum only valid when volume confirms the trend."""
    if len(df) < max(lookback, 20)+1: return pd.Series(0,index=df.index), {}
    mom = df['close'].pct_change(lookback)
    vol_ok = df['volume'] > vol_mult * df['volume'].rolling(20).mean()
    sig = pd.Series(0, index=df.index)
    sig.loc[(mom > 0) & vol_ok] = 1
    sig.loc[(mom < 0) & vol_ok] = -1
    return sig.ffill().fillna(0), {}

def rule_momentum_adaptive(df, lookback=14):
    """Adaptive lookback: penalize whipsaws by forcing flat when 14-bar mom is near zero."""
    if len(df) < lookback+20: return pd.Series(0,index=df.index), {}
    mom = df['close'].pct_change(lookback)
    threshold = mom.abs().rolling(100).quantile(0.4)  # 40th percentile of abs mom
    sig = pd.Series(0, index=df.index)
    long_cond = (mom > 0) & (mom.abs() > threshold)
    short_cond = (mom < 0) & (mom.abs() > threshold)
    sig.loc[long_cond] = 1
    sig.loc[short_cond] = -1
    return sig.ffill().fillna(0), {}

RULES = {
    'baseline_14': rule_momentum_baseline,
    'vol_scaled': rule_momentum_vol_scaled,
    'rsi_filter': rule_momentum_rsi_filter,
    'double_10_30': rule_momentum_double,
    'vol_confirm': rule_momentum_volume_confirm,
    'adaptive': rule_momentum_adaptive,
}

def eval_strat(prices, pos):
    p = np.asarray(prices, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if len(p) < 2: return {'sharpe': 0.0, 'return': 0.0, 'wr': 0.0, 'trades': 0}
    ret = np.diff(p) / p[:-1]
    cost = (FEE_BP + 0.5) / 10000.0
    pos_aligned = pos[:-1] if len(pos) == len(p) else pos
    trades = np.abs(np.diff(pos))
    strat = pos_aligned * ret - cost * trades
    trades_n = int(np.nansum(trades))
    if trades_n < 2 or np.nanstd(strat) == 0:
        return {'sharpe': 0.0, 'return': 0.0, 'wr': 0.0, 'trades': trades_n}
    sharpe = np.nanmean(strat) / np.nanstd(strat) * np.sqrt(365)
    nz = strat[strat != 0]
    wr = float(np.mean(nz > 0)) if len(nz) else 0.0
    return {'sharpe': float(sharpe), 'return': float(np.nansum(strat)*100), 'wr': float(wr*100), 'trades': trades_n}

def main():
    rows = []
    for sym, ex, approx_bars in PAIRS:
        df = load_pair(sym, ex)
        if df.empty: continue
        prices = df['close'].values
        print(f'\n=== {sym} ({len(df)} bars) ===')
        for name, fn in RULES.items():
            pos, _ = fn(df)
            s = eval_strat(prices, pos.values)
            rows.append((sym, name, s['sharpe'], s['return'], s['wr'], s['trades']))
            print(f'  {name:18s} Sharpe={s["sharpe"]:6.3f} return={s["return"]:7.2f}% wr={s["wr"]:5.1f}% trades={s["trades"]:3d}')

    top = sorted(rows, key=lambda x: x[2], reverse=True)[:25]
    print('\n=== Top 25 momentum variants ===')
    for r in top:
        print(f'{r[0]:14s} {r[1]:18s} Sharpe={r[2]:6.3f} return={r[3]:7.2f}% wr={r[4]:5.1f}% trades={r[5]:3d}')

    outfile = OUT / f'momentum_variants_{pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")}.json'
    outfile.write_text(json.dumps(rows, indent=2))
    print(f'\nWrote {outfile}')

if __name__ == '__main__':
    main()
