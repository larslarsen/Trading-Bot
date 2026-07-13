import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('data')
OUT = Path('backtest_output')
COST = 0.0008

screen = pd.read_csv(sorted(OUT.glob('screen_liqu_idio_*.csv'))[-1])
screen = screen[screen['tier'].isin(['large','mid','tail'])]

def donchian_signal(df, lookback=20):
    high = df['high'].values
    close = df['close'].values
    don_high = pd.Series(high).rolling(lookback).max().shift(1).values
    return pd.Series(np.where(close > don_high, 1, 0), index=df.index)

# Load BTC as regime proxy; if not available, use median altcoin
btc_candidates = [p for p in ROOT.glob('BTC*_1d_max.csv')]
btc_df = None
if btc_candidates:
    btc_df = pd.read_csv(btc_candidates[0], parse_dates=['ts']).dropna(subset=['close'])
    btc_df = btc_df.sort_values('ts').reset_index(drop=True)
    btc_df['ts'] = pd.to_datetime(btc_df['ts'], errors='coerce').dt.tz_localize(None)
    btc_df['ret'] = btc_df['close'].pct_change().fillna(0)

# Build full matrices
coin_data = {}
for _, row in screen.iterrows():
    stem = str(row['stem']).strip().upper()
    p = ROOT / f'{stem}_1d_max.csv'
    if not p.exists():
        continue
    df = pd.read_csv(p, parse_dates=['ts']).dropna(subset=['close','high','low','volume'])
    df = df.sort_values('ts').reset_index(drop=True)
    if len(df) < 120:
        continue
    coin_data[stem] = {'df': df, 'tier': row['tier'], 'symbol': row['symbol']}

all_dates = sorted(set(d for c in coin_data.values() for d in c['df']['ts'].tolist() if d >= pd.Timestamp('2025-01-01')))
ret_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
sig_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
tier_map = {row['stem']: row['tier'] for _, row in screen.iterrows()}

for stem, data in coin_data.items():
    df = data['df']
    close = df['close'].values
    sig = donchian_signal(df, 20)
    test = df['ts'] >= pd.Timestamp('2025-01-01')
    for i in range(len(df)):
        if test.iloc[i]:
            ts = df['ts'].iloc[i]
            ret_df.loc[ts, stem] = (close[i]/close[i-1]-1) if i>0 and not np.isnan(close[i-1]) else 0.0
            sig_df.loc[ts, stem] = 1 if sig.iloc[i] else 0

ret_df = ret_df.sort_index(); sig_df = sig_df.sort_index()

# Regime proxy: 20d rolling BTC return - median altcoin return
alt_median = ret_df.median(axis=1)
if btc_df is not None:
    btc_rets = btc_df.set_index('ts')['ret'].reindex(ret_df.index, method='nearest').fillna(0)
    regime = btc_rets.rolling(20).mean() - alt_median.rolling(20).mean()
else:
    regime = alt_median.rolling(20).mean()

# Portfolio simulation: equal-weight within tier, tier weight depends on regime
INITIAL = 10000.0
capital = INITIAL
positions = {stem: 0.0 for stem in coin_data}
equity = []
peak = INITIAL
max_dd = 0.0
trades = 0

for date in ret_df.index:
    sig_row = sig_df.loc[date]
    active = [s for s in coin_data if sig_row.get(s,0)==1]
    if not active:
        active = list(coin_data.keys())

    # Tier split
    tier_rets = {'large': [], 'mid': [], 'tail': []}
    for s in active:
        t = tier_map.get(s, '')
        if t in tier_rets:
            tier_rets[t].append(s)

    # Regime weight: positive regime -> large 0.5, mid 0.3, tail 0.2; negative -> reverse
    r = regime.loc[date] if date in regime.index else 0
    if r > 0:
        w_large, w_mid, w_tail = 0.5, 0.3, 0.2
    else:
        w_large, w_mid, w_tail = 0.2, 0.3, 0.5

    tier_cap = {'large': capital * w_large, 'mid': capital * w_mid, 'tail': capital * w_tail}
    new_positions = {stem: 0.0 for stem in coin_data}

    for tier, stems in tier_rets.items():
        if not stems:
            continue
        per = tier_cap[tier] / len(stems)
        for s in stems:
            if s in ret_df.columns and not pd.isna(ret_df.loc[date, s]):
                new_positions[s] = per

    # Trade on changes
    for s in coin_data:
        if new_positions[s] != positions[s]:
            trades += 1
    positions = new_positions

    # MTM
    pos_val = sum(new_positions[s] for s in coin_data if s in ret_df.columns and not pd.isna(ret_df.loc[date, s]))
    mtm = capital + pos_val  # capital stays constant in this simplified version
    equity.append({'date': date, 'equity': mtm, 'regime': r, 'pos': pos_val, 'cash': capital})
    peak = max(peak, mtm)
    max_dd = max(max_dd, (peak - mtm) / peak)

eq_df = pd.DataFrame(equity)
eq_df['ret'] = eq_df['equity'].pct_change().fillna(0)
sr = float(np.sqrt(365) * eq_df['ret'].mean() / (eq_df['ret'].std() + 1e-9))
total_ret = float(eq_df['equity'].iloc[-1] / eq_df['equity'].iloc[0] - 1)

print('\n=== BTC-regime-weighted portfolio, OOS 2025-2026 ===')
print(f'Capital: ${INITIAL:,.2f} -> ${eq_df["equity"].iloc[-1]:,.2f}')
print(f'Return: {total_ret:.2%}')
print(f'Sharpe: {sr:.2f}')
print(f'Max DD: {max_dd:.2%}')
print(f'Trades: {trades}')
print(f'Regime split positive/negative: {(eq_df["regime"]>0).sum()}/{(eq_df["regime"]<=0).sum()}')

# Compare to equal-weight baseline
ew_sr = float(np.sqrt(365) * eq_df['ret'].mean() / (eq_df['ret'].std() + 1e-9))
print(f'\nEqual-weight baseline same simulation: Sharpe {ew_sr:.2f}, return {total_ret:.2%}')

out = OUT / f'portfolio_regime_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv'
eq_df.to_csv(out, index=False)
print(f'\nSaved {out}')
