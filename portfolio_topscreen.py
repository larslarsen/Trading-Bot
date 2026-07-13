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

# Load all coin data into memory
coin_data = {}
for _, row in screen.iterrows():
    stem = str(row['stem']).strip().upper()
    p = ROOT / f'{stem}_1d_max.csv'
    if not p.exists():
        continue
    df = pd.read_csv(p, parse_dates=['ts']).dropna(subset=['close','high','low','volume'])
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce').dt.tz_localize(None)
    df = df.sort_values('ts').reset_index(drop=True)
    n = len(df)
    if n < 120:
        continue
    coin_data[stem] = {
        'df': df,
        'tier': row['tier'],
        'symbol': row['symbol'],
        'idio_vol': row['idio_vol'],
    }

# Build full test-window daily return matrix
all_dates = sorted(set(d for c in coin_data.values() for d in c['df']['ts'].tolist() if d >= pd.Timestamp('2025-01-01')))
stem_list = list(coin_data.keys())
ret_df = pd.DataFrame(index=all_dates, columns=stem_list, dtype=float)
sig_df = pd.DataFrame(index=all_dates, columns=stem_list, dtype=int)

for stem, data in coin_data.items():
    df = data['df']
    close = df['close'].values
    sig = donchian_signal(df, 20)
    test_mask = df['ts'] >= pd.Timestamp('2025-01-01')
    for i in range(len(df)):
        if test_mask.iloc[i]:
            ts = df['ts'].iloc[i]
            ret = (close[i] / close[i-1] - 1) if i > 0 and not np.isnan(close[i-1]) else 0.0
            ret_df.loc[ts, stem] = ret
            sig_df.loc[ts, stem] = 1 if sig.iloc[i] else 0

ret_df = ret_df.sort_index()
sig_df = sig_df.sort_index()

# Monthly rebalance: last day of each month
months = pd.date_range('2025-01-01', '2026-07-12', freq='ME')
rebal_dates = [m for m in months if m in ret_df.index]
if ret_df.index[-1] not in rebal_dates:
    rebal_dates.append(ret_df.index[-1])

# Ranking metrics: prior month mean daily return
def rank_coins(date, n):
    # Use prior month performance as rank
    prior = ret_df.loc[ret_df.index < date]
    if len(prior) < 20:
        return []
    recent = prior.tail(20)
    scores = recent.mean().dropna()
    top = scores.nlargest(n).index.tolist()
    return top

results = []
for n in [10, 20, 30, 50, 99]:
    port_rets = []
    rebalance_count = 0
    for i, date in enumerate(ret_df.index):
        if i == 0 or date in rebal_dates:
            rebalance_count += 1
            selected = rank_coins(date, n)
            if not selected:
                selected = stem_list[:n]
        # only include coins with signal=1 and in selected
        active = [s for s in selected if s in sig_df.columns and sig_df.loc[date, s] == 1]
        if not active:
            port_rets.append(0.0)
            continue
        w = 1.0 / len(active)
        port_ret = sum(w * ret_df.loc[date, s] for s in active if s in ret_df.columns and not pd.isna(ret_df.loc[date, s])) - COST
        port_rets.append(port_ret)

    pr = pd.Series(port_rets).dropna()
    if len(pr) < 20:
        continue
    sr = float(np.sqrt(365) * pr.mean() / (pr.std() + 1e-9))
    ret = float((1 + pr).prod() - 1)
    results.append({
        'n': n, 'rebalances': rebalance_count,
        'sharpe': sr, 'return': ret,
        'mean_daily': pr.mean(), 'std_daily': pr.std(),
        'days': len(pr),
    })

res = pd.DataFrame(results)
print('\n=== Portfolio construction: equal-weight top-N, monthly rebalance ===')
print(res[['n','rebalances','sharpe','return','mean_daily','std_daily']].to_string(index=False))

# Compare top-N vs all
base = res[res['n']==99].iloc[0]
print(f'\nBaseline all 99: Sharpe {base["sharpe"]:.2f}, return {base["return"]:.2%}')
for _, row in res.iterrows():
    if row['n'] == 99:
        continue
    print(f'Top {int(row["n"])}: Sharpe {row["sharpe"]:.2f}, return {row["return"]:.2%}, improvement {row["sharpe"] - base["sharpe"]:.2f}')

out = OUT / f'portfolio_topscreen_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv'
res.to_csv(out, index=False)
print(f'\nSaved {out}')
