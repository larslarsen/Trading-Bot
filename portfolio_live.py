import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('data')
OUT = Path('backtest_output')
TRADES_CSV = OUT / 'portfolio_trades.csv'
EQUITY_CSV = OUT / 'portfolio_equity.csv'

# Load screened universe
screen = pd.read_csv(sorted(OUT.glob('screen_liqu_idio_*.csv'))[-1])
screen = screen[screen['tier'].isin(['large','mid','tail'])]

def donchian_signal(df, lookback=20):
    high = df['high'].values
    close = df['close'].values
    don_high = pd.Series(high).rolling(lookback).max().shift(1).values
    return pd.Series(np.where(close > don_high, 1, 0), index=df.index)

# Load all coin data
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
    }

# Build full return and signal matrix
all_dates = sorted(set(d for c in coin_data.values() for d in c['df']['ts'].tolist() if d >= pd.Timestamp('2025-01-01')))
stem_list = list(coin_data.keys())
ret_df = pd.DataFrame(index=all_dates, columns=stem_list, dtype=float)
sig_df = pd.DataFrame(index=all_dates, columns=stem_list, dtype=int)
close_df = pd.DataFrame(index=all_dates, columns=stem_list, dtype=float)

for stem, data in coin_data.items():
    df = data['df']
    close = df['close'].values
    sig = donchian_signal(df, 20)
    test_mask = df['ts'] >= pd.Timestamp('2025-01-01')
    for i in range(len(df)):
        if test_mask.iloc[i]:
            ts = df['ts'].iloc[i]
            close_df.loc[ts, stem] = close[i]
            ret = (close[i] / close[i-1] - 1) if i > 0 and not np.isnan(close[i-1]) else 0.0
            ret_df.loc[ts, stem] = ret
            sig_df.loc[ts, stem] = 1 if sig.iloc[i] else 0

ret_df = ret_df.sort_index()
sig_df = sig_df.sort_index()
close_df = close_df.sort_index()

# Monthly rebalance dates: last trading day of each month
months = pd.date_range('2025-01-01', '2026-07-12', freq='ME')
rebal_dates = [m for m in months if m in ret_df.index]
if ret_df.index[-1] not in rebal_dates:
    rebal_dates.append(ret_df.index[-1])

# Portfolio simulation
INITIAL_CAPITAL = 10000.0
COST_PER_TRADE = 0.0008
MAX_POSITIONS = 99  # trade all screened coins

capital = INITIAL_CAPITAL
positions = {stem: 0.0 for stem in stem_list}  # value per coin
trades = []
equity_curve = []
peak = capital
max_dd = 0.0

prev_active = []

for i, date in enumerate(ret_df.index):
    # Rebalance at month-end
    if date in rebal_dates:
        # Select all coins with signal=1 on this date
        active = [s for s in stem_list if s in sig_df.columns and sig_df.loc[date, s] == 1]
        if not active:
            active = prev_active if prev_active else stem_list[:MAX_POSITIONS]
        if len(active) > MAX_POSITIONS:
            # Use idio_vol as secondary sort (higher idio first)
            idio_map = screen.set_index('stem')['idio_vol'].to_dict()
            active = sorted(active, key=lambda s: idio_map.get(s, 0), reverse=True)[:MAX_POSITIONS]

        # Close positions no longer active
        for stem in positions:
            if stem not in active and positions[stem] > 0:
                price = close_df.loc[date, stem] if stem in close_df.columns else 0
                if price > 0:
                    proceeds = positions[stem] * (1 - COST_PER_TRADE)
                    capital += proceeds
                    trades.append({'date': date, 'symbol': stem, 'side': 'SELL', 'price': price, 'value': positions[stem]})
                positions[stem] = 0.0

        # Open new positions
        weight_per = capital / len(active) if active else 0
        for stem in active:
            price = close_df.loc[date, stem] if stem in close_df.columns else 0
            if price <= 0:
                continue
            alloc = weight_per
            if positions.get(stem, 0) == 0:
                # Buy
                shares = alloc / price
                positions[stem] = alloc
                capital -= alloc * (1 + COST_PER_TRADE)
                trades.append({'date': date, 'symbol': stem, 'side': 'BUY', 'price': price, 'shares': shares, 'value': alloc})

        prev_active = active

    # Mark-to-market equity
    mtm = capital + sum(positions.get(s, 0) for s in stem_list if s in close_df.columns and not pd.isna(close_df.loc[date, s]) and positions.get(s, 0) > 0 for s in [s])
    pos_value = 0.0
    for s in stem_list:
        if s in positions and positions[s] > 0 and s in close_df.columns and not pd.isna(close_df.loc[date, s]):
            pos_value += positions[s]
    mtm = capital + pos_value
    equity_curve.append({'date': date, 'equity': mtm, 'capital': capital, 'positions_value': pos_value})
    peak = max(peak, mtm)
    max_dd = max(max_dd, (peak - mtm) / peak)

trades_df = pd.DataFrame(trades)
equity_df = pd.DataFrame(equity_curve)

# Daily return series
equity_df['ret'] = equity_df['equity'].pct_change().fillna(0)
sr = float(np.sqrt(365) * equity_df['ret'].mean() / (equity_df['ret'].std() + 1e-9))
total_ret = float(equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0] - 1)
days = len(equity_df)

print('\n=== Portfolio backtest: Donchian 20, monthly rebalance, equal-weight ===')
print(f'Period: {equity_df["date"].iloc[0]} -> {equity_df["date"].iloc[-1]} ({days} days)')
print(f'Initial capital: ${INITIAL_CAPITAL:,.2f}')
print(f'Final equity: ${equity_df["equity"].iloc[-1]:,.2f}')
print(f'Total return: {total_ret:.2%}')
print(f'Sharpe (annualized): {sr:.2f}')
print(f'Max drawdown: {max_dd:.2%}')
print(f'Total trades: {len(trades_df)}')
print(f'Buy trades: {(trades_df["side"]=="BUY").sum()}')
print(f'Sell trades: {(trades_df["side"]=="SELL").sum()}')

# Monthly return table
equity_df['month'] = equity_df['date'].dt.to_period('M')
monthly = equity_df.groupby('month')['equity'].last().pct_change().dropna()
print('\nMonthly returns:')
print(monthly.describe().to_string())

trades_df.to_csv(TRADES_CSV, index=False)
equity_df.to_csv(EQUITY_CSV, index=False)
print(f'\nSaved trades to {TRADES_CSV}')
print(f'Saved equity to {EQUITY_CSV}')
