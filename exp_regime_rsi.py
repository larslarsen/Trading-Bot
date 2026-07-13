
"""Regime fallback: d40 in trending, RSI momentum in choppy."""
import sys

import numpy as np
import pandas as pd
from pathlib import Path

from engine import simulate_portfolio, donchian_signal

ROOT = Path('data')
OUT = Path('backtest_output')
INITIAL = 1000.0
MAX_POSITIONS = 5
MAX_POS_PCT = 0.20
COST_BPS = 8
SLIPPAGE_BPS = 5
MIN_EQUITY = 100.0
VOL_LOOKBACK = 20
VOL_THRESHOLD = 0.25
VOL_CONSEC = 5

screen = pd.read_csv(sorted(OUT.glob('screen_liqu_idio_*.csv'))[-1])
screen = screen[screen.tier.isin(['large','mid','tail'])]


def load_coins():
    coin_data = {}
    seen = set()
    for _, row in screen.iterrows():
        stem = str(row['stem']).strip().upper()
        if stem in seen:
            continue
        seen.add(stem)
        p = ROOT / f'{stem}_1d_max.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=['ts']).dropna(subset=['close','high','low','volume'])
        df = df.sort_values('ts').reset_index(drop=True)
        if len(df[(df['ts'] >= '2025-01-01') & (df['ts'] <= '2026-07-12')]) < 60:
            continue
        coin_data[stem] = df
    return coin_data


def rsi_signal(close, entry_low=45, entry_high=55, exit_thr=70):
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    entry = ((rsi > entry_high) & (rsi.shift(1) < entry_low)).astype(int)
    exit_sig = (rsi > exit_thr).astype(int)
    return entry, exit_sig


print('Loading...', flush=True)
coin_data = load_coins()
all_dates = sorted(set(d for c in coin_data.values() for d in c.loc[(c['ts'] >= '2025-01-01') & (c['ts'] <= '2026-07-12'), 'ts']))
print(f'Coins: {len(coin_data)}, Dates: {len(all_dates)}', flush=True)

price_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
sig_d40 = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_rsi_entry = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_rsi_exit = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)

for stem, df in coin_data.items():
    close = df['close'].values
    mask = (df['ts'] >= '2025-01-01') & (df['ts'] <= '2026-07-12')
    for i in range(len(df)):
        if mask.iloc[i]:
            ts = df['ts'].iloc[i]
            price_df.loc[ts, stem] = close[i]
            d40 = donchian_signal(df['high'], df['low'], df['close'], 40)
            sig_d40.loc[ts, stem] = int(d40.iloc[i]) if pd.notna(d40.iloc[i]) else 0
            e, x = rsi_signal(close[:i+1])
            sig_rsi_entry.loc[ts, stem] = int(e.iloc[i]) if pd.notna(e.iloc[i]) else 0
            sig_rsi_exit.loc[ts, stem] = int(x.iloc[i]) if pd.notna(x.iloc[i]) else 0

price_df = price_df.sort_index()
sig_d40 = sig_d40.sort_index()
sig_rsi_entry = sig_rsi_entry.sort_index()
sig_rsi_exit = sig_rsi_exit.sort_index()

# portfolio realized vol for regime detection
all_prices = price_df.ffill().bfill()
port_ret = all_prices.pct_change().mean(axis=1)
realized_vol = port_ret.rolling(VOL_LOOKBACK).std() * np.sqrt(365)
realized_vol = realized_vol.reindex(price_df.index).fillna(0.0)

full_dates = price_df.index.tolist()
recent_dates = full_dates[-90:]

print('\n=== d40 baseline (full) ===', flush=True)
res = simulate_portfolio(price_df, sig_d40, initial=INITIAL, max_positions=MAX_POSITIONS)
print(f"baseline_d40: ret={res['return_pct']:.1f}% sharpe={res['sharpe']:.2f} dd={res['max_dd_pct']:.1f}% trades={res['trades']}", flush=True)

print('\n=== rsi_momentum (full) ===', flush=True)
res_rsi = simulate_portfolio(price_df, sig_rsi_entry, initial=INITIAL, max_positions=MAX_POSITIONS, exit_signal_df=sig_rsi_exit)
print(f"rsi_momentum: ret={res_rsi['return_pct']:.1f}% sharpe={res_rsi['sharpe']:.2f} dd={res_rsi['max_dd_pct']:.1f}% trades={res_rsi['trades']}", flush=True)

print('\n=== regime fallback d40->rsi (full) ===', flush=True)
cash = INITIAL
positions = {}
trades = 0
equity_curve = []
peak = INITIAL
max_dd = 0.0
daily_pnl = 0.0
ret_history = []
vol_counter = 0

for day_i, day in enumerate(price_df.index):
    row_prices = price_df.loc[day]
    row_d40 = sig_d40.loc[day]
    row_rsi = sig_rsi_entry.loc[day]
    row_rsi_exit = sig_rsi_exit.loc[day]
    vol = realized_vol.get(day, 0.0)

    if vol > VOL_THRESHOLD:
        vol_counter += 1
    else:
        vol_counter = max(0, vol_counter - 1)
    regime = 'choppy' if vol_counter >= VOL_CONSEC else 'trending'

    mtm = sum(positions[s]['shares'] * row_prices.get(s, 0) for s in positions if pd.notna(row_prices.get(s, 0)) and row_prices.get(s, 0) > 0)
    equity = cash + mtm
    equity_curve.append(equity)
    peak = max(peak, equity)
    dd = (peak - equity) / peak if peak > 0 else 0.0
    max_dd = max(max_dd, dd)
    if len(equity_curve) > 1:
        day_ret = equity / equity_curve[-2] - 1
        ret_history.append(day_ret)
        daily_pnl += day_ret

    if equity < MIN_EQUITY or daily_pnl < -0.03 * peak:
        for s in list(positions.keys()):
            px = row_prices.get(s, 0)
            if pd.notna(px) and px > 0:
                cash += positions[s]['shares'] * px * (1 - COST_BPS/10000)
                trades += 1
        positions.clear()
        daily_pnl = 0.0
        continue

    row_exit = row_rsi_exit if regime == 'choppy' else row_d40
    for s in list(positions.keys()):
        if s not in positions:
            continue
        if int(row_exit.get(s, 0)) == 0:
            px = row_prices.get(s, 0)
            if pd.notna(px) and px > 0:
                cash += positions[s]['shares'] * px * (1 - COST_BPS/10000)
                trades += 1
                positions.pop(s)

    row_entry = row_rsi if regime == 'choppy' else row_d40
    if len(positions) < MAX_POSITIONS:
        active = [s for s in row_entry.index if pd.notna(row_entry.get(s, 0)) and int(row_entry.get(s, 0)) == 1 and s not in positions][:MAX_POSITIONS - len(positions)]
        for s in active:
            px = row_prices.get(s, 0)
            if pd.isna(px) or px <= 0:
                continue
            size_usd = cash * MAX_POS_PCT
            if size_usd <= 0 or cash < size_usd:
                continue
            fill = px * (1 + (SLIPPAGE_BPS + COST_BPS) / 10000)
            positions[s] = {'shares': size_usd / fill}
            cash -= size_usd
            if len(positions) >= MAX_POSITIONS:
                break

eq_arr = np.array(equity_curve, dtype=float)
ret_arr = np.diff(eq_arr) / eq_arr[:-1] if len(eq_arr) > 1 else np.array([])
sharpe = float(np.mean(ret_arr) / (np.std(ret_arr) + 1e-12) * np.sqrt(365)) if ret_arr.size and np.std(ret_arr) > 0 else 0.0
total_ret = float((float(eq_arr[-1]) / INITIAL - 1) * 100) if len(eq_arr) else 0.0
peak = np.maximum.accumulate(eq_arr)
max_dd = float(np.max((peak - eq_arr) / np.where(np.abs(peak) < 1e-12, np.nan, peak)) * 100) if len(eq_arr) else 0.0
print(f"regime_d40_rsi full: ret={total_ret:.1f}% sharpe={sharpe:.2f} dd={max_dd:.1f}% trades={trades}", flush=True)

# 90-day regime fallback
print('\n=== regime fallback d40->rsi (90d) ===', flush=True)
cash = INITIAL
positions = {}
trades = 0
equity_curve = []
peak = INITIAL
max_dd = 0.0
daily_pnl = 0.0
ret_history = []
vol_counter = 0

for day_i, day in enumerate(recent_dates):
    row_prices = price_df.loc[day]
    row_d40 = sig_d40.loc[day]
    row_rsi = sig_rsi_entry.loc[day]
    row_rsi_exit = sig_rsi_exit.loc[day]
    vol = realized_vol.get(day, 0.0)

    if vol > VOL_THRESHOLD:
        vol_counter += 1
    else:
        vol_counter = max(0, vol_counter - 1)
    regime = 'choppy' if vol_counter >= VOL_CONSEC else 'trending'

    mtm = sum(positions[s]['shares'] * row_prices.get(s, 0) for s in positions if pd.notna(row_prices.get(s, 0)) and row_prices.get(s, 0) > 0)
    equity = cash + mtm
    equity_curve.append(equity)
    peak = max(peak, equity)
    dd = (peak - equity) / peak if peak > 0 else 0.0
    max_dd = max(max_dd, dd)
    if len(equity_curve) > 1:
        day_ret = equity / equity_curve[-2] - 1
        ret_history.append(day_ret)
        daily_pnl += day_ret

    if equity < MIN_EQUITY or daily_pnl < -0.03 * peak:
        for s in list(positions.keys()):
            px = row_prices.get(s, 0)
            if pd.notna(px) and px > 0:
                cash += positions[s]['shares'] * px * (1 - COST_BPS/10000)
                trades += 1
        positions.clear()
        daily_pnl = 0.0
        continue

    row_exit = row_rsi_exit if regime == 'choppy' else row_d40
    for s in list(positions.keys()):
        if s not in positions:
            continue
        if int(row_exit.get(s, 0)) == 0:
            px = row_prices.get(s, 0)
            if pd.notna(px) and px > 0:
                cash += positions[s]['shares'] * px * (1 - COST_BPS/10000)
                trades += 1
                positions.pop(s)

    row_entry = row_rsi if regime == 'choppy' else row_d40
    if len(positions) < MAX_POSITIONS:
        active = [s for s in row_entry.index if pd.notna(row_entry.get(s, 0)) and int(row_entry.get(s, 0)) == 1 and s not in positions][:MAX_POSITIONS - len(positions)]
        for s in active:
            px = row_prices.get(s, 0)
            if pd.isna(px) or px <= 0:
                continue
            size_usd = cash * MAX_POS_PCT
            if size_usd <= 0 or cash < size_usd:
                continue
            fill = px * (1 + (SLIPPAGE_BPS + COST_BPS) / 10000)
            positions[s] = {'shares': size_usd / fill}
            cash -= size_usd
            if len(positions) >= MAX_POSITIONS:
                break

eq_arr = np.array(equity_curve, dtype=float)
ret_arr = np.diff(eq_arr) / eq_arr[:-1] if len(eq_arr) > 1 else np.array([])
sharpe = float(np.mean(ret_arr) / (np.std(ret_arr) + 1e-12) * np.sqrt(365)) if ret_arr.size and np.std(ret_arr) > 0 else 0.0
total_ret = float((float(eq_arr[-1]) / INITIAL - 1) * 100) if len(eq_arr) else 0.0
peak = np.maximum.accumulate(eq_arr)
max_dd = float(np.max((peak - eq_arr) / np.where(np.abs(peak) < 1e-12, np.nan, peak)) * 100) if len(eq_arr) else 0.0
print(f"regime_d40_rsi 90d: ret={total_ret:.1f}% sharpe={sharpe:.2f} dd={max_dd:.1f}% trades={trades}", flush=True)
