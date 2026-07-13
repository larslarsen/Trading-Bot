"""Regime fallback: d40 trending -> TRIX choppy.
Tuning knobs at top.
"""
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from engine import simulate_portfolio, donchian_signal

# ---- tuning knobs ----
VOL_LOOKBACK = 20
VOL_THRESHOLD = 0.25
VOL_CONSEC = 5
MAX_POSITIONS = 5
MAX_POS_PCT = 0.20
COST_BPS = 8
SLIPPAGE_BPS = 5
MIN_EQUITY = 100.0
DAILY_LOSS_PCT = 0.03

ROOT = Path('data')
OUT = Path('backtest_output')
INITIAL = 1000.0

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


def trix_signals(df):
    close = df['close'].values
    ema1 = pd.Series(close).ewm(span=1, adjust=False).mean()
    ema2 = ema1.ewm(span=1, adjust=False).mean()
    ema3 = ema2.ewm(span=1, adjust=False).mean()
    trix = ema3.pct_change() * 100
    entry = pd.Series(((trix > 0) & (trix.diff() > 0)).astype(int), index=df.index)
    exit_sig = pd.Series((trix < 0).astype(int), index=df.index)
    return entry, exit_sig


print('Loading...', flush=True)
coin_data = load_coins()
all_dates = sorted(set(d for c in coin_data.values() for d in c.loc[(c['ts'] >= '2025-01-01') & (c['ts'] <= '2026-07-12'), 'ts']))
print(f'Coins: {len(coin_data)}, Dates: {len(all_dates)}', flush=True)

price_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
sig_d40 = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_trix_entry = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_trix_exit = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)

for stem, df in coin_data.items():
    close = df['close'].values
    e, x = trix_signals(df)
    mask = (df['ts'] >= '2025-01-01') & (df['ts'] <= '2026-07-12')
    s = donchian_signal(df['high'], df['low'], df['close'], 40)
    for i in range(len(df)):
        if mask.iloc[i]:
            ts = df['ts'].iloc[i]
            price_df.loc[ts, stem] = close[i]
            sig_d40.loc[ts, stem] = int(s.iloc[i]) if pd.notna(s.iloc[i]) else 0
            sig_trix_entry.loc[ts, stem] = int(e.iloc[i]) if pd.notna(e.iloc[i]) else 0
            sig_trix_exit.loc[ts, stem] = int(x.iloc[i]) if pd.notna(x.iloc[i]) else 0

price_df = price_df.sort_index()
sig_d40 = sig_d40.sort_index()
sig_trix_entry = sig_trix_entry.sort_index()
sig_trix_exit = sig_trix_exit.sort_index()

all_prices = price_df.ffill().bfill()
port_ret = all_prices.pct_change().mean(axis=1)
realized_vol = port_ret.rolling(VOL_LOOKBACK).std() * np.sqrt(365)
realized_vol = realized_vol.reindex(price_df.index).fillna(0.0)

full_dates = price_df.index.tolist()
recent_dates = full_dates[-90:]
TRADING_DAYS = 252


def run_regime(dates, price_sub, sig_d40_sub, sig_entry_sub, sig_exit_sub, vol_sub):
    cash = INITIAL
    positions = {}
    trades = 0
    equity_curve = []
    peak = INITIAL
    max_dd = 0.0
    daily_pnl = 0.0
    ret_history = []
    vol_counter = 0

    for day_i, day in enumerate(dates):
        row_prices = price_sub.loc[day]
        row_d40 = sig_d40_sub.loc[day]
        row_entry = sig_entry_sub.loc[day]
        row_exit = sig_exit_sub.loc[day]
        vol = float(vol_sub.get(day, 0.0))

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

        if equity < MIN_EQUITY or daily_pnl < -DAILY_LOSS_PCT:
            for s in list(positions.keys()):
                px = row_prices.get(s, 0)
                if pd.notna(px) and px > 0:
                    cash += positions[s]['shares'] * px * (1 - COST_BPS/10000)
                    trades += 1
            positions.clear()
            daily_pnl = 0.0
            continue

        row_exit_use = row_exit if regime == 'choppy' else row_d40
        row_entry_use = row_entry if regime == 'choppy' else row_d40
        for s in list(positions.keys()):
            if s not in positions:
                continue
            if int(row_exit_use.get(s, 0)) == 0:
                px = row_prices.get(s, 0)
                if pd.notna(px) and px > 0:
                    cash += positions[s]['shares'] * px * (1 - COST_BPS/10000)
                    trades += 1
                    positions.pop(s)

        if len(positions) < MAX_POSITIONS:
            active = [s for s in row_entry_use.index if pd.notna(row_entry_use.get(s, 0)) and int(row_entry_use.get(s, 0)) == 1 and s not in positions][:MAX_POSITIONS - len(positions)]
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

    eq = np.array(equity_curve, dtype=float)
    ret = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
    sharpe = float(np.mean(ret) / (np.std(ret) + 1e-12) * np.sqrt(TRADING_DAYS)) if ret.size and np.std(ret) > 0 else 0.0
    total_ret = float((float(eq[-1]) / INITIAL - 1) * 100) if len(eq) else 0.0
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max((peak - eq) / np.where(np.abs(peak) < 1e-12, np.nan, peak)) * 100) if len(eq) else 0.0
    return {'return_pct': total_ret, 'sharpe': sharpe, 'max_dd_pct': max_dd, 'trades': trades}


print('\n=== d40 baseline (full) ===', flush=True)
res = simulate_portfolio(price_df, sig_d40, initial=INITIAL, max_positions=MAX_POSITIONS)
print(f"baseline_d40: ret={res['return_pct']:.1f}% sharpe={res['sharpe']:.2f} dd={res['max_dd_pct']:.1f}% trades={res['trades']}", flush=True)

print('\n=== trix baseline (full) ===', flush=True)
res_i = simulate_portfolio(price_df, sig_trix_entry, initial=INITIAL, max_positions=MAX_POSITIONS, exit_signal_df=sig_trix_exit)
print(f"trix: ret={res_i['return_pct']:.1f}% sharpe={res_i['sharpe']:.2f} dd={res_i['max_dd_pct']:.1f}% trades={res_i['trades']}", flush=True)

print('\n=== regime fallback d40->trix (full) ===', flush=True)
res_reg = run_regime(full_dates, price_df, sig_d40, sig_trix_entry, sig_trix_exit, realized_vol)
print(f"regime_d40_trix full: ret={res_reg['return_pct']:.1f}% sharpe={res_reg['sharpe']:.2f} dd={res_reg['max_dd_pct']:.1f}% trades={res_reg['trades']}", flush=True)

print('\n=== regime fallback d40->trix (90d) ===', flush=True)
res_reg90 = run_regime(recent_dates, price_df.loc[recent_dates], sig_d40.loc[recent_dates], sig_trix_entry.loc[recent_dates], sig_trix_exit.loc[recent_dates], realized_vol.loc[recent_dates])
print(f"regime_d40_trix 90d: ret={res_reg90['return_pct']:.1f}% sharpe={res_reg90['sharpe']:.2f} dd={res_reg90['max_dd_pct']:.1f}% trades={res_reg90['trades']}", flush=True)
