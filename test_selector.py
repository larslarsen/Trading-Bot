#!/usr/bin/env python3
"""
Test 5-coin selection criteria for Donchian 40.
Baseline: first-come
Candidates: ADV rank, idio_vol rank, recent return rank
"""
import sys

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('data')
OUT = Path('backtest_output')
INITIAL = 1000.0
MAX_POSITIONS = 5
MAX_POSITION_PCT = 0.20
COST_BPS = 8
SLIPPAGE_BPS = 5
MIN_EQUITY = 100.0
LOOKBACK = 40
MOMENT_LOOKBACK = 10  # for recent return ranking

screen = pd.read_csv(sorted(OUT.glob('screen_liqu_idio_*.csv'))[-1])
screen = screen[screen.tier.isin(['large','mid','tail'])]

# Merge ADV and idio_vol from screen
adv_map = dict(zip(screen['stem'].str.upper(), screen['adv']))
idio_map = dict(zip(screen['stem'].str.upper(), screen['idio_vol']))

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
        test = df[df['ts'] >= pd.Timestamp('2025-01-01')]
        if len(test) < 60:
            continue
        coin_data[stem] = {'df': df, 'adv': adv_map.get(stem, 0), 'idio': idio_map.get(stem, 0)}
    return coin_data

print('Loading...', flush=True)
coin_data = load_coins()
all_dates = sorted(set(d for c in coin_data.values() for d in c['df']['ts'].tolist() if d >= pd.Timestamp('2025-01-01')))
print(f'Dates: {len(all_dates)}, {all_dates[0]} -> {all_dates[-1]}', flush=True)

price_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
sig_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
mom_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)

for stem, data in coin_data.items():
    df = data['df']
    close = df['close'].values
    high = df['high'].values
    don_high = pd.Series(high).rolling(LOOKBACK).max().shift(1).values
    sig = np.where(close > don_high, 1, 0)
    ret = pd.Series(close).pct_change()
    mom = ret.rolling(MOMENT_LOOKBACK).sum()
    test = df['ts'] >= pd.Timestamp('2025-01-01')
    for i in range(len(df)):
        if test.iloc[i]:
            ts = df['ts'].iloc[i]
            sig_df.loc[ts, stem] = int(sig[i])
            price_df.loc[ts, stem] = close[i]
            mom_df.loc[ts, stem] = mom.iloc[i] if not np.isnan(mom.iloc[i]) else np.nan

price_df = price_df.sort_index(); sig_df = sig_df.sort_index(); mom_df = mom_df.sort_index()


def run_portfolio(selector_fn, name):
    cash = INITIAL
    positions = {}
    trades = []
    equity_curve = []
    peak = cash
    max_dd = 0.0

    for day_i, day in enumerate(all_dates):
        row_prices = price_df.loc[day]
        row_sig = sig_df.loc[day]
        row_mom = mom_df.loc[day]

        mtm = 0.0
        for sym, pos in positions.items():
            px = row_prices.get(sym)
            if pd.notna(px) and px > 0:
                mtm += pos['shares'] * px
        equity = cash + mtm
        equity_curve.append(equity)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        if equity < MIN_EQUITY:
            print(f'{name} HALT equity floor {equity:.2f}', flush=True)
            break

        # exits
        for sym in list(positions.keys()):
            if row_sig.get(sym, 0) == 0:
                px = row_prices.get(sym)
                if pd.notna(px) and px > 0:
                    p = positions.pop(sym)
                    proceeds = p['shares'] * px * (1 - COST_BPS/10000)
                    pnl = proceeds - p['shares'] * p['entry']
                    fees = p['shares'] * p['entry'] * COST_BPS/10000 + p['shares'] * px * COST_BPS/10000
                    cash += proceeds
                    trades.append({'day': str(day), 'sym': sym, 'side': 'SELL', 'pnl': pnl, 'fees': fees})

        # entries
        active = [s for s in row_sig.index if row_sig.get(s, 0) == 1 and s not in positions]
        if active and len(positions) < MAX_POSITIONS:
            ranked = selector_fn(active, row_sig, row_mom, coin_data)
            open_slots = MAX_POSITIONS - len(positions)
            candidates = ranked[:open_slots]
            size_usd = (cash if cash > 0 else equity) * MAX_POSITION_PCT
            for sym in candidates:
                px = row_prices.get(sym)
                if pd.isna(px) or px <= 0:
                    continue
                fill = px * (1 + SLIPPAGE_BPS/10000 + COST_BPS/10000)
                shares = size_usd / fill
                positions[sym] = {'shares': shares, 'entry': fill}
                cash -= size_usd

    print(f'\n=== {name} ===', flush=True)
    print(f'Final equity: ${equity:.2f}', flush=True)
    print(f'Return: {(equity/INITIAL - 1)*100:.2f}%', flush=True)
    eq_arr = np.array(equity_curve)
    ret_arr = np.diff(eq_arr) / eq_arr[:-1]
    sharpe = np.mean(ret_arr) / np.std(ret_arr) * np.sqrt(365) if len(ret_arr) > 0 and np.std(ret_arr) > 0 else 0.0
    print(f'Sharpe: {sharpe:.2f}', flush=True)
    print(f'Max DD: {max_dd*100:.2f}%', flush=True)
    print(f'Trades: {len(trades)}', flush=True)
    return {'name': name, 'final': float(equity), 'return': float((equity/INITIAL - 1)*100), 'sharpe': float(sharpe), 'max_dd': float(max_dd*100), 'trades': len(trades)}

# Selectors
def selector_first_come(active, row_sig, row_mom, coin_data):
    return active[:MAX_POSITIONS]

def selector_adv(active, row_sig, row_mom, coin_data):
    return sorted(active, key=lambda s: coin_data.get(s, {}).get('adv', 0), reverse=True)[:MAX_POSITIONS]

def selector_idio(active, row_sig, row_mom, coin_data):
    return sorted(active, key=lambda s: coin_data.get(s, {}).get('idio', 0), reverse=True)[:MAX_POSITIONS]

def selector_momentum(active, row_sig, row_mom, coin_data):
    return sorted(active, key=lambda s: row_mom.get(s, -999), reverse=True)[:MAX_POSITIONS]

results = []
print('\n--- Testing selectors ---', flush=True)
results.append(run_portfolio(selector_first_come, 'first_come'))
results.append(run_portfolio(selector_adv, 'adv'))
results.append(run_portfolio(selector_idio, 'idio_vol'))
results.append(run_portfolio(selector_momentum, 'momentum'))

print('\n=== SUMMARY ===', flush=True)
for r in results:
    print(f"{r['name']}: ret={r['return']:.1f}%, sharpe={r['sharpe']:.2f}, dd={r['max_dd']:.1f}%, trades={r['trades']}", flush=True)

pd.DataFrame(results).to_csv(OUT / 'selector_test_results.csv', index=False)
