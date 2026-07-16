#!/usr/bin/env python3
"""
Donchian 20 paper trader on screened universe.
Default: offline with existing OHLCV files only.
Set FETCH=True to also load today's kline from MEXC.
"""
import json
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path('data')
OUT = Path('backtest_output')
STATE_FILE = OUT / 'paper_state.json'
TRADES_FILE = OUT / 'paper_trades.csv'
COST = 0.0008
INITIAL = 10000.0
MAX_POSITIONS = 5
MAX_POSITION_PCT = 0.20
MAX_DRAWDOWN_FLATTEN = 0.20

screen = pd.read_csv(sorted(OUT.glob('screen_liqu_idio_*.csv'))[-1])
screen = screen[screen['tier'].isin(['large','mid','tail'])]
stems = screen['stem'].tolist()
print(f'Loaded {len(stems)} screened coins')

state = {
    'capital': INITIAL,
    'positions': {},
    'peak': INITIAL,
    'max_dd': 0.0,
    'trades': 0,
    'last_run': None,
}
if STATE_FILE.exists():
    with open(STATE_FILE) as f:
        state = json.load(f)


def save_state():
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def fetch_latest(stem):
    try:
        url = f'https://api.mexc.com/api/v3/klines?symbol={stem}&interval=1d&limit=2'
        req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.88.1'})
        with urllib.request.urlopen(req, timeout=3) as r:
            raw = json.load(r)
        if not raw:
            return None
        k = raw[-1]
        return {
            'ts': pd.to_datetime(int(k[0]), unit='ms', utc=True).tz_localize(None),
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
            'volume': float(k[5]),
        }
    except Exception:
        return None


def donchian_signal(df, lookback=40):
    high = df['high'].values
    close = df['close'].values
    don_high = pd.Series(high).rolling(lookback).max().shift(1).values
    return pd.Series(np.where(close > don_high, 1, 0), index=df.index)

prices = {}
for stem in stems:
    p = ROOT / f'{stem}_1d_max.csv'
    if not p.exists():
        continue
    df = pd.read_csv(p, parse_dates=['ts']).dropna(subset=['close','high','low','volume'])
    df = df.sort_values('ts').reset_index(drop=True)
    if len(df) < 21:
        continue
    if FETCH:
        latest = fetch_latest(stem)
        if latest and latest['ts'] > df['ts'].iloc[-1]:
            df = pd.concat([df, pd.DataFrame([latest])], ignore_index=True)
            df = df.sort_values('ts').reset_index(drop=True)
    prices[stem] = df

print(f'Got data for {len(prices)}/{len(stems)} coins')

signals = {}
for stem, df in prices.items():
    sig = donchian_signal(df, LOOKBACK)
    signals[stem] = int(sig.iloc[-1])

active = [s for s, v in signals.items() if v == 1]
print(f'Active signals: {len(active)}/{len(signals)}')

for stem in list(state['positions']):
    if stem not in active:
        alloc = state['positions'].pop(stem)
        state['capital'] += alloc * (1 - COST)
        state['trades'] += 1

to_open = [s for s in active if s not in state['positions'] and s in prices]
if to_open:
    per = state['capital'] / len(to_open)
    for stem in to_open:
        state['positions'][stem] = per
        state['capital'] -= per * (1 + COST)
        state['trades'] += 1

mtm = state['capital']
for stem, alloc in state['positions'].items():
    if stem in prices:
        mtm += alloc

state['peak'] = max(state['peak'], mtm)
state['max_dd'] = max(state['max_dd'], (state['peak'] - mtm) / state['peak'])
state['last_run'] = datetime.now(timezone.utc).isoformat()

print('\n=== Paper Trader State ===')
print(f'Capital:     ${state["capital"]:,.2f}')
print(f'Positions:   {len(state["positions"])} coins')
print(f'MTM equity:  ${mtm:,.2f}')
print(f'Peak:        ${state["peak"]:,.2f}')
print(f'Max drawdown:{state["max_dd"]:.2%}')
print(f'Trades:      {state["trades"]}')

save_state()
print(f'\nState saved to {STATE_FILE}')
