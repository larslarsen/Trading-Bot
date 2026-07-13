#!/usr/bin/env python3
"""
Live paper trader: Donchian 20 on screened alts.
Validated benchmark: 10 max positions, no per-trade stops,
20% portfolio drawdown flatten, long-only.
"""
import sys
print('START paper_trader_multi.py', flush=True)
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from order_manager_multi import (
    MultiPositionState,
    MAX_POSITIONS,
    MAX_DRAWDOWN_PCT,
    MAX_POSITION_PCT,
    MIN_EQUITY_TO_TRADE,
    COST_BPS,
    SLIPPAGE_BPS,
)
from engine import compute_live_regime, get_regime_signals

ROOT = Path('data')
OUT = Path('backtest_output')
INITIAL = 10000.0
FETCH = True
LOOKBACK = 40


def fetch_latest(stem):
    try:
        import urllib.request
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



screen = pd.read_csv(sorted(OUT.glob('screen_liqu_idio_*.csv'))[-1])
screen = screen[screen.tier.isin(['large','mid','tail'])]
stems = screen.stem.tolist()
state = MultiPositionState(initial_capital=INITIAL)

prices = {}
raw_signals = {}
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

print(f'Loaded {len(prices)} coins')

# Compute market regime from recent data
# Use improved regime detector (literature-backed ADX + vol + Kaufman ER + hysteresis)
# Set use_improved_regime=False to fall back to the original simple gate
use_improved_regime = True
if use_improved_regime:
    try:
        from engine import improved_compute_live_regime
        regime = improved_compute_live_regime(prices)
    except Exception:
        regime = compute_live_regime(prices)
else:
    regime = compute_live_regime(prices)

trend_rule = "cci"          # Best from regime_backtest.py (CCI-trend/Williams-chop)
chop_rule = "williams_r"   # Best performer in chop regime
active_rule = trend_rule if regime == 'trend' else chop_rule
print(f'Regime: {regime} → using {active_rule} (improved={use_improved_regime})')

# Compute signals with the chosen rule
for stem, df in prices.items():
    try:
        entry, exit_sig = get_regime_signals(active_rule, df)
        last_entry = int(entry.iloc[-1]) if len(entry) > 0 else 0
        last_exit = int(exit_sig.iloc[-1]) if len(exit_sig) > 0 else 0
        raw_signals[stem] = {'entry': last_entry, 'exit': last_exit}
    except Exception as e:
        raw_signals[stem] = {'entry': 0, 'exit': 0}

# Build active list: open on entry; keep if no exit (or re-evaluate on entry for simplicity)
active = []
for sym in list(state.positions.keys()):
    # If we have an exit signal for current rule, we'll close below
    pass

for sym, sigs in raw_signals.items():
    if sigs['entry']:
        active.append(sym)

# For coins we are in, force close on explicit exit
positions_to_close = []
for sym in list(state.positions.keys()):
    if sym in raw_signals and raw_signals[sym]['exit']:
        positions_to_close.append(sym)

print(f'Active entry signals ({active_rule}): {len(active)}')

# Reconcile stale positions
for sym in list(state.positions.keys()):
    px_row = prices.get(sym)
    if px_row is None:
        px_row = fetch_latest(sym)
    if px_row is None:
        print(f'Stale position {sym}: no price, closing at entry')
        state.close_position(sym, state.positions[sym].entry)

# Circuit breaker check
first_price = None
for sym in list(state.positions.keys()) + active[:1]:
    px_row = prices.get(sym)
    if px_row is not None:
        first_price = float(px_row['close'].iloc[-1])
        break

if first_price is not None:
    ok, reason = state.check_circuit_breakers(first_price)
    if not ok:
        if state.halted and 'drawdown' in str(state.halt_reason):
            print(f'CIRCUIT BREAKER {reason}, flattening to cash')
            mtm_prices = {}
            for sym in list(state.positions.keys()):
                px_row = prices.get(sym)
                if px_row is not None:
                    mtm_prices[sym] = float(px_row['close'].iloc[-1])
            state.flatten_all(mtm_prices)
        else:
            print(f'CIRCUIT BREAKER: {reason}, staying flat')
            state.save()
            exit(0)

# Close positions no longer breaking out OR explicit exit from regime rule
for sym in list(state.positions.keys()):
    should_close = sym not in active
    if sym in raw_signals and raw_signals[sym].get('exit'):
        should_close = True
    if should_close:
        px_row = prices.get(sym)
        px = float(px_row['close'].iloc[-1]) if px_row is not None else None
        if px is None or px <= 0:
            continue
        state.close_position(sym, px)
        print(f'CLOSE {sym} @ {px:.4f} (regime rule or no signal)')

# Open new positions
for sym in active:
    if sym in state.positions:
        continue
    if len(state.positions) >= MAX_POSITIONS:
        break
    px_row = prices.get(sym)
    px = float(px_row['close'].iloc[-1]) if px_row is not None else None
    if px is None or px <= 0:
        continue
    ok, reason = state.check_circuit_breakers(px)
    if not ok:
        print(f'CIRCUIT BREAKER: {reason}, staying flat')
        break
    size = state.equity * MAX_POSITION_PCT * state.vol_scale()
    pos = state.open_position(sym, px, size)
    if pos:
        print(f'OPEN {sym} @ {px:.4f}, size=${size:.2f}')

# Mark to market
mtm_prices = {sym: float(df['close'].iloc[-1]) for sym, df in prices.items() if len(df)}
eq = state.update_equity_from_mtm(mtm_prices)
current_dd = (state.peak_equity - eq) / state.peak_equity if state.peak_equity > 0 else 0
state.equity_history.append(float(eq))
if len(state.equity_history) > 365:
    state.equity_history = state.equity_history[-365:]
state.save()

print(f'MTM equity: ${eq:.2f}')
print(f'Peak: ${state.peak_equity:.2f}')
print(f'Max DD: {state.max_dd:.2%}')
print(f'Positions: {len(state.positions)}')

if current_dd > MAX_DRAWDOWN_PCT:
    print(f'Flattening: DD {current_dd:.2%} > {MAX_DRAWDOWN_PCT:.2%}')
    state.flatten_all(mtm_prices)
    state.halt(f'live_drawdown_flatten_{pd.Timestamp.now().isoformat()}')
    state.save()


# Optional: regime quality report (uncomment or pass --regime-stats)
if 'prices' in dir() and len(prices) > 0:
    try:
        from engine import load_screened_universe, analyze_regime_quality, print_regime_stats
        # Rebuild quick market proxy for the run
        closes = []
        for df in prices.values():
            if len(df) > 0:
                closes.append(df['close'].iloc[-60:].reset_index(drop=True))
        if closes:
            minl = min(len(c) for c in closes)
            mkt = pd.concat([c.iloc[-minl:] for c in closes], axis=1).mean(axis=1)
            if len(mkt) > 30:
                stats = analyze_regime_quality(mkt, adx_threshold=22, vol_threshold=0.22, er_threshold=0.35)
                print("\n=== Regime quality for this run ===")
                print_regime_stats(stats)
    except Exception as e:
        pass  # silent in normal runs

