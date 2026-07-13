"""
Live regime-aware paper trader for screened altcoins.

Uses centralized signals from engine.py (cci/rei/williams_r + others).
Improved regime detector (ADX + vol + Kaufman ER + hysteresis). Optional Hurst persistence filter.
Defaults: REI (trend) / Williams %R (chop) per verification backtests.
Long-only, hard cap 5 positions (ranked by signal strength), 20% equity/trade, 20% DD halt, costs.
Paranoid vol target (regime-gated in chop) available via USE_PARANOID_VOL_TARGET flag.
"""
import sys
from datetime import datetime
print(f"\n=== paper_trader_multi run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
print('START paper_trader_multi.py', flush=True)
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from order_manager_multi import (
    ENABLE_VOL_TARGET,
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

# === Paranoia mode (vol targeting) ===
# Set USE_PARANOID_VOL_TARGET = True when you want to be more defensive.
# It applies vol scaling *only* when the regime is "chop".
# In "trend" regimes you get full position sizing (1.0 scale).
USE_PARANOID_VOL_TARGET = False
VOL_TARGET = 0.15

# Hurst regime (literature-backed persistence filter)
# H > 0.5 suggests persistent/trending behavior -> favor momentum rules
USE_HURST_REGIME = False
HURST_WINDOW = 60
HURST_THRESHOLD = 0.5


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
use_improved_regime = True  # always use improved regime detector
USE_MA_REGIME = True  # Try friend suggestion: MA crossover (short > long) as regime filter
if USE_HURST_REGIME:
    from engine import hurst_regime, load_screened_universe
    # Build market close proxy for Hurst
    closes = []
    for df in prices.values():
        if len(df) > 0:
            closes.append(df['close'].iloc[-HURST_WINDOW-5:].reset_index(drop=True))
    if closes:
        minl = min(len(c) for c in closes)
        mkt = pd.concat([c.iloc[-minl:] for c in closes], axis=1).mean(axis=1)
        regime = hurst_regime(mkt, len(mkt)-1, window=HURST_WINDOW, threshold=HURST_THRESHOLD)
    else:
        regime = "chop"
elif use_improved_regime:
    try:
        from engine import improved_compute_live_regime, compute_regime
        if USE_MA_REGIME:
            # Use simple MA crossover as regime (short MA > long MA = trend)
            from engine import load_screened_universe
            closes = []
            for df in prices.values():
                if len(df) > 0:
                    closes.append(df['close'].iloc[-60:].reset_index(drop=True))
            if closes:
                minl = min(len(c) for c in closes)
                mkt = pd.concat([c.iloc[-minl:] for c in closes], axis=1).mean(axis=1)
                regime = compute_regime(mkt, len(mkt)-1, method="ma")
            else:
                regime = "chop"
        else:
            regime = improved_compute_live_regime(prices)
    except Exception:
        regime = compute_live_regime(prices)
else:
    regime = compute_live_regime(prices)

trend_rule = "rei"          # Best from verification (REI-trend/Williams-chop) - stronger on recent OOS/WF
chop_rule = "williams_r"   # Best performer in chop regime
active_rule = trend_rule if regime == 'trend' else chop_rule
# Available centralized rules include: ma30 / ma30_recapture (new), cci, rei, williams_r
print(f'Regime: {regime} → using {active_rule} (improved={use_improved_regime}, ma={USE_MA_REGIME})')

# Compute daily vol scale factor
# - If USE_PARANOID_VOL_TARGET: only apply vol scaling in chop regime
# - Otherwise fall back to the global ENABLE_VOL_TARGET behavior
vol_scale_factor = 1.0
if USE_PARANOID_VOL_TARGET:
    if regime == "chop":
        vol_scale_factor = state.vol_scale()
    else:
        vol_scale_factor = 1.0
elif ENABLE_VOL_TARGET:
    vol_scale_factor = state.vol_scale()

# Compute signals with the chosen rule + strength for ranking
for stem, df in prices.items():
    try:
        entry, exit_sig = get_regime_signals(active_rule, df)
        last_entry = int(entry.iloc[-1]) if len(entry) > 0 else 0
        last_exit = int(exit_sig.iloc[-1]) if len(exit_sig) > 0 else 0

        strength = 0.0
        if active_rule == "williams_r":
            # Williams %R strength: higher = more oversold (wr is negative, e.g. -95 is stronger than -82)
            high = pd.Series(df["high"].values, index=df.index)
            low = pd.Series(df["low"].values, index=df.index)
            close = pd.Series(df["close"].values, index=df.index)
            period = 14
            highest = high.rolling(period, min_periods=1).max()
            lowest = low.rolling(period, min_periods=1).min()
            wr = -100 * (highest - close) / (highest - lowest + 1e-12)
            last_wr = float(wr.iloc[-1]) if len(wr) > 0 else -50.0
            # Add small bonus for how much it has risen (positive diff)
            wr_diff = float(wr.diff().iloc[-1]) if len(wr) > 1 else 0.0
            strength = (-last_wr) + max(0.0, wr_diff * 2.0)
        elif active_rule == "rei":
            # REI strength: higher positive rei is better
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
            strength = last_rei + max(0.0, rei_diff * 0.5)
        else:
            strength = 1.0  # fallback

        raw_signals[stem] = {'entry': last_entry, 'exit': last_exit, 'strength': strength}
    except Exception as e:
        raw_signals[stem] = {'entry': 0, 'exit': 0, 'strength': 0.0}

# Build active list: open on entry; keep if no exit (or re-evaluate on entry for simplicity)
active = []
for sym in list(state.positions.keys()):
    # If we have an exit signal for current rule, we'll close below
    pass

for sym, sigs in raw_signals.items():
    if sigs['entry']:
        active.append((sym, sigs.get('strength', 0.0)))

# Rank by strength descending (best first)
active = [sym for sym, _ in sorted(active, key=lambda x: -x[1])]

# For coins we are in, force close on explicit exit
positions_to_close = []
for sym in list(state.positions.keys()):
    if sym in raw_signals and raw_signals[sym]['exit']:
        positions_to_close.append(sym)

raw_active_count = len([s for s in raw_signals.values() if s.get('entry')])
print(f'Active entry signals ({active_rule}): {raw_active_count} (ranked, taking top {MAX_POSITIONS} by strength)')

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
for sym in list(state.positions.keys()) + (active[:1] if active else []):
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
    size = state.equity * MAX_POSITION_PCT * vol_scale_factor
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

print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M")}] MTM equity: ${eq:.2f}')
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

