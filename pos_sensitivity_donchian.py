import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('data')
OUT = Path('backtest_output')
INITIAL = 1000.0
COST = 0.0013
LOOKBACK = 40
POSITIONS_OPTIONS = [5, 6]
TEST_START = pd.Timestamp('2025-01-01')

screen = pd.read_csv(sorted(OUT.glob('screen_liqu_idio_*.csv'))[-1])
screen = screen[screen['tier'].isin(['large','mid','tail'])]
coin_dfs = {}
for _, row in screen.iterrows():
    stem = str(row['stem']).strip().upper()
    p = ROOT / f'{stem}_1d_max.csv'
    if not p.exists():
        continue
    df = pd.read_csv(p, parse_dates=['ts']).dropna(subset=['close','high','low','volume'])
    df = df.sort_values('ts').reset_index(drop=True)
    df = df[df['ts'] >= TEST_START]
    if len(df) < LOOKBACK + 60:
        continue
    coin_dfs[stem] = df

stems = sorted(coin_dfs.keys())
all_dates = sorted({d for df in coin_dfs.values() for d in df['ts'].tolist()})
print(f'Loaded {len(stems)} coins spanning {len(all_dates)} test days, {all_dates[0]} -> {all_dates[-1]}', flush=True)

sig = {}
price = {}
for s, df in coin_dfs.items():
    c = df['close'].values
    h = df['high'].values
    sig_arr = np.zeros(len(df), dtype=int)
    price_arr = np.zeros(len(df), dtype=float)
    for i in range(LOOKBACK, len(df)):
        n90 = max(i - 90 + 1, 0)
        don_high = pd.Series(h[n90:i+1]).rolling(LOOKBACK).max().shift(1).iloc[-1]
        sig_arr[i] = 1 if c[i] > don_high else 0
        price_arr[i] = c[i]
    sig[s] = pd.Series(sig_arr, index=df['ts'])
    price[s] = pd.Series(price_arr, index=df['ts'])

price_df = pd.DataFrame({s: price[s] for s in stems}).sort_index().reindex(all_dates)
sig_df = pd.DataFrame({s: sig[s] for s in stems}).sort_index().reindex(all_dates)
ret_df = price_df.pct_change()

print('\nExample close panel:')
print(price_df.head(3).to_string())
print('\nExample signal panel:')
print(sig_df.head(3).to_string())

print('\n=== Running portfolio simulations ===')
records = []

for max_pos in POSITIONS_OPTIONS:
    cash = INITIAL
    slot_coin = [None] * max_pos
    slot_val = [0.0] * max_pos
    trades = 0
    eq_curve = []

    for t in range(len(all_dates)):
        day = all_dates[t]
        sig_t = sig_df.loc[day]
        ret_t = ret_df.loc[day]

        # 1) Exits first, on exact signal change
        for i in range(max_pos):
            coin = slot_coin[i]
            if coin is None:
                continue
            if sig_t.get(coin, 0) == 0:
                r = ret_t.get(coin, 0.0)
                if pd.notna(r):
                    slot_val[i] *= (1 + r)
                cash += slot_val[i] * (1 - COST)
                slot_coin[i] = None
                slot_val[i] = 0.0
                trades += 1

        # 2) Mark to market open positions using day return
        mtm = 0.0
        for i in range(max_pos):
            coin = slot_coin[i]
            if coin is None:
                continue
            r = ret_t.get(coin, 0.0)
            if pd.notna(r):
                slot_val[i] *= (1 + r)
            mtm += slot_val[i]

        equity = cash + mtm
        eq_curve.append(equity)

        # 3) Opens: fill empty slots with highest-ranked active breaks
        open_slots = sum(1 for x in slot_coin if x is None)
        if open_slots > 0:
            active = sorted([c for c in stems if sig_t.get(c, 0) == 1 and c not in slot_coin])
            picks = active[:open_slots]
            target = equity / max_pos
            j = 0
            for i in range(max_pos):
                if j >= len(picks):
                    break
                if slot_coin[i] is None:
                    coin = picks[j]
                    slot_val[i] = target * (1 - COST)
                    cash -= target
                    slot_coin[i] = coin
                    trades += 1
                    j += 1

    eq = np.array(eq_curve, dtype=np.float64)
    denom = np.where(np.abs(eq[:-1]) < 1e-12, np.nan, eq[:-1])
    ret_arr = np.diff(eq) / denom
    ret_arr = ret_arr[~np.isnan(ret_arr)]
    sharpe = float((ret_arr.mean() / (ret_arr.std() + 1e-12)) * np.sqrt(365)) if ret_arr.std() > 0 else 0.0
    total_ret = float(eq[-1] / INITIAL - 1)
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max((peak - eq) / np.where(np.abs(peak) < 1e-12, np.nan, peak)))
    records.append({
        'positions': max_pos,
        'final_equity': round(float(eq[-1]), 2),
        'return_pct': round(total_ret * 100, 2),
        'sharpe': round(sharpe, 2),
        'max_dd_pct': round(max_dd * 100, 2),
        'trades': trades
    })
    print(f'positions={max_pos} final=${eq[-1]:.2f} return={total_ret*100:.2f}% sharpe={sharpe:.2f} maxDD={max_dd*100:.2f}% trades={trades}', flush=True)

print('\n=== Sensitivity: 5 vs 6 positions ===')
print(pd.DataFrame.from_records(records).to_string(index=False))
pd.DataFrame.from_records(records).to_csv(OUT / 'pos_sensitivity_donchian40_90d.csv', index=False)
print(f'Saved {OUT / "pos_sensitivity_donchian40_90d.csv"}')
