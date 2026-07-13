import sys

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path('data')
OUT = Path('backtest_output')
INITIAL = 1000.0
COST_BPS = 8
SLIPPAGE_BPS = 5
MIN_EQUITY = 100.0
MAX_POSITIONS = 5
MAX_POSITION_PCT = 0.20
RSI_PERIOD = 14
RSI_OVERSOLD = 30
LOOKBACK_RET = 5
DROP_THRESHOLD = -0.10
DATA_START = pd.Timestamp('2025-01-01')

screen = pd.read_csv(sorted(OUT.glob('screen_liqu_idio_*.csv'))[-1])

def load_coins(filter_to_screened=True):
    coin_data = {}
    seen = set()
    sources = screen[screen.tier.isin(['large','mid','tail'])] if filter_to_screened else screen
    for _, row in sources.iterrows():
        stem = str(row['stem']).strip().upper()
        if stem in seen:
            continue
        seen.add(stem)
        p = ROOT / f'{stem}_1d_max.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=['ts']).dropna(subset=['close','high','low','volume'])
        df = df.sort_values('ts').reset_index(drop=True)
        test = df['ts'] >= DATA_START
        if len(df[test]) < 60:
            continue
        coin_data[stem] = {'df': df}
    return coin_data

print('Loading screened universe...', flush=True)
coin_data_screened = load_coins(True)
print(f'Screened: {len(coin_data_screened)} coins')

print('Loading broad universe...', flush=True)
coin_data_broad = load_coins(False)
print(f'Broad: {len(coin_data_broad)} coins')

all_screened_dates = sorted(set(d for c in coin_data_screened.values() for d in c['df']['ts'].tolist() if d >= DATA_START))
all_broad_dates = sorted(set(d for c in coin_data_broad.values() for d in c['df']['ts'].tolist() if d >= DATA_START))
recent_dates = sorted(set(all_screened_dates + all_broad_dates))[-90:]

price_s = pd.DataFrame(index=all_screened_dates, columns=list(coin_data_screened.keys()), dtype=float)
sig_mr_s = pd.DataFrame(index=all_screened_dates, columns=list(coin_data_screened.keys()), dtype=int)
price_b = pd.DataFrame(index=all_broad_dates, columns=list(coin_data_broad.keys()), dtype=float)
sig_mr_b = pd.DataFrame(index=all_broad_dates, columns=list(coin_data_broad.keys()), dtype=int)
sig_d40_s = pd.DataFrame(index=all_screened_dates, columns=list(coin_data_screened.keys()), dtype=int)

for stem, data in coin_data_screened.items():
    df = data['df']; close = df['close'].values; high = df['high'].values; volume = df['volume'].values; test = df['ts'] >= DATA_START
    dh40 = pd.Series(high).rolling(40).max().shift(1).values
    sig40 = np.where(close > dh40, 1, 0)
    delta = pd.Series(close).diff(); gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean(); loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan); rsi = (100 - (100 / (1 + rs))).fillna(50).values
    ret5 = pd.Series(close).pct_change(LOOKBACK_RET).values
    for i in range(len(df)):
        if test.iloc[i]:
            ts = df['ts'].iloc[i]
            price_s.loc[ts, stem] = close[i]; sig_d40_s.loc[ts, stem] = int(sig40[i])
            if rsi[i] < RSI_OVERSOLD and ret5[i] < DROP_THRESHOLD:
                sig_mr_s.loc[ts, stem] = 1

for stem, data in coin_data_broad.items():
    df = data['df']; close = df['close'].values; test = df['ts'] >= DATA_START
    delta = pd.Series(close).diff(); gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean(); loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan); rsi = (100 - (100 / (1 + rs))).fillna(50).values
    ret5 = pd.Series(close).pct_change(LOOKBACK_RET).values
    for i in range(len(df)):
        if test.iloc[i]:
            ts = df['ts'].iloc[i]
            price_b.loc[ts, stem] = close[i]
            if rsi[i] < RSI_OVERSOLD and ret5[i] < DROP_THRESHOLD:
                sig_mr_b.loc[ts, stem] = 1

price_s = price_s.sort_index(); sig_mr_s = sig_mr_s.sort_index(); sig_d40_s = sig_d40_s.sort_index()
price_b = price_b.sort_index(); sig_mr_b = sig_mr_b.sort_index()

print('\n=== FULL WINDOW 2025-2026 ===', flush=True)

def build_rsi(price_df, stems):
    out = {}
    for stem in stems:
        if stem not in price_df.columns:
            continue
        close = price_df[stem].dropna().values
        if len(close) < RSI_PERIOD + 1:
            continue
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).fillna(50)
        out[stem] = dict(zip(price_df.index, rsi))
    return out

def run_once(price_df, sig_df, label, dates, exit_rule='rsi70'):
    cash = INITIAL
    positions = {}
    trades = 0
    equity_curve = []
    peak = cash
    max_dd = 0.0
    daily_pnl = 0.0
    rsi_map = build_rsi(price_df, sig_df.columns) if 'rsi' in str(exit_rule) else {}
    for day in dates:
        row_prices = price_df.loc[day]; row_sig = sig_df.loc[day]
        mtm = 0.0
        for sym, pos in positions.items():
            px = row_prices.get(sym, 0)
            if pd.notna(px) and px > 0:
                mtm += pos['shares'] * px
        equity = cash + mtm; equity_curve.append(equity); peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0; max_dd = max(max_dd, dd)
        if equity < MIN_EQUITY:
            break
        if daily_pnl < -0.03 * peak:
            for sym in list(positions.keys()):
                px = row_prices.get(sym, 0)
                if pd.notna(px) and px > 0:
                    cash += positions[sym]['shares'] * px * (1 - COST_BPS/10000)
                    positions.pop(sym)
            daily_pnl = 0.0; continue
        for sym in list(positions.keys()):
            if sym not in positions:
                continue
            ex = False
            if exit_rule == 'sig':
                ex = row_sig.get(sym, 0) == 0
            elif 'rsi' in str(exit_rule):
                thresh = float(str(exit_rule).replace('rsi',''))
                ex = rsi_map.get(sym, {}).get(day, 50) > thresh
            elif str(exit_rule).startswith('time') and 'entry_day' in positions[sym]:
                ex = (pd.Timestamp(day) - pd.Timestamp(positions[sym]['entry_day'])).days >= int(str(exit_rule).replace('time',''))
            elif str(exit_rule).startswith('high10') and 'entry_day' in positions[sym]:
                hist = price_df.loc[:day, sym].dropna().iloc[-11:]
                ex = len(hist) >= 11 and row_prices.get(sym, 0) < hist.iloc[:-1].max()
            if ex:
                px = row_prices.get(sym, 0)
                if pd.notna(px) and px > 0:
                    cash += positions[sym]['shares'] * px * (1 - COST_BPS/10000)
                    trades += 1
                    positions.pop(sym)
        if len(positions) < MAX_POSITIONS:
            active = [s for s in row_sig.index if row_sig.get(s, 0) == 1 and s not in positions]
            slots = MAX_POSITIONS - len(positions)
            for sym in active[:slots]:
                px = row_prices.get(sym, 0)
                if pd.isna(px) or px <= 0:
                    continue
                size_usd = cash * MAX_POSITION_PCT if cash > 0 else 0
                if size_usd <= 0:
                    break
                fill = px * (1 + (SLIPPAGE_BPS + COST_BPS)/10000)
                positions[sym] = {'shares': size_usd / fill, 'entry_day': day}
                cash -= size_usd
                if len(positions) >= MAX_POSITIONS:
                    break
    eq_arr = np.array(equity_curve)
    ret_arr = np.diff(eq_arr) / eq_arr[:-1] if len(eq_arr) > 1 else np.array([])
    sharpe = float(np.mean(ret_arr) / (np.std(ret_arr) + 1e-12) * np.sqrt(365)) if ret_arr.size and np.std(ret_arr) > 0 else 0.0
    total_ret = float((float(eq_arr[-1]) / INITIAL - 1) * 100) if len(eq_arr) else 0.0
    peak = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peak - eq_arr) / np.where(np.abs(peak) < 1e-12, np.nan, peak)) * 100) if len(eq_arr) else 0.0
    print(f"  {label}: final=${eq_arr[-1]:.2f} ret={total_ret:.1f}% sharpe={sharpe:.2f} dd={max_dd:.1f}% trades={trades}", flush=True)

run_once(price_s, sig_d40_s, 'd40_screened_full', all_screened_dates, 'sig')
run_once(price_s, sig_mr_s, 'mr_screened_rsi70', all_screened_dates, 'rsi70')
run_once(price_s, sig_mr_s, 'mr_screened_rsi60', all_screened_dates, 'rsi60')
run_once(price_s, sig_mr_s, 'mr_screened_time20', all_screened_dates, 'time20')
run_once(price_s, sig_mr_s, 'mr_screened_high10', all_screened_dates, 'high10')
run_once(price_b, sig_mr_b, 'mr_broad_rsi70', all_broad_dates, 'rsi70')
run_once(price_b, sig_mr_b, 'mr_broad_rsi60', all_broad_dates, 'rsi60')
run_once(price_b, sig_mr_b, 'mr_broad_time20', all_broad_dates, 'time20')
run_once(price_b, sig_mr_b, 'mr_broad_high10', all_broad_dates, 'high10')

print('\n=== RECENT 90d ===', flush=True)
run_once(price_s, sig_d40_s, 'd40_screened_90d', recent_dates, 'sig')
run_once(price_s, sig_mr_s, 'mr_screened_90d_rsi70', recent_dates, 'rsi70')
run_once(price_s, sig_mr_s, 'mr_screened_90d_time20', recent_dates, 'time20')
run_once(price_s, sig_mr_s, 'mr_screened_90d_high10', recent_dates, 'high10')

print('\n=== EARLIEST WINDOW SUB-PERIOD ===', flush=True)
from datetime import timedelta
sub_size = min(150, len(all_screened_dates))
sub_dates = all_screened_dates[:sub_size]
run_once(price_s, sig_mr_s, 'mr_screened_early_rsi70', sub_dates, 'rsi70')
run_once(price_s, sig_mr_s, 'mr_screened_early_time20', sub_dates, 'time20')
run_once(price_s, sig_mr_s, 'mr_screened_early_high10', sub_dates, 'high10')
