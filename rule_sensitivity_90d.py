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
        test = df[df['ts'] >= pd.Timestamp('2025-01-01')]
        if len(test) < 60:
            continue
        coin_data[stem] = {'df': df}
    return coin_data

print('Loading...', flush=True)
coin_data = load_coins()
all_dates = sorted(set(d for c in coin_data.values() for d in c['df']['ts'].tolist() if d >= pd.Timestamp('2025-01-01')))
print(f'Dates: {len(all_dates)}, {all_dates[0]} -> {all_dates[-1]}', flush=True)

price_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
sig_d40 = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_d10 = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_ens = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sma_50_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
sma_100_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
sma_200_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
vol_ma_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)

for stem, data in coin_data.items():
    df = data['df']
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    test = df['ts'] >= pd.Timestamp('2025-01-01')

    dh40 = pd.Series(high).rolling(40).max().shift(1).values
    sig40 = np.where(close > dh40, 1, 0)

    dh10 = pd.Series(high).rolling(10).max().shift(1).values
    sig10 = np.where(close > dh10, 1, 0)

    ens = np.zeros(len(df), dtype=int)
    for lb in [20, 40, 60, 90]:
        dh = pd.Series(high).rolling(lb).max().shift(1).values
        ens = np.where((ens == 1) | (close > dh), 1, 0)
    sig_ensemble = ens

    sma50 = pd.Series(close).rolling(50).mean().values
    sma100 = pd.Series(close).rolling(100).mean().values
    sma200 = pd.Series(close).rolling(200).mean().values
    vol_ma = pd.Series(volume).rolling(1).mean().shift(1).values

    for i in range(len(df)):
        if test.iloc[i]:
            ts = df['ts'].iloc[i]
            sig_d40.loc[ts, stem] = int(sig40[i])
            sig_d10.loc[ts, stem] = int(sig10[i])
            sig_ens.loc[ts, stem] = int(sig_ensemble[i])
            price_df.loc[ts, stem] = close[i]
            sma_50_df.loc[ts, stem] = sma50[i] if not pd.isna(sma50[i]) else np.nan
            sma_100_df.loc[ts, stem] = sma100[i] if not pd.isna(sma100[i]) else np.nan
            sma_200_df.loc[ts, stem] = sma200[i] if not pd.isna(sma200[i]) else np.nan
            vol_ma_df.loc[ts, stem] = vol_ma[i] if not pd.isna(vol_ma[i]) else np.nan

price_df = price_df.sort_index()
sig_d40 = sig_d40.sort_index()
sig_d10 = sig_d10.sort_index()
sig_ens = sig_ens.sort_index()
sma_50_df = sma_50_df.sort_index()
sma_100_df = sma_100_df.sort_index()
sma_200_df = sma_200_df.sort_index()
vol_ma_df = vol_ma_df.sort_index()

recent_dates = price_df.index.tolist()[-90:]
print(f'Running 90-day window: {recent_dates[0]} -> {recent_dates[-1]}', flush=True)

def run_once(sig_df, label, sma_filter=None, vol_filter=False):
    cash = INITIAL
    positions = {}
    trades = []
    equity_curve = []
    peak = cash
    max_dd = 0.0
    daily_pnl = 0.0

    for day in recent_dates:
        row_prices = price_df.loc[day]
        row_sig = sig_df.loc[day]
        row_sma = None
        if sma_filter == 50:
            row_sma = sma_50_df.loc[day]
        elif sma_filter == 100:
            row_sma = sma_100_df.loc[day]
        elif sma_filter == 200:
            row_sma = sma_200_df.loc[day]
        row_vol_ma = vol_ma_df.loc[day] if vol_filter else None

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
            break
        if daily_pnl < -0.03 * peak:
            for sym in list(positions.keys()):
                px = row_prices.get(sym)
                if pd.notna(px) and px > 0:
                    p = positions.pop(sym)
                    proceeds = p['shares'] * px * (1 - COST_BPS/10000)
                    fees = p['shares'] * p['entry'] * COST_BPS/10000 + p['shares'] * px * COST_BPS/10000
                    cash += proceeds
                    trades.append({'side':'SELL_DAILY','pnl': proceeds - p['shares']*p['entry'], 'fees': fees})
            daily_pnl = 0
            continue

        active = []
        for sym in row_sig.index:
            sig = row_sig.get(sym, 0)
            if sig != 1 or sym in positions:
                continue
            if sma_filter is not None:
                sma_v = row_sma.get(sym)
                px = row_prices.get(sym)
                if pd.isna(sma_v) or pd.isna(px) or px <= 0 or px < sma_v:
                    continue
            if vol_filter:
                v = row_vol_ma.get(sym)
                vol = coin_data.get(sym, {}).get('df', pd.DataFrame()).loc[coin_data.get(sym, {}).get('df', pd.DataFrame())['ts'] == day, 'volume'].values
                if pd.isna(v) or v <= 0 or (len(vol) and vol[0] < v):
                    continue
            active.append(sym)

        for sym in list(positions.keys()):
            if sym in positions and row_sig.get(sym, 0) == 0:
                px = row_prices.get(sym)
                if pd.notna(px) and px > 0:
                    p = positions.pop(sym)
                    proceeds = p['shares'] * px * (1 - COST_BPS/10000)
                    fees = p['shares'] * p['entry'] * COST_BPS/10000 + p['shares'] * px * COST_BPS/10000
                    cash += proceeds
                    trades.append({'side':'SELL_SIG','pnl': proceeds - p['shares']*p['entry'], 'fees': fees})

        if active and len(positions) < MAX_POSITIONS:
            open_slots = MAX_POSITIONS - len(positions)
            candidates = active[:open_slots]
            size_usd = cash * MAX_POSITION_PCT if cash > 0 else 0
            for sym in candidates:
                px = row_prices.get(sym)
                if pd.isna(px) or px <= 0:
                    continue
                fill = px * (1 + (SLIPPAGE_BPS + COST_BPS)/10000)
                shares = size_usd / fill
                positions[sym] = {'shares': shares, 'entry': fill}
                cash -= size_usd

    eq_arr = np.array(equity_curve)
    ret_arr = np.diff(eq_arr) / eq_arr[:-1] if len(eq_arr) > 1 else np.array([])
    sharpe = float(np.mean(ret_arr) / (np.std(ret_arr) + 1e-12) * np.sqrt(365)) if ret_arr.size and np.std(ret_arr) > 0 else 0.0
    total_ret = float((float(eq_arr[-1]) / INITIAL - 1) * 100) if len(eq_arr) else 0.0
    peak = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peak - eq_arr) / np.where(np.abs(peak) < 1e-12, np.nan, peak)) * 100) if len(eq_arr) else 0.0
    return {
        'name': label,
        'final_equity': round(float(eq_arr[-1]), 2),
        'return_pct': round(total_ret, 2),
        'sharpe': round(sharpe, 2),
        'max_dd_pct': round(max_dd, 2),
        'trades': len(trades),
    }

records = []
records.append(run_once(sig_d40, 'baseline_d40'))
records.append(run_once(sig_d10, 'd10'))
records.append(run_once(sig_ens, 'ensemble_20_40_60_90'))
records.append(run_once(sig_d40, 'd40_sma50_filter', sma_filter=50))
records.append(run_once(sig_d40, 'd40_sma100_filter', sma_filter=100))
records.append(run_once(sig_d40, 'd40_sma200_filter', sma_filter=200))
records.append(run_once(sig_d40, 'd40_vol_filter', vol_filter=True))

summary = pd.DataFrame.from_records(records)
summary = summary[['name','final_equity','return_pct','sharpe','max_dd_pct','trades']]
out_path = OUT / 'rule_sensitivity_90d.csv'
summary.to_csv(out_path, index=False)
print('\n=== RULE SENSITIVITY: 90d ===')
print(summary.to_string(index=False))
print(f'Saved {out_path}')
