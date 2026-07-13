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
TIME_EXIT = 20
VOL_LOOKBACK = 20
TARGET_VOL = 0.15
VOL_BOUNDS = (0.25, 1.5)
CHOPPY_VOL = 0.25
CHOPPY_DAYS = 5

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
        test = df['ts'] >= pd.Timestamp('2025-01-01')
        if len(df[test]) < 60:
            continue
        coin_data[stem] = {'df': df}
    return coin_data

print('Loading...', flush=True)
coin_data = load_coins()
all_dates = sorted(set(d for c in coin_data.values() for d in c['df']['ts'].tolist() if d >= pd.Timestamp('2025-01-01')))
print(f'Dates: {len(all_dates)}, {all_dates[0]} -> {all_dates[-1]}', flush=True)

price_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
sig_d40 = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_mr = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)

for stem, data in coin_data.items():
    df = data['df']
    close = df['close'].values
    high = df['high'].values
    test = df['ts'] >= pd.Timestamp('2025-01-01')

    dh40 = pd.Series(high).rolling(40).max().shift(1).values
    sig40 = np.where(close > dh40, 1, 0)

    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values
    ret5 = pd.Series(close).pct_change(LOOKBACK_RET).values

    for i in range(len(df)):
        if test.iloc[i]:
            ts = df['ts'].iloc[i]
            sig_d40.loc[ts, stem] = int(sig40[i])
            price_df.loc[ts, stem] = close[i]
            if rsi[i] < RSI_OVERSOLD and ret5[i] < DROP_THRESHOLD:
                sig_mr.loc[ts, stem] = 1

price_df = price_df.sort_index()
sig_d40 = sig_d40.sort_index()
sig_mr = sig_mr.sort_index()

for window_name, dates in [('full_500d', None), ('recent_90d', None)]:
    if dates is None:
        dates = price_df.index.tolist()
    else:
        dates = price_df.index.tolist()[-90:]
    print(f'\n=== {window_name} ===', flush=True)

    def run_regime_fallback(label, use_mr_always=False):
        cash = INITIAL
        positions = {}
        trades = 0
        equity_curve = []
        peak = cash
        max_dd = 0.0
        daily_pnl = 0.0
        ret_history = []
        choppy_counter = 0
        use_mr = False

        # precompute MR RSI for time-exit is not needed; time exit uses entry_day
        for day in dates:
            row_prices = price_df.loc[day]
            row_d40 = sig_d40.loc[day]
            row_mr = sig_mr.loc[day]

            mtm = 0.0
            for sym, pos in positions.items():
                px = row_prices.get(sym, 0)
                if pd.notna(px) and px > 0:
                    mtm += pos['shares'] * px
            equity = cash + mtm
            equity_curve.append(equity)
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

            if len(equity_curve) > 1:
                ret_history.append(equity / equity_curve[-2] - 1)

            if not use_mr_always:
                if len(ret_history) >= VOL_LOOKBACK:
                    vol = float(np.std(np.array(ret_history[-VOL_LOOKBACK:])) * np.sqrt(365))
                    if vol > CHOPPY_VOL:
                        choppy_counter += 1
                    else:
                        choppy_counter = 0
                    use_mr = choppy_counter >= CHOPPY_DAYS

            if equity < MIN_EQUITY:
                break
            if daily_pnl < -0.03 * peak:
                for sym in list(positions.keys()):
                    px = row_prices.get(sym, 0)
                    if pd.notna(px) and px > 0:
                        cash += positions[sym]['shares'] * px * (1 - COST_BPS/10000)
                        positions.pop(sym)
                daily_pnl = 0.0
                continue

            # scale = vol target if MR mode
            scale = 1.0
            if use_mr or use_mr_always:
                if len(ret_history) >= VOL_LOOKBACK:
                    vol = float(np.std(np.array(ret_history[-VOL_LOOKBACK:])) * np.sqrt(365))
                    if vol > 0:
                        scale = float(np.clip(TARGET_VOL / vol, VOL_BOUNDS[0], VOL_BOUNDS[1]))

            # exits
            for sym in list(positions.keys()):
                if sym not in positions:
                    continue
                ex = False
                if use_mr or use_mr_always:
                    # time20 exit for MR
                    if 'entry_day' in positions[sym]:
                        days_held = (pd.Timestamp(day) - pd.Timestamp(positions[sym]['entry_day'])).days
                        if days_held >= TIME_EXIT:
                            ex = True
                else:
                    # d40 exit
                    if row_d40.get(sym, 0) == 0:
                        ex = True
                if ex:
                    px = row_prices.get(sym, 0)
                    if pd.notna(px) and px > 0:
                        cash += positions[sym]['shares'] * px * (1 - COST_BPS/10000)
                        trades += 1
                        positions.pop(sym)

            # entries
            if len(positions) < MAX_POSITIONS:
                active = []
                if use_mr or use_mr_always:
                    active = [s for s in row_mr.index if row_mr.get(s, 0) == 1 and s not in positions]
                else:
                    active = [s for s in row_d40.index if row_d40.get(s, 0) == 1 and s not in positions]
                slots = MAX_POSITIONS - len(positions)
                for sym in active[:slots]:
                    px = row_prices.get(sym, 0)
                    if pd.isna(px) or px <= 0:
                        continue
                    size_usd = cash * MAX_POSITION_PCT * scale if cash > 0 else 0
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

    run_regime_fallback('baseline_d40', use_mr_always=False)
    run_regime_fallback('mr_time20_always', use_mr_always=True)
    run_regime_fallback('regime_fallback_d40->mr', use_mr_always=False)
