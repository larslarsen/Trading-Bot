from pathlib import Path

triads = {
    'exp_distance_ma.py': """\
\"\"\"Literature experiment: Distance from Moving Average breakout.\"\"\"
import sys
print('START exp_distance_ma', flush=True)
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

print('Loading...', flush=True)
coin_data = load_coins()
all_dates = sorted(set(d for c in coin_data.values() for d in c.loc[(c['ts'] >= '2025-01-01') & (c['ts'] <= '2026-07-12'), 'ts']))
print(f'Coins: {len(coin_data)}, Dates: {len(all_dates)}', flush=True)

price_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
sig_entry = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_exit = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)

for stem, df in coin_data.items():
    close = df['close'].values
    ma = pd.Series(close).rolling(20).mean()
    dist = (close / ma.values - 1) * 100
    entry = ((dist > 0) & (pd.Series(dist).diff() > 0)).astype(int)
    exit_sig = (dist < -5).astype(int)
    mask = (df['ts'] >= '2025-01-01') & (df['ts'] <= '2026-07-12')
    for i in range(len(df)):
        if mask.iloc[i]:
            ts = df['ts'].iloc[i]
            price_df.loc[ts, stem] = close[i]
            sig_entry.loc[ts, stem] = int(entry.iloc[i]) if pd.notna(entry.iloc[i]) else 0
            sig_exit.loc[ts, stem] = int(exit_sig.iloc[i]) if pd.notna(exit_sig.iloc[i]) else 0

price_df = price_df.sort_index()
sig_entry = sig_entry.sort_index()
sig_exit = sig_exit.sort_index()
full_dates = price_df.index.tolist()
recent_dates = full_dates[-90:]

print(chr(10) + '=== d40 baseline (full) ===', flush=True)
sig_d40 = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
for stem, df in coin_data.items():
    s = donchian_signal(df['high'], df['low'], df['close'], 40)
    mask = (df['ts'] >= '2025-01-01') & (df['ts'] <= '2026-07-12')
    for i in range(len(df)):
        if mask.iloc[i]:
            ts = df['ts'].iloc[i]
            sig_d40.loc[ts, stem] = int(s.iloc[i]) if pd.notna(s.iloc[i]) else 0
sig_d40 = sig_d40.sort_index()
res = simulate_portfolio(price_df, sig_d40, initial=INITIAL, max_positions=MAX_POSITIONS)
print(f"baseline_d40: ret={res['return_pct']:.1f}% sharpe={res['sharpe']:.2f} dd={res['max_dd_pct']:.1f}% trades={res['trades']}", flush=True)

print(chr(10) + '=== distance_ma (full) ===', flush=True)
res_full = simulate_portfolio(price_df, sig_entry, initial=INITIAL, max_positions=MAX_POSITIONS, exit_signal_df=sig_exit)
print(f"distance_ma: ret={res_full['return_pct']:.1f}% sharpe={res_full['sharpe']:.2f} dd={res_full['max_dd_pct']:.1f}% trades={res_full['trades']}", flush=True)

print(chr(10) + '=== distance_ma (90d) ===', flush=True)
res_90 = simulate_portfolio(price_df.loc[recent_dates], sig_entry.loc[recent_dates], initial=INITIAL, exit_signal_df=sig_exit.loc[recent_dates])
print(f"distance_ma_90d: ret={res_90['return_pct']:.1f}% sharpe={res_90['sharpe']:.2f} dd={res_90['max_dd_pct']:.1f}% trades={res_90['trades']}", flush=True)
""",
    'exp_price_momentum.py': """\
\"\"\"Literature experiment: price momentum threshold.\"\"\"
import sys
print('START exp_price_momentum', flush=True)
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

print('Loading...', flush=True)
coin_data = load_coins()
all_dates = sorted(set(d for c in coin_data.values() for d in c.loc[(c['ts'] >= '2025-01-01') & (c['ts'] <= '2026-07-12'), 'ts']))
print(f'Coins: {len(coin_data)}, Dates: {len(all_dates)}', flush=True)

price_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
sig_entry = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_exit = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)

for stem, df in coin_data.items():
    close = df['close'].values
    ret5 = pd.Series(close).pct_change(5) * 100
    entry = (ret5 > 5).astype(int)
    exit_sig = (ret5 < -5).astype(int)
    mask = (df['ts'] >= '2025-01-01') & (df['ts'] <= '2026-07-12')
    for i in range(len(df)):
        if mask.iloc[i]:
            ts = df['ts'].iloc[i]
            price_df.loc[ts, stem] = close[i]
            sig_entry.loc[ts, stem] = int(entry.iloc[i]) if pd.notna(entry.iloc[i]) else 0
            sig_exit.loc[ts, stem] = int(exit_sig.iloc[i]) if pd.notna(exit_sig.iloc[i]) else 0

price_df = price_df.sort_index()
sig_entry = sig_entry.sort_index()
sig_exit = sig_exit.sort_index()
full_dates = price_df.index.tolist()
recent_dates = full_dates[-90:]

print(chr(10) + '=== d40 baseline (full) ===', flush=True)
sig_d40 = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
for stem, df in coin_data.items():
    s = donchian_signal(df['high'], df['low'], df['close'], 40)
    mask = (df['ts'] >= '2025-01-01') & (df['ts'] <= '2026-07-12')
    for i in range(len(df)):
        if mask.iloc[i]:
            ts = df['ts'].iloc[i]
            sig_d40.loc[ts, stem] = int(s.iloc[i]) if pd.notna(s.iloc[i]) else 0
sig_d40 = sig_d40.sort_index()
res = simulate_portfolio(price_df, sig_d40, initial=INITIAL, max_positions=MAX_POSITIONS)
print(f"baseline_d40: ret={res['return_pct']:.1f}% sharpe={res['sharpe']:.2f} dd={res['max_dd_pct']:.1f}% trades={res['trades']}", flush=True)

print(chr(10) + '=== price_momentum (full) ===', flush=True)
res_full = simulate_portfolio(price_df, sig_entry, initial=INITIAL, max_positions=MAX_POSITIONS, exit_signal_df=sig_exit)
print(f"price_momentum: ret={res_full['return_pct']:.1f}% sharpe={res_full['sharpe']:.2f} dd={res_full['max_dd_pct']:.1f}% trades={res_full['trades']}", flush=True)

print(chr(10) + '=== price_momentum (90d) ===', flush=True)
res_90 = simulate_portfolio(price_df.loc[recent_dates], sig_entry.loc[recent_dates], initial=INITIAL, exit_signal_df=sig_exit.loc[recent_dates])
print(f"price_momentum_90d: ret={res_90['return_pct']:.1f}% sharpe={res_90['sharpe']:.2f} dd={res_90['max_dd_pct']:.1f}% trades={res_90['trades']}", flush=True)
""",
    'exp_donchian_vol_filter.py': """\
\"\"\"Literature experiment: Donchian 40 with realized-volatility filter.\"\"\"
import sys
print('START exp_donchian_vol_filter', flush=True)
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

print('Loading...', flush=True)
coin_data = load_coins()
all_dates = sorted(set(d for c in coin_data.values() for d in c.loc[(c['ts'] >= '2025-01-01') & (c['ts'] <= '2026-07-12'), 'ts']))
print(f'Coins: {len(coin_data)}, Dates: {len(all_dates)}', flush=True)

price_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
sig_d40 = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_vol = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)

for stem, df in coin_data.items():
    close = df['close'].values
    sig_d40[stem] = donchian_signal(df['high'], df['low'], df['close'], 40).values
    sig_vol[stem] = pd.Series(close).pct_change().rolling(20).std().values * np.sqrt(365)
    mask = (df['ts'] >= '2025-01-01') & (df['ts'] <= '2026-07-12')
    for i in range(len(df)):
        if mask.iloc[i]:
            ts = df['ts'].iloc[i]
            price_df.loc[ts, stem] = close[i]

price_df = price_df.sort_index()
sig_d40 = sig_d40.sort_index()
sig_vol = sig_vol.sort_index()
full_dates = price_df.index.tolist()
recent_dates = full_dates[-90:]

print(chr(10) + '=== d40 baseline (full) ===', flush=True)
res = simulate_portfolio(price_df, sig_d40, initial=INITIAL, max_positions=MAX_POSITIONS)
print(f"baseline_d40: ret={res['return_pct']:.1f}% sharpe={res['sharpe']:.2f} dd={res['max_dd_pct']:.1f}% trades={res['trades']}", flush=True)

print(chr(10) + '=== d40_vol_filter (full) ===', flush=True)
sig_entry = sig_d40.copy(deep=True)
rolling_mean = sig_vol.rolling(len(sig_vol), min_periods=1).mean()
sig_entry = sig_entry.where(sig_vol < rolling_mean, 0).astype(int)
res_full = simulate_portfolio(price_df, sig_entry, initial=INITIAL, max_positions=MAX_POSITIONS)
print(f"d40_vol_filter: ret={res_full['return_pct']:.1f}% sharpe={res_full['sharpe']:.2f} dd={res_full['max_dd_pct']:.1f}% trades={res_full['trades']}", flush=True)

print(chr(10) + '=== d40_vol_filter (90d) ===', flush=True)
sig_entry90 = sig_d40.loc[recent_dates].copy(deep=True)
sig_entry90 = sig_entry90.where(sig_vol.loc[recent_dates] < rolling_mean.loc[recent_dates], 0).astype(int)
res_90 = simulate_portfolio(price_df.loc[recent_dates], sig_entry90, initial=INITIAL, max_positions=MAX_POSITIONS)
print(f"d40_vol_filter_90d: ret={res_90['return_pct']:.1f}% sharpe={res_90['sharpe']:.2f} dd={res_90['max_dd_pct']:.1f}% trades={res_90['trades']}", flush=True)
""",
}

for filename, body in triads.items():
    Path('/home/lars/trading-bot', filename).write_text(body)
    print(f'written {filename} {len(body)}')
