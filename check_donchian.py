import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('data')
OUT = Path('backtest_output')
screen = pd.read_csv(sorted(OUT.glob('screen_liqu_idio_*.csv'))[-1])
screen = screen[screen['tier'].isin(['large','mid','tail'])]

# Donchian signal check
stem = screen['stem'].iloc[0]
p = ROOT / f'{stem}_1d_max.csv'
df = pd.read_csv(p, parse_dates=['ts']).dropna(subset=['close','high','low','volume']).sort_values('ts').reset_index(drop=True)

high = df['high'].values
close = df['close'].values
don_high = pd.Series(high).rolling(20).max().shift(1).values
sig = pd.Series(np.where(close > don_high, 1, 0), index=df.index)

print(f'Latest 20 days for {stem}:')
print(df[['ts','close','high']].tail(20))
print(f'\nSignal today: {sig.iloc[-1]}')
print(f'High today: {high[-1]}')
print(f'Donchian high prev 20: {don_high[-1]}')
