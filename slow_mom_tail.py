import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('data')
OUT = Path('backtest_output')
COST = 0.0008

LIQUID_TAIL = [
    'BLUB_USDT','RATS_USDT','HIPPO_USDT','SOLV_USDT','TOWNS_USDT',
    'ACT_USDT','ARPA_USDT','AGT_USDT','PIXEL_USDT','VEX_USDT',
    'BMT_USDT','FHE_USDT','RESOLV_USDT','BIGTIME_USDT','PROMPT_USDT',
    'COOKIE_USDT','PARTI_USDT','SHELL_USDT','SAGA_USDT','SWARMS_USDT',
    'PI_USDT','ZEUS_USDT','WCT_USDT','THE_USDT','SUNDOG_USDT',
    'A8_USDT','MITO_USDT','GTC_USDT','TREE_USDT','UPC_USDT',
    'FTT_USDT','PYR_USDT','TRADOOR_USDT','PSG_USDT','BEL_USDT',
]

def _pick(symbol):
    for suffix in ['', '_binance', '_mexc']:
        p = ROOT / f'{symbol}{suffix}_1d_max.csv'
        if p.exists():
            return p
    return None

def load_symbol(symbol):
    p = _pick(symbol)
    if p is None:
        print(f'missing {symbol}')
        return None
    df = pd.read_csv(p, parse_dates=['ts']).dropna(subset=['close','high','low','volume']).sort_values('ts').reset_index(drop=True)
    return df

def backtest_slow_mom(df, lookback, hold=None):
    df = df.copy()
    df['mom'] = df['close'] / df['close'].shift(lookback) - 1
    # long-only trend-following: enter when mom > 0, exit when mom <= 0
    df['pos'] = (df['mom'] > 0).astype(int)
    df['exit'] = (df['mom'] <= 0).astype(int)
    df = df.dropna(subset=['pos','exit','close']).reset_index(drop=True)
    if len(df) < 120:
        return None
    n = len(df)
    train_n = int(n*0.7)
    price = df['close'].values
    position = 0
    entry = 0.0
    rets = []
    for idx in range(train_n, n):
        if position == 0 and df.iloc[idx]['pos'] == 1:
            position = 1
            entry = float(price[idx])
        elif position == 1 and df.iloc[idx]['exit'] == 1:
            r = (float(price[idx])/entry - 1) - 2*COST
            rets.append(r)
            position = 0
    if position == 1 and len(price) > 1:
        r = (float(price[-1])/entry - 1) - 2*COST
        rets.append(r)
    if len(rets) < 2:
        return None
    arr = np.array(rets, dtype=float)
    sharpe = float(np.sqrt(365) * np.mean(arr) / (np.std(arr, ddof=1)+1e-9))
    total = float(np.prod(1+arr) - 1)
    wr = float(np.mean(arr > 0))
    return {
        'sharpe': sharpe,
        'return': total,
        'wr': wr,
        'trades': int(len(arr)),
        'max_dd': float(np.min(np.cumprod(1+arr) - np.maximum.accumulate(np.cumprod(1+arr)))),
        'port_rets': arr.tolist(),
    }

rows=[]
for lb in [60, 90]:
    for coin in LIQUID_TAIL:
        df = load_symbol(coin)
        if df is None:
            continue
        res = backtest_slow_mom(df, lb)
        if res is None:
            continue
        res.update({'coin': coin, 'lookback': lb, 'mode': f'mom_{lb}'})
        rows.append(res)

rdf = pd.DataFrame(rows)
if rdf.empty:
    print('No results')
    raise SystemExit

print('\n=== Slow momentum results on 35 liquid tail coins ===')
for lb in [60, 90]:
    sub = rdf[rdf['lookback']==lb].sort_values('sharpe', ascending=False)
    print(f'\n--- Lookback {lb} ---')
    print(sub[['coin','sharpe','return','wr','trades']].to_string(index=False))
    print(f"mean Sharpe={sub['sharpe'].mean():.3f} median Sharpe={sub['sharpe'].median():.3f} pct positive={np.mean(sub['sharpe']>0)*100:.1f}%")

out = OUT / f'slow_mom_tail_{pd.Timestamp.now():%Y%m%d_%H%M%S}.json'
rdf.to_json(out, orient='records', indent=2, date_format='iso')
print('\nWrote', out)
