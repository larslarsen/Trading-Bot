import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path('data')
OUT = Path('backtest_output')
COST = 0.0008

LARGE = ['DOGE_USDT', 'ENS_USDT', 'LRC_USDT', 'CRV_USDT', 'EDEN_USDT']
TAIL_FILES = sorted(ROOT.glob('*_mexc_1d_max.csv'))
TAIL = [p.stem.replace('_mexc_1d_max', '') for p in TAIL_FILES]
ALL_COINS = {'large': LARGE, 'tail': TAIL}
def load_coin(fname):
    p = ROOT / f'{fname}_1d_max.csv'
    if not p.exists():
        # try mexc-prefixed variant
        hits = list(ROOT.glob(f'{fname}*_1d_max.csv'))
        if hits:
            p = hits[0]
    if not p.exists():
        print(f'missing {fname}')
        return None
    df = pd.read_csv(p, parse_dates=['ts'])
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
    df = df.dropna(subset=['close','high','low','volume']).sort_values('ts').reset_index(drop=True)
    return df

def triple_rsi_series(df):
    delta = df['close'].diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean() + 1e-9)
    rsi = 100 - 100/(1+rs)
    return rsi

def walk_forward_trade_rets(df, pos_fn, hold=14):
    pos, exit_ = pos_fn(df)
    df = df.copy()
    df['pos'] = pos
    df['exit'] = exit_
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
    # forced exit at last bar if still holding
    if position == 1:
        r = (float(price[-1])/entry - 1) - 2*COST
        rets.append(r)
    if len(rets) < 2:
        return None
    arr = np.array(rets, dtype=float)
    sharpe = float(np.sqrt(365) * np.mean(arr) / (np.std(arr, ddof=1)+1e-9))
    total = float(np.prod(1+arr) - 1)
    wr = float(np.mean(arr > 0))
    return {'sharpe': sharpe, 'return': total, 'wr': wr, 'trades': int(len(arr)), 'port_rets': arr.tolist()}

def triple_rsi(df):
    rsi = triple_rsi_series(df)
    pos = (rsi < 30).astype(int)
    exit_ = (rsi > 70).astype(int)
    return pos, exit_

rows=[]
for tier, coins in ALL_COINS.items():
    for coin in coins:
        df = load_coin(coin)
        if df is None:
            continue
        res = walk_forward_trade_rets(df, triple_rsi)
        if res is None:
            continue
        res.update({'tier': tier, 'coin': coin, 'rule': 'triple_rsi'})
        rows.append(res)

rdf = pd.DataFrame(rows)
if rdf.empty:
    print('No results')
    raise SystemExit

print('\n=== Tail vs Large triple_rsi (all 45 tail coins) ===')
print(rdf[['tier','coin','sharpe','return','wr','trades']].sort_values(['tier','sharpe'], ascending=[True,False]).to_string(index=False))

print('\n=== Distribution by tier ===')
for tier in ['tail','large']:
    sub = rdf[rdf['tier']==tier]
    print(f"{tier}: n={len(sub)} median_sharpe={sub['sharpe'].median():.3f} mean_sharpe={sub['sharpe'].mean():.3f} pct_positive={np.mean(sub['sharpe']>0)*100:.1f}% median_wr={sub['wr'].median()*100:.1f}%")

out = OUT / f'triple_rsi_tail_vs_large_full_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
rdf.to_json(out, orient='records', indent=2, date_format='iso')
print('\nWrote', out)
