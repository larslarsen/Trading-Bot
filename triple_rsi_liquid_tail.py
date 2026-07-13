import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path('data')
OUT = Path('backtest_output')
COST = 0.0008

LARGE_RAW = ['DOGE_USDT', 'ENS_USDT', 'LRC_USDT', 'CRV_USDT', 'EDEN_USDT']

def _pick(symbol):
    cands = [
        ROOT / f'{symbol}_1d_max.csv',
        ROOT / f'{symbol}_binance_1d_max.csv',
        ROOT / f'{symbol}_mexc_1d_max.csv',
    ]
    for c in cands:
        if c.exists():
            return c
    return None

LIQUID_TAIL = [
    'BLUB_USDT','RATS_USDT','HIPPO_USDT','SOLV_USDT','TOWNS_USDT',
    'ACT_USDT','ARPA_USDT','AGT_USDT','PIXEL_USDT','VEX_USDT',
    'BMT_USDT','FHE_USDT','RESOLV_USDT','BIGTIME_USDT','PROMPT_USDT',
    'COOKIE_USDT','PARTI_USDT','SHELL_USDT','SAGA_USDT','SWARMS_USDT',
    'PI_USDT','ZEUS_USDT','WCT_USDT','THE_USDT','SUNDOG_USDT',
    'A8_USDT','MITO_USDT','GTC_USDT','TREE_USDT','UPC_USDT',
    'FTT_USDT','PYR_USDT','TRADOOR_USDT','PSG_USDT','BEL_USDT',
]

def load_symbol(symbol):
    p = _pick(symbol)
    if p is None:
        print(f'missing {symbol}')
        return None
    df = pd.read_csv(p, parse_dates=['ts']).dropna(subset=['close','high','low','volume']).sort_values('ts').reset_index(drop=True)
    return df

def triple_rsi(df):
    delta = df['close'].diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean() + 1e-9)
    rsi = 100 - 100/(1+rs)
    pos = (rsi < 30).astype(int)
    exit_ = (rsi > 70).astype(int)
    return pos, exit_

def backtest(df, pos_fn):
    pos, exit_ = pos_fn(df)
    df = df.copy()
    df['pos'] = pos
    df['exit'] = exit_
    df = df.dropna(subset=['pos','exit','close']).reset_index(drop=True)
    if len(df) < 120:
        return None
    n = len(df)
    train_n = int(n*0.7)
    test = df.iloc[train_n:].copy().reset_index(drop=True)
    if len(test) < 30:
        return None
    price = test['close'].values
    position = 0
    entry = 0.0
    trade_rets = []
    daily_rets = [0.0] * len(test)
    for idx in range(len(test)):
        if idx > 0 and position == 1:
            daily_rets[idx] = float(price[idx]) / float(test.iloc[idx-1]['close']) - 1.0
        if position == 0 and test.iloc[idx]['pos'] == 1:
            position = 1
            entry = float(price[idx])
        elif position == 1 and test.iloc[idx]['exit'] == 1:
            r = (float(price[idx])/entry - 1) - 2*COST
            trade_rets.append(r)
            position = 0
            daily_rets[idx] = r
    if position == 1 and len(price) > 1:
        r = (float(price[-1])/entry - 1) - 2*COST
        trade_rets.append(r)
        daily_rets[-1] = r
    if len(trade_rets) < 2:
        return None
    arr = np.array(trade_rets, dtype=float)
    daily_arr = np.array(daily_rets, dtype=float)
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
        'daily_rets': daily_arr.tolist(),
        'daily_len': int(len(daily_arr)),
    }

rows=[]
for tier, symbols in [('large', LARGE_RAW), ('tail', LIQUID_TAIL)]:
    for symbol in symbols:
        df = load_symbol(symbol)
        if df is None:
            continue
        res = backtest(df, triple_rsi)
        if res is None:
            continue
        res.update({'tier': tier, 'coin': symbol, 'rule': 'triple_rsi'})
        rows.append(res)

rdf = pd.DataFrame(rows)
if rdf.empty:
    print('No results')
    raise SystemExit

print('\n=== Triple RSI on liquid tail vs large ===')
print(rdf[['tier','coin','sharpe','return','wr','trades']].sort_values(['tier','sharpe'], ascending=[True,False]).to_string(index=False))

print('\n=== Distribution by tier ===')
for tier in ['tail','large']:
    sub = rdf[rdf['tier']==tier]
    if len(sub)==0:
        continue
    print(f"{tier}: n={len(sub)} median_sharpe={sub['sharpe'].median():.3f} mean_sharpe={sub['sharpe'].mean():.3f} pct_positive={np.mean(sub['sharpe']>0)*100:.1f}% median_wr={sub['wr'].median()*100:.1f}%")

out = OUT / f'triple_rsi_liquid_tail_vs_large_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
rdf.to_json(out, orient='records', indent=2, date_format='iso')
print('\nWrote', out)
