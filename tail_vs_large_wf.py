import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('data')
COINS = {
    'tail': ['PYR_USDT_mexc', 'PARTI_USDT_mexc', 'TREE_USDT_mexc', 'EVAA_USDT_mexc', 'BEL_USDT_mexc'],
    'large': ['DOGE_USDT', 'ENS_USDT', 'LRC_USDT', 'CRV_USDT', 'EDEN_USDT'],
}
COST = 0.0008

def load_coin(fname):
    p = ROOT / f'{fname}_1d_max.csv'
    if not p.exists():
        print(f'missing {p}')
        return None
    df = pd.read_csv(p, parse_dates=['ts'])
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
    df = df.dropna(subset=['close','high','low','volume']).sort_values('ts').reset_index(drop=True)
    return df

def triple_rsi(df):
    delta = df['close'].diff()
    rs = delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0).rolling(14).mean() + 1e-9)
    rsi = 100 - 100/(1+rs)
    pos = (rsi < 30).astype(int)
    exit_ = (rsi > 70).astype(int)
    return pos, exit_

def donchian(df):
    h = df['high'].rolling(20).max()
    l = df['low'].rolling(20).min()
    pos = (df['close'] > h.shift(1)).astype(int)
    exit_ = (df['close'] < l.shift(1)).astype(int)
    return pos, exit_

def trend_ma(df):
    sma20 = df['close'].rolling(20).mean()
    sma50 = df['close'].rolling(50).mean()
    pos = (sma20 > sma50).astype(int)
    exit_ = (sma20 < sma50).astype(int)
    return pos, exit_

def macd_cross(df):
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    pos = (macd > sig).astype(int)
    exit_ = (macd < sig).astype(int)
    return pos, exit_

RULES = {'triple_rsi': triple_rsi, 'donchian': donchian, 'trend_ma': trend_ma, 'macd': macd_cross}

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
    # Build aligned daily return series for test set only
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
    if len(rets) < 2:
        return None
    arr = np.array(rets, dtype=float)
    sharpe = np.sqrt(365) * np.mean(arr) / (np.std(arr, ddof=1)+1e-9)
    total = float(np.prod(1+arr) - 1)
    wr = float(np.mean(arr > 0))
    return {
        'sharpe': float(sharpe),
        'return': float(total),
        'wr': float(wr),
        'trades': int(len(arr)),
        'max_dd': float(np.min(np.cumprod(1+arr) - np.maximum.accumulate(np.cumprod(1+arr)))),
        'port_rets': [float(x) for x in arr],
    }

def main():
    rows = []
    for tier, coins in COINS.items():
        for coin in coins:
            df = load_coin(coin)
            if df is None:
                continue
            for rule_name, rule_fn in RULES.items():
                res = backtest(df, rule_fn)
                if res is None:
                    continue
                res.update({'tier': tier, 'coin': coin, 'rule': rule_name})
                rows.append(res)
    if not rows:
        print('No results')
        return
    rdf = pd.DataFrame(rows)
    print('\n=== All Results ===')
    print(rdf[['tier','coin','rule','sharpe','return','wr','trades']].sort_values(['tier','rule','sharpe'], ascending=[True,True,False]).to_string(index=False))
    print('\n=== Summary by Tier ===')
    print(rdf.groupby('tier')[['sharpe','return','wr']].mean().to_string())

    out = Path('backtest_output') / f'tail_vs_large_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
    out.parent.mkdir(exist_ok=True)
    rdf.to_json(out, orient='records', indent=2, date_format='iso')
    print(f'\nWrote {out}')

if __name__ == '__main__':
    main()
