#!/usr/bin/env python3
"""Collect ALL available 1d history for retail alt pairs."""
import json, time, warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import ccxt

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
CACHE = DATA / 'fetch_state.json'
DATA.mkdir(exist_ok=True)

EX = {
    'binance': ccxt.binance({'enableRateLimit': True}),
    'mexc': ccxt.mexc({'enableRateLimit': True}),
    'bybit': ccxt.bybit({'enableRateLimit': True}),
    'okx': ccxt.okx({'enableRateLimit': True}),
}

# Full alt universe across exchanges
PAIRS = [
    ('DOGE/USDT', 'binance'), ('ENS/USDT', 'binance'), ('LRC/USDT', 'binance'),
    ('CRV/USDT', 'binance'), ('APE/USDT', 'binance'), ('ALGO/USDT', 'binance'),
    ('ADA/USDT', 'binance'), ('ARB/USDT', 'binance'), ('MATIC/USDT', 'binance'),
    ('AVAX/USDT', 'binance'), ('ETH/USDT', 'binance'), ('SOL/USDT', 'binance'),
    ('BTC/USDT', 'binance'), ('XRP/USDT', 'binance'), ('LINK/USDT', 'binance'),
    ('DOT/USDT', 'binance'), ('UNI/USDT', 'binance'), ('ATOM/USDT', 'binance'),
    ('FTM/USDT', 'binance'), ('NEAR/USDT', 'binance'), ('APT/USDT', 'binance'),
    ('SUI/USDT', 'binance'), ('SEI/USDT', 'binance'), ('TIA/USDT', 'binance'),
    ('WIF/USDT', 'binance'), ('BONK/USDT', 'binance'), ('OP/USDT', 'binance'),
    ('BLUR/USDT', 'binance'), ('ETHFI/USDT', 'binance'),
    ('GALA/USDT', 'bybit'), ('TON/USDT', 'bybit'), ('NOT/USDT', 'bybit'),
    ('TRX/USDT', 'bybit'), ('SAND/USDT', 'okx'), ('MANA/USDT', 'okx'),
    ('EGLD/USDT', 'okx'), ('HNT/USDT', 'okx'), ('KAS/USDT', 'okx'),
    ('EDEN/USDT', 'mexc'), ('BIGTIME/USDT', 'mexc'), ('BLUB/USDT', 'mexc'),
]

def fetch_with_resume(symbol, ex_name):
    ex = EX[ex_name]
    fname = DATA / f'{symbol.replace("/","_")}_{ex_name}_1d_max.csv'
    
    since = None
    if fname.exists():
        try:
            df = pd.read_csv(fname)
            df['ts'] = pd.to_datetime(df['ts'], utc=True)
            ts_max = df['ts'].max()
            since = int(ts_max.timestamp() * 1000) + 86400000  # next day
            if ts_max >= pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1):
                return symbol, ex_name, pd.DataFrame(), f'already_current_{len(df)}bars'
        except Exception:
            since = None
    
    all_bars = []
    attempts = 0
    empty_streak = 0
    while attempts < 600:
        try:
            bars = ex.fetch_ohlcv(symbol, timeframe='1d', since=since, limit=1000)
        except ccxt.BadSymbol:
            return symbol, ex_name, pd.DataFrame(), 'bad_symbol'
        except ccxt.RateLimitExceeded:
            time.sleep(2.0)
            continue
        except Exception as e:
            msg = str(e)
            if 'IP' in msg or 'banned' in msg.lower() or '429' in msg:
                time.sleep(5.0)
                continue
            return symbol, ex_name, pd.DataFrame(), f'ERR:{msg[:60]}'
        
        if not bars:
            empty_streak += 1
            if empty_streak >= 3:
                break
            time.sleep(1.0)
            continue
        empty_streak = 0
        all_bars.extend(bars)
        since = bars[-1][0] + 1
        attempts += 1
        time.sleep(0.25)  # gentle on exchange
    
    if not all_bars and since is None:
        return symbol, ex_name, pd.DataFrame(), 'empty'
    
    new_df = pd.DataFrame(all_bars, columns=['ts','open','high','low','close','volume'])
    new_df['ts'] = pd.to_datetime(new_df['ts'], unit='ms', utc=True)
    
    if fname.exists():
        old_df = pd.read_csv(fname)
        old_df['ts'] = pd.to_datetime(old_df['ts'], utc=True)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    else:
        combined = new_df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    
    combined.to_csv(fname, index=False)
    return symbol, ex_name, combined, f'ok_{len(combined)}bars'

def main():
    t0 = time.time()
    manifest = {'timestamp': datetime.now(timezone.utc).isoformat(), 'symbols': {}}
    
    for i, (sym, ex_name) in enumerate(PAIRS):
        print(f'[{i+1}/{len(PAIRS)}] {sym}@{ex_name} ...', end='', flush=True)
        try:
            symbol, ex, df, status = fetch_with_resume(sym, ex_name)
            if df.empty:
                print(f' {status}')
                manifest['symbols'][f'{sym}@{ex_name}'] = {'status': status}
                continue
            
            first = df['ts'].min().isoformat()
            last = df['ts'].max().isoformat()
            print(f' {status} ({first} -> {last})')
            manifest['symbols'][f'{sym}@{ex_name}'] = {
                'status': status,
                'bars': len(df),
                'start': first,
                'end': last,
            }
        except Exception as e:
            print(f' ERR: {e}')
            manifest['symbols'][f'{sym}@{ex_name}'] = {'status': f'ERR:{str(e)[:60]}'}
    
    CACHE.write_text(json.dumps(manifest, indent=2))
    print(f'\nDone in {time.time()-t0:.1f}s')
    print(f'Manifest: {CACHE}')

if __name__ == '__main__':
    main()
