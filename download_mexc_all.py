#!/usr/bin/env python3
"""
Bulk 1d OHLCV downloader from MEXC public API.
No auth required. Rate-limited; uses polite sleep.
"""
import time
import json
from pathlib import Path
from datetime import datetime, timezone
import urllib.request
import pandas as pd

ROOT = Path('data')
BATCH_SLEEP = 0.3
MAX_BARS = 1500

def get_mexc_symbols():
    url = 'https://api.mexc.com/api/v3/exchangeInfo'
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    symbols = [s['symbol'] for s in data.get('symbols', []) if s.get('status') == '1']
    # keep USDT/USDC pairs only
    return [s for s in symbols if s.endswith('USDT') or s.endswith('USDC')]

def fetch_klines(symbol, interval='1d', limit=1000):
    url = f'https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = json.load(r)
    if not raw:
        return None
    rows = []
    for k in raw:
        # [openTime, open, high, low, close, volume, closeTime, ...]
        ts = pd.to_datetime(int(k[0]), unit='ms', utc=True).tz_localize(None)
        rows.append({
            'ts': ts,
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
            'volume': float(k[5]),
        })
    df = pd.DataFrame(rows).sort_values('ts').reset_index(drop=True)
    return df

def main():
    ROOT.mkdir(exist_ok=True)
    symbols = get_mexc_symbols()
    print(f'MEXC spot pairs: {len(symbols)}')

    # Skip symbols we already have healthy files for
    have = set()
    for p in ROOT.glob('*_1d_max.csv'):
        sym = p.name.replace('_1d_max.csv', '')
        try:
            df = pd.read_csv(p, parse_dates=['ts'])
            if len(df) >= 300:
                have.add(sym)
        except Exception:
            pass
    targets = [s for s in symbols if s not in have]
    print(f'Targets to download: {len(targets)} (have {len(have)})')

    failed = []
    for i, sym in enumerate(targets):
        try:
            df = fetch_klines(sym, interval='1d', limit=MAX_BARS)
            if df is None or len(df) < 60:
                print(f'  skip {sym}: not enough bars')
                failed.append(sym)
                continue
            out = ROOT / f'{sym}_1d_max.csv'
            df.to_csv(out, index=False)
            print(f'  {sym:20s} bars={len(df)} last={str(df["ts"].iloc[-1].date())}')
        except Exception as e:
            failed.append(sym)
            print(f'  FAIL {sym}: {e}')
        time.sleep(BATCH_SLEEP)

    print(f'\nDone. Downloaded {len(targets)-len(failed)}, failed {len(failed)}')
    print('Failed:', ' '.join(sorted(failed)[:20]))

if __name__ == '__main__':
    main()
