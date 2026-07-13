#!/usr/bin/env python3
"""Select 50 tail coins from enriched universe and download 1d OHLCV."""
import csv, json, time
from pathlib import Path
import urllib.request

import pandas as pd
import ccxt

ROOT = Path('data')
ROOT.mkdir(exist_ok=True)

# 1. Load enriched universe
coins = []
with open(ROOT / 'universe_enriched.csv') as f:
    for row in csv.DictReader(f):
        coins.append(row)

# 2. Select top 50 tail coins by volume
tail = [c for c in coins if c['tier'] == 'tail' and c.get('volume_24h_usd') and float(c['volume_24h_usd']) > 0]
tail.sort(key=lambda x: float(x['volume_24h_usd']), reverse=True)
selected = tail[:50]

print(f'Selected {len(selected)} tail coins:')
for c in selected:
    print(f"  {c['symbol']:10s} {c['name']:30s} {c['exchange']:12s} ${float(c['volume_24h_usd']):>12,.0f}")

# Save selection manifest
with open(ROOT / 'tail_download_list.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=coins[0].keys())
    writer.writeheader()
    writer.writerows(selected)

# 3. Downloader helpers
def download_ccxt_1d(symbol, exchange_id='mexc', limit=730):
    ex = getattr(ccxt, exchange_id)()
    bars = ex.fetch_ohlcv(symbol, timeframe='1d', limit=limit)
    df = pd.DataFrame(bars, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df

# 4. Download
downloaded = 0
for coin in selected:
    sym = coin['symbol']
    saved = False
    for quote in ['USDT', 'USDC']:
        df = None
        try:
            df = download_ccxt_1d(f'{sym}/{quote}', 'mexc')
        except Exception as e:
            pass
        if df is not None and len(df) >= 60:
            fname = ROOT / f'{sym}_{quote}_mexc_1d_max.csv'
            df.to_csv(fname, index=False)
            print(f"  {sym} {quote}: {len(df)} bars -> {fname.name}")
            downloaded += 1
            saved = True
            break
    if not saved:
        print(f"  {sym}: FAILED")

print(f'\nDownloaded: {downloaded}/{len(selected)}')
