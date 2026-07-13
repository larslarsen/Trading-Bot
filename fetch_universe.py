#!/usr/bin/env python3
"""Fetch and save pair universe from Kraken + MEXC public APIs."""
import json, csv
from pathlib import Path
import urllib.request

OUT_DIR = Path('data')
OUT_DIR.mkdir(exist_ok=True)

def fetch(url):
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())

# Kraken
print('Fetching Kraken...')
k = fetch('https://api.kraken.com/0/public/AssetPairs')
kraken_rows = []
for pid, info in k.get('result', {}).items():
    base = info.get('base', '')
    quote = info.get('quote', '')
    altname = info.get('altname', pid)
    # only USD/USDT pairs
    if quote not in ('USD', 'USDT'):
        continue
    kraken_rows.append({
        'exchange': 'kraken',
        'pair_id': pid,
        'symbol': altname,
        'base': base,
        'quote': quote,
        'raw': json.dumps(info),
    })
with open(OUT_DIR / 'kraken_pairs.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['exchange', 'pair_id', 'symbol', 'base', 'quote', 'raw'])
    w.writeheader()
    w.writerows(kraken_rows)
print(f'  Kraken: {len(kraken_rows)} USD/USDT pairs saved')

# MEXC
print('Fetching MEXC...')
m = fetch('https://api.mexc.com/api/v3/exchangeInfo')
mexc_rows = []
for s in m.get('symbols', []):
    if s.get('status') != '1':
        continue
    sym = s.get('symbol', '')
    base = s.get('baseAsset', '')
    quote = s.get('quoteAsset', '')
    if quote not in ('USDC', 'USDT'):
        continue
    mexc_rows.append({
        'exchange': 'mexc',
        'pair_id': sym,
        'symbol': sym,
        'base': base,
        'quote': quote,
        'raw': json.dumps(s),
    })
with open(OUT_DIR / 'mexc_pairs.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['exchange', 'pair_id', 'symbol', 'base', 'quote', 'raw'])
    w.writeheader()
    w.writerows(mexc_rows)
print(f'  MEXC: {len(mexc_rows)} USDC/USDT pairs saved')

# unified candidates = base assets appearing in either, with quote = USDT/USDC
kraken_bases = {r['base'] for r in kraken_rows}
mexc_bases = {r['base'] for r in mexc_rows}
all_bases = sorted(kraken_bases | mexc_bases)
print(f'  Combined unique base assets: {len(all_bases)}')
with open(OUT_DIR / 'universe_candidates.txt', 'w') as f:
    f.write('\n'.join(all_bases))
print(f'  Saved universe_candidates.txt')
