#!/usr/bin/env python3
"""Fetch crypto universe from CoinGecko free API with volume/category data."""
import json, csv, time
from pathlib import Path
import urllib.request

OUT = Path('data')
OUT.mkdir(exist_ok=True)

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

all_coins = []
page = 1
while True:
    url = f'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=volume_desc&per_page=250&page={page}&sparkline=false'
    try:
        data = fetch(url)
        if not data:
            break
        all_coins.extend(data)
        print(f'Page {page}: {len(data)} coins')
        page += 1
        time.sleep(1.5)
    except Exception as e:
        print(f'Stopped at page {page}: {e}')
        break

print(f'Total fetched: {len(all_coins)}')

# Save full JSON
with open(OUT / 'coingecko_universe_raw.json', 'w') as f:
    json.dump(all_coins, f, indent=2)

# Filter: volume > 100k USD, exclude stablecoins by name heuristic
stable_keywords = ['usdt','usdc','busd','tusd','dai','frax','usdd','usde','pyusd','rlusd','fdusd','euroc','eurc']
filtered = []
for c in all_coins:
    vol = c.get('total_volume') or 0
    if vol < 100000:
        continue
    sym = (c.get('symbol') or '').lower()
    name = (c.get('name') or '').lower()
    if any(k in sym or k in name for k in stable_keywords):
        continue
    filtered.append({
        'id': c.get('id'),
        'symbol': c.get('symbol','').upper(),
        'name': c.get('name',''),
        'market_cap_rank': c.get('market_cap_rank'),
        'market_cap_usd': c.get('market_cap'),
        'volume_24h_usd': c.get('total_volume'),
        'current_price_usd': c.get('current_price'),
        'price_change_24h': c.get('price_change_percentage_24h'),
        'categories': ','.join(c.get('categories',[]) or []),
    })

print(f'After volume/category filter: {len(filtered)}')

with open(OUT / 'coingecko_universe_filtered.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id','symbol','name','market_cap_rank','market_cap_usd','volume_24h_usd','current_price_usd','price_change_24h','categories'])
    writer.writeheader()
    writer.writerows(filtered)

print('Saved data/coingecko_universe_filtered.csv')
