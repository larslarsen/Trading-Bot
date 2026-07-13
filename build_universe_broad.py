#!/usr/bin/env python3
"""
Build broad crypto universe from MEXC + Kraken + CoinGecko.
Exchange-native approach: use all listed pairs, enrich with mcap tiers.
"""
import time
import json
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('data')
ROOT.mkdir(exist_ok=True)

# 1) Load MEXC pairs (already have exchangeInfo)
print('Loading MEXC pairs...')
mexc_url = 'https://api.mexc.com/api/v3/exchangeInfo'
req = urllib.request.Request(mexc_url, headers={'User-Agent': 'curl/7.88.1'})
with urllib.request.urlopen(req, timeout=30) as r:
    mdata = json.load(r)
mexc_symbols = [s['symbol'] for s in mdata.get('symbols', []) if s.get('status') == '1']
mexc_bases = set(s.replace('USDT','').replace('USDC','').replace('BTC','').replace('ETH','') 
                  for s in mexc_symbols 
                  if s.endswith('USDT') or s.endswith('USDC'))
print(f'MEXC: {len(mexc_symbols)} pairs, {len(mexc_bases)} unique bases')

# 2) Load Kraken pairs
print('Loading Kraken pairs...')
kraken_url = 'https://api.kraken.com/0/public/AssetPairs'
req = urllib.request.Request(kraken_url, headers={'User-Agent': 'curl/7.88.1'})
with urllib.request.urlopen(req, timeout=30) as r:
    kdata = json.load(r)
kraken_pairs = list(kdata.get('result', {}).keys())
kraken_usdt = [p for p in kraken_pairs if '.t' in p or p.endswith('USDT') or p.endswith('USD')]
kraken_bases = set()
for p in kraken_pairs:
    if 'X' in p and 'Z' in p:
        base = p.split('.')[0].replace('X','').replace('Z','')
        kraken_bases.add(base)
print(f'Kraken: {len(kraken_pairs)} pairs, {len(kraken_bases)} unique bases')

# 3) Load CoinGecko top 2000 by mcap for enrichment
print('Loading CoinGecko market data...')
cg_url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page=1'
cg_data = []
for page in range(1, 9):  # up to 2000 coins
    url = f'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page={page}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            coins = json.load(r)
            if not coins:
                break
            cg_data.extend(coins)
            print(f'  page {page}: got {len(coins)} coins')
        time.sleep(2.5)  # increased backoff
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f'  page {page}: 429 rate limit, sleeping 30s')
            time.sleep(30)
            continue
        print(f'  page {page} failed: {e}')
        break
    except Exception as e:
        print(f'  page {page} failed: {e}')
        break

if cg_data:
    cg_df = pd.DataFrame(cg_data)
    cg_df = cg_df[['id', 'symbol', 'name', 'market_cap', 'market_cap_rank', 'total_volume']].copy()
    cg_df['symbol'] = cg_df['symbol'].str.upper()
    cg_df = cg_df.drop_duplicates(subset=['symbol'], keep='first')
    print(f'CoinGecko: {len(cg_df)} coins fetched, deduped')
else:
    print('WARNING: CoinGecko fetch failed')
    cg_df = pd.DataFrame(columns=['id', 'symbol', 'name', 'market_cap', 'market_cap_rank', 'total_volume'])

# 4) Build unified symbol list
all_bases = mexc_bases | kraken_bases
print(f'Total unique bases: {len(all_bases)}')

# 5) Match with CoinGecko
cg_map = cg_df.set_index('symbol').to_dict('index') if len(cg_df) > 0 else {}

universe_rows = []
for base in sorted(all_bases):
    # Find MEXC pairs
    mexc_has = any((base + q) in mexc_symbols for q in ['USDT', 'USDC', 'BTC', 'ETH'])
    # Find Kraken (approximate)
    has_kraken = False
    for kp in kraken_pairs:
        if base in kp:
            has_kraken = True
            break
    
    # CoinGecko enrichment
    cg = cg_map.get(base, {})
    mcap_rank = cg.get('market_cap_rank', '')
    mcap = cg.get('market_cap', '')
    volume = cg.get('total_volume', '')
    
    # Tier assignment based on mcap_rank
    if mcap_rank != '' and not pd.isna(mcap_rank):
        rank = int(mcap_rank) if mcap_rank else 99999
        if rank <= 500:
            tier = 'large'
        elif rank <= 1500:
            tier = 'mid'
        else:
            tier = 'tail'
    else:
        tier = 'unknown'
    
    universe_rows.append({
        'symbol': base,
        'mexc': mexc_has,
        'kraken': has_kraken,
        'tier': tier,
        'mcap_rank': mcap_rank,
        'mcap_usd': mcap,
        'volume_24h_usd': volume,
        'coingecko_id': cg.get('id', ''),
        'name': cg.get('name', '')
    })

u = pd.DataFrame(universe_rows)
print(f'\nUniverse: {len(u)} coins')
print('Tier counts:', u['tier'].value_counts().to_dict())
print('Exchange coverage:', f"MEXC:{u['mexc'].sum()}, Kraken:{u['kraken'].sum()}, Both:{(u['mexc'] & u['kraken']).sum()}")

# Save
u.to_csv(ROOT / 'universe_broad.csv', index=False)
print('\nSaved data/universe_broad.csv')
print(f'\nTop 20 by mcap:')
top20 = u.copy()
top20['mcap_rank_num'] = pd.to_numeric(top20['mcap_rank'], errors='coerce')
top20 = top20.nsmallest(20, 'mcap_rank_num')[['symbol', 'tier', 'mcap_rank', 'mexc', 'kraken']]
print(top20.to_string(index=False))
