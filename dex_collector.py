#!/usr/bin/env python3
"""
DexScreener token-discovery collector.

Goal: find genuinely new / low-cap tokens with recent volume spikes.
Free endpoints used:
  - /token-profiles/latest
  - /search/?q=
  - /token-pairs/v1/{chain}/{token_address}
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(__file__).parent / 'dex_data'
OUT_DIR.mkdir(exist_ok=True)

BASE = 'https://api.dexscreener.com/latest/dex'
CHAINS = ['solana', 'ethereum', 'base', 'bsc', 'arbitrum', 'avalanche', 'fantom']

# Focus on tokens that look like retail momentum candidates:
# - very recent on DEX
# - some liquidity
# - enough activity to be tradeable
MIN_LIQUIDITY_USD = 200
MIN_VOLUME_USD = 500
MAX_PAIR_AGE_HOURS = 168  # 7 days


def fetch_token_profiles():
    try:
        r = requests.get(f'{BASE}/token-profiles/latest', timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f'token-profiles error: {e}')
        return []


def search_pairs(chain):
    try:
        r = requests.get(f'{BASE}/search', params={'q': chain}, timeout=15)
        r.raise_for_status()
        return r.json().get('pairs', [])
    except Exception as e:
        print(f'  search {chain}: ERROR {e}')
        return []


def fetch_token_pairs(chain, address):
    try:
        r = requests.get(f'{BASE}/token-pairs/v1/{chain}/{address}', timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def normalize_ts(v):
    if not v:
        return None
    try:
        return pd.to_datetime(v, utc=True)
    except Exception:
        try:
            return pd.to_datetime(int(v), unit='ms', utc=True)
        except Exception:
            return None


def collect():
    now = datetime.now(timezone.utc)
    candidates = []
    seen = set()

    # 1) pull possible new pairs from additional searches
    queries = [
        'SOL', 'ETH', 'BASE', 'BSC', 'ARB', 'AVAX', 'FTM', 'MATIC', 'SUI', 'INJ',
        'DOGE', 'XRP', 'LINK', 'UNI', 'AAVE', 'MKR', 'PEPE', 'WIF', 'BONK', 'FLOKI',
    ]
    print(f'Queries: {len(queries)}')
    for q in queries:
        for chain in CHAINS:
            pairs = search_pairs(f'{q} {chain}')
            for pair in pairs:
                addr = pair.get('baseToken', {}).get('address') or pair.get('address')
                if not addr:
                    continue
                key = (chain.lower(), str(addr).lower())
                if key in seen:
                    continue
                seen.add(key)
                created = normalize_ts(pair.get('pairCreatedAt') or pair.get('createdAt'))
                if not created:
                    continue
                age_hours = (now - created).total_seconds() / 3600
                vol = float(pair.get('volume', {}).get('h24', 0) or 0)
                liq = float(pair.get('liquidity', {}).get('usd', 0) or 0)
                if age_hours > MAX_PAIR_AGE_HOURS:
                    continue
                if vol < MIN_VOLUME_USD and liq < MIN_LIQUIDITY_USD:
                    continue
                candidates.append({
                    'chain': pair.get('chainId'),
                    'dex': pair.get('dexId'),
                    'pair': pair.get('pairAddress'),
                    'base_symbol': pair.get('baseToken', {}).get('symbol'),
                    'quote_symbol': pair.get('quoteToken', {}).get('symbol'),
                    'address': addr,
                    'price_usd': pair.get('priceUsd'),
                    'vol_24h': vol,
                    'liquidity_usd': liq,
                    'age_hours': age_hours,
                    'txns_24h': pair.get('txns', {}).get('h24', {}).get('buys', 0) + pair.get('txns', {}).get('h24', {}).get('sells', 0),
                    'price_chg_24h': pair.get('priceChange', {}).get('h24'),
                    'created_at': created.isoformat() if created else None,
                })

    # 2) supplement with direct chain searches
    for chain in CHAINS:
        pairs = search_pairs(chain)
        for pair in pairs:
            addr = pair.get('baseToken', {}).get('address') or pair.get('address')
            if not addr:
                continue
            key = (chain.lower(), str(addr).lower())
            if key in seen:
                continue
            seen.add(key)
            created = normalize_ts(pair.get('pairCreatedAt') or pair.get('createdAt'))
            if not created:
                continue
            age_hours = (now - created).total_seconds() / 3600
            vol = float(pair.get('volume', {}).get('h24', 0) or 0)
            liq = float(pair.get('liquidity', {}).get('usd', 0) or 0)
            if age_hours > MAX_PAIR_AGE_HOURS:
                continue
            if vol < MIN_VOLUME_USD and liq < MIN_LIQUIDITY_USD:
                continue
            candidates.append({
                'chain': pair.get('chainId'),
                'dex': pair.get('dexId'),
                'pair': pair.get('pairAddress'),
                'base_symbol': pair.get('baseToken', {}).get('symbol'),
                'quote_symbol': pair.get('quoteToken', {}).get('symbol'),
                'address': addr,
                'price_usd': pair.get('priceUsd'),
                'vol_24h': vol,
                'liquidity_usd': liq,
                'age_hours': age_hours,
                'txns_24h': pair.get('txns', {}).get('h24', {}).get('buys', 0) + pair.get('txns', {}).get('h24', {}).get('sells', 0),
                'price_chg_24h': pair.get('priceChange', {}).get('h24'),
                'created_at': created.isoformat() if created else None,
            })

    df = pd.DataFrame(candidates)
    if df.empty:
        print('No candidates found')
        return
    df.drop_duplicates(subset=['chain', 'address'], inplace=True)
    df.sort_values('vol_24h', ascending=False, inplace=True)
    ts = now.strftime('%Y%m%d_%H%M%S')
    out = OUT_DIR / f'dex_candidates_{ts}.csv'
    df.to_csv(out, index=False)
    print(f'\nTop candidates:\n{df.head(20).to_string()}')
    print(f'\nSaved {len(df)} candidates to {out}')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    collect()
