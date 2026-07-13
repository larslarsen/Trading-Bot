#!/usr/bin/env python3
"""
Gentle DEX scout: polls free sources at a low rate to find new retail pairs.

Designed to NOT trigger rate limits by default.
If it sees sustained 429/403, it backs off and stops.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

OUT_DIR = Path(__file__).parent / 'dex_data'
OUT_DIR.mkdir(exist_ok=True)

# ---- Low-rate configuration ----
MIN_INTERVAL_SEC = 30          # minimum between same-source queries
BACKOFF_FACTOR = 2             # double wait time on rate limit
MAX_BACKOFF_SEC = 600          # cap backoff at 10 minutes
ERROR_THRESHOLD = 5            # consecutive errors before long sleep
STATE_FILE = OUT_DIR / 'dex_scout_state.json'

# Sources to poll (1 request per source per cycle)
SOURCES = {
    'geckoterminal_pools': {
        'url': 'https://api.geckoterminal.com/api/v2/networks/solana/pools',
        'params': {'limit': 50},
    },
    'birdeye_trending': {
        'url': 'https://public-api.birdeye.so/public/token_list/top_trending',
        'params': {'limit': 50},
    },
    # Could add more Birdeye/Gecko endpoints here later
}

CHAINS = ['solana', 'ethereum', 'base', 'bsc', 'arbitrum', 'avalanche', 'fantom']

# Filtering thresholds
MIN_VOLUME_24H_USD = 5_000         # at least $5k 24h volume
MIN_LIQUIDITY_USD = 10_000         # at least $10k liquidity
MIN_AGE_DAYS = 3                   # at least 3 days old
MAX_PRICE_CHG_24H_PCT = 150        # ignore extreme pumps/rugs

def passes_filters(row):
    try:
        if row.get('source') != 'geckoterminal':
            return True
        vol = float(row.get('volume_24h') or 0)
        liq = float(row.get('liquidity_usd') or 0)
        chg = float(row.get('price_chg_24h') or 0)
        created = row.get('created_at')
        age_days = 9999
        if created:
            try:
                created_dt = datetime.fromisoformat(str(created).replace('Z', '+00:00'))
                age_days = (datetime.now(timezone.utc) - created_dt).days
            except Exception:
                age_days = 0
        if vol < MIN_VOLUME_24H_USD:
            return False
        if liq < MIN_LIQUIDITY_USD:
            return False
        if age_days < MIN_AGE_DAYS:
            return False
        if abs(chg) > MAX_PRICE_CHG_24H_PCT:
            return False
        return True
    except Exception:
        return False


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        'last_poll': {},
        'backoff_until': {},
        'consecutive_errors': {},
        'found_pairs': [],
    }


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def should_poll(source_name):
    """Check if enough time has passed since last poll."""
    state = load_state()
    backoff_key = 'backoff_until'
    if backoff_key in state and state[backoff_key].get(source_name):
        until = datetime.fromisoformat(state[backoff_key][source_name])
        if datetime.now(timezone.utc) < until:
            return False, f'backoff until {until.isoformat()}'
    
    last = state.get('last_poll', {}).get(source_name)
    if last and (time.time() - last) < MIN_INTERVAL_SEC:
        return False, f'rate limited ({time.time()-last:.0f}s ago)'
    return True, 'ok'


def mark_success(source_name):
    state = load_state()
    state['last_poll'][source_name] = time.time()
    state['backoff_until'][source_name] = None
    state['consecutive_errors'][source_name] = 0
    save_state(state)


def mark_error(source_name, backoff=False):
    state = load_state()
    errors = state.get('consecutive_errors', {}).get(source_name, 0) + 1
    state['consecutive_errors'][source_name] = errors
    
    if backoff or errors >= ERROR_THRESHOLD:
        # Exponential backoff
        wait = MIN_INTERVAL_SEC * (BACKOFF_FACTOR ** min(errors, 10))
        wait = min(wait, MAX_BACKOFF_SEC)
        until = (datetime.now(timezone.utc).timestamp() + wait)
        state['backoff_until'][source_name] = datetime.fromtimestamp(until, timezone.utc).isoformat()
        print(f'[{source_name}] rate limited, backing off {wait}s')
    
    save_state(state)


def poll_geckoterminal():
    """Poll GeckoTerminal Solana pools list."""
    source = 'geckoterminal_pools'
    ok, reason = should_poll(source)
    if not ok:
        print(f'[{source}] skip: {reason}')
        return []
    
    try:
        r = requests.get(
            SOURCES[source]['url'],
            params=SOURCES[source]['params'],
            timeout=10,
            headers={'accept': 'application/json'}
        )
        
        if r.status_code == 429 or r.status_code == 403:
            mark_error(source, backoff=True)
            return []
        
        r.raise_for_status()
        data = r.json().get('data', [])
        mark_success(source)
        
        # Extract pool info
        results = []
        for pool in data[:50]:
            attr = pool.get('attributes', {})
            results.append({
                'source': 'geckoterminal',
                'pool_id': pool.get('id'),
                'name': attr.get('name'),
                'base_symbol': attr.get('base_token_symbol'),
                'quote_symbol': attr.get('quote_token_symbol'),
                'base_price_usd': attr.get('base_token_price_usd'),
                'price_usd': attr.get('quote_token_price_usd'),
                'volume_24h': attr.get('volume_usd', {}).get('h24'),
                'liquidity_usd': attr.get('reserve_in_usd'),
                'price_chg_24h': attr.get('price_change_percentage', {}).get('h24'),
                'txns_24h': attr.get('tx_count', {}).get('h24'),
                'created_at': attr.get('pool_created_at'),
            })
        
        # Save raw snapshot
        if results:
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            pd.DataFrame(results).to_csv(OUT_DIR / f'gecko_pools_{ts}.csv', index=False)
        
        return results
    
    except Exception as e:
        print(f'[{source}] error: {e}')
        mark_error(source)
        return []


def poll_birdeye():
    """Poll Birdeye trending tokens."""
    source = 'birdeye_trending'
    ok, reason = should_poll(source)
    if not ok:
        print(f'[{source}] skip: {reason}')
        return []
    
    try:
        url = SOURCES[source]['url']
        r = requests.get(
            url,
            params=SOURCES[source]['params'],
            timeout=10,
            headers={'accept': 'application/json'}
        )
        
        if r.status_code == 429 or r.status_code == 403:
            mark_error(source, backoff=True)
            return []
        
        r.raise_for_status()
        data = r.json().get('data', [])
        mark_success(source)
        
        # Normalize to pool-like format
        results = []
        for token in data[:50]:
            results.append({
                'source': 'birdeye',
                'pool_id': token.get('address'),
                'name': token.get('symbol'),
                'base_symbol': token.get('symbol'),
                'quote_symbol': 'USDT',
                'base_price_usd': token.get('price'),
                'price_usd': token.get('price'),
                'volume_24h': token.get('volume_24h'),
                'liquidity_usd': token.get('liquidity'),
                'price_chg_24h': token.get('price_change_24h'),
                'txns_24h': None,
                'created_at': None,
            })
        
        # Save raw snapshot
        if results:
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            pd.DataFrame(results).to_csv(OUT_DIR / f'birdeye_trending_{ts}.csv', index=False)
        
        return results
    
    except Exception as e:
        print(f'[{source}] error: {e}')
        mark_error(source)
        return []


def main():
    print('[dex_scout] starting gentle polling mode')
    print(f'[dex_scout] min interval: {MIN_INTERVAL_SEC}s per source')
    print(f'[dex_scout] will backoff on rate limits, stop after {ERROR_THRESHOLD} consecutive errors')
    
    all_pairs = []
    
    # Poll each source
    for source_name in SOURCES:
        if 'geckoterminal' in source_name:
            pairs = poll_geckoterminal()
        elif 'birdeye' in source_name:
            pairs = poll_birdeye()
        else:
            continue
        
        all_pairs.extend(pairs)
        
        # Small delay between sources
        if source_name != list(SOURCES.keys())[-1]:
            time.sleep(5)
    
    # Deduplicate and log new findings
    if all_pairs:
        df = pd.DataFrame(all_pairs)
        before = len(df)
        df = df[df.apply(passes_filters, axis=1)]
        dropped = before - len(df)
        seen_file = OUT_DIR / 'dex_scout_seen.csv'
        
        if seen_file.exists():
            seen = pd.read_csv(seen_file)
            df_seen = pd.concat([seen, df], ignore_index=True).drop_duplicates(subset=['pool_id'], keep='first')
        else:
            df_seen = df.drop_duplicates(subset=['pool_id'], keep='first')
        
        df_seen.to_csv(seen_file, index=False)
        print(f'[dex_scout] {len(all_pairs)} raw, {len(df)} after filters, {len(df_seen)} total tracked')
        
        # Log top volume/momentum candidates
        if 'volume_24h' in df.columns and 'price_chg_24h' in df.columns:
            df['volume_24h'] = pd.to_numeric(df['volume_24h'], errors='coerce')
            df['price_chg_24h'] = pd.to_numeric(df['price_chg_24h'], errors='coerce')
            top = df.nlargest(10, 'volume_24h')
            print(f'[dex_scout] top by volume:\n{top[["source","name","volume_24h","price_chg_24h"]].to_string(index=False)}')
    else:
        print('[dex_scout] no pairs this cycle')
    
    print(f'[dex_scout] cycle complete. Next cycle in {MIN_INTERVAL_SEC}s')
    print(f'[dex_scout] total tracked pairs: {len(pd.read_csv(OUT_DIR/"dex_scout_seen.csv")) if (OUT_DIR/"dex_scout_seen.csv").exists() else 0}')


if __name__ == '__main__':
    main()
