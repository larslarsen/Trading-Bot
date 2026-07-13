#!/usr/bin/env python3
"""
Direct Solana DEX data fetcher from public RPC nodes.

Rebuilds OHLCV bars from on-chain swap transactions.
No API keys, no GeckoTerminal/Birdeye middleware.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

OUT_DIR = Path(__file__).parent / 'dex_data'
OUT_DIR.mkdir(exist_ok=True)

# Public Solana RPC endpoints (free, no key)
RPC_ENDPOINTS = [
    'https://api.mainnet-beta.solana.com',
    'https://solana-api.projectserum.com',
    'https://rpc.ankr.com/solana',
]

# DEX Program IDs that emit swap transactions
DEX_PROGRAMS = {
    'Raydium': '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',
    'Orca': 'DjVE6JNiYqPL2QXpzUUQantxfuWGe9A9UQA3DGqwbLAd',
    'Jupiter': 'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4',
    'OrcaWhirl': 'whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc',
    'Meteora': 'Eo7WjKq67rjJQSZxS6zYsYvz69SN7x2A5VBmjKGvCGBT',
}

# Known pool addresses for popular retail tokens
# Format: (base_mint, quote_mint, pool_address, dex)
KNOWN_POOLS = {
    'WIF/USDT_solana': ('WIF address', 'USDT address', 'Hjj93YiyaFYY8zY2EW6FM2i2gd4rxzhoLCLFabrRpump', 'pumpswap'),
    'DOGE/USDT_solana': ('DOGE address', 'USDT address', 'Gzsz28orh7mRUmcoBYRS4NBCN1HUUz9ixZdn24B6KjG2', 'raydium'),
    'AVAX/USDT_solana': ('AVAX address', 'USDT address', '6KBf3BwHxcczzNHUKgax3WsNfDH9vwMVZXPujdkEYcbE', 'raydium'),
    'SUI/USDT_solana': ('SUI address', 'USDT address', '7WmTQbAL4M54umYjiraZLaiMFtnjVBiNdq9wowsXenY4', 'raydium'),
    'XRP/USDT_solana': ('XRP address', 'USDT address', 'ET122tFEx9jbq1gVrYxdnAyq1gxpGXyBvran3c3Pi5N5', 'raydium'),
    'INJ/USDT_solana': ('INJ address', 'USDT address', 'BeYA9PGXQe6w6QUtzky9JXzuBCrP7pmkZcjWqdYHpNsh', 'raydium'),
    'XRP/ETH_base': ('XRP address', 'ETH address', '0x72F7433E850e87b26ac2a5ad001F92E36CdCa696', 'uniswap'),
}


def rpc_call(method, params, retries=3):
    """Call Solana RPC with failover."""
    payload = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': method,
        'params': params,
    }
    last_err = None
    for endpoint in RPC_ENDPOINTS:
        for attempt in range(retries):
            try:
                r = requests.post(endpoint, json=payload, timeout=15)
                r.raise_for_status()
                result = r.json()
                if 'error' in result:
                    last_err = result['error']
                    continue
                return result.get('result')
            except Exception as e:
                last_err = str(e)
                time.sleep(0.5)
    print(f'RPC {method} failed after {retries * len(RPC_ENDPOINTS)} attempts: {last_err}')
    return None


def get_swap_signatures(pool_address, start_slot=None, limit=1000):
    """Fetch swap transaction signatures for a pool."""
    # Query the pool's token account for swap transactions
    params = [
        {
            'mentions': [pool_address],
        },
        {
            'commitment': 'confirmed',
            'limit': limit,
        },
    ]
    if start_slot:
        params[1]['minSlot'] = start_slot
    result = rpc_call('searchTransactions', params)
    if not result or not result.get('Transactions'):
        return []
    return [tx for tx in result['Transactions'] if is_swap_tx(tx)]


def is_swap_tx(tx):
    """Check if transaction is a swap based on program IDs."""
    try:
        instructions = tx.get('transaction', {}).get('message', {}).get('instructions', [])
        for inst in instructions:
            program_id = inst.get('programId')
            if program_id in DEX_PROGRAMS.values():
                return True
    except Exception:
        pass
    return False


def get_swap_data(tx_signature):
    """Fetch full swap transaction details."""
    params = [
        tx_signature,
        {
            'encoding': 'json',
            'commitment': 'confirmed',
            'maxSupportedTransactionVersion': 0,
        },
    ]
    result = rpc_call('getTransaction', params)
    if not result:
        return None
    return parse_swap_transaction(result)


def parse_swap_transaction(tx_data):
    """Extract price/amount/swap data from transaction."""
    try:
        meta = tx_data.get('meta', {})
        pre_balances = meta.get('preBalances', [])
        post_balances = meta.get('postBalances', [])
        
        if len(pre_balances) < 4 or len(post_balances) < 4:
            return None
        
        # Calculate price from balance changes
        base_in = pre_balances[2] - post_balances[2]
        quote_out = post_balances[3] - pre_balances[3]
        
        if quote_out > 0:
            price = base_in / quote_out
            amount = quote_out
            return {
                'price': price,
                'amount': amount,
                'timestamp': tx_data.get('blockTime'),
                'slot': tx_data.get('slot'),
            }
    except Exception:
        pass
    return None


def build_ohlcv_from_swaps(swaps, timeframe='1d'):
    """Rebuild OHLCV bars from a list of swaps."""
    if not swaps:
        return pd.DataFrame()
    
    df = pd.DataFrame(swaps)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Resample to requested timeframe
    freq_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '1d': '1d'}
    freq = freq_map.get(timeframe, '1h')
    
    o = df['price'].resample(freq).first()
    h = df['price'].resample(freq).max()
    l = df['price'].resample(freq).min()
    c = df['price'].resample(freq).last()
    v = df['amount'].resample(freq).sum()
    
    out = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}, index=c.index)
    out.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    return out


def fetch_pool_swaps(pool_address, pool_name, max_transactions=5000):
    """Fetch all available swaps for a pool and save to CSV."""
    print(f'Fetching swaps for {pool_name} ({pool_address}) ...')
    all_swaps = []
    start_slot = None
    last_len = 0
    
    # Paginate through historical swaps
    for page in range(50):  # max 50 pages * 1000 = 50k transactions
        signatures = get_swap_signatures(pool_address, start_slot=start_slot, limit=1000)
        if not signatures:
            break
        
        for sig in signatures[:1000]:
            swap = get_swap_data(sig.get('signature'))
            if swap:
                all_swaps.append(swap)
        
        if len(all_swaps) >= max_transactions:
            break
        
        if len(all_swaps) == last_len:
            break  # no new data
        last_len = len(all_swaps)
        
        # Progress reporting
        if page % 10 == 0 and page > 0:
            print(f'  {pool_name}: {len(all_swaps)} swaps fetched ...')
            time.sleep(1)  # rate limit courtesy
    
    if not all_swaps:
        print(f'  {pool_name}: no swap data found')
        return None, None
    
    df_swaps = pd.DataFrame(all_swaps)
    df_swaps['timestamp'] = pd.to_datetime(df_swaps['timestamp'], unit='s', utc=True)
    df_swaps.sort_values('timestamp', inplace=True)
    
    # Save raw swaps
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    swap_out = OUT_DIR / f'{pool_name}_swaps_{ts}.csv'
    df_swaps.to_csv(swap_out, index=False)
    
    # Build OHLCV bars
    ohlcv = {
        '1h': build_ohlcv_from_swaps(all_swaps, '1h'),
        '1d': build_ohlcv_from_swaps(all_swaps, '1d'),
    }
    
    ohlcv_paths = {}
    for tf, df in ohlcv.items():
        if not df.empty:
            out_path = OUT_DIR / f'{pool_name}_{tf}_{ts}.csv'
            df.to_csv(out_path, index=True)
            ohlcv_paths[tf] = str(out_path)
    
    print(f'  {pool_name}: {len(df_swaps)} swaps -> {len(ohlcv_paths)} OHLCV files')
    return str(swap_out), ohlcv_paths


def main():
    t0 = time.time()
    print('[onchain] fetching DEX swap data directly from Solana RPC ...')
    print('[onchain] NOTE: this is OUR data - no API keys needed')
    
    results = {}
    for pool_name, (base_mint, quote_mint, pool_address, dex) in KNOWN_POOLS.items():
        if 'Placeholder' in pool_address:
            print(f'  {pool_name}: placeholder pool address, skipping')
            continue
        
        swap_path, ohlcv_paths = fetch_pool_swaps(pool_address, pool_name)
        results[pool_name] = {
            'pool': pool_address,
            'dex': dex,
            'swaps': swap_path,
            'ohlcv': ohlcv_paths,
            'swap_count': len(pd.read_csv(swap_path)) if swap_path and Path(swap_path).exists() else 0,
        }
    
    # Save manifest
    manifest = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'rpc_endpoints': RPC_ENDPOINTS,
        'dex_programs': DEX_PROGRAMS,
        'results': results,
    }
    manifest_path = OUT_DIR / f'onchain_manifest_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json'
    with open(manifest_path, 'w') as f:
        json.dump({k: (v if not isinstance(v, Path) else str(v)) for k, v in manifest.items()}, f, indent=2, default=str)
    
    print(f'\n[onchain] completed in {time.time()-t0:.1f}s')
    print(f'[onchain] manifest saved to {manifest_path}')


if __name__ == '__main__':
    main()
