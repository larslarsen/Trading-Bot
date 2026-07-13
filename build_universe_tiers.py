#!/usr/bin/env python3
"""
Build exchange-aware universe with cap tiers.
- One coin per quote currency per exchange
- Tier by market-cap rank tertiles
- Merge with CoinGecko filtered list
"""
import csv, json
from pathlib import Path
from collections import defaultdict

ROOT = Path('data')

# 1. Load CoinGecko filtered list
cg = {}
with open(ROOT / 'coingecko_universe_filtered.csv') as f:
    for row in csv.DictReader(f):
        sym = row['symbol'].upper()
        cg[sym] = {
            'id': row['id'],
            'name': row['name'],
            'mcap_rank': int(row['market_cap_rank']) if row['market_cap_rank'] else None,
            'mcap_usd': float(row['market_cap_usd']) if row['market_cap_usd'] else None,
            'volume_24h': float(row['volume_24h_usd']) if row['volume_24h_usd'] else 0,
            'price_usd': float(row['current_price_usd']) if row['current_price_usd'] else None,
            'chg_24h': float(row['price_change_24h']) if row['price_change_24h'] else None,
            'categories': row['categories'],
        }
print(f'CoinGecko filtered entries: {len(cg)}')

# 2. Load Kraken + MEXC pair lists
kraken_pairs, mexc_pairs = [], []
with open(ROOT / 'kraken_pairs.csv') as f:
    for row in csv.DictReader(f):
        kraken_pairs.append(row)
with open(ROOT / 'mexc_pairs.csv') as f:
    for row in csv.DictReader(f):
        mexc_pairs.append(row)
print(f'Kraken pairs: {len(kraken_pairs)}, MEXC pairs: {len(mexc_pairs)}')

# 3. Deduplicate: one coin per quote per exchange
# Prefer USDT > USDC > USD > ZUSD
QUOTE_PRIORITY = {'USDT': 0, 'USDC': 1, 'USDC': 1, 'USD': 2, 'ZUSD': 2}
def quote_priority(row):
    q = row.get('quote', '').upper()
    return QUOTE_PRIORITY.get(q, 99)

kraken_best = {}
for row in kraken_pairs:
    base = row['base'].upper()
    q = row['quote'].upper()
    if q not in ('ZUSD', 'USDT', 'USDC'):
        continue
    if base not in kraken_best or quote_priority(row) < quote_priority(kraken_best[base]):
        kraken_best[base] = row

mexc_best = {}
for row in mexc_pairs:
    base = row['base'].upper()
    q = row['quote'].upper()
    if q not in ('USDT', 'USDC'):
        continue
    if base not in mexc_best or quote_priority(row) < quote_priority(mexc_best[base]):
        mexc_best[base] = row

print(f'Kraken unique bases: {len(kraken_best)}, MEXC unique bases: {len(mexc_best)}')

# 4. Build unified exchange-aware list
all_exchange_coins = {}
for base, row in kraken_best.items():
    all_exchange_coins[base] = {**row, 'exchange': 'kraken'}
for base, row in mexc_best.items():
    if base in all_exchange_coins:
        all_exchange_coins[base]['exchange'] = 'kraken+mexc'
    else:
        all_exchange_coins[base] = {**row, 'exchange': 'mexc'}

print(f'Combined unique bases: {len(all_exchange_coins)}')

# 5. Merge with CoinGecko on symbol
merged = []
for base, ex_row in all_exchange_coins.items():
    cg_row = cg.get(base)
    if cg_row:
        mcap_rank = cg_row['mcap_rank'] or 99999
        merged.append({
            'symbol': base,
            'name': cg_row['name'],
            'exchange': ex_row['exchange'],
            'quote': ex_row['quote'],
            'mcap_rank': mcap_rank,
            'mcap_usd': cg_row['mcap_usd'],
            'volume_24h_usd': cg_row['volume_24h'],
            'current_price_usd': cg_row['price_usd'],
            'price_change_24h': cg_row['chg_24h'],
            'categories': cg_row['categories'],
        })
    else:
        # No CG data — still include, no mcap
        merged.append({
            'symbol': base,
            'name': base,
            'exchange': ex_row['exchange'],
            'quote': ex_row['quote'],
            'mcap_rank': 99999,
            'mcap_usd': None,
            'volume_24h_usd': None,
            'current_price_usd': None,
            'price_change_24h': None,
            'categories': '',
        })

print(f'Merged with CG: {sum(1 for m in merged if m["mcap_rank"] != 99999)} with mcap, '
      f'{sum(1 for m in merged if m["mcap_rank"] == 99999)} without')

# 6. Assign tier by mcap rank tertiles (among coins with mcap data)
ranked = sorted([m for m in merged if m['mcap_rank'] and m['mcap_rank'] != 99999], key=lambda x: x['mcap_rank'])
n = len(ranked)
tertile_size = max(1, n // 3)
for i, m in enumerate(ranked):
    if i < tertile_size:
        m['tier'] = 'large'
    elif i < 2 * tertile_size:
        m['tier'] = 'mid'
    else:
        m['tier'] = 'tail'

# Fill tier for unranked coins as 'tail'
for m in merged:
    if 'tier' not in m:
        m['tier'] = 'tail'

# 7. Summary by tier
from collections import Counter
tier_counts = Counter(m['tier'] for m in merged)
print(f'Tiers: {dict(tier_counts)}')

# 8. Save
fields = ['symbol','name','exchange','quote','tier','mcap_rank','mcap_usd','volume_24h_usd','current_price_usd','price_change_24h','categories']
with open(ROOT / 'universe_enriched.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(merged)

with open(ROOT / 'universe_tail_only.txt', 'w') as f:
    tail = sorted([m for m in merged if m['tier'] == 'tail'], key=lambda x: x['mcap_rank'] if x['mcap_rank'] != 99999 else 99999)
    for m in tail:
        f.write(f"{m['symbol']} | {m['name']} | {m['exchange']} | mcap_rank={m['mcap_rank'] or '?'} | vol=${m['volume_24h_usd'] or 0:,.0f}\n")
    f.write(f'\nTotal tail coins: {len(tail)}\n')

print('Saved data/universe_enriched.csv and data/universe_tail_only.txt')

# Print top 20 tail coins by volume
tail = [m for m in merged if m['tier'] == 'tail' and m['volume_24h_usd']]
tail_sorted = sorted(tail, key=lambda x: x['volume_24h_usd'] or 0, reverse=True)
print('\nTop 20 tail coins by volume:')
for m in tail_sorted[:20]:
    print(f"  {m['symbol']:10s} {m['name']:30s} {m['exchange']:12s} vol=${m['volume_24h_usd']:>12,.0f} mcap_r={m['mcap_rank'] or '?':>5} cat={m['categories'][:40]}")
