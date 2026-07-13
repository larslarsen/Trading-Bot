import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('data')
UNIVERSE = ROOT / 'universe_broad.csv'
OUT = Path('backtest_output')

# Load universe
u = pd.read_csv(UNIVERSE)
u = u.dropna(subset=['symbol'])
u['symbol'] = u['symbol'].astype(str).str.strip()
u['tier'] = u['tier'].fillna('unknown').str.strip().str.lower()

# Filter to enriched + known tier
u = u[u['tier'].isin(['large','mid','tail'])]
print(f'Enriched coins: {len(u)}')
print(u['tier'].value_counts().to_dict())

# Load OHLCV for each coin, compute ADV and idiosyncratic vol
available = sorted(ROOT.glob('*_1d_max.csv'))

rows = []
for _, row in u.iterrows():
    sym = str(row['symbol']).strip().upper()
    candidates = [p.name.replace('_1d_max.csv', '').upper() for p in available if p.name.replace('_1d_max.csv', '').upper().startswith(sym)]
    if not candidates:
        continue
    stem = max(candidates, key=len)
    p = ROOT / f'{stem}_1d_max.csv'
    try:
        df = pd.read_csv(p, parse_dates=['ts']).dropna(subset=['close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], errors='coerce').dt.tz_localize(None)
        if len(df) < 60:
            continue
        df['ret'] = df['close'].pct_change()
        adv = df['volume'].mean()
        idio = df['ret'].rolling(30).std().mean()
        rows.append({
            'symbol': sym,
            'stem': stem,
            'tier': row.get('tier',''),
            'bars': len(df),
            'adv': adv,
            'idio_vol': idio,
            'mcap_rank': row.get('mcap_rank', None),
        })
    except Exception:
        continue

screen = pd.DataFrame(rows)
screen['adv_rank'] = screen['adv'].rank(pct=True)

# Liquidity filter: top 500 by ADV
screen = screen.sort_values('adv', ascending=False).head(500)
print(f'After ADV filter: {len(screen)}')

# Rank by idiosyncratic vol within tier
screen['idio_rank_in_tier'] = screen.groupby('tier')['idio_vol'].rank(pct=True)

# Top quintile of idiosyncratic vol per tier
selected_parts = []
for tier, grp in screen.groupby('tier'):
    n = max(3, int(len(grp) * 0.2))
    selected_parts.append(grp.nlargest(n, 'idio_vol'))
selected = pd.concat(selected_parts).sort_values('idio_vol', ascending=False).reset_index(drop=True)

print('\n=== Top idiosyncratic vol by tier ===')
print(selected[['symbol','tier','bars','adv','idio_vol']].head(20).to_string(index=False))

out_csv = OUT / f'screen_liqu_idio_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv'
selected.to_csv(out_csv, index=False)
print(f'\nSaved {out_csv}')
