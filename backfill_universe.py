import time
import urllib.request
import pandas as pd
from pathlib import Path

ROOT = Path('data')
LIMIT = 1500
BATCH_SLEEP = 0.25

# Universe CSV -> unique stems we should have
universe = pd.read_csv(ROOT / 'universe_broad.csv')
universe = universe.dropna(subset=['symbol'])
universe['symbol'] = universe['symbol'].astype(str).str.strip()

existing = {p.name.replace('_1d_max.csv', '').upper(): p for p in ROOT.glob('*_1d_max.csv')}

missing = []
for _, row in universe.iterrows():
    sym = str(row['symbol']).strip().upper()
    if sym not in existing:
        candidates = [s for s in existing if s.startswith(sym)]
        if not candidates:
            missing.append(sym)
    else:
        bars = sum(1 for _ in open(existing[sym])) - 1
        if bars < 300:
            missing.append(sym)

print(f'Targeting backfill for {len(missing)} coins')

downloaded = 0
failed = 0
for sym in missing:
    for quote in ['USDT', 'USDC']:
        stem = f'{sym}{quote}'
        p = ROOT / f'{stem}_1d_max.csv'
        if p.exists():
            continue
        try:
            url = f'https://api.mexc.com/api/v3/klines?symbol={stem}&interval=1d&limit={LIMIT}'
            req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.88.1'})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = __import__('json').load(r)
            if not raw:
                continue
            rows = []
            for k in raw:
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
            df.to_csv(p, index=False)
            downloaded += 1
            print(f'  {stem:20s} bars={len(df)}')
        except Exception as e:
            failed += 1
        time.sleep(BATCH_SLEEP)
    if downloaded >= 200:
        print('Batch cap reached, stopping early')
        break

print(f'Downloaded {downloaded}, failures {failed}')
