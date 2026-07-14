"""
BloFin backfill (1d OHLCV) via ccxt public API. Pulls daily bars for the pairs in
blofin_swap_pairs.txt (swap/perpetual USDT pairs) and saves to
data/<SYMBOL>_blofin_1d_max.csv. BloFin serves 1000 bars/call (back to ~2023-10).

Symbol format in blofin_swap_pairs.txt is bare (e.g. BTCUSDT); ccxt wants
'BTC/USDT:USDT'. Convert: quote = USDC if endswith USDC else USDT; base = symbol[:-4].
Save with _blofin suffix to avoid colliding with same-named MEXC files.
"""
import time
import pandas as pd
import ccxt
from pathlib import Path

ROOT = Path('data')
PAUSE = 0.25
LIMIT = 1000


def to_ccxt(sym):
    quote = 'USDC' if sym.endswith('USDC') else 'USDT'
    base = sym[:-4]
    return f"{base}/{quote}:{quote}"


def main():
    ex = ccxt.blofin({'enableRateLimit': True})
    pairs = [l.strip() for l in open(ROOT / 'blofin_swap_pairs.txt') if l.strip()]
    print(f'BloFin backfill: {len(pairs)} pairs')
    done = 0
    for sym in pairs:
        try:
            out = ROOT / f"{sym}_blofin_1d_max.csv"
            if out.exists():
                continue
            ccxt_sym = to_ccxt(sym)
            rows = ex.fetchOHLCV(ccxt_sym, '1d', limit=LIMIT)
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
            df.to_csv(out, index=False)
            done += 1
            if done % 25 == 0:
                print(f'  {sym:14s} {len(df)} bars {df["ts"].min().date()}..{df["ts"].max().date()}')
        except Exception as e:
            print(f'  {sym} ERR {str(e)[:60]}')
        time.sleep(PAUSE)
    print(f'BloFin backfill DONE. New files: {done}')


if __name__ == '__main__':
    main()
