#!/usr/bin/env python3
"""
Retail shitcoin screener: free, tiered, with live pull.

Tiers:
1. Discovery: DexScreener search + Birdeye trending tokens + GeckoTerminal trending
2. Data fetch: Binance public klines 1d/4h, MEXC public klines 1d/4h
3. Simple rule scoring: SMA crossover 20/50 + volume, Triple RSI, momentum breakout
4. Output: ranked list with confidence/volume/volume_z_score/age_days

No API keys required. Designed for high-throughput screening, not deep research.
"""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import requests
import vectorbt as vbt

try:
    import ccxt
except ImportError:
    raise SystemExit('pip install ccxt')

OUT_DIR = Path(__file__).parent / 'dex_data'
OUT_DIR.mkdir(exist_ok=True)

# ---- Free API endpoints ----
DEXSCREENER_BASE = 'https://api.dexscreener.com/latest/dex'
BIRDEYE_BASE = 'https://public-api.birdeye.so/public'
GEcko_BASE = 'https://www.geckoterminal.com/api/v1'

# ---- VENUES ----
CEX = {
    'binance': ccxt.binance({'enableRateLimit': True}),
    'mexc': ccxt.mexc({'enableRateLimit': True}),
}

# ---- Screener configuration ----
MIN_VOL_USD = 50_000        # high 24h volume filter
MAX_LIQUIDITY_USD = 2_000_000  # low liquidity cap
MIN_LIQUIDITY_USD = 500      # must have some liquidity
MAX_AGE_DAYS = 120           # focus on reasonably young tokens
SAMPLE_DAYS = 180            # pull 6 months of OHLCV for backtesting

# ---- Simple rule library ----
def rule_sma_cross_vol(df):
    """SMA(20) > SMA(50) + volume > 1.2x average."""
    sma_fast = df['close'].rolling(20).mean()
    sma_slow = df['close'].rolling(50).mean()
    cross_up = (sma_fast > sma_slow).astype(bool)
    vol = df['volume']
    vol_ma = vol.rolling(20).mean()
    vol_ok = (vol > 1.2 * vol_ma).astype(bool)
    signal = pd.Series(0, index=df.index)
    signal.loc[cross_up & vol_ok] = 1
    signal.loc[(~cross_up) & vol_ok] = -1
    return signal


def rule_triple_rsi(df):
    """RSI(14) momentum: long if > 55, short if < 45, with volume filter."""
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    vol = df['volume']
    vol_ma = vol.rolling(20).mean()
    vol_ok = (vol > 1.2 * vol_ma).astype(bool)
    signal = pd.Series(0, index=df.index)
    signal.loc[(rsi > 55) & vol_ok] = 1
    signal.loc[(rsi < 45) & vol_ok] = -1
    return signal


def rule_vol_breakout(df, lookback=20):
    """Price breaks highest high of last 20 bars with above-average volume."""
    high_n = df['high'].rolling(lookback).max()
    low_n = df['low'].rolling(lookback).min()
    vol = df['volume']
    vol_ma = vol.rolling(20).mean()
    vol_ok = (vol > 1.5 * vol_ma).astype(bool)
    signal = pd.Series(0, index=df.index)
    signal.loc[(df['close'] > high_n.shift(1)) & vol_ok] = 1
    signal.loc[(df['close'] < low_n.shift(1)) & vol_ok] = -1
    return signal


def score_rule(signal, df, freq, rule_name):
    """Run rule through vectorbt, return simplified metrics."""
    trade = signal.reindex(df.index).fillna(0)
    price = df['close'].values.astype(float)
    trade_arr = trade.values.astype(int)
    entries_long = trade_arr == 1
    entries_short = trade_arr == -1
    prev = np.roll(trade_arr, 1)
    prev[0] = 0
    exits = (prev != 0) & (trade_arr == 0)
    short_exits = (prev == -1) & (trade_arr == 0)
    pf = vbt.Portfolio.from_signals(
        price, entries=entries_long, exits=exits,
        short_entries=entries_short, short_exits=short_exits,
        freq=freq, init_cash=10_000, size=100, size_type='value',
        fees=0.0008, slippage=0.0005,
    )
    try:
        trades = pf.trades
        wr = float(trades.win_rate()) if trades.count() > 0 else 0.0
    except Exception:
        wr = 0.0
    try:
        sr = float(pf.sharpe_ratio())
    except Exception:
        sr = 0.0
    return {
        'rule': rule_name,
        'return': float(pf.total_return()),
        'sharpe': sr,
        'win_rate': wr,
        'trades': int(pf.trades.count()),
        'signals': int((trade_arr != 0).sum()),
        'bars': int(len(df)),
    }


def fetch_ohlcv(symbol, timeframe='1d', since_days=180):
    """Try multiple CEXes for OHLCV."""
    for name, ex in CEX.items():
        try:
            since = ex.parse8601((pd.Timestamp.utcnow() - pd.Timedelta(days=since_days)).isoformat())
            all_rows = []
            while True:
                ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
                if not ohlcv:
                    break
                all_rows.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                if len(ohlcv) < 1000:
                    break
                time.sleep(0.1)
            if all_rows and len(all_rows) > 100:
                df = pd.DataFrame(all_rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df.set_index('ts', inplace=True)
                df.sort_index(inplace=True)
                return df
        except Exception:
            continue
    return pd.DataFrame()


# ---- Discovery functions ----
def dexscreen_search(query='new', limit=100):
    """Search DexScreener for recent pairs."""
    try:
        r = requests.get(f'{DEXSCREENER_BASE}/search', params={'q': query}, timeout=10)
        r.raise_for_status()
        return r.json().get('pairs', [])[:limit]
    except Exception:
        return []


def birdeye_trending(limit=50):
    """Get trending tokens from Birdeye."""
    try:
        r = requests.get(f'{BIRDEYE_BASE}/token_list/top_trending', timeout=10)
        r.raise_for_status()
        data = r.json().get('data', [])
        return data[:limit]
    except Exception:
        return []


def normalize_ts(v):
    if not v:
        return None
    if isinstance(v, str):
        try:
            return pd.to_datetime(v, utc=True)
        except Exception:
            try:
                return pd.to_datetime(int(v), unit='ms', utc=True)
            except Exception:
                return None
    try:
        vi = int(v)
        return pd.to_datetime(vi, unit='ms', utc=True)
    except Exception:
        try:
            return pd.to_datetime(v, utc=True)
        except Exception:
            return None


def normalize_pair(p):
    """Extract standardized symbol from DexScreener pair."""
    base = p.get('baseToken', {}).get('symbol') or p.get('baseToken', {}).get('address', '')
    quote = p.get('quoteToken', {}).get('symbol') or 'USDT'
    if not base:
        return None
    if quote not in ('USDT', 'USD', 'USDC', 'BUSD', 'DAI'):
        quote = 'USDT'
    return f'{base}/{quote}'


def discover():
    """Tier 1: discover candidate tokens from multiple free sources."""
    candidates = []
    seen = set()

    # DexScreener token-specific queries
    for q in ['SOL','ETH','BASE','BSC','ARB','AVAX','FTM','MATIC','SUI','INJ','DOGE','XRP','LINK','UNI','AAVE','MKR','PEPE','WIF','BONK','FLOKI']:
        for p in dexscreen_search(q, limit=50):
            sym = normalize_pair(p)
            if not sym:
                continue
            created = normalize_ts(p.get('pairCreatedAt') or p.get('createdAt'))
            age_days = (datetime.now(timezone.utc) - created).days if created else None
            vol = float(p.get('volume', {}).get('h24', 0) or 0)
            liq = float(p.get('liquidity', {}).get('usd', 0) or 0)
            key = sym.lower()
            if key in seen:
                continue
            seen.add(key)
            if age_days is None or age_days > MAX_AGE_DAYS:
                continue
            if vol < MIN_VOL_USD or liq > MAX_LIQUIDITY_USD or liq < MIN_LIQUIDITY_USD:
                continue
            candidates.append({
                'source': 'dexscreener',
                'symbol': sym,
                'chain': p.get('chainId'),
                'dex': p.get('dexId'),
                'address': p.get('baseToken', {}).get('address'),
                'volume_24h': vol,
                'liquidity_usd': liq,
                'price_usd': p.get('priceUsd'),
                'age_days': age_days,
                'txns_24h': p.get('txns', {}).get('h24', {}).get('buys', 0) + p.get('txns', {}).get('h24', {}).get('sells', 0),
                'price_chg_24h': p.get('priceChange', {}).get('h24'),
            })

    # Birdeye trending
    for t in birdeye_trending(limit=50):
        sym = t.get('symbol') or t.get('address')
        if not sym:
            continue
        sym = f'{sym}/USDT'
        if sym.lower() in seen:
            continue
        seen.add(sym.lower())
        age_days = t.get('age_days')
        if age_days is None or age_days > MAX_AGE_DAYS:
            continue
        candidates.append({
            'source': 'birdeye',
            'symbol': sym,
            'chain': t.get('chain'),
            'dex': None,
            'address': t.get('address'),
            'volume_24h': float(t.get('volume_24h', 0) or 0),
            'liquidity_usd': float(t.get('liquidity', 0) or 0),
            'price_usd': t.get('price'),
            'age_days': age_days,
            'txns_24h': None,
            'price_chg_24h': t.get('price_change_24h'),
        })

    return candidates


def screen(candidates):
    """Tier 2 + 3: fetch data, run rules, rank."""
    results = []
    skipped = []
    total = len(candidates)
    for i, c in enumerate(candidates):
        sym = c['symbol']
        print(f'[{i+1}/{total}] {sym} ...')
        best = None
        for tf in ['1d', '4h']:
            try:
                df = fetch_ohlcv(sym, timeframe=tf, since_days=SAMPLE_DAYS)
                if len(df) < 200:
                    continue
                for rule_fn, rname in [
                    (rule_sma_cross_vol, 'sma_cross_vol'),
                    (rule_triple_rsi, 'triple_rsi'),
                    (rule_vol_breakout, 'vol_breakout'),
                ]:
                    sig = rule_fn(df)
                    score = score_rule(sig, df, tf, rname)
                    if best is None or score['sharpe'] > best['sharpe']:
                        best = {**score, 'timeframe': tf}
            except Exception as e:
                print(f'  {sym} {tf}: ERROR {e}')
                continue
        if best:
            out = dict(c)
            out['best_timeframe'] = best.get('timeframe')
            out['best_rule'] = best.get('rule')
            out['sharpe'] = best.get('sharpe')
            out['return'] = best.get('return')
            out['win_rate'] = best.get('win_rate')
            out['trades'] = best.get('trades')
            results.append(out)
        else:
            skipped.append(sym)
            print(f'  {sym}: skipped - no CEX OHLCV data')
    if skipped:
        print(f'\nSkipped {len(skipped)} tokens without CEX data: {skipped}')
    return results


def main():
    t0 = time.time()
    print('[dex_screener] discovering candidates ...')
    candidates = discover()
    print(f'[dex_screener] discovered {len(candidates)} candidates')
    if not candidates:
        print('No candidates found — try again later or widen filters')
        return
    ranked = screen(candidates)
    ranked.sort(key=lambda x: x.get('sharpe', -999), reverse=True)
    out_cols = ['symbol', 'chain', 'age_days', 'volume_24h', 'liquidity_usd',
                'best_rule', 'best_timeframe', 'sharpe', 'return', 'win_rate', 'trades']
    df = pd.DataFrame(ranked)
    out_path = OUT_DIR / 'volume_spike_rank.csv'
    df_cols = [c for c in out_cols if c in df.columns]
    df[df_cols].to_csv(out_path, index=False)
    print(f'\n[dex_screener] top candidates:')
    for r in ranked[:10]:
        print(f"  {r['symbol']} | {r.get('chain')} | {r.get('age_days')}d | ${r.get('volume_24h',0):,.0f} vol ${r.get('liquidity_usd',0):,.0f} liq | best={r.get('best_rule')} {r.get('best_timeframe')} sharpe={r.get('sharpe')} return={r.get('return')} wr={r.get('win_rate')} trades={r.get('trades')}")
    print(f'\nSaved {len(ranked)} ranked candidates to {out_path}')
    print(f'Elapsed: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
