#!/usr/bin/env python3
"""Batch pull + scan TradingView watchlist for 1d simple-rule edge on crypto pairs."""
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import ccxt

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
OUT = ROOT / 'backtest_output'
OUT.mkdir(exist_ok=True)
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)

EXCHANGES = {
    'binance': ccxt.binance({'enableRateLimit': True}),
    'mexc': ccxt.mexc({'enableRateLimit': True}),
}

# Mapping from TradingView watchlist -> (exchange, symbol)
# Prioritizing Bybit public klines first, then MEXC fallback.
TV_CRYPTO = [
    ("ADA/USDT",  "binance"),
    ("ALGO/USDT", "binance"),
    ("APE/USDT",  "binance"),
    ("APX/USDT",  "binance"),
    ("ARB/USDT",  "binance"),
    ("ARVN/USDT","mexc"),
    ("AVAX/USDT","binance"),
    ("AXL/USDT",  "binance"),
    ("BIGTIME/USDT","mexc"),
    ("BLUB/USDT", "mexc"),
    ("BTBT/USDT", "binance"),
    ("BUCKAZOIDS/USDT","mexc"),
    ("SOL/USDT",  "binance"),
]

# Confirmed winners from prior walk-forward
WINNERS = [
    ("DOGE/USDT", "binance"),
    ("ENS/USDT",  "binance"),
    ("LRC/USDT",  "binance"),
    ("CRV/USDT",  "binance"),
    ("EDEN/USDT","mexc"),
]

PAIRS = TV_CRYPTO + WINNERS
TIMEFRAME = '1d'
SINCE_DAYS = 730  # 2 years
FETCH_LIMIT = 1000

FEE_BP = 0.8
SLIP_BP = 0.5


def ts_days_ago(days):
    return int((datetime.now(timezone.utc).timestamp() - days * 86400) * 1000)


def fetch_ohlcv(symbol, exchange_name):
    since = ts_days_ago(SINCE_DAYS)
    ex = EXCHANGES[exchange_name]
    all_bars = []
    while True:
        bars = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=FETCH_LIMIT)
        if not bars:
            break
        all_bars.extend(bars)
        since = bars[-1][0] + 1
        if len(bars) < FETCH_LIMIT:
            break
        time.sleep(0.1)
    if not all_bars:
        return pd.DataFrame()
    df = pd.DataFrame(all_bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    return df


# ---- Simple rule library (cost-adjusted) ----


def sma_cross_vol(df, fast=20, slow=50):
    if len(df) < slow:
        return pd.Series([0]*len(df), index=df.index), {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':0}
    sma_f = df['close'].rolling(fast).mean()
    sma_s = df['close'].rolling(slow).mean()
    signal = (sma_f > sma_s).astype(int)
    signal[signal.diff() == 0] = 0
    pos = signal.replace(0, np.nan).ffill().fillna(0)
    ret = df['close'].pct_change()
    cost = (FEE_BP + SLIP_BP) / 10000.0
    strat = pos.shift(1) * ret - cost * np.abs(pos.diff().fillna(0))
    trades = int(np.abs(pos.diff()).sum())
    if strat.std() == 0 or trades < 2:
        return pos, {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':trades}
    sharpe = strat.mean() / strat.std() * np.sqrt(365)
    wr = (strat > 0).mean()
    return pos, {'sharpe': round(float(sharpe),3), 'return': round(float(strat.sum()*100),2), 'wr': round(float(wr*100),1), 'trades': trades}


def triple_rsi(df, p1=14, p2=28, p3=56):
    if len(df) < max(p1,p2,p3):
        return pd.Series([0]*len(df), index=df.index), {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':0}
    r1 = df['close'].pct_change().rolling(p1).apply(lambda x: (x>0).mean(), raw=True)
    r2 = df['close'].pct_change().rolling(p2).apply(lambda x: (x>0).mean(), raw=True)
    r3 = df['close'].pct_change().rolling(p3).apply(lambda x: (x>0).mean(), raw=True)
    sig = ((r1 > 0.5) & (r2 > 0.5) & (r3 > 0.5)).astype(int)
    pos = sig.replace(0, np.nan).ffill().fillna(0)
    ret = df['close'].pct_change()
    cost = (FEE_BP + SLIP_BP) / 10000.0
    strat = pos.shift(1) * ret - cost * np.abs(pos.diff().fillna(0))
    trades = int(np.abs(pos.diff()).sum())
    if strat.std() == 0 or trades < 2:
        return pos, {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':trades}
    sharpe = strat.mean() / strat.std() * np.sqrt(365)
    wr = (strat > 0).mean()
    return pos, {'sharpe': round(float(sharpe),3), 'return': round(float(strat.sum()*100),2), 'wr': round(float(wr*100),1), 'trades': trades}


def donchian(df, lookback=20):
    if len(df) < lookback:
        return pd.Series([0]*len(df), index=df.index), {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':0}
    roll_high = df['high'].rolling(lookback).max()
    roll_low = df['low'].rolling(lookback).min()
    sig = pd.Series(0, index=df.index)
    sig[df['close'] > roll_high.shift(1)] = 1
    sig[df['close'] < roll_low.shift(1)] = -1
    pos = sig.replace(0, np.nan).ffill().fillna(0)
    ret = df['close'].pct_change()
    cost = (FEE_BP + SLIP_BP) / 10000.0
    strat = pos.shift(1) * ret - cost * np.abs(pos.diff().fillna(0))
    trades = int(np.abs(pos.diff()).sum())
    if strat.std() == 0 or trades < 2:
        return pos, {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':trades}
    sharpe = strat.mean() / strat.std() * np.sqrt(365)
    wr = (strat > 0).mean()
    return pos, {'sharpe': round(float(sharpe),3), 'return': round(float(strat.sum()*100),2), 'wr': round(float(wr*100),1), 'trades': trades}


def momentum(df, lookback=10):
    if len(df) < lookback + 1:
        return pd.Series([0]*len(df), index=df.index), {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':0}
    mom = df['close'].pct_change(lookback)
    sig = (mom > 0).astype(int)
    pos = sig.replace(0, np.nan).ffill().fillna(0)
    ret = df['close'].pct_change()
    cost = (FEE_BP + SLIP_BP) / 10000.0
    strat = pos.shift(1) * ret - cost * np.abs(pos.diff().fillna(0))
    trades = int(np.abs(pos.diff()).sum())
    if strat.std() == 0 or trades < 2:
        return pos, {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':trades}
    sharpe = strat.mean() / strat.std() * np.sqrt(365)
    wr = (strat > 0).mean()
    return pos, {'sharpe': round(float(sharpe),3), 'return': round(float(strat.sum()*100),2), 'wr': round(float(wr*100),1), 'trades': trades}


def vol_breakout(df, lookback=20, mult=1.5):
    if len(df) < lookback:
        return pd.Series([0]*len(df), index=df.index), {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':0}
    avg_vol = df['volume'].rolling(lookback).mean()
    sig = (df['volume'] > avg_vol.shift(1) * mult).astype(int)
    pos = sig.replace(0, np.nan).ffill().fillna(0)
    ret = df['close'].pct_change()
    cost = (FEE_BP + SLIP_BP) / 10000.0
    strat = pos.shift(1) * ret - cost * np.abs(pos.diff().fillna(0))
    trades = int(np.abs(pos.diff()).sum())
    if strat.std() == 0 or trades < 2:
        return pos, {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':trades}
    sharpe = strat.mean() / strat.std() * np.sqrt(365)
    wr = (strat > 0).mean()
    return pos, {'sharpe': round(float(sharpe),3), 'return': round(float(strat.sum()*100),2), 'wr': round(float(wr*100),1), 'trades': trades}


def macd(df, fast=12, slow=26, signal=9):
    if len(df) < slow + signal:
        return pd.Series([0]*len(df), index=df.index), {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':0}
    ema_f = df['close'].ewm(span=fast, adjust=False).mean()
    ema_s = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    pos_series = (macd_line > sig_line).astype(int)
    pos = pos_series.replace(0, np.nan).ffill().fillna(0)
    ret = df['close'].pct_change()
    cost = (FEE_BP + SLIP_BP) / 10000.0
    strat = pos.shift(1) * ret - cost * np.abs(pos.diff().fillna(0))
    trades = int(np.abs(pos.diff()).sum())
    if strat.std() == 0 or trades < 2:
        return pos, {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':trades}
    sharpe = strat.mean() / strat.std() * np.sqrt(365)
    wr = (strat > 0).mean()
    return pos, {'sharpe': round(float(sharpe),3), 'return': round(float(strat.sum()*100),2), 'wr': round(float(wr*100),1), 'trades': trades}


RULES = {
    'sma_cross_vol': sma_cross_vol,
    'triple_rsi': triple_rsi,
    'vol_breakout': vol_breakout,
    'donchian': donchian,
    'macd': macd,
    'momentum': momentum,
}


def score_pair(symbol, exchange_name):
    safe = symbol.replace('/', '_') + '_' + exchange_name
    cache = DATA / f'{safe}_1d.csv'
    if cache.exists() and cache.stat().st_size > 100:
        df = pd.read_csv(cache, parse_dates=['ts'])
    else:
        df = fetch_ohlcv(symbol, exchange_name)
        if df.empty:
            return None
        df.to_csv(cache, index=False)
    
    df = df.sort_values('ts').reset_index(drop=True)
    rows = []
    for rule_name, rule_fn in RULES.items():
        _, stats = rule_fn(df)
        rows.append({
            'symbol': symbol,
            'exchange': exchange_name,
            'rule': rule_name,
            'timeframe': '1d',
            'sharpe': stats['sharpe'],
            'return_pct': stats['return'],
            'winrate_pct': stats['wr'],
            'trades': stats['trades'],
            'bars': len(df),
        })
    return rows


def main():
    t0 = time.time()
    print(f'[{datetime.now(timezone.utc).isoformat()}] batch scan starting')
    print(f'Pairs: {len(PAIRS)}')
    
    all_results = []
    for i, (symbol, exchange_name) in enumerate(PAIRS):
        print(f'[{i+1}/{len(PAIRS)}] {symbol}@{exchange_name} ...', end='', flush=True)
        try:
            rows = score_pair(symbol, exchange_name)
            if rows:
                all_results.extend(rows)
                best = max(rows, key=lambda r: r['sharpe'])
                print(f" {len(rows)} rules, best={best['rule']} Sharpe={best['sharpe']}")
            else:
                print(' no data')
        except Exception as e:
            print(f' error: {e}')
        time.sleep(0.2)
    
    if not all_results:
        print('No results collected')
        return
    
    df = pd.DataFrame(all_results)
    report_path = OUT / f'watchlist_scan_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(report_path, index=False)
    
    # Ranking
    best = df.sort_values('sharpe', ascending=False).groupby('symbol').first().reset_index()
    best = best.sort_values('sharpe', ascending=False).head(20)
    
    print(f'\n--- Top pairs by best rule Sharpe ---')
    print(best[['symbol','rule','timeframe','sharpe','return_pct','winrate_pct','trades']].to_string(index=False))
    
    print(f'\nWrote {report_path}')
    print(f'Completed in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
