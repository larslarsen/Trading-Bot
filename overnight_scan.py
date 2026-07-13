#!/usr/bin/env python3
"""Overnight extended watchlist scan: max history, proper multi-fold WF, fixed rule logic."""
import json, time, warnings
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

EXCHANGE = ccxt.binance({'enableRateLimit': True})
FALLBACK = ccxt.mexc({'enableRateLimit': True})
TIMEFRAME = '1d'
FETCH_LIMIT = 1000

FEE_BP = 0.8
SLIP_BP = 0.5

# Verified pairs from TradingView watchlist + confirmed winners
PAIRS = [
    ('ADA/USDT', 'binance'),
    ('ALGO/USDT', 'binance'),
    ('APE/USDT', 'binance'),
    ('ARB/USDT', 'binance'),
    ('AVAX/USDT', 'binance'),
    ('AXL/USDT', 'binance'),
    ('SOL/USDT', 'binance'),
    ('DOGE/USDT', 'binance'),
    ('ENS/USDT', 'binance'),
    ('LRC/USDT', 'binance'),
    ('CRV/USDT', 'binance'),
    ('EDEN/USDT', 'mexc'),
]

def fetch_max_ohlcv(symbol, exchange_name):
    since = None
    ex = EXCHANGE if exchange_name == 'binance' else FALLBACK
    all_bars = []
    pages = 0
    while pages < 40:
        try:
            bars = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=FETCH_LIMIT)
        except ccxt.BadSymbol as e:
            return pd.DataFrame(), str(e)
        except Exception as e:
            return pd.DataFrame(), str(e)
        if not bars:
            break
        all_bars.extend(bars)
        since = bars[-1][0] + 1
        pages += 1
        if len(bars) < FETCH_LIMIT:
            break
        time.sleep(0.2)
    if not all_bars:
        return pd.DataFrame(), 'empty'
    df = pd.DataFrame(all_bars, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    return df, 'ok'


# ---- Fixed rule implementations matching simple_rules_compare.py logic ----

def _cost():
    return (FEE_BP + SLIP_BP) / 10000.0

def _eval(price, pos):
    ret = pd.Series(price).pct_change().values
    cost = _cost()
    pos = np.asarray(pos, dtype=float)
    strat = pos[:-1] * ret[1:] - cost * np.abs(np.diff(pos))
    trades = int(np.nansum(np.abs(np.diff(pos))))
    if len(strat) == 0 or np.nanstd(strat) == 0 or trades < 2:
        return {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':trades}
    sharpe = np.nanmean(strat) / np.nanstd(strat) * np.sqrt(365)
    wr = float(np.mean(strat[strat != 0] > 0)) if np.any(strat != 0) else 0.0
    return {
        'sharpe': round(float(sharpe),3),
        'return': round(float(np.nansum(strat)*100),2),
        'wr': round(float(wr*100),1),
        'trades': trades,
    }

def rule_sma_cross_vol(df):
    if len(df) < 50:
        return pd.Series([0]*len(df), index=df.index), _stats()
    sma_fast = df['close'].rolling(20).mean()
    sma_slow = df['close'].rolling(50).mean()
    cross_up = (sma_fast > sma_slow).astype(bool)
    vol = df['volume']
    vol_ma = vol.rolling(20).mean()
    vol_ok = (vol > 1.2 * vol_ma).astype(bool)
    sig = pd.Series(0, index=df.index)
    sig.loc[cross_up & vol_ok] = 1
    sig.loc[(~cross_up) & vol_ok] = -1
    return sig, _eval(df['close'].values, sig)

def rule_triple_rsi(df):
    if len(df) < 56:
        return pd.Series([0]*len(df), index=df.index), _stats()
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
    sig = pd.Series(0, index=df.index)
    sig.loc[(rsi > 55) & vol_ok] = 1
    sig.loc[(rsi < 45) & vol_ok] = -1
    return sig, _eval(df['close'].values, sig)

def rule_vol_breakout(df):
    if len(df) < 20:
        return pd.Series([0]*len(df), index=df.index), _stats()
    high_n = df['high'].rolling(20).max()
    low_n = df['low'].rolling(20).min()
    vol = df['volume']
    vol_ma = vol.rolling(20).mean()
    vol_ok = (vol > 1.5 * vol_ma).astype(bool)
    sig = pd.Series(0, index=df.index)
    sig.loc[(df['close'] > high_n.shift(1)) & vol_ok] = 1
    sig.loc[(df['close'] < low_n.shift(1)) & vol_ok] = -1
    return sig, _eval(df['close'].values, sig)

def rule_donchian(df):
    if len(df) < 20:
        return pd.Series([0]*len(df), index=df.index), _stats()
    high_n = df['high'].rolling(20).max()
    low_n = df['low'].rolling(20).min()
    sig = pd.Series(0, index=df.index)
    sig.loc[df['close'] > high_n.shift(1)] = 1
    sig.loc[df['close'] < low_n.shift(1)] = -1
    return sig, _eval(df['close'].values, sig)

def rule_macd(df):
    if len(df) < 35:
        return pd.Series([0]*len(df), index=df.index), _stats()
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    cross_up = (macd_line > signal_line).astype(bool)
    sig = pd.Series(0, index=df.index)
    sig.loc[cross_up] = 1
    sig.loc[~cross_up] = -1
    return sig, _eval(df['close'].values, sig)

def rule_momentum(df):
    if len(df) < 14:
        return pd.Series([0]*len(df), index=df.index), _stats()
    mom = df['close'].pct_change(14)
    sig = pd.Series(0, index=df.index)
    sig.loc[mom > 0] = 1
    sig.loc[mom < 0] = -1
    return sig, _eval(df['close'].values, sig)

def _stats():
    return {'sharpe':0.0,'return':0.0,'wr':0.0,'trades':0}

RULES = {
    'sma_cross_vol': rule_sma_cross_vol,
    'triple_rsi': rule_triple_rsi,
    'vol_breakout': rule_vol_breakout,
    'donchian': rule_donchian,
    'macd': rule_macd,
    'momentum': rule_momentum,
}


def overlapping_wf(df):
    n = len(df)
    # Adapt windows to data length: target ~3-6 folds
    if n >= 2000:
        train, test, step = 500, 500, 250
    elif n >= 1200:
        train, test, step = 250, 250, 125
    elif n >= 800:
        train, test, step = 200, 200, 100
    else:
        train, test, step = 125, 125, 62
    folds = []
    start = 0
    while start + train + test <= n:
        folds.append((df.iloc[start:start+train].copy(), df.iloc[start+train:start+train+test].copy()))
        start += step
    return folds, train, test, step


def main():
    t0 = time.time()
    print(f'[{datetime.now(timezone.utc).isoformat()}] overnight scan starting')
    report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'pairs': []}
    best_overall = []

    for i, (symbol, ex_name) in enumerate(PAIRS):
        print(f'\n[{i+1}/{len(PAIRS)}] {symbol}@{ex_name} ...', end='', flush=True)
        try:
            df, status = fetch_max_ohlcv(symbol, ex_name)
            if df.empty:
                print(f' skip: {status}')
                continue
            df = df.sort_values('ts').reset_index(drop=True)
            cache = DATA / f'{symbol.replace("/","_")}_1d_max.csv'
            df.to_csv(cache, index=False)
            folds, train, test, step = overlapping_wf(df)
            if not folds:
                print(f' {len(df)} bars, 0 folds, skip')
                continue
            print(f' {len(df)} bars, {len(folds)} folds (t={train}/test={test}/step={step})')
            pair_result = {'symbol': symbol, 'bars': len(df), 'folds': len(folds), 'rules': {}}
            for rule_name, rule_fn in RULES.items():
                fold_sharpes, fold_returns, fold_wrs, fold_trades = [], [], [], []
                for train_df, test_df in folds:
                    _, stats = rule_fn(test_df)
                    fold_sharpes.append(stats['sharpe'])
                    fold_returns.append(stats['return'])
                    fold_wrs.append(stats['wr'])
                    fold_trades.append(stats['trades'])
                mean_sharpe = float(np.nanmean(fold_sharpes)) if fold_sharpes else 0.0
                std_sharpe = float(np.nanstd(fold_sharpes)) if fold_sharpes else 0.0
                mean_return = float(np.nanmean(fold_returns)) if fold_returns else 0.0
                mean_wr = float(np.nanmean(fold_wrs)) if fold_wrs else 0.0
                total_trades = int(np.nansum(fold_trades)) if fold_trades else 0
                pair_result['rules'][rule_name] = {
                    'mean_sharpe': round(mean_sharpe,3),
                    'std_sharpe': round(std_sharpe,3),
                    'mean_return_pct': round(mean_return,2),
                    'mean_winrate_pct': round(mean_wr,1),
                    'total_trades': total_trades,
                    'folds': len(folds),
                    'fold_sharpes': [round(float(x),3) for x in fold_sharpes],
                }
                print(f'    {rule_name}: Sharpe={mean_sharpe:.3f} return={mean_return:.2f}% wr={mean_wr:.1f}% trades={total_trades}')
                best_overall.append((symbol, rule_name, mean_sharpe, mean_return, mean_wr, total_trades, len(folds)))
        except Exception as e:
            print(f' ERR: {e}')
        time.sleep(0.3)

    report['best_overall'] = [
        {'symbol': s, 'rule': r, 'sharpe': sh, 'return_pct': ret, 'winrate_pct': wr, 'trades': tr, 'folds': f}
        for s,r,sh,ret,wr,tr,f in best_overall
    ]
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    report_path = OUT / f'overnight_wf_{ts}.json'
    report_path.write_text(json.dumps(report, indent=2))

    top = sorted(best_overall, key=lambda x: x[2], reverse=True)[:20]
    print(f'\n=== Top 20 rule/pairs by mean Sharpe ===')
    for s,r,sh,ret,wr,tr,f in top:
        print(f'{s:15s} {r:15s} Sharpe={sh:6.3f} return={ret:7.2f}% wr={wr:5.1f}% trades={tr:3d} folds={f}')
    print(f'\nWrote {report_path}')
    print(f'Overnight scan complete in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
