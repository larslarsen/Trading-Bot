#!/usr/bin/env python3
"""
MA crossover + volume across multiple timeframes and pairs.
Ordered by data efficiency: 1d → 4h → 1h → 5m.
If 1d/4h fails, simpler rules are dead on these assets.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import time
import numpy as np
import pandas as pd
import vectorbt as vbt

OUT_DIR = Path(__file__).parent / 'backtest_output'
OUT_DIR.mkdir(exist_ok=True)
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT',
           'DOGEUSDT', 'XRPUSDT', 'BNBUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT', 'UNIUSDT',
           'AAVEUSDT', 'MKRUSDT']
TIMEFRAMES = ['1d', '4h', '1h', '5min']
FEE_BP = 0.8
SLIP_BP = 0.5
INIT_CASH = 10_000


def load_data():
    df = pd.read_csv('multi_5m.csv')
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[df['symbol'].isin(SYMBOLS)].copy()
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df.sort_values(['symbol', 'ts'], inplace=True)
    frames = []
    for sym, g in df.groupby('symbol'):
        g = g.set_index('ts').sort_index()
        o = g['open'].resample('5min').first()
        h = g['high'].resample('5min').max()
        l = g['low'].resample('5min').min()
        c = g['close'].resample('5min').last()
        v = g['volume'].resample('5min').sum()
        base = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}, index=c.index)
        base.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        base['symbol'] = sym
        frames.append(base)
    return pd.concat(frames)


def resample(df, timeframe):
    g = df.copy()
    if timeframe == '5min':
        return g
    o = g['open'].resample(timeframe).first()
    h = g['high'].resample(timeframe).max()
    l = g['low'].resample(timeframe).min()
    c = g['close'].resample(timeframe).last()
    v = g['volume'].resample(timeframe).sum()
    out = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}, index=c.index)
    out.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    out['symbol'] = g['symbol'].iloc[0]
    return out


def rule_sma_cross_vol(df):
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


def vectorbt_run(price, entries, exits, short_entries, short_exits, freq):
    pf = vbt.Portfolio.from_signals(
        price, entries=entries, exits=exits,
        short_entries=short_entries, short_exits=short_exits,
        freq=freq, init_cash=INIT_CASH, size=100, size_type='value',
        fees=FEE_BP / 10_000.0, slippage=SLIP_BP / 10_000.0,
    )
    trades = pf.trades
    return {
        'total_return': float(pf.total_return()),
        'sharpe_ratio': float(pf.sharpe_ratio()),
        'max_drawdown': float(pf.max_drawdown()),
        'win_rate': float(trades.win_rate()) if trades.count() else None,
        'total_trades': int(trades.count()),
        'final_equity': float(pf.value().iloc[-1]) if len(pf.value()) else None,
    }


def eval_rule(signal, df, freq):
    trade = signal.reindex(df.index).fillna(0)
    price = df['close'].values.astype(float)
    trade_arr = trade.values.astype(int)
    entries_long = (trade_arr == 1)
    entries_short = (trade_arr == -1)
    prev = np.roll(trade_arr, 1)
    prev[0] = 0
    exits = (prev != 0) & (trade_arr == 0)
    short_exits = (prev == -1) & (trade_arr == 0)
    stats = vectorbt_run(price, entries_long, exits, entries_short, short_exits, freq)
    stats['nonzero_trades'] = int((trade_arr != 0).sum())
    stats['bars'] = int(len(df))
    return stats


def main():
    t0 = time.time()
    full = load_data()
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'rule': 'sma_cross_vol',
        'cases': {},
        'elapsed_sec': 0,
    }
    for tf in TIMEFRAMES:
        report['cases'][tf] = {}
        for sym, df in full.groupby('symbol'):
            if len(df) < 100:
                continue
            df_tf = resample(df, tf)
            if len(df_tf) < 200:
                continue
            sig = rule_sma_cross_vol(df_tf)
            stats = eval_rule(sig, df_tf, tf)
            stats['nonzero_signals'] = int((sig != 0).sum())
            report['cases'][tf][sym] = stats
    report['elapsed_sec'] = round(time.time() - t0, 1)
    out = OUT_DIR / 'sma_cross_timeframe_report.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('[MA] saved', out)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
