#!/usr/bin/env python3
"""
Multi-asset 1h backtest: Triple RSI + volume across available pairs.

Uses local multi_5m.csv resampled to 1h, so no network fetch needed.
Tests whether retail-driven inefficiency exists in altcoins vs BTC.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

OUT_DIR = Path(__file__).parent / 'backtest_output'
OUT_DIR.mkdir(exist_ok=True)
INIT_CASH = 10_000
FEE_BP = 0.8
SLIP_BP = 0.5
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']


def load_resample(path='multi_5m.csv'):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[df['symbol'].isin(SYMBOLS)].copy()
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df.sort_values(['symbol', 'ts'], inplace=True)
    # resample 5m -> 1h per symbol
    frames = []
    for sym, g in df.groupby('symbol'):
        g = g.set_index('ts').sort_index()
        o = g['open'].resample('1h').first()
        h = g['high'].resample('1h').max()
        l = g['low'].resample('1h').min()
        c = g['close'].resample('1h').last()
        v = g['volume'].resample('1h').sum()
        out = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}, index=c.index)
        out.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        out['symbol'] = sym
        frames.append(out)
    return pd.concat(frames)


def rule_triple_rsi(df):
    close = df['close']
    vol = df['volume']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi_4h = rsi.rolling(4).mean()
    rsi_1d = rsi.rolling(24).mean()

    oversold = (rsi < 30) & (rsi_4h < 30) & (rsi_1d < 30)
    overbought = (rsi > 70) & (rsi_4h > 70) & (rsi_1d > 70)

    vol_ma = vol.rolling(20).mean()
    vol_ok = vol > 1.1 * vol_ma

    signal = pd.Series(0, index=df.index)
    signal[oversold & vol_ok] = 1
    signal[overbought & vol_ok] = -1
    return signal


def vectorbt_run(price, entries, exits, short_entries, short_exits):
    pf = vbt.Portfolio.from_signals(
        price, entries=entries, exits=exits,
        short_entries=short_entries, short_exits=short_exits,
        freq='1h', init_cash=INIT_CASH, size=100, size_type='value',
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


def eval_rule(signal, df_test):
    trade = signal.reindex(df_test.index).fillna(0)
    price = df_test['close'].values.astype(float)
    trade_arr = trade.values.astype(int)
    entries_long = (trade_arr == 1)
    entries_short = (trade_arr == -1)
    prev = np.roll(trade_arr, 1)
    prev[0] = 0
    exits = (prev != 0) & (trade_arr == 0)
    short_exits = (prev == -1) & (trade_arr == 0)
    return vectorbt_run(price, entries_long, exits, entries_short, short_exits)


def main():
    t0 = time.time()
    data = load_resample()
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'timeframe': '1h',
        'fee_bp': FEE_BP,
        'slip_bp': SLIP_BP,
        'cases': {},
        'elapsed_sec': 0,
    }
    for sym, df in data.groupby('symbol'):
        if len(df) < 200:
            continue
        sig = rule_triple_rsi(df)
        stats = eval_rule(sig, df)
        stats['nonzero_trades'] = int((sig != 0).sum())
        stats['bars'] = int(len(df))
        report['cases'][sym] = stats
    report['elapsed_sec'] = round(time.time() - t0, 1)
    out = OUT_DIR / 'multi_asset_1h_report.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('[MULTI] saved', out)


if __name__ == '__main__':
    import time
    import warnings
    warnings.filterwarnings('ignore')
    main()
