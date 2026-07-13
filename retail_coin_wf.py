#!/usr/bin/env python3
"""
Walk-forward validation: SMA crossover + volume across retail alts on 1d/4h/1h/5min timeframes.
Tests whether the edge exists on shorter TFs, not just 1d.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import vectorbt as vbt

try:
    import ccxt
except ImportError:
    raise SystemExit('pip install ccxt')

OUT_DIR = Path(__file__).parent / 'backtest_output'
OUT_DIR.mkdir(exist_ok=True)

INIT_CASH = 10_000
FEE_BP = 0.8
SLIP_BP = 0.5
TIMEFRAMES = ['1d', '4h', '1h', '5m']
SYMBOLS = ['DOGE/USDT', 'CRV/USDT', 'ENS/USDT', 'WIF/USDT', 'LRC/USDT', 'AUSDT/USDT', 'EDEN/USDT']
TRAIN_BARS = 500
TEST_BARS = 500
MIN_TEST = 100
MAX_SINCE_DAYS = {
    'DOGE/USDT': 2920,
    'CRV/USDT': 1460,
    'ENS/USDT': 1460,
    'WIF/USDT': 1460,
    'LRC/USDT': 1460,
    'AUSDT/USDT': 1460,
    'EDEN/USDT': 1095,
}


def fetch_ohlcv(symbol, timeframe, since_days=1460):
    ex = ccxt.binance({'enableRateLimit': True})
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
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    df.sort_index(inplace=True)
    return df


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
    stats['bars'] = int(len(df))
    stats['nonzero_signals'] = int((trade_arr != 0).sum())
    return stats


def make_splits(n, train=TRAIN_BARS, test=TEST_BARS, min_test=MIN_TEST):
    splits = []
    start = 0
    while start + train + test <= n:
        tr = list(range(start, start + train))
        te = list(range(start + train, start + train + test))
        if len(te) < min_test:
            break
        splits.append({'train_idx': tr, 'test_idx': te})
        start += test
    return splits


def main():
    t0 = time.time()
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'rule': 'sma_cross_vol_wf',
        'cases': {},
        'elapsed_sec': 0,
    }
    for sym in SYMBOLS:
        report['cases'][sym] = {}
        for tf in TIMEFRAMES:
            try:
                since_days = MAX_SINCE_DAYS.get(sym, 1460)
                df = fetch_ohlcv(sym, tf, since_days=since_days)
                if len(df) < TRAIN_BARS + TEST_BARS:
                    print(f'  {sym} {tf}: too short {len(df)}')
                    report['cases'][sym][tf] = {'error': f'too short {len(df)}'}
                    continue
                sig = rule_sma_cross_vol(df)
                splits = make_splits(len(df))
                folds = []
                for i, sp in enumerate(splits):
                    test_df = df.iloc[sp['test_idx']].copy()
                    test_sig = sig.iloc[sp['test_idx']].copy()
                    stats = eval_rule(test_sig, test_df, tf)
                    stats['fold'] = i
                    stats['test_rows'] = len(sp['test_idx'])
                    folds.append(stats)
                report['cases'][sym][tf] = {
                    'folds': folds,
                    'summary': {
                        'folds': len(folds),
                        'total_return_mean': float(np.mean([f['total_return'] for f in folds])),
                        'total_return_std': float(np.std([f['total_return'] for f in folds])),
                        'sharpe_mean': float(np.mean([f['sharpe_ratio'] for f in folds])),
                        'sharpe_std': float(np.std([f['sharpe_ratio'] for f in folds])),
                        'win_rate_mean': float(np.mean([f['win_rate'] for f in folds if f['win_rate'] is not None])) if any(f['win_rate'] is not None for f in folds) else None,
                        'gated_trades': int(sum(f['total_trades'] for f in folds)),
                    }
                }
                s = report['cases'][sym][tf]['summary']
                print(f'  {sym} {tf}: folds={s["folds"]}, ret={s["total_return_mean"]:.2%}±{s["total_return_std"]:.2%}, sharpe={s["sharpe_mean"]:.2f}±{s["sharpe_std"]:.2f}, trades={s["gated_trades"]}')
            except Exception as e:
                print(f'  {sym} {tf}: ERROR {e}')
                report['cases'][sym][tf] = {'error': str(e)}
    report['elapsed_sec'] = round(time.time() - t0, 1)
    out = OUT_DIR / 'retail_coin_wf_report.json'
    out.write_text(json.dumps(report, indent=2))
    print(f'\nSaved {out}')
    print(f'Elapsed: {report["elapsed_sec"]}s')


if __name__ == '__main__':
    main()
