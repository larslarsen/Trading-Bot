#!/usr/bin/env python3
"""
Multi-asset cross features for BTC/USDT prediction.
Uses the universal multi-asset CSV schema: ts, symbol, open, high, low, close, volume
"""
import pandas as pd
from pathlib import Path

PAIRS = ['ETHUSDT', 'SOLUSDT', 'BTCUSDT']
LAGS = [1, 3, 6, 12, 24]  # bars ahead for return lookback on 5m series


def load_multi(path):
    p = Path(path)
    if not p.exists():
        print(f'[multi] {path} not found, skipping cross-asset features')
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=['ts'])
    if 'ts' in df.columns and df['ts'].dt.tz is None:
        df['ts'] = df['ts'].dt.tz_localize('UTC')
    df = df.rename(columns={'ts': 'timestamp'})
    df = df[df['symbol'].isin(PAIRS)].copy()
    return df


def make_cross_features(df_btc, df_multi) -> pd.DataFrame:
    if df_multi is None or df_multi.empty or 'symbol' not in df_multi.columns:
        return df_btc
    syms = sorted(df_multi['symbol'].unique())
    if 'BTCUSDT' not in syms or len(syms) < 2:
        print(f'[multi] insufficient symbols found: {syms}')
        return df_btc

    btc = df_multi[df_multi['symbol'] == 'BTCUSDT'][['timestamp', 'close', 'volume']].copy()
    btc = btc.rename(columns={'close': 'btc_close', 'volume': 'btc_volume'})
    others = [s for s in syms if s != 'BTCUSDT']

    cross = btc.copy()
    for sym in others:
        oth = df_multi[df_multi['symbol'] == sym][['timestamp', 'close', 'volume']].copy()
        oth = oth.rename(columns={'close': f'{sym}_close', 'volume': f'{sym}_volume'})
        cross = pd.merge(cross, oth, on='timestamp', how='inner')
        cross[f'{sym}_returns'] = cross[f'{sym}_close'].pct_change(1)
        cross[f'{sym}_btc_ratio'] = cross[f'{sym}_close'] / cross['btc_close']
        cross[f'{sym}_btc_ratio_chg'] = cross[f'{sym}_btc_ratio'].pct_change(1)
        cross[f'{sym}_volume_z'] = (cross[f'{sym}_volume'] - cross[f'{sym}_volume'].rolling(24).mean()) / (cross[f'{sym}_volume'].rolling(24).std() + 1e-9)
        for l in [6, 12, 24, 48]:
            cross[f'{sym}_btc_ret_rel_{l}'] = cross[f'{sym}_close'].pct_change(l) - cross['btc_close'].pct_change(l)
            cross[f'btc_corr_{sym}_{l}'] = cross['btc_close'].pct_change(1).rolling(l).corr(cross[f'{sym}_close'].pct_change(1))

    keep = ['timestamp'] + [c for c in cross.columns if c not in [
        'timestamp', 'btc_close', 'btc_volume'] + [f'{s}_close' for s in others] + [f'{s}_volume' for s in others]]
    cross = cross[keep].copy()

    if 'timestamp' in df_btc.columns:
        df_btc = pd.merge(df_btc, cross, on='timestamp', how='left')
    return df_btc


def add_multi_asset_features(df, multi_path='multi_5m.csv'):
    df_multi = load_multi(multi_path)
    df = make_cross_features(df, df_multi)
    return df
