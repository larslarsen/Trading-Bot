#!/usr/bin/env python3
"""Multi-asset cross features for single-pair ML models.

Builds cross-asset context (relative returns, ratios, rolling correlation vs
the BTC benchmark) from a universal multi-asset CSV (ts, symbol, open, high,
low, close, volume). The file is the source of truth for which assets are
available; BTCUSDT must be present as the benchmark base.
"""
import pandas as pd
from pathlib import Path

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
    # Accept whatever symbols the file carries (BTCUSDT is required as the
    # benchmark base; see make_cross_features). Do NOT whitelist — the file is
    # the source of truth for which cross-assets are available for the era.
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

    # Join cross-asset features onto the target frame by its DatetimeIndex.
    # The target's time is the index (named 'ts'), NOT a 'timestamp' column, so
    # a column-name merge would silently skip and drop all cross features.
    cross = cross.set_index('timestamp')
    if not isinstance(df_btc.index, pd.DatetimeIndex):
        # Defensive: if caller passed a 'timestamp' column instead of an index,
        # align on it rather than dropping the features.
        if 'timestamp' in df_btc.columns:
            df_btc = df_btc.set_index('timestamp')
        else:
            print('[multi] target frame has no time index/column; skipping cross features')
            return df_btc
    return df_btc.join(cross, how='left')


def add_multi_asset_features(df, multi_path='multi_5m.csv'):
    """Attach cross-asset features. Returns (df, added_cols) where added_cols
    is the list of new column names (so callers can include them in the model
    feature set — they are generated dynamically and are NOT in ALL_FEATURES).
    """
    cols_before = set(df.columns)
    df_multi = load_multi(multi_path)
    df = make_cross_features(df, df_multi)
    added = [c for c in df.columns if c not in cols_before]
    return df, added
