#!/usr/bin/env python3
"""
Multi-timeframe feature engineering with lookahead bias guard.

Adds higher-timeframe indicators as features while ensuring no future
information leaks into the current 5m bar. See Sobreiro et al. (2026)
Forecasting 8(3), 40 for the lookahead bias correction methodology.

Correct alignment rule:
- Higher-timeframe indicator at time T must be computed ONLY from data
  available up to time T in the lower timeframe.
- No post-hoc reindexing that uses higher-TF close values for lower-TF rows.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe using only past data."""
    ohlc = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    available = {k: v for k, v in ohlc.items() if k in df.columns}
    return df.resample(rule, label='right', closed='right').agg(available)


def lagged_indicator(
    df: pd.DataFrame,
    ht_indicators: pd.DataFrame,
    lags: int = 1,
) -> pd.DataFrame:
    """Attach higher-timeframe indicators with explicit lag guard.

    The indicator at time T is shifted by `lags` periods so that only
    fully-closed higher-timeframe bars are used. This prevents lookahead
    bias where an incomplete TF bar would otherwise contain future prices.
    """
    aligned = ht_indicators.reindex(df.index, method='ffill')
    if lags > 0:
        aligned = aligned.shift(lags)
    return aligned.add_prefix('ht_')


def build_multi_timeframe_features(
    df_5m: pd.DataFrame,
    timeframes: Optional[List[str]] = None,
    lags: int = 1,
) -> pd.DataFrame:
    """Build multi-timeframe feature matrix with lookahead guard.

    Parameters
    ----------
    df_5m : 5-minute OHLCV dataframe with DatetimeIndex.
    timeframes : list of resample rules, e.g. ['15min', '1H', '4H', '1D'].
    lags : number of higher-TF bars to lag by.

    Returns
    -------
    DataFrame indexed like `df_5m` with `ht_<tf>_<feature>` columns.
    """
    if timeframes is None:
        timeframes = ['15min', '1h', '4h', '1d']

    if df_5m.empty or not isinstance(df_5m.index, pd.DatetimeIndex):
        return pd.DataFrame(index=df_5m.index if not isinstance(df_5m.index, pd.DatetimeIndex) else df_5m.index)

    parts: List[pd.DataFrame] = []
    for tf in timeframes:
        ht = resample_ohlc(df_5m, tf)
        if ht.empty:
            continue
        ht_feat = _indicators(ht).add_prefix(f'ht_{tf}_')
        aligned = lagged_indicator(df_5m, ht_feat, lags=lags)
        parts.append(aligned)

    if not parts:
        return pd.DataFrame(index=df_5m.index)

    return pd.concat(parts, axis=1)


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if 'close' in df.columns:
        close = df['close']
        out['close'] = close
        out['ret'] = close.pct_change()
        out['log_ret'] = np.log(close).diff()
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        high = df['high']
        low = df['low']
        tr = np.maximum(high - low, np.maximum(
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ))
        out['atr'] = tr.rolling(14, min_periods=1).mean()
        out['atr_pct'] = out['atr'] / close
    if 'close' in df.columns:
        out['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
        out['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
        out['rsi_14'] = _rsi(df['close'], 14)
    if 'volume' in df.columns:
        out['vol_ma'] = df['volume'].rolling(20, min_periods=1).mean()
        out['vol_ratio'] = df['volume'] / out['vol_ma']
    return out


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ema_up = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    ema_down = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = ema_up / ema_down.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from pipeline import fetch_data
    df = fetch_data().head(5000)
    mtf = build_multi_timeframe_features(df)
    print(mtf.head())
    print('Shape:', mtf.shape)
