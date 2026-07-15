#!/usr/bin/env python3
"""
Equities/ETF regime feature engineering.

Loads the equities_daily_*.csv collector output and produces regime flags
plus macro risk indicators aligned to the BTC 5-minute index.

Does NOT add equities prices as raw features; adds only regime/state flags
that are unlikely to leak future information if forward-filled with 1-bar lag.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / 'data'


def latest_equities_file() -> Path | None:
    files = sorted(DATA_DIR.glob('equities_daily_*.csv'))
    return files[-1] if files else None


def load_equities() -> pd.DataFrame:
    path = latest_equities_file()
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date').sort_index()
    # Normalize possible weird columns
    for c in list(df.columns):
        if not c.endswith('_close') and not c.endswith('_volume'):
            df = df.drop(columns=[c])
    return df


def _ma_signal(series: pd.Series, window: int = 20) -> pd.Series:
    ma = series.rolling(window, min_periods=1).mean()
    sig = (series - ma) / (ma.abs() + 1e-10)
    return sig


def build_equities_regime(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Returns macro regime dataframe aligned to `df_5m.index`.

    Columns:
      - eq_spy_signal, eq_gold_signal, eq_tlt_signal, eq_uup_signal, eq_vix_signal
      - eq_risk_on, eq_risk_off
      - eq_vix_spike
      - eq_rates_falling, eq_credit_stress
    """
    eq = load_equities()
    if eq.empty:
        empty = pd.DataFrame(index=df_5m.index)
        for c in ['eq_spy_signal', 'eq_gold_signal', 'eq_tlt_signal',
                  'eq_uup_signal', 'eq_vix_signal', 'eq_risk_on', 'eq_risk_off',
                  'eq_vix_spike', 'eq_rates_falling', 'eq_credit_stress']:
            empty[c] = 0
        return empty

    spy_close = eq.get('SPY_close')
    gld_close = eq.get('GLD_close')
    tlt_close = eq.get('TLT_close')
    uup_close = eq.get('UUP_close')
    vix_close = eq.get('^VIX_close') if '^VIX_close' in eq.columns else eq.get('VIX_close')
    hyg_close = eq.get('HYG_close')
    lqd_close = eq.get('LQD_close')

    regime = pd.DataFrame(index=df_5m.index)

    if spy_close is not None:
        s = _ma_signal(spy_close, 20)
        regime['eq_spy_signal'] = s.reindex(df_5m.index, method='ffill')
    else:
        regime['eq_spy_signal'] = 0.0

    if gld_close is not None:
        regime['eq_gold_signal'] = _ma_signal(gld_close, 20).reindex(df_5m.index, method='ffill')
    else:
        regime['eq_gold_signal'] = 0.0

    if tlt_close is not None:
        regime['eq_tlt_signal'] = _ma_signal(tlt_close, 20).reindex(df_5m.index, method='ffill')
    else:
        regime['eq_tlt_signal'] = 0.0

    if uup_close is not None:
        regime['eq_uup_signal'] = _ma_signal(uup_close, 20).reindex(df_5m.index, method='ffill')
    else:
        regime['eq_uup_signal'] = 0.0

    if vix_close is not None:
        vix_ma = vix_close.rolling(20, min_periods=1).mean()
        vix_spike = (vix_close / (vix_ma + 1e-10) - 1.0).clip(lower=0)
        regime['eq_vix_signal'] = vix_spike.reindex(df_5m.index, method='ffill')
        regime['eq_vix_spike'] = (regime['eq_vix_signal'] > 0.5).astype(int)
    else:
        regime['eq_vix_signal'] = 0.0
        regime['eq_vix_spike'] = 0

    if tlt_close is not None:
        r = tlt_close.pct_change(20)
        regime['eq_rates_falling'] = (r < 0).astype(int).reindex(df_5m.index, method='ffill')
    else:
        regime['eq_rates_falling'] = 0

    if hyg_close is not None and lqd_close is not None:
        spread_proxy = (hyg_close / (lqd_close + 1e-10)).rolling(20, min_periods=1).mean()
        credit_stress = ((hyg_close / (lqd_close + 1e-10)) - spread_proxy).clip(lower=0)
        regime['eq_credit_stress'] = credit_stress.reindex(df_5m.index, method='ffill').fillna(0)
    else:
        regime['eq_credit_stress'] = 0.0

    # Risk-on/risk-off based on SPY+VIX combo
    spy_up = (regime['eq_spy_signal'] > 0).astype(int)
    vix_calm = (regime['eq_vix_spike'] == 0).astype(int)
    regime['eq_risk_on'] = (spy_up & vix_calm).astype(int)
    regime['eq_risk_off'] = ((~spy_up.astype(bool)) | (~vix_calm.astype(bool))).astype(int)

    return regime.fillna(0)
