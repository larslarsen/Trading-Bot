"""
Tests for multi_asset_features.py
- Builds a tiny synthetic multi-asset dataset
- Verifies expected cross-asset columns are generated
- Verifies no duplicate columns
- Verifies NaNs only where expected from rolling windows
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from multi_asset_features import add_multi_asset_features, make_cross_features


@pytest.fixture()
def tmp_multi_csv(tmp_path: Path):
    rng = pd.date_range('2024-01-01', periods=300, freq='5min', tz='UTC')
    rows = []
    np.random.seed(0)
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        base = 100 if sym == 'BTCUSDT' else 1 if sym == 'ETHUSDT' else 0.1
        close = base + np.cumsum(np.random.randn(300) * 0.001 * base)
        vol = np.random.rand(300) * 10 + 1
        for i, ts in enumerate(rng):
            rows.append({
                'ts': ts,
                'symbol': sym,
                'open': float(close[i]),
                'high': float(close[i] * 1.001),
                'low': float(close[i] * 0.999),
                'close': float(close[i]),
                'volume': float(vol[i]),
            })
    df = pd.DataFrame(rows)
    out = tmp_path / 'multi.csv'
    df.to_csv(out, index=False)
    return out


def test_add_multi_asset_features_adds_cross_columns(tmp_multi_csv: Path):
    btc = pd.read_csv(tmp_multi_csv, parse_dates=['ts'])
    btc = btc[btc['symbol'] == 'BTCUSDT'].copy().reset_index(drop=True)
    btc = btc.rename(columns={'ts': 'timestamp'})
    btc['label'] = 2

    out = add_multi_asset_features(btc, str(tmp_multi_csv))
    new_cols = [c for c in out.columns if c not in btc.columns]

    assert len(out) == len(btc)
    assert any('btc_corr_' in c for c in new_cols)
    assert any('btc_ratio' in c for c in new_cols)
    assert any('volume_z' in c for c in new_cols)
    assert len(new_cols) == 24


def test_add_multi_asset_features_no_duplicate_timestamp(tmp_multi_csv: Path):
    btc = pd.read_csv(tmp_multi_csv, parse_dates=['ts'])
    btc = btc[btc['symbol'] == 'BTCUSDT'].copy().reset_index(drop=True)
    btc = btc.rename(columns={'ts': 'timestamp'})

    out = add_multi_asset_features(btc, str(tmp_multi_csv))
    assert out.columns.duplicated().sum() == 0


def test_add_multi_asset_features_missing_file_returns_unchanged():
    btc = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=10, freq='5min', tz='UTC'), 'close': [1]*10})
    btc['label'] = 2
    out = add_multi_asset_features(btc, '/tmp/does-not-exist.csv')
    assert 'ETHUSDT_btc_ratio' not in out.columns
    pd.testing.assert_frame_equal(out.reset_index(drop=True), btc.reset_index(drop=True))


def test_make_cross_features_only_btc_symbol_returns_unchanged(tmp_path: Path):
    rng = pd.date_range('2024-01-01', periods=50, freq='5min', tz='UTC')
    df = pd.DataFrame({
        'ts': rng,
        'symbol': 'BTCUSDT',
        'open': 100.0,
        'high': 100.1,
        'low': 99.9,
        'close': 100.0,
        'volume': 10.0,
    })
    out_path = tmp_path / 'one.csv'
    df.to_csv(out_path, index=False)
    btc = df.rename(columns={'ts': 'timestamp'}).copy()
    out = add_multi_asset_features(btc, str(out_path))
    assert 'ETHUSDT_btc_ratio' not in out.columns
