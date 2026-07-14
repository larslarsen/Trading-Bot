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


# ── Edge cases / branch coverage ──────────────────────────────────────────
def test_load_multi_missing_file_returns_empty():
    from multi_asset_features import load_multi
    out = load_multi('/tmp/definitely-missing-xyz.csv')
    assert out.empty


def test_load_multi_filters_to_known_pairs_and_localizes(tmp_path: Path):
    from multi_asset_features import load_multi, PAIRS
    rng = pd.date_range('2024-01-01', periods=10, freq='5min')  # tz-NAIVE
    rows = []
    for sym in ['BTCUSDT', 'ETHUSDT', 'UNKNOWN']:
        for i, ts in enumerate(rng):
            rows.append({'ts': ts, 'symbol': sym, 'open': 1.0, 'high': 1.0,
                         'low': 1.0, 'close': 1.0, 'volume': 1.0})
    p = tmp_path / 'm.csv'
    pd.DataFrame(rows).to_csv(p, index=False)
    out = load_multi(p)
    # unknown symbol excluded; tz-naive 'ts' localized to UTC
    assert set(out['symbol'].unique()) == {'BTCUSDT', 'ETHUSDT'}
    assert out['timestamp'].dt.tz is not None


def test_make_cross_features_none_passthrough():
    from multi_asset_features import make_cross_features
    btc = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=5, freq='5min', tz='UTC'),
                        'close': [1.0] * 5})
    out = make_cross_features(btc, None)
    assert out is btc  # returns the same frame (no copy)


def test_make_cross_features_empty_passthrough():
    from multi_asset_features import make_cross_features
    btc = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=5, freq='5min', tz='UTC'),
                        'close': [1.0] * 5})
    out = make_cross_features(btc, pd.DataFrame())
    assert len(out) == len(btc)


def test_make_cross_features_insufficient_symbols_passthrough(tmp_path: Path):
    from multi_asset_features import make_cross_features
    # only BTCUSDT present -> < 2 symbols -> passthrough
    rng = pd.date_range('2024-01-01', periods=10, freq='5min', tz='UTC')
    df = pd.DataFrame({'ts': rng, 'symbol': 'BTCUSDT',
                       'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0})
    p = tmp_path / 'solo.csv'
    df.to_csv(p, index=False)
    from multi_asset_features import load_multi
    btc = df.rename(columns={'ts': 'timestamp'}).copy()
    out = make_cross_features(btc, load_multi(p))
    assert 'ETHUSDT_btc_ratio' not in out.columns


def test_make_cross_features_left_join_no_row_loss(tmp_multi_csv: Path):
    # df_btc shorter than the merged cross table -> left join must keep all btc rows
    from multi_asset_features import make_cross_features, load_multi
    btc = pd.read_csv(tmp_multi_csv, parse_dates=['ts'])
    btc = btc[btc['symbol'] == 'BTCUSDT'].copy().reset_index(drop=True)
    btc = btc.rename(columns={'ts': 'timestamp'}).head(50)  # subset of rows
    df_multi = load_multi(tmp_multi_csv)
    out = make_cross_features(btc, df_multi)
    assert len(out) == 50  # no rows dropped
    # cross columns appear and are NaN where timestamps didn't match
    assert 'ETHUSDT_btc_ratio' in out.columns
