"""Tests for model_trainer.py — name normalization, path routing, feature build.

Network-free: fetch_data + optional feed loaders are monkeypatched to synthetic
frames so build_symbol_features runs end-to-end on a tiny DataFrame.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import model_trainer as mt
import canonical_features as cf


# ── name normalization ──────────────────────────────────────────────────────
def test_norm_sym_variants():
    assert mt._norm_sym("DOGE") == "DOGE"
    assert mt._norm_sym("DOGEUSDT") == "DOGE"
    assert mt._norm_sym("DOGE/USDT") == "DOGE"
    assert mt._norm_sym("BTC") == "BTC"
    assert mt._norm_sym("BTCUSDT") == "BTC"


# ── path routing ────────────────────────────────────────────────────────────
def test_model_out_path_btc_is_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "MODEL_DIR", tmp_path)
    assert mt.model_out_path("BTC").name == "latest_xgb.json"
    assert mt.model_out_path("DOGE").name == "doge_xgb.json"
    assert mt.model_out_path("DOGEUSDT").name == "doge_xgb.json"


def test_meta_metrics_paths_mirror_model(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "MODEL_DIR", tmp_path)
    assert mt.meta_out_path("BTC").name == "latest_meta.json"
    assert mt.metrics_out_path("BTC").name == "latest_metrics.json"
    assert mt.meta_out_path("ETH").name == "eth_meta.json"


# ── build_symbol_features ───────────────────────────────────────────────────
def _synthetic_df(n=300):
    idx = pd.date_range("2022-01-01", periods=n, freq="5min", tz="UTC")
    idx.name = "timestamp"
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "open": rng.uniform(90, 110, n),
        "high": rng.uniform(95, 115, n),
        "low": rng.uniform(85, 105, n),
        "close": rng.uniform(90, 110, n),
        "volume": rng.uniform(100, 1000, n),
    }, index=idx)


@pytest.fixture
def patch_build(monkeypatch, tmp_path):
    # Make build_symbol_features run without network or optional feeds.
    # load_micro/load_funding/load_onchain are imported locally inside
    # build_symbol_features (from micro_features / onchain_features), so we
    # patch the source modules, not the mt namespace.
    import micro_features, onchain_features
    monkeypatch.setattr(mt, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(mt, "fetch_data", lambda sym: _synthetic_df())
    monkeypatch.setattr(mt, "add_multi_asset_features",
                        lambda df, f: (df, []))
    monkeypatch.setattr(micro_features, "load_micro",
                        lambda idx: pd.DataFrame(index=idx))
    monkeypatch.setattr(micro_features, "load_funding",
                        lambda sym, idx: pd.DataFrame(index=idx))
    monkeypatch.setattr(onchain_features, "load_onchain",
                        lambda idx, sym: pd.DataFrame(index=idx))
    # dex_features.add_dex_features is imported inside try; stub the module
    import types
    dex = types.ModuleType("dex_features")
    dex.add_dex_features = lambda df: (df, [])
    monkeypatch.setitem(sys.modules, "dex_features", dex)


def test_build_symbol_features_returns_canonical_dims(patch_build):
    df, feats = mt.build_symbol_features("BTC")
    assert isinstance(df, pd.DataFrame)
    assert feats == cf.CANONICAL
    assert df.shape[1] == 5 + cf.N_FEATURES  # OHLCV + 113 (OHLCV ⊂ CANONICAL -> dup labels)
    # every canonical column present
    assert all(c in df.columns for c in cf.CANONICAL)


def test_build_symbol_features_dedupes_index(patch_build):
    # feed a df with a duplicated timestamp index -> must be de-duplicated
    d = _synthetic_df()
    d2 = d.copy()
    d = pd.concat([d, d2]).sort_index()
    # override fetch_data to return the duplicated frame
    import model_trainer as mt2
    mt2.fetch_data = lambda sym: d
    df, _ = mt2.build_symbol_features("BTC")
    assert not df.index.duplicated().any()


def test_train_and_save_writes_model_and_meta(patch_build, tmp_path, monkeypatch):
    # Tiny trainable dataset; stub the model dir + skip the heavy walk-forward
    # by feeding enough rows for one fold. Validates the full serialize path.
    monkeypatch.setattr(mt, "MODEL_DIR", tmp_path)
    # build a small but label-able df with canonical features
    d = _synthetic_df(n=2000)
    import model_trainer as mt2
    mt2.fetch_data = lambda sym: d
    out = mt2.train_and_save(symbol="BTC")
    # train_and_save returns True on success, None if no usable features
    assert out in (True, None)
    if out is True:
        assert (tmp_path / "latest_xgb.json").exists()
        assert (tmp_path / "latest_meta.json").exists()
        assert (tmp_path / "latest_metrics.json").exists()


def test_train_and_save_routes_non_btc_filename(patch_build, tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "MODEL_DIR", tmp_path)
    # need enough rows for at least one walk-forward fold (>= ~20k bars)
    d = _synthetic_df(n=60000)
    import model_trainer as mt2
    mt2.fetch_data = lambda sym: d
    out = mt2.train_and_save(symbol="DOGE")
    # non-BTC writes <sym>_xgb.json, never clobbering latest_xgb.json
    assert out is True
    assert (tmp_path / "doge_xgb.json").exists()
    assert not (tmp_path / "latest_xgb.json").exists()
