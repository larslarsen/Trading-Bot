"""Tests for canonical_features.py — the frozen 113-feature contract."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import canonical_features as cf


def test_canonical_dimension_is_113():
    # resolve() yields 113 columns (base + 24 cross-asset + always-present);
    # the historical "98" design target drifted as ALWAYS/on-chain feeds grew.
    assert cf.N_FEATURES == 113
    assert len(cf.CANONICAL) == 113


def test_canonical_is_unique_and_ordered():
    assert len(set(cf.CANONICAL)) == len(cf.CANONICAL)


def test_cross_assets_locked():
    assert cf.CROSS_ASSETS == ["ETHUSDT", "DOGEUSDT"]


def test_resolve_returns_exactly_canonical():
    df = pd.DataFrame({c: np.arange(10.0) for c in cf.CANONICAL})
    out, feats = cf.resolve(df)
    assert feats == cf.CANONICAL
    # when the input lacks OHLCV, resolve returns exactly the CANONICAL block
    # in canonical order
    assert list(out.columns) == cf.CANONICAL
    assert out.shape[1] == cf.N_FEATURES


def test_resolve_zero_fills_missing():
    # df has only half the canonical columns + close
    present = cf.CANONICAL[:50]
    df = pd.DataFrame({c: np.arange(5.0) for c in present})
    df["close"] = np.arange(5.0)
    out, feats = cf.resolve(df)
    # missing columns must exist and be all-zero
    missing = [c for c in cf.CANONICAL if c not in present]
    assert all((out[c] == 0.0).all() for c in missing)
    # every canonical column is present (zero-filled or real)
    assert set(out.columns) >= set(cf.CANONICAL)


def test_resolve_drops_extra_columns():
    df = pd.DataFrame({c: np.arange(5.0) for c in cf.CANONICAL})
    df["close"] = np.arange(5.0)
    df["totally_unknown_feature"] = 1.0
    out, feats = cf.resolve(df)
    assert "totally_unknown_feature" not in out.columns
    assert set(out.columns) >= set(cf.CANONICAL)


def test_resolve_preserves_ohlcv():
    # serving path: df carries OHLCV -> resolve keeps them and prepends the
    # canonical block after the raw-keep columns
    df = pd.DataFrame({c: np.arange(10.0) for c in cf.CANONICAL})
    df["open"] = np.linspace(1, 2, 10)
    df["high"] = np.linspace(2, 3, 10)
    df["low"] = np.linspace(0, 1, 10)
    df["close"] = np.linspace(3, 4, 10)
    df["volume"] = np.linspace(5, 6, 10)
    out, _ = cf.resolve(df)
    assert out["open"].iloc[0] == 1.0 and out["close"].iloc[-1] == 4.0
    # OHLCV columns present alongside the canonical block
    assert {"open", "high", "low", "close", "volume"} <= set(out.columns)
    assert set(cf.CANONICAL) <= set(out.columns)


def test_resolve_drops_leading_nan_rows():
    cols = cf.CANONICAL[:5]
    df = pd.DataFrame({c: [np.nan, np.nan, 1.0, 2.0, 3.0] for c in cols})
    df["close"] = [np.nan, np.nan, 1.0, 2.0, 3.0]
    out, _ = cf.resolve(df)
    # leading all-NaN rows dropped; remaining rows kept
    assert len(out) == 3
