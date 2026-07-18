"""Tests for pipeline.py pure functions — labels, splits, filters, regime.

Network-free: these operate on synthetic DataFrames, no CSV/network needed.
"""
import numpy as np
import pandas as pd
import pytest

import pipeline as pl


def _ohlc(n=500, seed=1, start="2022-01-01"):
    idx = pd.date_range("2022-01-01", periods=n, freq="5min", tz="UTC")
    idx.name = "timestamp"
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open": close,
        "high": close + rng.uniform(0, 1, n),
        "low": close - rng.uniform(0, 1, n),
        "close": close,
        "volume": rng.uniform(100, 1000, n),
    }, index=idx)


# ── triple_barrier_labels ───────────────────────────────────────────────────
def test_triple_barrier_labels_shape_and_domain():
    df = _ohlc(n=400)
    out = pl.triple_barrier_labels(df, horizon=12)
    assert "label" in out.columns
    labs = out["label"].dropna()
    assert set(labs.unique()).issubset({0, 1, 2})


def test_triple_barrier_labels_short_horizon_returns_flat_only():
    # too few rows for a full horizon -> no barrier is ever touched ->
    # labels stay at the all-flat (2) initialization
    df = _ohlc(n=5)
    out = pl.triple_barrier_labels(df, horizon=12)
    assert (out["label"] == 2).all()


# ── walk_forward_splits ─────────────────────────────────────────────────────
def test_walk_forward_splits_returns_list_of_dicts():
    # 200k rows clears val(5k)+test(15k)+2*step(15k) so 3 folds fit
    df = _ohlc(n=200_000)
    splits = pl.walk_forward_splits(df, folds=3)
    assert isinstance(splits, list) and len(splits) == 3
    for s in splits:
        assert {"train_idx", "val_idx", "test_idx"} <= set(s.keys())


def test_walk_forward_splits_expanding():
    df = _ohlc(n=200_000)
    splits = pl.walk_forward_splits(df, folds=3)
    # each fold's train window ends before the next fold's train window ends
    ends = [max(s["train_idx"]) for s in splits]
    assert ends[0] < ends[1] < ends[2]


def test_walk_forward_splits_empty_for_tiny_df():
    df = _ohlc(n=10)
    assert pl.walk_forward_splits(df, folds=3) == []


# ── cost_aware_filter ───────────────────────────────────────────────────────
def test_cost_aware_filter_holds_on_strong_signal():
    # strong flat prob -> desired 2 -> return 2
    probs = np.array([0.05, 0.05, 0.90])
    assert pl.cost_aware_filter(probs, prev_pos=0) == 2


def test_cost_aware_filter_flattens_on_weak_signal():
    # desired differs from prev but margin <= cost threshold -> stay flat
    probs = np.array([0.34, 0.34, 0.32])
    # desired=1 (p_short not > p_long -> else), prev=2, margin 0 <= req
    assert pl.cost_aware_filter(probs, prev_pos=2) == 2


def test_cost_aware_filter_flips_on_strong_reversal():
    # strong short signal from a flat position -> take short (0)
    probs = np.array([0.90, 0.05, 0.05])
    assert pl.cost_aware_filter(probs, prev_pos=2) == 0


def test_cost_aware_filter_takes_long_when_margin_exceeds_cost():
    # strong long signal from flat -> long (1)
    probs = np.array([0.05, 0.90, 0.05])
    assert pl.cost_aware_filter(probs, prev_pos=2) == 1

# ── detect_regime ───────────────────────────────────────────────────────────
def test_detect_regime_adds_columns():
    # detect_regime requires volatility_24 + bb_mid, produced by derive_features
    df = pl.derive_features(_ohlc(n=300))
    out = pl.detect_regime(df)
    assert "regime_high_vol" in out.columns
    assert "regime_trending" in out.columns
    assert out["regime_high_vol"].isin([0, 1]).all()
    assert out["regime_trending"].isin([0, 1]).all()


def test_detect_regime_unknown_without_features():
    # no vol features -> labels everything "unknown", no crash
    df = _ohlc(n=50)
    out = pl.detect_regime(df)
    assert (out["regime"] == "unknown").all()


# ── add_resampled_features ──────────────────────────────────────────────────
def test_add_resampled_features_returns_merged_frame():
    df = _ohlc(n=300)
    res = pl.add_resampled_features(df)
    out = df.join(res, how="left")
    assert len(out) == len(df)
    # resampled columns should be present (some are not all-NaN)
    assert res.shape[1] > 0
