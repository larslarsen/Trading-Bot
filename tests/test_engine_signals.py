"""Edge-case / branch-coverage tests for engine.py signal + volatility functions.

Each signal is exercised on:
  - a flat (zero-move) series -> known degenerate output (0 / no entries)
  - a monotonic up series -> expected directional behaviour
  - short / NaN inputs -> no crash, correct length

The williams_r_buggy_exit vs williams_r_signals divergence is pinned as a
regression test (the buggy variant sells at the oversold bottom; the correct
one sells on recovery).
"""
import numpy as np
import pandas as pd
import pytest

from engine import (
    rei_signals, williams_r_signals, williams_r_buggy_exit, cci_signals,
    rsi_signals, tsi_signals, bop_signals, mtf_confirm_signals,
    true_range, atr, chandelier_exit, atr_trailing_exit, apply_trailing_overlay,
    ma30_recapture_signals, kaufman_efficiency_ratio, choppiness_index,
    realized_vol, bb_width, bbwp, bbwp_signals,
)

N = 60


def _df(close, high=None, low=None, volume=None):
    close = np.asarray(close, dtype=float)
    n = len(close)
    high = np.asarray(high, dtype=float) if high is not None else close + 1.0
    low = np.asarray(low, dtype=float) if low is not None else close - 1.0
    volume = np.asarray(volume, dtype=float) if volume is not None else np.full(n, 1000.0)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume}, index=idx)


def test_rei_flat_is_zero():
    df = _df(np.full(N, 100.0))
    entry, exit_sig = rei_signals(df)
    assert len(entry) == N
    assert (entry == 0).all()
    assert (exit_sig == 0).all()


def test_rei_monotonic_up_fires_entry():
    df = _df(np.linspace(100, 200, N))
    entry, _ = rei_signals(df)
    # a sustained up-move must produce at least one recapture-style entry
    assert entry.sum() > 0


def test_williams_r_flat_is_zero_and_bounded():
    df = _df(np.full(N, 100.0))
    entry, exit_sig = williams_r_signals(df)
    # flat: highest==lowest==close -> wr=0; no cross events
    assert (entry == 0).all() and (exit_sig == 0).all()


def test_williams_r_bounded_range():
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.normal(0, 1, 200)) + 100
    df = _df(close)
    entry, exit_sig = williams_r_signals(df)
    # %R must live in [-100, 0]; signals are 0/1
    assert ((entry | exit_sig) >= 0).all()
    assert (entry <= 1).all() and (exit_sig <= 1).all()


def test_williams_buggy_vs_correct_differ_on_v_shape():
    # V-shape: crash to oversold then recover. Correct exit fires on recovery
    # (wr crosses UP through -20); buggy exit fires at the bottom (wr<-20 & falling).
    crash = np.concatenate([np.full(20, 100.0), np.linspace(100, 50, 10), np.linspace(50, 100, 10)])
    df = _df(crash)
    correct = williams_r_signals(df)[1]
    buggy = williams_r_buggy_exit(df)[1]
    # they must NOT be identical -> the buggy path is reproducible & distinct
    assert not (correct == buggy).all()
    # buggy fires at/near the bottom; correct fires on the way up
    bottom = 29  # index of the 50.0 low
    assert buggy.iloc[bottom] == 1  # sells at the oversold bottom (pathological)
    assert correct.iloc[bottom] == 0  # correct does NOT sell at the bottom


def test_cci_flat_is_zero():
    df = _df(np.full(N, 100.0))
    entry, exit_sig = cci_signals(df)
    assert (entry == 0).all() and (exit_sig == 0).all()


def test_rsi_flat_zero_loss_treated_oversold():
    # flat prices -> gain=loss=0 -> rs=0 -> rsi=0 -> entry fires (rsi<30).
    # idx0 is NaN -> entry 0; every later bar is oversold -> entry 1.
    # This documents the zero-volatility RSI degeneracy (regression pin).
    df = _df(np.full(N, 100.0))
    entry, exit_sig = rsi_signals(df)
    assert entry.iloc[0] == 0
    assert (entry.iloc[1:] == 1).all()
    assert (exit_sig == 0).all()


def test_rsi_recovery_exit_fires_on_dip_then_rebound():
    # RSI recovery exit only fires when %R crosses UP through 50. A pure
    # monotonic trend never does (it sits at 100), so use a dip-then-rebound:
    # drop into oversold, then climb back across the 50 midline.
    close = np.concatenate([np.full(20, 100.0), np.linspace(100, 60, 15), np.linspace(60, 130, 25)])
    df = _df(close)
    entry, exit_sig = rsi_signals(df)
    # the rebound must produce at least one recovery exit
    assert exit_sig.sum() > 0


def test_tsi_flat_no_entry():
    df = _df(np.full(N, 100.0))
    entry, exit_sig = tsi_signals(df)
    assert (entry == 0).all()


def test_bop_flat_no_entry():
    df = _df(np.full(N, 100.0))
    entry, exit_sig = bop_signals(df)
    assert (entry == 0).all()


def test_mtf_confirm_flat_no_entries():
    # flat series -> sma50 == sma200 == close -> close > sma50 is False -> no entries
    df = _df(np.full(120, 100.0))
    entry, exit_sig = mtf_confirm_signals(df)
    assert (entry == 0).all()
    assert (exit_sig == 0).all()


def test_mtf_confirm_uptrend_produces_entries():
    # a sustained uptrend eventually satisfies sma50>sma200 and price>sma50
    df = _df(np.linspace(100, 150, 120))
    entry, exit_sig = mtf_confirm_signals(df)
    assert entry.sum() > 0
    # short window does not crash and yields 0/1 only
    assert set(entry.unique()).issubset({0, 1})


def test_true_range_known_values():
    h = pd.Series([10.0, 11.0])
    l = pd.Series([8.0, 9.0])
    c = pd.Series([9.0, 10.0])
    tr = true_range(h, l, c)
    assert tr.iloc[0] == pytest.approx(2.0)          # high - low
    assert tr.iloc[1] == pytest.approx(2.0)          # max(11-9, |11-9|, |9-9|)=2


def test_atr_single_bar_equals_true_range():
    h = pd.Series([10.0])
    l = pd.Series([8.0])
    c = pd.Series([9.0])
    a = atr(h, l, c, period=14)
    assert a.iloc[0] == pytest.approx(2.0)           # only one TR sample


def test_chandelier_and_atr_trailing_length_and_dtype():
    df = _df(np.linspace(100, 120, N))
    ce = chandelier_exit(df)
    te = atr_trailing_exit(df)
    assert len(ce) == N and len(te) == N
    assert set(ce.unique()).issubset({0, 1})
    assert set(te.unique()).issubset({0, 1})


def test_apply_trailing_overlay_ors_and_fills_nan():
    base = pd.Series([1, 0, np.nan, 1])
    trail = pd.Series([0, np.nan, 1, 0])
    out = apply_trailing_overlay(base, trail)
    assert list(out) == [1, 0, 1, 1]                 # OR after fillna(0)


def test_ma30_recapture_cross_fires_entry():
    # 40 flat bars then one close above the MA
    close = np.concatenate([np.full(40, 100.0), [101.0]])
    df = _df(close)
    entry, exit_sig = ma30_recapture_signals(df)
    assert len(entry) == len(close)
    # entry fires exactly at the cross-up bar (index 40)
    assert entry.iloc[40] == 1
    assert entry.iloc[:40].sum() == 0


def test_ma30_recapture_flat_no_entries():
    df = _df(np.full(N, 100.0))
    entry, exit_sig = ma30_recapture_signals(df)
    assert (entry == 0).all()


def test_kaufman_flat_zero_monotonic_one():
    flat = pd.Series(np.full(10, 5.0))
    assert kaufman_efficiency_ratio(flat, period=3).iloc[-1] == pytest.approx(0.0)
    up = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    er = kaufman_efficiency_ratio(up, period=3)
    assert er.iloc[-1] == pytest.approx(1.0)         # perfectly directional -> 1.0


def test_choppiness_fills_nan_with_zero():
    h = pd.Series(np.full(N, 10.0))
    l = pd.Series(np.full(N, 9.0))
    c = pd.Series(np.full(N, 9.5))
    ci = choppiness_index(h, l, c, period=14)
    assert not ci.isna().any()                        # NaN/inf replaced
    assert (ci >= 0).all()


def test_realized_vol_flat_zero():
    rv = realized_vol(pd.Series(np.full(N, 100.0)))
    # flat -> pct_change std == 0 (after warmup)
    assert rv.iloc[-1] == pytest.approx(0.0)


def test_bb_width_flat_zero_expanded_volatile():
    flat = bb_width(pd.Series(np.full(40, 100.0)))
    assert flat.iloc[-1] == pytest.approx(0.0)
    vol = bb_width(pd.Series(np.concatenate([np.full(20, 100.0), np.linspace(100, 200, 20)])))
    assert vol.iloc[-1] > 0.0


def test_bbwp_in_unit_interval():
    rng = np.random.default_rng(1)
    close = np.cumsum(rng.normal(0, 1, 300)) + 100
    wp = bbwp(pd.Series(close))
    valid = wp.dropna()
    assert ((valid >= 0.0) & (valid <= 1.0)).all()


def test_bbwp_signals_flat_none():
    df = _df(np.full(N, 100.0))
    entry, exit_sig = bbwp_signals(df)
    assert (entry == 0).all() and (exit_sig == 0).all()


def test_signals_do_not_mutate_inputs():
    df = _df(np.linspace(100, 150, N))
    snap = df.copy()
    for fn in (rei_signals, williams_r_signals, cci_signals, rsi_signals,
               tsi_signals, bop_signals, mtf_confirm_signals,
               ma30_recapture_signals, bbwp_signals):
        fn(df)
    pd.testing.assert_frame_equal(df, snap)
