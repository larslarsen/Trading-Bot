"""Edge-case / branch-coverage tests for engine.py portfolio simulation + metrics."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from engine import (
    simulate_portfolio, deflated_sharpe_ratio, probabilistic_sharpe_ratio,
    studentized_performance, stepwise_spa_test,
)

N = 60


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=N, freq="B")
    close = np.cumsum(rng.normal(0, 1, N)) + 100
    df = pd.DataFrame({"X": close, "Y": close[::-1]}, index=dates)
    sig = pd.DataFrame(0, index=dates, columns=["X", "Y"])
    sig.iloc[5, :] = 1  # enter on day 5
    return df, sig


# ── Metrics ────────────────────────────────────────────────────────────────
def test_dsr_positive_high_n_is_near_one():
    dsr = deflated_sharpe_ratio(2.0, 10, 252)
    assert 0.0 <= dsr <= 1.0
    assert dsr > 0.9


def test_dsr_negative_is_low():
    dsr = deflated_sharpe_ratio(-2.0, 10, 252)
    assert dsr < 0.1


def test_dsr_nan_input_is_effectively_zero():
    # nan SR is coerced to 0.0, producing a hugely negative z -> DSR ~ 0
    dsr = deflated_sharpe_ratio(float("nan"), 10, 252)
    assert dsr < 1e-50  # effectively zero (float-epsilon, not exactly 0.0)


def test_dsr_single_trial_expected_max_zero():
    dsr = deflated_sharpe_ratio(1.0, 1, 252)
    assert 0.0 <= dsr <= 1.0


def test_psr_positive_above_half():
    psr = probabilistic_sharpe_ratio(1.0, benchmark_sr=0.0, T=252)
    assert psr > 0.5


def test_psr_negative_below_half():
    psr = probabilistic_sharpe_ratio(-1.0, benchmark_sr=0.0, T=252)
    assert psr < 0.5


def test_psr_zero_sr_near_half():
    psr = probabilistic_sharpe_ratio(0.0, benchmark_sr=0.0, T=252)
    assert psr == pytest.approx(0.5, abs=1e-6)


def test_studentized_too_short_returns_nan():
    T, mu, v = studentized_performance([0.01, 0.02], [0.0, 0.0])
    assert np.isnan(T) and np.isnan(mu) and np.isnan(v)


def test_studentized_zero_variance_returns_nan():
    # LATENT: a zero-variance (constant) strategy yields v->0 -> T=nan.
    # stepwise_spa_test then silently treats it as "not significant" (not counted).
    # Pinned as a regression so the behavior is explicit, not silent.
    T, mu, v = studentized_performance([0.01] * 30, [0.0] * 30)
    assert np.isnan(T)
    res = stepwise_spa_test([[0.01] * 30], [0.0] * 30)
    assert res["n_tested"] == 0
    assert res["significant"] is False


def test_studentized_basic_noisy():
    strat = [0.011, 0.012, 0.009, 0.013, 0.010, 0.012, 0.008, 0.014, 0.011, 0.012]
    bench = [0.0] * 10
    T, mu, v = studentized_performance(strat, bench)
    assert np.isfinite(T) and mu > 0 and v > 0


def test_stepwise_spa_empty_is_not_significant():
    res = stepwise_spa_test([], [0.0, 0.0, 0.0])
    assert res["significant"] is False
    assert res["n_tested"] == 0


def test_stepwise_spa_one_winner():
    rng = np.random.default_rng(0)
    strat = (rng.normal(0.01, 0.001, 40)).tolist()  # winning but with variance
    bench = [0.0] * 40
    res = stepwise_spa_test([strat], bench)
    assert res["significant"] is True
    assert res["surviving_indices"] == [0]


def test_stepwise_spa_multiple_some_survive():
    rng = np.random.default_rng(1)
    good = (rng.normal(0.02, 0.001, 40)).tolist()
    bad = (rng.normal(-0.01, 0.001, 40)).tolist()
    bench = [0.0] * 40
    res = stepwise_spa_test([good, bad], bench)
    assert 0 in res["surviving_indices"]
    # the losing strategy should not survive
    assert 1 not in res["surviving_indices"]


# ── simulate_portfolio ──────────────────────────────────────────────────────
def test_simulate_flat_prices_no_trades():
    dates = pd.date_range("2025-01-01", periods=N, freq="B")
    df = pd.DataFrame({"X": np.full(N, 100.0)}, index=dates)
    sig = pd.DataFrame(0, index=dates, columns=["X"])
    sig.iloc[5, 0] = 1
    res = simulate_portfolio(df, sig, initial=1000.0)
    assert res["final_equity"] == pytest.approx(1000.0, abs=1.0)
    assert "sharpe" in res and "dsr" in res and "psr" in res


def test_simulate_empty_signals_no_trades():
    dates = pd.date_range("2025-01-01", periods=N, freq="B")
    df = pd.DataFrame({"X": np.linspace(100, 200, N)}, index=dates)
    sig = pd.DataFrame(0, index=dates, columns=["X"])
    res = simulate_portfolio(df, sig, initial=1000.0)
    assert res["trades"] == 0
    assert res["final_equity"] == pytest.approx(1000.0, abs=1.0)


def test_simulate_nan_price_skips_position():
    dates = pd.date_range("2025-01-01", periods=N, freq="B")
    close = np.linspace(100, 200, N).astype(float)
    close[6] = np.nan  # corrupt price on entry day
    df = pd.DataFrame({"X": close}, index=dates)
    sig = pd.DataFrame(0, index=dates, columns=["X"])
    sig.iloc[5, 0] = 1
    res = simulate_portfolio(df, sig, initial=1000.0)
    assert np.isfinite(res["final_equity"])


def test_simulate_max_positions_one():
    df, sig = _toy()
    res = simulate_portfolio(df, sig, initial=1000.0, max_positions=1)
    assert res["final_equity"] > 0


def test_simulate_vol_scale_fn_applied():
    df, sig = _toy()
    calls = []

    def vol_fn(day_i, equity, ret_history):
        calls.append(day_i)
        return 0.5

    res = simulate_portfolio(df, sig, initial=1000.0, vol_scale_fn=vol_fn)
    assert len(calls) > 0


def test_simulate_regime_swap():
    df, sig = _toy()
    alt = sig.copy()
    alt.iloc[:, :] = 0
    dates = df.index
    regime = pd.Series(["trend"] * len(dates), index=dates)

    def regime_fn(close_market, day_i):
        return regime.iloc[day_i]

    res = simulate_portfolio(
        df, sig, initial=1000.0,
        regime_fn=regime_fn,
        regime_rule_map={"trend": sig, "chop": alt},
    )
    assert np.isfinite(res["final_equity"])


def test_simulate_fair_compare_writes_csv(tmp_path):
    df, sig = _toy()
    out = tmp_path / "fair.csv"
    res = simulate_portfolio(df, sig, initial=1000.0,
                              fair_compare_path=out, fair_compare_rule="toy")
    assert out.exists()
    text = out.read_text().strip().splitlines()
    assert text[0].startswith("rule_name")
    assert "toy" in text[1]
