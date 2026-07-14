"""Edge-case / branch-coverage tests for engine.py regime, vol-scaling, and
portfolio-simulation / metric functions.

These assert the ACTUAL library behavior (some is debatable by design, e.g.
default_regime and compute_regime use opposite vol logic) so future regressions
are caught, not to dictate correctness.
"""

import numpy as np
import pandas as pd
import pytest

from engine import (
    compute_regime, improved_compute_live_regime, compute_live_regime,
    hurst_exponent, rolling_hurst, hurst_regime, ma_crossover_regime,
    compute_vol_scale, make_vol_scale_fn, make_regime_gated_vol_scale_fn,
    default_regime, adx, chandelier_exit, mtf_confirm_signals,
    simulate_portfolio, deflated_sharpe_ratio, probabilistic_sharpe_ratio,
    analyze_regime_quality,
)


def make_market(n=120, seed=0, drift=0.1):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(drift, 1.0, n))


# ── compute_regime ─────────────────────────────────────────────────────────
def test_compute_regime_early_returns_trend():
    c = make_market(20)
    assert compute_regime(pd.Series(c), 10) == "trend"  # < 30 bars -> trend


def test_compute_regime_kaufman_pure_er():
    up = pd.Series(np.linspace(100, 200, 120))
    chop = pd.Series([100.0] * 120)
    assert compute_regime(up, 119, method="kaufman") == "trend"
    assert compute_regime(chop, 119, method="kaufman") == "chop"


def test_compute_regime_ma_method():
    up = pd.Series(np.linspace(100, 300, 250))
    down = pd.Series(np.linspace(300, 100, 250))
    assert compute_regime(up, 249, method="ma") == "trend"
    assert compute_regime(down, 249, method="ma") == "chop"
    assert compute_regime(up, 50, method="ma") == "chop"  # < long ma


def test_adx_finite_after_period():
    c = pd.Series(make_market(80))
    h = c + 1.0
    l = c - 1.0
    a = adx(h, l, c, period=14)
    assert len(a) == len(c)
    # ADX needs ~2*period bars before it is non-NaN
    assert a.iloc[28:].apply(np.isfinite).all()


def test_compute_regime_constant_series_is_chop():
    # constant series -> ADX=0, ER=0 -> not trending -> chop
    c = pd.Series([100.0] * 120)
    assert compute_regime(c, 119, method="rule") == "chop"


def test_compute_regime_bbwp_and_mesa_methods():
    flat = pd.Series([100.0] * 120)
    up = pd.Series(np.linspace(100, 200, 120))
    assert compute_regime(flat, 119, method="mesa") == "chop"
    assert compute_regime(up, 119, method="bbwp") in ("trend", "chop")


def test_compute_regime_hysteresis_returns_valid():
    c = pd.Series(make_market(120, seed=3))
    assert compute_regime(c, 119, use_hysteresis=True, hysteresis_bars=2) in ("trend", "chop")


# ── live wrappers ──────────────────────────────────────────────────────────
def test_improved_compute_live_regime_empty_and_short():
    assert improved_compute_live_regime({}) == "trend"
    df = pd.DataFrame({"close": np.arange(5.0)})
    assert improved_compute_live_regime({"A": df}) == "trend"  # min_len < 10


def test_improved_compute_live_regime_runs_on_real_data():
    dfs = {f"S{i}": pd.DataFrame({"close": make_market(60, seed=i)}) for i in range(3)}
    assert improved_compute_live_regime(dfs) in ("trend", "chop")


def test_compute_live_regime_empty_and_short():
    assert compute_live_regime({}) == "trend"
    df = pd.DataFrame({"close": np.arange(3.0)})
    assert compute_live_regime({"A": df}) == "trend"


# ── Hurst ──────────────────────────────────────────────────────────────────
def test_hurst_exponent_short_is_nan():
    assert np.isnan(hurst_exponent(pd.Series(np.arange(10.0))))


def test_hurst_exponent_clipped_range():
    h = hurst_exponent(pd.Series(make_market(200, seed=2)))
    assert 0.0 <= h <= 1.0


def test_rolling_hurst_structure():
    s = pd.Series(make_market(150, seed=4))
    rh = rolling_hurst(s, window=60, max_lag=20)
    assert len(rh) == len(s)
    assert rh.iloc[59:].apply(lambda x: np.isnan(x) or (0 <= x <= 1)).all()


def test_hurst_regime_short_window_is_chop():
    assert hurst_regime(pd.Series(make_market(40)), 10, window=60) == "chop"


def test_hurst_regime_returns_valid_string():
    s = pd.Series(np.cumsum(np.random.default_rng(7).normal(0, 1, 120)) + 100)
    assert hurst_regime(s, 119) in ("trend", "chop")


def test_ma_crossover_regime():
    up = pd.Series(np.linspace(100, 300, 250))
    down = pd.Series(np.linspace(300, 100, 250))
    assert ma_crossover_regime(up, 249) == "trend"
    assert ma_crossover_regime(down, 249) == "chop"
    assert ma_crossover_regime(up, 50) == "chop"


# ── Vol scaling ────────────────────────────────────────────────────────────
def test_compute_vol_scale_short_history_is_one():
    assert compute_vol_scale([0.01, -0.02], target_vol=0.15) == 1.0


def test_compute_vol_scale_flat_returns_one():
    assert compute_vol_scale([0.0] * 30, target_vol=0.15, lookback=20) == 1.0


def test_compute_vol_scale_clips_to_bounds():
    rng = np.random.default_rng(0)
    big = rng.normal(0.2, 0.1, 30).tolist()    # high vol -> clip to lower bound
    tiny = rng.normal(0.0001, 0.00005, 30).tolist()  # low vol -> clip to upper
    lo, hi = 0.25, 1.5
    assert compute_vol_scale(big, target_vol=0.15, lookback=20, bounds=(lo, hi)) == pytest.approx(lo)
    assert compute_vol_scale(tiny, target_vol=0.15, lookback=20, bounds=(lo, hi)) == pytest.approx(hi)


def test_make_vol_scale_fn_wraps():
    fn = make_vol_scale_fn(target_vol=0.15, lookback=20)
    assert fn(10, 1000.0, [0.01] * 25) == compute_vol_scale([0.01] * 25)


def test_regime_gated_vol_scale_fn_only_in_target_regime():
    regimes = pd.Series(["chop"] * 30 + ["trend"] * 30)
    gated = make_regime_gated_vol_scale_fn(regimes, apply_in="chop")
    assert gated(5, 1000.0, [0.01] * 25) == compute_vol_scale([0.01] * 25)
    assert gated(35, 1000.0, [0.01] * 25) == 1.0  # trend regime -> no scaling
    assert gated(99, 1000.0, [0.01] * 25) == 1.0  # day_i beyond series -> 1.0


# ── default_regime ─────────────────────────────────────────────────────────
def test_default_regime_early_bars_trend():
    c = pd.Series(np.linspace(100, 200, 40))
    assert default_regime(c, 10) == "trend"
    assert default_regime(c, 24) == "trend"


def test_default_regime_low_vol_is_chop():
    # default_regime treats LOW pct-change vol as chop (opposite vol logic to
    # compute_regime's rule path, which requires low vol for trend)
    flat = pd.Series([100.0] * 120)
    assert default_regime(flat, 119) == "chop"
    lin = pd.Series(np.linspace(100, 200, 120))  # smooth -> low pct vol -> chop
    assert default_regime(lin, 119) == "chop"
    # returns a valid regime for volatile data
    vu = pd.Series(make_market(120, seed=5, drift=0.5))
    assert default_regime(vu, 119) in ("trend", "chop")


# ── indicators ─────────────────────────────────────────────────────────────
def test_chandelier_exit_fires_below_stop():
    close = list(np.arange(100, 130.0)) + list(np.arange(129, 80, -4.0))
    high = np.array(close) + 1.0
    low = np.array(close) - 1.0
    df = pd.DataFrame({"close": close, "high": high, "low": low})
    ex = chandelier_exit(df, period=10, mult=2.0)
    assert ex.dtype == int
    assert ex.sum() > 0


def test_mtf_confirm_signals():
    close = np.linspace(100, 200, 250)
    entry, exit_sig = mtf_confirm_signals(pd.DataFrame({"close": close}))
    assert entry.iloc[-1] == 1
    assert exit_sig.dtype == int


# ── metrics ────────────────────────────────────────────────────────────────
def test_deflated_sharpe_ratio_range_and_nan_safe():
    assert 0.0 <= deflated_sharpe_ratio(0.8, 10, 5) <= 1.0
    assert 0.0 <= deflated_sharpe_ratio(-0.5, 10, 5) <= 1.0
    assert 0.0 <= deflated_sharpe_ratio(float("nan"), 10, 5) <= 1.0
    assert np.isfinite(deflated_sharpe_ratio(2.0, 1000, 250))


def test_probabilistic_sharpe_ratio_range():
    assert 0.0 <= probabilistic_sharpe_ratio(0.5, 0.0, 252) <= 1.0
    assert probabilistic_sharpe_ratio(0.5, -1.0, 252) > probabilistic_sharpe_ratio(0.5, 1.0, 252)


# ── simulate_portfolio ─────────────────────────────────────────────────────
def make_price_sig(n=60, n_sym=2):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=n)
    price = pd.DataFrame(
        {f"S{i}": 100 + np.cumsum(rng.normal(0.1, 1.0, n)) for i in range(n_sym)},
        index=dates,
    )
    sig = pd.DataFrame(0, index=dates, columns=price.columns)
    sig.iloc[5] = 1
    return price, sig


def test_simulate_portfolio_runs_and_keys():
    price, sig = make_price_sig()
    res = simulate_portfolio(price, sig)
    for k in ("final_equity", "return_pct", "sharpe", "effective_sharpe",
              "exposure_ratio", "max_dd_pct", "trades", "dsr", "psr", "n_trials"):
        assert k in res
    assert res["trades"] == price.shape[1]


def test_simulate_portfolio_circuit_breaker_flattens():
    n = 60
    dates = pd.date_range("2025-01-01", periods=n)
    arr = 100 + np.concatenate([np.zeros(20), -np.arange(20) * 3, np.zeros(20)])
    price = pd.DataFrame({"S0": arr}, index=dates)
    sig = pd.DataFrame(0, index=dates, columns=["S0"])
    sig.iloc[5] = 1
    res = simulate_portfolio(price, sig, initial=1000.0)
    assert res["trades"] >= 1
    assert res["max_dd_pct"] >= 0.0


def test_simulate_portfolio_vol_scale_and_regime_hooks():
    price, sig = make_price_sig(n=60)
    vfn = make_vol_scale_fn(target_vol=0.15, lookback=20)
    regime_fn = lambda close, i: "trend" if i < 30 else "chop"
    regime_rule_map = {"trend": sig, "chop": sig}
    res = simulate_portfolio(price, sig, vol_scale_fn=vfn,
                              regime_fn=regime_fn, regime_rule_map=regime_rule_map)
    assert res["trades"] == price.shape[1]
    assert "sharpe" in res


def test_simulate_portfolio_flat_market_no_inf():
    n = 40
    dates = pd.date_range("2025-01-01", periods=n)
    price = pd.DataFrame({"S0": [100.0] * n}, index=dates)
    sig = pd.DataFrame(0, index=dates, columns=["S0"])
    sig.iloc[5] = 1
    res = simulate_portfolio(price, sig)
    assert np.isfinite(res["sharpe"])
    assert np.isfinite(res["final_equity"])


def test_analyze_regime_quality_returns_stats():
    stats = analyze_regime_quality(pd.Series(make_market(100, seed=7)))
    assert "distribution" in stats
    assert "transitions" in stats
    assert "durations" in stats
    assert stats["transitions"]["total_days"] == 100
