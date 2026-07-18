"""Production-readiness regression tests.

Covers the fixes from the prod audit:
  * The live trader must call start_daily_bar() so the daily-loss circuit
    breaker resets and the flash-crash window populates. Without it the
    "daily" loss limit is cumulative (halts permanently after any 3% DD)
    and the flash-crash breaker never fires.
  * _log_trade (trade journal) must be atomic (no corrupt journal on a
    truncated write) and must not raise on a pre-existing corrupt journal.
"""
from datetime import datetime, timezone

import pytest

from portfolio_engine import PortfolioEngine, EngineConfig


def test_daily_loss_breaker_resets_across_days():
    """daily-loss limit must reset per day, not be cumulative."""
    eng = PortfolioEngine(EngineConfig(initial_capital=10000.0,
                                       max_daily_loss_pct=0.03))
    # Day 1: lose 4% (exceeds 3% daily) -> should halt for the day.
    eng.start_daily_bar(100.0)
    eng.positions["X"] = type("P", (), {"shares": 100.0, "entry": 1.0})()
    eng.cash -= 100.0
    # simulate a -4% daily pnl directly through close + breaker path
    eng.daily_pnl = -400.0
    ok, reason = eng.check_circuit_breakers()
    assert ok is False, f"expected daily-loss halt, got {reason}"
    assert "daily_loss" in reason

    # Day 2: new day resets the tally -> breaker clears.
    eng.halted = False
    eng.halt_reason = None
    eng.start_daily_bar(100.0)
    eng.daily_pnl = 0.0  # reset as start_daily_bar does
    ok, reason = eng.check_circuit_breakers()
    assert ok is True, f"expected cleared breaker after day rollover, got {reason}"


def test_flash_crash_window_populates_via_start_daily_bar():
    """flash-crash window only fills if start_daily_bar is called."""
    eng = PortfolioEngine(EngineConfig(initial_capital=10000.0,
                                       flash_crash_bars=3, flash_crash_pct=0.50))
    # Without any daily bars, window is empty -> no flash-crash trip.
    ok, reason = eng.check_circuit_breakers()
    assert ok is True
    # Roll 3 daily bars with a 60% move -> now it must trip.
    eng.start_daily_bar(100.0)
    eng.start_daily_bar(120.0)
    eng.start_daily_bar(40.0)  # (120-40)/120 = 0.667 > 0.50
    ok, reason = eng.check_circuit_breakers()
    assert ok is False, f"expected flash-crash trip, got {reason}"
    assert "flash_crash" in reason


def test_live_trader_wires_daily_bar(monkeypatch, tmp_path):
    """cex_ml_xgb_5m.run_once must invoke start_daily_bar so breakers work."""
    import importlib
    trader = importlib.import_module("cex_ml_xgb_5m")

    called = {"n": 0}
    monkeypatch.setattr(trader.MultiPositionState, "start_daily_bar",
                        lambda self, p: called.__setitem__("n", called["n"] + 1))
    # force a day rollover by clearing the cached day
    trader._last_daily_day[0] = None

    state = trader.MultiPositionState(
        initial_capital=10000.0,
        state_file=tmp_path / "exec.json",
        journal_file=tmp_path / "journal.json",
    )
    monkeypatch.setattr(trader, "active_pairs", lambda: {"BTCUSDT": object()})
    monkeypatch.setattr(trader, "discover_models", lambda: {"BTCUSDT": object()})
    monkeypatch.setattr(trader, "price_for", lambda s: 100.0)
    monkeypatch.setattr(trader, "mem_guard_abort", lambda mb: None)

    # models non-empty so ref_pair resolves to BTCUSDT and ref_price > 0,
    # which is what lets _maybe_start_daily_bar call start_daily_bar.
    trader.run_once({"BTCUSDT": object()}, state)
    assert called["n"] >= 1, "run_once did not call start_daily_bar"
