"""
Tests for order_manager.py state, circuit breakers, fees, and journaling.
"""
import json
import order_manager
import pytest
from pathlib import Path
from unittest.mock import patch

from order_manager import ExecutionState, PositionSide, simulate_fill


@pytest.fixture()
def fresh_state(tmp_path: Path, monkeypatch):
    journal = tmp_path / 'trade_journal.json'
    state_file = tmp_path / 'execution_state.json'
    monkeypatch.setattr('order_manager.TRADE_JOURNAL', journal)
    monkeypatch.setattr('order_manager.STATE_FILE', state_file)
    return ExecutionState()


def test_default_state(fresh_state: ExecutionState):
    assert fresh_state.equity == 1000.0
    assert fresh_state.peak_equity == 1000.0
    assert fresh_state.daily_pnl == 0.0
    assert fresh_state.current_position == PositionSide.FLAT
    assert fresh_state.halted is False
    assert fresh_state.halt_reason is None


def test_record_trade_updates_equity_and_peak(fresh_state: ExecutionState):
    fresh_state.record_trade(PositionSide.LONG, 100.0, 101.0, 1.0, 1.0, 0.0)
    assert fresh_state.equity == pytest.approx(1001.0)
    assert fresh_state.peak_equity == pytest.approx(1001.0)
    assert len(fresh_state.trades) == 1

    # losing trade drops peak? no, peak stays at historical max
    fresh_state.record_trade(PositionSide.SHORT, 101.0, 103.0, 1.0, -2.0, 0.0)
    assert fresh_state.equity == pytest.approx(999.0)
    assert fresh_state.peak_equity == pytest.approx(1001.0)


def test_record_trade_writes_journal(fresh_state: ExecutionState):
    # fresh_state already redirects TRADE_JOURNAL to tmp_path via fixture
    fresh_state.record_trade(PositionSide.LONG, 100.0, 101.0, 1.0, 1.0, 0.0)
    data = json.loads(order_manager.TRADE_JOURNAL.read_text())
    assert len(data) == 1
    assert data[0]['side'] == 'LONG'
    assert data[0]['pnl'] == pytest.approx(1.0)


def test_circuit_breaker_daily_loss(fresh_state: ExecutionState):
    ok, reason = fresh_state.check_circuit_breakers(50000.0)
    assert ok is True
    fresh_state.daily_pnl = -0.04 * fresh_state.peak_equity
    ok, reason = fresh_state.check_circuit_breakers(50000.0)
    assert ok is False
    assert 'exceeds limit' in reason


def test_circuit_breaker_drawdown(fresh_state: ExecutionState):
    fresh_state.equity = 850.0
    fresh_state.peak_equity = 1000.0
    ok, reason = fresh_state.check_circuit_breakers(50000.0)
    assert ok is False
    assert 'drawdown=' in reason


def test_circuit_breaker_flash_crash(fresh_state: ExecutionState):
    fresh_state.last_bar_prices = [100.0, 101.0, 103.26, 103.0, 106.0]
    ok, reason = fresh_state.check_circuit_breakers(106.0)
    assert ok is False
    assert 'flash_crash' in reason


def test_circuit_breaker_equity_floor(fresh_state: ExecutionState):
    fresh_state.equity = 50.0
    ok, reason = fresh_state.check_circuit_breakers(50000.0)
    assert ok is False
    assert 'equity=50.00' in reason


def test_reset_daily_clears_pnl_only(fresh_state: ExecutionState):
    fresh_state.daily_pnl = -50.0
    fresh_state.equity = 950.0
    fresh_state.reset_daily()
    assert fresh_state.daily_pnl == 0.0
    assert fresh_state.equity == pytest.approx(950.0)


def test_simulate_fill_long_and_short():
    fill, fee = simulate_fill(PositionSide.LONG, 100.0, slippage_bps=0.0005, fee_bps=0.0005)
    assert fee == pytest.approx(0.05)
    assert fill == pytest.approx(100.1)

    fill, fee = simulate_fill(PositionSide.SHORT, 100.0, slippage_bps=0.0005, fee_bps=0.0005)
    assert fee == pytest.approx(0.05)
    assert fill == pytest.approx(99.9)

    fill, fee = simulate_fill(PositionSide.FLAT, 100.0)
    assert fee == 0.0
    assert fill == 100.0
