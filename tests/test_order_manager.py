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


# ── Edge cases / branch coverage ──────────────────────────────────────────
def test_halted_state_blocks_trading_early(fresh_state: ExecutionState):
    fresh_state.halted = True
    fresh_state.halt_reason = "max_drawdown"
    ok, reason = fresh_state.check_circuit_breakers(50000.0)
    assert ok is False
    assert "HALTED" in reason


def test_halt_persists_to_state_file(fresh_state: ExecutionState, tmp_path: Path):
    fresh_state.halt("manual")
    data = json.loads(order_manager.STATE_FILE.read_text())
    assert data["halted"] is True
    assert data["halt_reason"] == "manual"


def test_record_trade_with_fees_reduces_equity(fresh_state: ExecutionState):
    # equity += pnl - fees (not pnl alone)
    fresh_state.record_trade(PositionSide.LONG, 100.0, 101.0, 1.0, 1.0, 0.3)
    assert fresh_state.equity == pytest.approx(1000.0 + 1.0 - 0.3)
    assert fresh_state.trades[-1]["equity_after"] == pytest.approx(fresh_state.equity)


def test_position_sizing_uses_max_position_pct(fresh_state: ExecutionState):
    fresh_state.equity = 5000.0
    assert fresh_state.position_sizing() == pytest.approx(500.0)  # 10% of equity
    fresh_state.equity = 1234.5
    assert fresh_state.position_sizing() == pytest.approx(123.45)


def test_simulate_fill_custom_bps():
    fill, fee = simulate_fill(PositionSide.LONG, 200.0, slippage_bps=0.001, fee_bps=0.002)
    assert fee == pytest.approx(200.0 * 0.002)
    assert fill == pytest.approx(200.0 * (1 + 0.001 + 0.002))
    fill, fee = simulate_fill(PositionSide.SHORT, 200.0, slippage_bps=0.001, fee_bps=0.002)
    assert fill == pytest.approx(200.0 * (1 - 0.001 - 0.002))


def test_flash_crash_not_triggered_below_threshold(fresh_state: ExecutionState):
    # 6-sample window with a move just under FLASH_CRASH_PCT (3%) -> ok
    fresh_state.last_bar_prices = [100.0, 100.0, 100.0, 100.0, 100.0]
    ok, reason = fresh_state.check_circuit_breakers(101.0)
    assert ok is True
    assert reason is None


def test_flash_crash_zero_price_no_division_error(fresh_state: ExecutionState):
    # all-zero prices: window_high == 0 -> guard skips, no ZeroDivisionError
    fresh_state.last_bar_prices = [0.0, 0.0, 0.0, 0.0, 0.0]
    ok, reason = fresh_state.check_circuit_breakers(0.0)
    assert ok is True


def test_circuit_breaker_skips_flash_until_window_full(fresh_state: ExecutionState):
    # fewer than FLASH_CRASH_BARS+1 samples -> flash branch not evaluated
    fresh_state.last_bar_prices = [100.0, 50.0]  # big move but tiny window
    ok, _ = fresh_state.check_circuit_breakers(50.0)
    assert ok is True  # not halted by flash (window too short to judge)


def test_check_circuit_breakers_appends_price_to_window(fresh_state: ExecutionState):
    # check_circuit_breakers appends current_price into last_bar_prices (rolling history)
    fresh_state.last_bar_prices = [100.0, 101.0]
    fresh_state.check_circuit_breakers(102.0)
    assert fresh_state.last_bar_prices[-1] == 102.0
    assert len(fresh_state.last_bar_prices) == 3


def test_save_roundtrip_preserves_state(fresh_state: ExecutionState, tmp_path: Path):
    fresh_state.equity = 1234.5
    fresh_state.peak_equity = 1400.0
    fresh_state.daily_pnl = -12.0
    fresh_state.halted = True
    fresh_state.halt_reason = "manual"
    # NOTE: save() persists only the circuit-breaker fields
    # (equity/peak/daily_pnl/halted/halt_reason), not current_position.
    fresh_state.save()
    reloaded = ExecutionState()
    assert reloaded.equity == pytest.approx(1234.5)
    assert reloaded.peak_equity == pytest.approx(1400.0)
    assert reloaded.daily_pnl == pytest.approx(-12.0)
    assert reloaded.halted is True
    assert reloaded.halt_reason == "manual"


def test_journal_appends_across_trades(fresh_state: ExecutionState):
    # a second trade must APPEND to the journal, not overwrite it
    fresh_state.record_trade(PositionSide.LONG, 100.0, 101.0, 1.0, 1.0, 0.0)
    fresh_state.record_trade(PositionSide.SHORT, 101.0, 103.0, 1.0, -2.0, 0.0)
    data = json.loads(order_manager.TRADE_JOURNAL.read_text())
    assert len(data) == 2
    assert data[0]['side'] == 'LONG' and data[1]['side'] == 'SHORT'


def test_load_from_existing_state_file(tmp_path: Path, monkeypatch):
    # pre-seed a state file, then construct -> loads it (not defaults)
    state_file = tmp_path / 'execution_state.json'
    state_file.write_text(json.dumps({
        "equity": 777.0, "peak_equity": 888.0, "daily_pnl": -5.0,
        "halted": True, "halt_reason": "test",
        # current_position is intentionally NOT loaded -> stays default FLAT
    }))
    monkeypatch.setattr('order_manager.STATE_FILE', state_file)
    monkeypatch.setattr('order_manager.TRADE_JOURNAL', tmp_path / 'j.json')
    st = ExecutionState()
    assert st.equity == pytest.approx(777.0)
    assert st.peak_equity == pytest.approx(888.0)
    assert st.daily_pnl == pytest.approx(-5.0)
    assert st.halted is True
    assert st.current_position == PositionSide.FLAT  # not persisted
