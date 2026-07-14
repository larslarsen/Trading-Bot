"""
Tests for paper_trader.py helpers and execution path.
"""
import json
from io import BytesIO
from unittest.mock import patch, MagicMock
import pytest

try:
    from paper_trader import fill_price, run_once, FEE_PROFILES, PROFILE, SLIPPAGE_BP, SIZE_USD, JOURNAL_FILE
    from order_manager import ExecutionState, PositionSide
    HAS_LEGACY = True
except Exception:
    HAS_LEGACY = False
    fill_price = run_once = None
    class ExecutionState: pass
    class PositionSide:
        LONG = "LONG"
        SHORT = "SHORT"
        FLAT = "FLAT"



@pytest.mark.skipif(not HAS_LEGACY, reason="legacy paper_trader not importable")
def test_fill_price_long_short_flat():
    assert fill_price(100.0, "LONG", 5) == pytest.approx(100.05)
    assert fill_price(100.0, "SHORT", 5) == pytest.approx(99.95)
    assert fill_price(100.0, "FLAT", 5) == pytest.approx(100.0)
    assert fill_price(100.0, "LONG", 0) == pytest.approx(100.0)


@pytest.mark.skipif(not HAS_LEGACY, reason="legacy paper_trader not importable")
def test_run_once_hold_when_flat(tmp_path, monkeypatch):
    state = ExecutionState()
    monkeypatch.setattr('order_manager.STATE_FILE', tmp_path / 'state.json')
    monkeypatch.setattr('paper_trader.JOURNAL_FILE', tmp_path / 'trade_journal.json')

    payload = json.dumps({"signal": "FLAT", "timestamp": "2026-01-01T00:00:00+00:00"}).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = payload

    with patch('paper_trader.get', side_effect=[
        {"equity": 1000.0, "peak_equity": 1000.0, "daily_pnl": 0.0, "halted": False, "halt_reason": None, "trade_count": 0, "recent_trades": []},
        {"signal": "FLAT", "timestamp": "2026-01-01T00:05:00+00:00"},
    ]):
        with patch('pandas.read_csv', return_value=_mock_history()):
            run_once(state)
    assert state.current_position == PositionSide.FLAT


@pytest.mark.skipif(not HAS_LEGACY, reason="legacy paper_trader not importable")
def test_run_once_open_long_triggers_trade(tmp_path, monkeypatch, capsys):
    state = ExecutionState()
    monkeypatch.setattr('order_manager.STATE_FILE', tmp_path / 'state.json')
    monkeypatch.setattr('paper_trader.JOURNAL_FILE', tmp_path / 'trade_journal.json')

    with patch('paper_trader.get', side_effect=[
        {"equity": 1000.0, "peak_equity": 1000.0, "daily_pnl": 0.0, "halted": False, "halt_reason": None, "trade_count": 0, "recent_trades": []},
        {"signal": "LONG", "confidence": 0.8, "timestamp": "2026-01-01T00:05:00+00:00"},
    ]):
        with patch('pandas.read_csv', return_value=_mock_history()):
            run_once(state)

    assert state.current_position == PositionSide.LONG
    assert state.position_size > 0
    # opening deducts fees -> equity drops below the starting 1000.0
    assert state.equity < 1000.0
    assert state.equity > 0


@pytest.mark.skipif(not HAS_LEGACY, reason="legacy paper_trader not importable")
def test_run_once_circuit_breaker_blocks(tmp_path, monkeypatch):
    state = ExecutionState()
    state.halted = True
    state.halt_reason = "daily_loss_limit"
    monkeypatch.setattr('order_manager.STATE_FILE', tmp_path / 'state.json')
    monkeypatch.setattr('paper_trader.JOURNAL_FILE', tmp_path / 'trade_journal.json')

    with patch('paper_trader.get', side_effect=[
        {"equity": 1000.0, "peak_equity": 1000.0, "daily_pnl": 0.0, "halted": True, "halt_reason": "daily_loss_limit", "trade_count": 0, "recent_trades": []},
        {"signal": "LONG", "timestamp": "2026-01-01T00:05:00+00:00"},
    ]):
        with patch('pandas.read_csv', return_value=_mock_history()):
            run_once(state)
    assert state.current_position == PositionSide.FLAT


def _mock_history():
    import pandas as pd
    return pd.DataFrame({"ts": pd.to_datetime(["2026-01-01T00:05:00+00:00"]), "close": [50000.0]})
