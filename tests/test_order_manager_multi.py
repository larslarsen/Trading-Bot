"""
Tests for the LIVE persistence layer (MultiPositionState) built on the shared engine.

These assert the live-specific behavior the engine itself doesn't cover:
  - state survives JSON round-trip through MultiPositionState.save/load
  - the trade journal is appended on every close (and matches the engine's trades)
  - atomic save writes a temp file then replaces (no partial state on disk)
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from order_manager_multi import MultiPositionState, EngineConfig, CONFIG
from portfolio_engine import PortfolioEngine


@pytest.fixture
def scratch_state(tmp_path):
    state_file = tmp_path / "execution_state_multi.json"
    journal_file = tmp_path / "trade_journal.json"
    # start fresh
    if state_file.exists():
        state_file.unlink()
    if journal_file.exists():
        journal_file.unlink()
    return MultiPositionState(
        initial_capital=10000.0,
        state_file=state_file,
        journal_file=journal_file,
    ), state_file, journal_file


def test_live_state_persists_cash_and_positions(scratch_state):
    eng, sf, jf = scratch_state
    eng.start_daily_bar(100.0)
    eng.open_position("AAA", 100.0, eng.equity * 0.20)
    eng.mark_to_market({"AAA": 105.0})
    eng.save()

    # reload from disk through a fresh instance
    reloaded = MultiPositionState(state_file=sf, journal_file=jf)
    assert reloaded.cash == pytest.approx(eng.cash, rel=1e-9)
    assert reloaded.equity == pytest.approx(eng.equity, rel=1e-9)
    assert len(reloaded.positions) == 1
    assert "AAA" in reloaded.positions


def test_live_journal_appends_on_close(scratch_state):
    eng, sf, jf = scratch_state
    eng.start_daily_bar(100.0)
    eng.open_position("AAA", 100.0, eng.equity * 0.20)
    eng.close_position("AAA", 110.0)
    # journal hook writes synchronously during close
    assert jf.exists(), "journal file should be created on trade"
    journal = json.loads(jf.read_text())
    assert len(journal) == 1
    assert journal[0]["symbol"] == "AAA"
    assert journal[0]["pnl"] == pytest.approx(eng.trades[0]["pnl"], rel=1e-9)
    assert len(eng.trades) == 1


def test_live_save_is_atomic_no_partial_file(scratch_state):
    eng, sf, jf = scratch_state
    eng.start_daily_bar(100.0)
    eng.open_position("AAA", 100.0, eng.equity * 0.20)
    eng.save()
    # a .tmp should NOT be left lying around after a successful save
    tmp_files = list(sf.parent.glob("*.tmp"))
    assert tmp_files == [], f"leftover temp state files: {tmp_files}"
    # the saved file is valid JSON and parseable back into the engine
    data = json.loads(sf.read_text())
    assert "cash" in data and "positions" in data


def test_live_lock_prevents_concurrent_corruption(scratch_state):
    """Two saves back-to-back (simulating overlap) must not corrupt state."""
    eng, sf, jf = scratch_state
    eng.start_daily_bar(100.0)
    eng.open_position("AAA", 100.0, eng.equity * 0.20)
    eng.save()
    eng.open_position("BBB", 50.0, eng.equity * 0.20)
    eng.save()
    reloaded = MultiPositionState(state_file=sf, journal_file=jf)
    assert len(reloaded.positions) == 2


def test_live_reset_daily_clears_flash_window_and_persists(scratch_state):
    eng, sf, jf = scratch_state
    eng._flash_window = [100.0, 200.0, 300.0]
    eng.daily_pnl = -42.0
    eng.reset_daily()
    assert eng._flash_window == []
    assert eng.daily_pnl == 0.0
    reloaded = MultiPositionState(state_file=sf, journal_file=jf)
    assert reloaded._flash_window == []
    assert reloaded.daily_pnl == 0.0


def test_live_save_persists_halt_state(scratch_state):
    eng, sf, jf = scratch_state
    eng.halt("test_halt")
    reloaded = MultiPositionState(state_file=sf, journal_file=jf)
    assert reloaded.halted is True
    assert reloaded.halt_reason == "test_halt"


def test_live_missing_state_file_is_fresh(scratch_state):
    # a brand-new instance pointed at a non-existent file starts at defaults
    _, sf, jf = scratch_state
    if sf.exists():
        sf.unlink()
    if jf.exists():
        jf.unlink()
    eng = MultiPositionState(state_file=sf, journal_file=jf)
    assert eng.equity == pytest.approx(eng.config.initial_capital)
    assert eng.halted is False
    assert len(eng.positions) == 0
