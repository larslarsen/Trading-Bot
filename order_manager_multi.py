#!/usr/bin/env python3
"""
Live multi-position execution engine for screened altcoins.

Persistence + trade journal are layered on top of the shared PortfolioEngine
(portfolio_engine.py). All cash/MTM/risk logic lives in the base engine so that
the live trader and the backtest replay use IDENTICAL execution math.
"""

import json
import fcntl
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import CONFIG
from portfolio_engine import PortfolioEngine, EngineConfig, Position

TRADE_JOURNAL = Path(__file__).parent / "trade_journal.json"
STATE_FILE = Path(__file__).parent / "execution_state_multi.json"


class MultiPositionState(PortfolioEngine):
    """Live state: adds JSON persistence + trade-journal append to the engine."""

    def __init__(self, initial_capital=10000.0, state_file=STATE_FILE, journal_file=TRADE_JOURNAL):
        super().__init__(EngineConfig(**{**CONFIG.__dict__}))
        self._state_file = Path(state_file)
        self._journal_file = Path(journal_file)
        # wire journal hook
        self._on_trade = self._log_trade
        self._load()

    # persistence ------------------------------------------------------------
    def _load(self):
        if self._state_file.exists():
            with open(self._state_file) as f:
                data = json.load(f)
            self.load_state_dict(data)

    def save(self):
        """Atomic + locked save so a second concurrent run can't corrupt state."""
        with open(self._state_file, "a+") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                tmp = self._state_file.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    # to_state_dict() returns only JSON-native types (float/int/
                    # bool/list/dict); drop default=str so a genuinely
                    # non-serializable field raises instead of being silently
                    # stringified into persisted state.
                    json.dump(self.to_state_dict(), f, indent=2)
                tmp.replace(self._state_file)
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)
        # `with` closes lockf on exit -> no fd leak across many save() calls.

    def _log_trade(self, trade):
        # Atomic + locked append so a crash mid-write cannot corrupt the journal
        # (same crash-safety discipline as save()). Reads the full list (trade
        # counts/day are bounded) then writes a tmp file + os.replace.
        if self._journal_file.exists():
            try:
                journal = json.loads(self._journal_file.read_text())
            except Exception:
                journal = []
        else:
            journal = []
        journal.append(trade)
        tmp = self._journal_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(journal, f, indent=2)
        tmp.replace(self._journal_file)

    # daily reset used by ops cron-style runs
    def reset_daily(self):
        self.daily_pnl = 0.0
        self._flash_window = []
        self.save()


if __name__ == "__main__":
    state = MultiPositionState()
    print(f"Multi-position state: equity={state.equity:.2f}, peak={state.peak_equity:.2f}, "
          f"halted={state.halted}, positions={len(state.positions)}")
