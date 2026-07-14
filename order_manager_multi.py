#!/usr/bin/env python3
"""
Live multi-position execution engine for screened altcoins.

Persistence + trade journal are layered on top of the shared PortfolioEngine
(portfolio_engine.py). All cash/MTM/risk logic lives in the base engine so that
the live trader and the backtest replay use IDENTICAL execution math.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from portfolio_engine import PortfolioEngine, EngineConfig, Position


# ── CONFIG (kept here for the live trader; tune in one place) ─────────────
CONFIG = EngineConfig(
    initial_capital=10000.0,
    max_daily_loss_pct=0.03,
    max_drawdown_pct=0.20,
    max_positions=5,
    max_position_pct=0.20,
    min_equity_to_trade=100.0,
    flash_crash_bars=5,
    flash_crash_pct=0.50,
    extreme_move_pct=0.90,
    cost_bps=8.0 / 10000.0,
    slippage_bps=5.0 / 10000.0,
    enable_vol_target=False,   # parity-vol target hook (off by default)
    vol_lookback=20,
    target_vol=0.15,
    min_vol_scale=0.25,
    max_vol_scale=1.5,
)

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
        tmp = self._state_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_state_dict(), f, indent=2, default=str)
        # atomic replace
        tmp.replace(self._state_file)

    def _log_trade(self, trade):
        if self._journal_file.exists():
            journal = json.loads(self._journal_file.read_text())
        else:
            journal = []
        journal.append(trade)
        self._journal_file.write_text(json.dumps(journal, indent=2))

    # daily reset used by ops cron-style runs
    def reset_daily(self):
        self.daily_pnl = 0.0
        self._flash_window = []
        self.save()


if __name__ == "__main__":
    state = MultiPositionState()
    print(f"Multi-position state: equity={state.equity:.2f}, peak={state.peak_equity:.2f}, "
          f"halted={state.halted}, positions={len(state.positions)}")
