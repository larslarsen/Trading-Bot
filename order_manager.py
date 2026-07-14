#!/usr/bin/env python3
"""
Order manager + circuit breaker.
Handles position sizing, order placement, kill switches, and trade journaling.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum

# ── CONFIG ──────────────────────────────────────────────────────────
MAX_DAILY_LOSS_PCT   = 0.03     # 3% daily loss → halt
MAX_DRAWDOWN_PCT     = 0.10     # 10% peak-to-trough → halt
MAX_POSITION_PCT     = 0.10     # 10% equity per trade
MIN_EQUITY_TO_TRADE  = 100.0    # USD
FLASH_CRASH_BARS     = 5
FLASH_CRASH_PCT      = 0.03     # 3% in 5 bars → halt
TRADE_JOURNAL = Path(__file__).parent / "trade_journal.json"
STATE_FILE    = Path(__file__).parent / "execution_state.json"

# ── STATE ──────────────────────────────────────────────────────────
class PositionSide(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1

class ExecutionState:
    def __init__(self):
        self.equity = 1000.0
        self.peak_equity = 1000.0
        self.daily_pnl = 0.0
        self.current_position = PositionSide.FLAT
        self.position_size = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.halted = False
        self.halt_reason = None
        self.last_bar_prices = []
        self._load()

    def _load(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                data = json.load(f)
            self.equity = data.get("equity", 1000.0)
            self.peak_equity = data.get("peak_equity", 1000.0)
            self.daily_pnl = data.get("daily_pnl", 0.0)
            self.halted = data.get("halted", False)
            self.halt_reason = data.get("halt_reason", None)

    def save(self):
        with open(STATE_FILE, "w") as f:
            json.dump({
                "equity": self.equity,
                "peak_equity": self.peak_equity,
                "daily_pnl": self.daily_pnl,
                "halted": self.halted,
                "halt_reason": self.halt_reason,
            }, f, indent=2)

    def record_trade(self, side, entry, exit_price, size, pnl, fees):
        self.equity += pnl - fees
        self.peak_equity = max(self.peak_equity, self.equity)
        self.daily_pnl += pnl - fees
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "side": side.name,
            "entry": entry,
            "exit": exit_price,
            "size": size,
            "pnl": pnl,
            "fees": fees,
            "equity_after": self.equity,
        }
        self.trades.append(trade)
        self._log_trade(trade)

    def _log_trade(self, trade):
        if TRADE_JOURNAL.exists():
            journal = json.loads(TRADE_JOURNAL.read_text())
        else:
            journal = []
        journal.append(trade)
        TRADE_JOURNAL.write_text(json.dumps(journal, indent=2))

    def check_circuit_breakers(self, current_price):
        """Return (ok_to_trade, reason_if_not)."""
        if self.halted:
            return False, f"HALTED: {self.halt_reason}"

        if self.equity < MIN_EQUITY_TO_TRADE:
            self.halt("equity_too_low")
            return False, f"equity={self.equity:.2f} < {MIN_EQUITY_TO_TRADE}"

        # Daily loss
        if self.daily_pnl < -MAX_DAILY_LOSS_PCT * self.peak_equity:
            self.halt("daily_loss_limit")
            return False, f"daily_pnl={self.daily_pnl:.2f} exceeds limit"

        # Drawdown
        dd = (self.peak_equity - self.equity) / self.peak_equity
        if dd > MAX_DRAWDOWN_PCT:
            self.halt("max_drawdown")
            return False, f"drawdown={dd:.2%} exceeds {MAX_DRAWDOWN_PCT:.2%}"

        # Flash crash
        self.last_bar_prices.append(current_price)
        if len(self.last_bar_prices) > FLASH_CRASH_BARS:
            self.last_bar_prices.pop(0)
            window_low = min(self.last_bar_prices)
            window_high = max(self.last_bar_prices)
            if window_high > 0 and (window_high - window_low) / window_high > FLASH_CRASH_PCT:
                self.halt("flash_crash")
                return False, f"flash_crash: {(window_high - window_low)/window_high:.2%} in {FLASH_CRASH_BARS} bars"

        return True, None

    def halt(self, reason):
        self.halted = True
        self.halt_reason = reason
        self.save()

    def reset_daily(self):
        """Call at UTC midnight."""
        self.daily_pnl = 0.0
        self.last_bar_prices = []
        self.save()

    def position_sizing(self) -> float:
        """Position size as a fraction of equity (MAX_POSITION_PCT).

        Confidence-weighted sizing was removed: the param was never wired to a
        signal and the 0.25 equity cap was always looser than MAX_POSITION_PCT,
        so this returns the flat cap unambiguously.
        """
        return self.equity * MAX_POSITION_PCT


def simulate_fill(side, price, slippage_bps=0.0005, fee_bps=0.0005):
    """Simulate realistic fill with slippage + fees."""
    slip = price * slippage_bps
    fee = price * fee_bps
    if side == PositionSide.LONG:
        fill = price + slip + fee
    elif side == PositionSide.SHORT:
        fill = price - slip - fee
    else:
        return price, 0.0
    return fill, fee


if __name__ == "__main__":
    state = ExecutionState()
    print(f"State: equity={state.equity:.2f}, peak={state.peak_equity:.2f}, halted={state.halted}")
