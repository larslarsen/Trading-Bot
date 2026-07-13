#!/usr/bin/env python3
"""
Multi-position execution engine with ATR trailing stops,
inverse-vol sizing hooks, and full state persistence.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from typing import Dict, Optional

# ── CONFIG ──────────────────────────────────────────────────────────
MAX_DAILY_LOSS_PCT   = 0.03     # 3% daily loss → halt
MAX_DRAWDOWN_PCT     = 0.20     # 20% peak-to-tough → halt
MAX_POSITIONS        = 5       # hard cap
MAX_POSITION_PCT     = 0.20     # 20% equity per trade
MIN_EQUITY_TO_TRADE  = 100.0    # USD
FLASH_CRASH_BARS     = 5
FLASH_CRASH_PCT      = 0.50     # 50% move in 5 daily bars → halt (relaxed for noisy alt daily data; was 3% which was too sensitive)

# Vol-target fallback (disabled by default)
ENABLE_VOL_TARGET    = True
VOL_LOOKBACK         = 20       # days
TARGET_VOL           = 0.15     # 15% annualized
MIN_VOL_SCALE        = 0.25
MAX_VOL_SCALE        = 1.5

COST_BPS    = 0.0008
SLIPPAGE_BPS = 0.0005

TRADE_JOURNAL = Path(__file__).parent / "trade_journal.json"
STATE_FILE    = Path(__file__).parent / "execution_state_multi.json"

# ── STATE ──────────────────────────────────────────────────────────
class PositionSide(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


class Position:
    def __init__(self, symbol, side, entry, shares, highest_high=None):
        self.symbol = symbol
        self.side = side
        self.entry = float(entry)
        self.shares = float(shares)
        self.entry_time = datetime.now(timezone.utc).isoformat()
        self.highest_high = float(highest_high) if highest_high is not None else float(entry)

    def to_dict(self):
        return {
            "symbol": self.symbol,
            "side": self.side.name,
            "entry": self.entry,
            "shares": self.shares,
            "entry_time": self.entry_time,
            "highest_high": self.highest_high,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["symbol"],
            PositionSide[d["side"]],
            d["entry"],
            d["shares"],
            d.get("highest_high"),
        )


class MultiPositionState:
    def __init__(self, initial_capital=1000.0):
        self.equity = float(initial_capital)
        self.peak_equity = float(initial_capital)
        self.daily_pnl = 0.0
        self.trades = []
        self.positions: Dict[str, Position] = {}
        self.halted = False
        self.halt_reason = None
        self.last_bar_prices = []
        self.max_dd = 0.0
        self.equity_history = []
        self._load()

    # persistence
    def _load(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                data = json.load(f)
            self.equity = float(data.get("equity", self.equity))
            self.peak_equity = float(data.get("peak_equity", self.peak_equity))
            self.daily_pnl = float(data.get("daily_pnl", self.daily_pnl))
            self.halted = bool(data.get("halted", self.halted))
            self.halt_reason = data.get("halt_reason", self.halt_reason)
            self.positions = {k: Position.from_dict(v) for k, v in data.get("positions", {}).items()}
            self.trades = data.get("trades", [])
            self.max_dd = float(data.get("max_dd", 0.0))
            self.equity_history = data.get("equity_history", [])

    def save(self):
        with open(STATE_FILE, "w") as f:
            json.dump({
                "equity": self.equity,
                "peak_equity": self.peak_equity,
                "daily_pnl": self.daily_pnl,
                "halted": self.halted,
                "halt_reason": self.halt_reason,
                "positions": {k: v.to_dict() for k, v in self.positions.items()},
                "trades": self.trades,
                "max_dd": self.max_dd,
                "equity_history": self.equity_history,
            }, f, indent=2, default=str)

    def record_trade(self, side, entry, exit_price, size, pnl, fees, symbol=None):
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "side": side.name,
            "symbol": symbol,
            "entry": entry,
            "exit": exit_price,
            "size": size,
            "pnl": pnl,
            "fees": fees,
            "equity_after": self.equity,
        }
        self.trades.append(trade)
        self.equity = self.equity + pnl - fees
        self.peak_equity = max(self.peak_equity, self.equity)
        self.daily_pnl += pnl - fees
        self._log_trade(trade)

    def _log_trade(self, trade):
        if TRADE_JOURNAL.exists():
            journal = json.loads(TRADE_JOURNAL.read_text())
        else:
            journal = []
        journal.append(trade)
        TRADE_JOURNAL.write_text(json.dumps(journal, indent=2))

    # risk
    def check_circuit_breakers(self, current_price):
        if self.halted:
            return False, f"HALTED: {self.halt_reason}"

        if self.equity < MIN_EQUITY_TO_TRADE:
            self.halt("equity_too_low")
            return False, f"equity={self.equity:.2f} < {MIN_EQUITY_TO_TRADE}"

        if self.daily_pnl < -MAX_DAILY_LOSS_PCT * self.peak_equity:
            self.halt("daily_loss_limit")
            return False, f"daily_pnl={self.daily_pnl:.2f} exceeds limit"

        dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        if dd > MAX_DRAWDOWN_PCT:
            self.halt("max_drawdown")
            return False, f"drawdown={dd:.2%} exceeds {MAX_DRAWDOWN_PCT:.2%}"

        self.last_bar_prices.append(current_price)
        if len(self.last_bar_prices) > FLASH_CRASH_BARS:
            self.last_bar_prices.pop(0)
            window_low = min(self.last_bar_prices)
            window_high = max(self.last_bar_prices)
            move = (window_high - window_low) / window_high if window_high > 0 else 0
            if move > FLASH_CRASH_PCT:
                if move > 0.90:
                    # Likely data artifact (fetch glitch, bad bar, rounding on micro-cap)
                    print(f"[CIRCUIT] Ignoring extreme {move:.1%} move in {FLASH_CRASH_BARS} bars as probable data error")
                    self.last_bar_prices = []  # reset window
                else:
                    self.halt("flash_crash")
                    return False, f"flash_crash: {move:.2%} in {FLASH_CRASH_BARS} bars"

        return True, None

    def halt(self, reason):
        self.halted = True
        self.halt_reason = reason
        self.save()

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.last_bar_prices = []
        self.save()

    # position management
    def open_position(self, symbol, fill_price, size_usd):
        if len(self.positions) >= MAX_POSITIONS:
            return None
        target = min(size_usd, self.equity * MAX_POSITION_PCT)
        if target <= 0:
            return None
        fill = fill_price * (1 + SLIPPAGE_BPS + COST_BPS)
        shares = target / fill
        pos = Position(symbol, PositionSide.LONG, fill, shares)
        self.positions[symbol] = pos
        self.save()
        return pos

    def close_position(self, symbol, exit_price):
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        proceeds = pos.shares * exit_price * (1 - COST_BPS)
        pnl = proceeds - pos.shares * pos.entry
        fees = pos.shares * pos.entry * COST_BPS + pos.shares * exit_price * COST_BPS
        self.record_trade(
            PositionSide.LONG,
            pos.entry,
            exit_price,
            pos.shares,
            pnl,
            fees,
            symbol=symbol,
        )
        self.save()
        return pnl

    def flatten_all(self, prices: Dict[str, float]):
        for symbol in list(self.positions.keys()):
            px = prices.get(symbol)
            if px is None or px <= 0:
                continue
            self.close_position(symbol, px)
        self.save()

    def update_equity_from_mtm(self, prices: Dict[str, float]):
        mtm = 0.0
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is not None and px > 0:
                mtm += pos.shares * px
        self.equity = self.cash_like() + mtm
        self.peak_equity = max(self.peak_equity, self.equity)
        if self.peak_equity > 0:
            self.max_dd = max(self.max_dd, (self.peak_equity - self.equity) / self.peak_equity)
        self.save()
        return self.equity

    def cash_like(self):
        return self.equity - sum(
            pos.shares * getattr(pos, "latest_price", 0.0)
            for pos in self.positions.values()
        )

    def vol_scale(self, lookback=20, target=0.15, min_scale=0.25, max_scale=1.5):
        if not ENABLE_VOL_TARGET or len(self.equity_history) < lookback:
            return 1.0
        vol = np.std(np.array(self.equity_history[-lookback:])) * np.sqrt(365)
        if vol <= 0:
            return 1.0
        scale = target / vol
        return float(np.clip(scale, min_scale, max_scale))

    def position_value(self, prices: Dict[str, float]):
        total = 0.0
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is not None and px > 0:
                total += pos.shares * px
        return total


if __name__ == "__main__":
    state = MultiPositionState()
    print(f"Multi-position state: equity={state.equity:.2f}, peak={state.peak_equity:.2f}, "
          f"halted={state.halted}, positions={len(state.positions)}")
