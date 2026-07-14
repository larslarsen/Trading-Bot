#!/usr/bin/env python3
"""
Shared portfolio execution engine for both LIVE paper trading and BACKTEST/replay.

This is the single source of truth for:
  - Position lifecycle (open / close / flatten)
  - Cash accounting (deduct on open, credit on close)
  - Mark-to-market (equity = cash + sum(shares * price))
  - Risk: circuit breakers (daily loss, max drawdown, equity floor, flash crash)
  - Halt state

It contains NO file IO and NO network access. Persistence/journalling (for the live
trader) and data fetching (for replay) live in the callers / subclasses.

Design notes (fixes carried over from the old split):
  - Sizing is equity-based, not cash-based, so the 20% target is actually reachable.
  - The flash-crash window is per-DAY (reset by start_daily_bar), not per check-call.
  - record_trade updates equity immediately AND MTM re-derives it, so equity is never
    left in a temporary wrong state between runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EngineConfig:
    """All tunable risk / cost parameters. Shared by live and replay."""
    initial_capital: float = 10000.0
    max_daily_loss_pct: float = 0.03      # 3% daily loss -> halt
    max_drawdown_pct: float = 0.20        # 20% peak-to-trough -> halt
    max_positions: int = 5                # hard cap
    max_position_pct: float = 0.20        # 20% equity per position
    min_equity_to_trade: float = 100.0    # USD floor
    flash_crash_bars: int = 5
    flash_crash_pct: float = 0.50        # 50% move in flash_crash_bars daily bars -> halt
    extreme_move_pct: float = 0.90        # above this we treat as data error, not crash
    cost_bps: float = 8.0 / 10000.0
    slippage_bps: float = 5.0 / 10000.0
    # Vol-target (optional, off by default in callers)
    enable_vol_target: bool = False
    vol_lookback: int = 20
    target_vol: float = 0.15
    min_vol_scale: float = 0.25
    max_vol_scale: float = 1.5


class PositionSide:
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


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
            "side": self.side,
            "entry": self.entry,
            "shares": self.shares,
            "entry_time": self.entry_time,
            "highest_high": self.highest_high,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["symbol"],
            d["side"],
            d["entry"],
            d["shares"],
            d.get("highest_high"),
        )


class PortfolioEngine:
    """
    Pure execution core. No persistence, no network, no printing.

    Live usage:
        eng = PortfolioEngine(EngineConfig(initial_capital=10000.0))
        eng.load_state_json(...)           # or subclass adds persistence
        eng.start_daily_bar(reference_price)
        ok, reason = eng.check_circuit_breakers()
        if not ok: eng.flatten_all(prices); eng.halt(reason)
        for sym, px in closes.items(): eng.close_position(sym, px)
        for sym, px in opens.items():
            if len(eng.positions) >= eng.config.max_positions: break
            eng.open_position(sym, px, eng.equity * eng.config.max_position_pct * scale)
        eq = eng.mark_to_market(prices)

    Replay usage is identical, just driven in a per-day loop with prefix data.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        ic = self.config.initial_capital
        self.cash = float(ic)
        self.equity = float(ic)
        self.peak_equity = float(ic)
        self.daily_pnl = 0.0
        self.trades: List[dict] = []
        self.positions: Dict[str, Position] = {}
        self.halted = False
        self.halt_reason: Optional[str] = None
        self.max_dd = 0.0
        self.equity_history: List[float] = []
        self._flash_window: List[float] = []
        # hook for subclasses (live journal); default no-op
        self._on_trade = None

    # ---- position lifecycle -------------------------------------------------
    def open_position(self, symbol, fill_price, size_usd):
        """Open a LONG position. size_usd is the TARGET notional.

        Sizing base is the engine's tracked capital base (peak_equity at first
        deposit), NOT the post-open equity, so 5 x 20% = 100% is reachable and
        each position is a stable fraction of capital.
        """
        if self.halted:
            return None
        if len(self.positions) >= self.config.max_positions:
            return None
        base = max(self.equity, self.peak_equity, self.config.initial_capital)
        target = min(size_usd, base * self.config.max_position_pct)
        if target <= 0:
            return None
        fill = fill_price * (1 + self.config.slippage_bps + self.config.cost_bps)
        if fill <= 0:
            return None
        shares = target / fill
        pos = Position(symbol, PositionSide.LONG, fill, shares)
        self.positions[symbol] = pos
        self.cash -= target  # deduct cash used; NAV unchanged at fill
        return pos

    def close_position(self, symbol, exit_price):
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        fill = exit_price * (1 - self.config.cost_bps)
        proceeds = pos.shares * fill
        cost_basis = pos.shares * pos.entry
        pnl = proceeds - cost_basis
        fees = pos.shares * pos.entry * self.config.cost_bps + pos.shares * exit_price * self.config.cost_bps
        self.cash += proceeds
        self.daily_pnl += pnl - fees
        self.equity = self.cash + self.position_value()  # always consistent
        self.peak_equity = max(self.peak_equity, self.equity)
        self._update_dd()
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "side": PositionSide.LONG,
            "symbol": symbol,
            "entry": pos.entry,
            "exit": exit_price,
            "size": pos.shares,
            "pnl": pnl,
            "fees": fees,
            "equity_after": self.equity,
        }
        self.trades.append(trade)
        if self._on_trade is not None:
            self._on_trade(trade)
        return pnl

    def flatten_all(self, prices: Dict[str, float]):
        for symbol in list(self.positions.keys()):
            px = prices.get(symbol)
            if px is not None and px > 0:
                self.close_position(symbol, px)

    # ---- valuation ---------------------------------------------------------
    def position_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        if prices is None:
            # value at entry (used only right after a close, before MTM)
            return sum(p.shares * p.entry for p in self.positions.values())
        return sum(
            pos.shares * px for pos, px in (
                (pos, prices.get(sym)) for sym, pos in self.positions.items()
            ) if px is not None and px > 0
        )

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        self.equity = self.cash + self.position_value(prices)
        self.peak_equity = max(self.peak_equity, self.equity)
        self._update_dd()
        return self.equity

    def _update_dd(self):
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity
            self.max_dd = max(self.max_dd, dd)

    # ---- risk / circuit breakers ------------------------------------------
    def start_daily_bar(self, reference_price: float):
        """Call ONCE per simulated/live day before risk checks.

        Resets the daily-loss tally and rolls the flash-crash window forward so
        the window is measured in DAYS, not in check-call count.
        """
        self.daily_pnl = 0.0
        if reference_price is not None and reference_price > 0:
            self._flash_window.append(float(reference_price))
            if len(self._flash_window) > self.config.flash_crash_bars:
                self._flash_window.pop(0)

    def check_circuit_breakers(self):
        """Returns (ok, reason). Halts the engine if a breaker trips."""
        if self.halted:
            return False, f"HALTED: {self.halt_reason}"

        if self.equity < self.config.min_equity_to_trade:
            self.halt("equity_too_low")
            return False, f"equity_too_low: equity={self.equity:.2f} < {self.config.min_equity_to_trade}"

        if self.daily_pnl < -self.config.max_daily_loss_pct * self.peak_equity:
            self.halt("daily_loss_limit")
            return False, f"daily_loss_limit: daily_pnl={self.daily_pnl:.2f} exceeds {self.config.max_daily_loss_pct*100:.2f}% limit"

        dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        if dd > self.config.max_drawdown_pct:
            self.halt("max_drawdown")
            return False, f"max_drawdown: drawdown={dd:.2%} exceeds {self.config.max_drawdown_pct:.2%}"

        if len(self._flash_window) >= self.config.flash_crash_bars:
            window_low = min(self._flash_window)
            window_high = max(self._flash_window)
            move = (window_high - window_low) / window_high if window_high > 0 else 0
            if move > self.config.flash_crash_pct:
                if move > self.config.extreme_move_pct:
                    # Probable data artifact (fetch glitch, micro-cap rounding)
                    self._flash_window = []
                else:
                    self.halt("flash_crash")
                    return False, f"flash_crash: {move:.2%} in {self.config.flash_crash_bars} bars"

        return True, None

    def halt(self, reason: str):
        self.halted = True
        self.halt_reason = reason

    # ---- vol target (optional) --------------------------------------------
    def vol_scale(self, lookback=None, target=None, min_scale=None, max_scale=None) -> float:
        if not self.config.enable_vol_target or len(self.equity_history) < (lookback or self.config.vol_lookback):
            return 1.0
        lb = lookback or self.config.vol_lookback
        vol = float(np.std(np.array(self.equity_history[-lb:])) * np.sqrt(365))
        if vol <= 0:
            return 1.0
        tgt = target or self.config.target_vol
        scale = tgt / vol
        return float(np.clip(scale, min_scale or self.config.min_vol_scale, max_scale or self.config.max_vol_scale))

    # ---- serialization helpers (used by subclasses) -----------------------
    def to_state_dict(self) -> dict:
        return {
            "cash": self.cash,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "daily_pnl": self.daily_pnl,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "trades": self.trades,
            "max_dd": self.max_dd,
            "equity_history": self.equity_history,
        }

    def load_state_dict(self, data: dict):
        self.cash = float(data.get("cash", self.cash))
        self.equity = float(data.get("equity", self.equity))
        self.peak_equity = float(data.get("peak_equity", self.peak_equity))
        self.daily_pnl = float(data.get("daily_pnl", self.daily_pnl))
        self.halted = bool(data.get("halted", self.halted))
        self.halt_reason = data.get("halt_reason", self.halt_reason)
        self.positions = {k: Position.from_dict(v) for k, v in data.get("positions", {}).items()}
        self.trades = data.get("trades", [])
        self.max_dd = float(data.get("max_dd", 0.0))
        self.equity_history = data.get("equity_history", [])
