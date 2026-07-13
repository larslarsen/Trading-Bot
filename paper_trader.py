#!/usr/bin/env python3
"""
Paper trader: polls /signal, simulates execution via order_manager,
writes to trade_journal, and respects circuit breakers.
"""
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from order_manager import ExecutionState, PositionSide
from pipeline import COST

# ── CONFIG ─────────────────────────────────────────────────────────────
SIGNAL_URL   = "http://127.0.0.1:8080/signal"
DASH_URL     = "http://127.0.0.1:8080/dashboard"
POLL_SEC     = 300        # 5 minutes, aligned to 5m bars
# Skip trades whose top-class confidence is below this.
CONFIDENCE_THRESHOLD = 0.60
FEE_PROFILES = {
    "bybit_vip0":        {"maker_bp": 0.20, "taker_bp": 0.55},
    "blofin_regular":    {"maker_bp": 0.20, "taker_bp": 0.60},
    "mexc_api":          {"maker_bp": 0.60, "taker_bp": 0.80},
    "woox_regular":      {"maker_bp": 0.60, "taker_bp": 2.50},
    "okx_us":            {"maker_bp": 2.00, "taker_bp": 3.50},
    "hyperliquid":       {"maker_bp": 0.15, "taker_bp": 0.45},
}
PROFILE = "blofin_regular"

SLIPPAGE_BP  = 5
SIZE_USD     = 100.0
JOURNAL_FILE = Path(__file__).parent / "trade_journal.json"

# ── HELPERS ─────────────────────────────────────────────────────────


def get(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def pct_bp(val: float) -> float:
    return val * 10_000.0


def fill_price(price: float, side: str, slippage_bp: float) -> float:
    s = slippage_bp / 10_000.0
    if side == "LONG":
        return price * (1.0 + s)
    if side == "SHORT":
        return price * (1.0 - s)
    return price


# ── MAIN LOOP ───────────────────────────────────────────────────────


def run_once(state: ExecutionState):
    dash = get(DASH_URL)
    signal_data = get(SIGNAL_URL)
    signal = signal_data.get("signal", "FLAT")
    confidence = float(signal_data.get("confidence", 0.0))
    ts = signal_data.get("timestamp") or datetime.now(timezone.utc).isoformat()

    if signal != "FLAT" and confidence < CONFIDENCE_THRESHOLD:
        print(f"[{ts}] confidence {confidence:.3f} < {CONFIDENCE_THRESHOLD}, staying FLAT")
        signal = "FLAT"
    price = None

    # Try to get last traded price from the local OHLCV history if signal payload lacks it.
    # We derive from the inference result's source data by reusing the data feed.
    try:
        from data_feed import HISTORY_CSV, LOCAL_CSV
        import pandas as pd
        hist = pd.read_csv(HISTORY_CSV, parse_dates=["ts"])
        local = pd.read_csv(LOCAL_CSV, parse_dates=["ts"])
        combined = pd.concat([local, hist], ignore_index=True).drop_duplicates("ts").sort_values("ts")
        if not combined.empty:
            last = combined.iloc[-1]
            price = float(last["close"])
    except Exception:
        price = None

    if price is None:
        print(f"[{ts}] no price available, skipping")
        return

    ok, reason = state.check_circuit_breakers(price)
    if not ok:
        print(f"[{ts}] CIRCUIT BREAKER: {reason}")
        return

    # Decide action based on current position and new signal
    current = state.current_position
    action = "HOLD"
    side = None
    entry = exit_price = None
    pnl = 0.0
    fees = 0.0

    if signal == "LONG" and current != PositionSide.LONG:
        if current == PositionSide.SHORT:
            # close short
            exit_price = fill_price(price, "SHORT", SLIPPAGE_BP)
            fees = SIZE_USD * FEE_PROFILES[PROFILE]["taker_bp"] / 10_000.0
            fees += SIZE_USD * FEE_PROFILES[PROFILE]["taker_bp"] / 10_000.0
            pnl_short = (state.entry_price - exit_price) / state.entry_price * state.position_size - fees
            entry_price = state.entry_price
            state.record_trade(PositionSide.SHORT, entry_price, exit_price, state.position_size, -pnl_short, fees)
            print(f"[{ts}] CLOSE SHORT exit={exit_price:.2f} pnl={-pnl_short:.2f} fees={fees:.2f} equity={state.equity:.2f}")
            state.current_position = PositionSide.FLAT
            state.position_size = 0.0
            state.entry_price = 0.0
            state.save()

        # open long
        entry = fill_price(price, "LONG", SLIPPAGE_BP)
        fee_rate = FEE_PROFILES[PROFILE]["taker_bp"] / 10_000.0
        fees = SIZE_USD * fee_rate + SIZE_USD * fee_rate
        state.entry_price = entry
        state.position_size = SIZE_USD / entry if entry > 0 else 0.0
        state.current_position = PositionSide.LONG
        print(f"[{ts}] OPEN LONG entry={entry:.2f} size={state.position_size:.6f} fees={fees:.2f} equity={state.equity:.2f}")

    elif signal == "SHORT" and current != PositionSide.SHORT:
        if current == PositionSide.LONG:
            # close long
            exit_price = fill_price(price, "LONG", SLIPPAGE_BP)
            fees = SIZE_USD * FEE_PROFILES[PROFILE]["taker_bp"] / 10_000.0
            fees += SIZE_USD * FEE_PROFILES[PROFILE]["taker_bp"] / 10_000.0
            pnl_long = (exit_price - state.entry_price) / state.entry_price * state.position_size - fees
            entry_price = state.entry_price
            state.record_trade(PositionSide.LONG, entry_price, exit_price, state.position_size, pnl_long, fees)
            print(f"[{ts}] CLOSE LONG exit={exit_price:.2f} pnl={pnl_long:.2f} fees={fees:.2f} equity={state.equity:.2f}")
            state.current_position = PositionSide.FLAT
            state.position_size = 0.0
            state.entry_price = 0.0
            state.save()

        # open short
        entry = fill_price(price, "SHORT", SLIPPAGE_BP)
        fee_rate = FEE_PROFILES[PROFILE]["taker_bp"] / 10_000.0
        fees = SIZE_USD * fee_rate + SIZE_USD * fee_rate
        state.entry_price = entry
        state.position_size = SIZE_USD / entry if entry > 0 else 0.0
        state.current_position = PositionSide.SHORT
        print(f"[{ts}] OPEN SHORT entry={entry:.2f} size={state.position_size:.6f} fees={fees:.2f} equity={state.equity:.2f}")
    else:
        print(f"[{ts}] signal={signal} current={current.name if current else 'UNKNOWN'} price={price:.2f} HOLD")

    print(f"[{ts}] equity={state.equity:.2f} peak={state.peak_equity:.2f} dd={((state.peak_equity-state.equity)/state.peak_equity*100 if state.peak_equity>0 else 0):.2f}%")


def main():
    print("Paper trader starting...")
    state = ExecutionState()
    while True:
        try:
            run_once(state)
        except Exception as e:
            print(f"[{datetime.now(timezone.utc).isoformat()}] error: {e}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
