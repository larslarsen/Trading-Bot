#!/usr/bin/env python3
"""Independent DEX paper trader (separate from the CEX live trader).

Mirrors paper_trader_multi.py logic but:
- trades DEX coins only (dex_data/ + latest screen_dex_idio_*),
- uses its OWN state file + trade journal (no collision with the CEX trader),
- writes its OWN cron log.

Same engine/config: rei-trend / cci-chop + d40 fill, 5 positions, 20% sizing,
20% DD halt, costs. Regime via the live MA method (method="ma") to stay
consistent with the running CEX trader's config.

Run from cron independently, e.g. 07:05 UTC (after the CEX run at 00:05 UTC,
or any time — DEX and CEX state files are disjoint).
"""
import sys
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from order_manager_multi import MultiPositionState
from config import (
    CONFIG, USE_IMPROVED_REGIME, USE_MA_REGIME, USE_HURST_REGIME,
    HURST_WINDOW, HURST_THRESHOLD, TREND_RULE, CHOP_RULE, SECONDARY_CHOP_RULE,
    USE_PARANOID_VOL_TARGET,
)
from engine import compute_live_regime, get_regime_signals, improved_compute_live_regime

# Self-contained state (independent of the CEX trader's files)
STATE_FILE = Path("execution_state_dex.json")
JOURNAL_FILE = Path("trade_journal_dex.json")
ROOT = Path("dex_data")
OUT = Path("backtest_output")
INITIAL = 10000.0
FETCH = True          # no live DEX fetch endpoint wired; uses dex_data/ on disk
LOOKBACK = 40

print(f"\n=== paper_trader_dex run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} ===", flush=True)
print("START paper_trader_dex.py", flush=True)


def _col(df, name):
    return pd.Series(df[name].values, index=df.index)


def load_dex_universe():
    """Preference: selection shortlist (76 coins ranked by OOS return). Fallback: latest DEX screen."""
    shortlist = Path("dex_selection_shortlist.csv")
    if shortlist.exists():
        sl = pd.read_csv(shortlist)
        stems = [str(s).strip().upper() for s in sl["symbol"].tolist()]
        print(f"Universe: dex_selection_shortlist.csv ({len(stems)} selected coins)")
        return stems
    screens = sorted(OUT.glob("screen_dex_idio_*.csv"))
    if not screens:
        raise FileNotFoundError("no dex_selection_shortlist.csv or screen_dex_idio_*.csv")
    screen = pd.read_csv(screens[-1])
    screen = screen[screen.tier.isin(["large", "mid", "tail"])]
    stems = [str(s).strip().upper() for s in (screen.stem.tolist() if "stem" in screen.columns else screen.symbol.tolist())]
    print(f"Universe: latest screen_dex_idio_* ({len(stems)} coins) [shortlist missing]")
    return stems


def main():
    # Independent state
    state = MultiPositionState(initial_capital=INITIAL,
                                state_file=STATE_FILE, journal_file=JOURNAL_FILE)

    stems = load_dex_universe()

    prices = {}
    for stem in stems:
        stem = str(stem).strip().upper()
        p = ROOT / f"{stem}_1d_max.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["ts"]).dropna(subset=["close", "high", "low", "volume"])
        df = df.sort_values("ts").reset_index(drop=True)
        if len(df) < 21:
            continue
        prices[stem] = df

    print(f"Loaded {len(prices)} DEX coins")

    # Regime (same MA method as the live CEX trader)
    try:
        if USE_HURST_REGIME:
            regime = compute_live_regime(prices)
        else:
            regime = improved_compute_live_regime(prices, method="ma" if USE_MA_REGIME else "rule")
    except Exception as e:
        print(f"REGIME FALLBACK ({e!r})")
        regime = "trend"

    active_rule = TREND_RULE if regime == "trend" else CHOP_RULE
    print(f"DEX Regime: {regime} -> using {active_rule} (ma={USE_MA_REGIME})")

    # Signals + strength ranking (same as CEX trader)
    raw_signals = {}
    for stem, df in prices.items():
        try:
            entry, exit_sig = get_regime_signals(active_rule, df)
            last_entry = int(entry.iloc[-1]) if len(entry) > 0 else 0
            last_exit = int(exit_sig.iloc[-1]) if len(exit_sig) > 0 else 0
            strength = 1.0
            if active_rule == "williams_r":
                high = _col(df, "high"); low = _col(df, "low"); close = _col(df, "close")
                period = 14
                highest = high.rolling(period, min_periods=1).max()
                lowest = low.rolling(period, min_periods=1).min()
                wr = -100 * (highest - close) / (highest - lowest + 1e-12)
                last_wr = float(wr.iloc[-1]) if len(wr) > 0 else -50.0
                wr_diff = float(wr.diff().iloc[-1]) if len(wr) > 1 else 0.0
                strength = (-last_wr) + max(0.0, wr_diff * 2.0)
            elif active_rule == "rei":
                close = _col(df, "close"); high = _col(df, "high"); low = _col(df, "low")
                up_move = high - high.shift(1)
                down_move = low.shift(1) - low
                up = up_move.where((up_move > 0) & (up_move > down_move), 0).fillna(0)
                down = down_move.where((down_move > 0) & (down_move > up_move), 0).fillna(0)
                rng = (high - low).rolling(14, min_periods=1).mean()
                rei = 100 * (up.rolling(14, min_periods=1).sum() - down.rolling(14, min_periods=1).sum()) / (rng + 1e-12)
                last_rei = float(rei.iloc[-1]) if len(rei) > 0 else 0.0
                rei_diff = float(rei.diff().iloc[-1]) if len(rei) > 1 else 0.0
                strength = last_rei + max(0.0, rei_diff * 0.5)
            elif active_rule == "donchian40":
                high = _col(df, "high"); close = _col(df, "close")
                upper = high.rolling(40, min_periods=1).max().shift(1)
                strength = float(((close.iloc[-1] - upper.iloc[-1]) / (upper.iloc[-1] + 1e-12)) * 100) if len(close) > 1 else 0.0
            raw_signals[stem] = {"entry": last_entry, "exit": last_exit, "strength": strength}
        except Exception as e:
            print(f"WARN regime signal failed for {stem}: {e!r}")
            raw_signals[stem] = {"entry": 0, "exit": 0, "strength": 0.0}

    active = [s for s in raw_signals if raw_signals[s]["entry"]]
    active = sorted(active, key=lambda s: -raw_signals[s]["strength"])

    # Secondary chop fill-in (same logic as CEX trader)
    if regime == "chop" and not active and SECONDARY_CHOP_RULE:
        raw_signals = {}
        for stem, df in prices.items():
            try:
                entry, exit_sig = get_regime_signals(SECONDARY_CHOP_RULE, df)
                last_entry = int(entry.iloc[-1]) if len(entry) > 0 else 0
                close = _col(df, "close")
                ma = close.ewm(span=30, adjust=False).mean()
                strength = float(((close.iloc[-1] - ma.iloc[-1]) / (ma.iloc[-1] + 1e-12)) * 100)
                raw_signals[stem] = {"entry": last_entry, "exit": 0, "strength": strength}
            except Exception as e:
                print(f"WARN regime signal failed for {stem}: {e!r}")
                raw_signals[stem] = {"entry": 0, "exit": 0, "strength": 0.0}
        active = [s for s in raw_signals if raw_signals[s]["entry"]]
        active = sorted(active, key=lambda s: -raw_signals[s]["strength"])

    # Close on explicit exit
    for sym in list(state.positions.keys()):
        if sym in raw_signals and raw_signals[sym].get("exit"):
            px_row = prices.get(sym)
            px = float(px_row["close"].iloc[-1]) if px_row is not None else None
            if px and px > 0:
                state.close_position(sym, px)
                print(f"CLOSE {sym} @ {px:.6f} (regime rule exit)")

    # Clear transient halts
    if state.halted and state.halt_reason in ("daily_loss_limit", "flash_crash"):
        print(f"Clearing transient halt: {state.halt_reason}")
        state.halted = False
        state.halt_reason = None

    # Circuit breaker
    ref_price = None
    for sym in list(state.positions.keys()) + (active[:1] if active else []):
        px_row = prices.get(sym)
        if px_row is not None:
            ref_price = float(px_row["close"].iloc[-1])
            break
    if ref_price is not None:
        state.start_daily_bar(ref_price)
        ok, reason = state.check_circuit_breakers()
        if not ok:
            if state.halted and state.halt_reason and "drawdown" in state.halt_reason:
                print(f"CIRCUIT BREAKER {reason}, flattening to cash")
                mtm = {s: float(prices[s]["close"].iloc[-1]) for s in state.positions if s in prices}
                state.flatten_all(mtm)
            else:
                print(f"CIRCUIT BREAKER: {reason}, staying flat")
                state.save()
                return

    # Open new positions
    for sym in active:
        if sym in state.positions:
            continue
        if len(state.positions) >= CONFIG.max_positions:
            break
        px_row = prices.get(sym)
        px = float(px_row["close"].iloc[-1]) if px_row is not None else None
        if not px or px <= 0:
            continue
        ok, reason = state.check_circuit_breakers()
        if not ok:
            print(f"CIRCUIT BREAKER: {reason}, staying flat")
            break
        size = state.equity * CONFIG.max_position_pct
        pos = state.open_position(sym, px, size)
        if pos:
            print(f"OPEN {sym} @ {px:.6f}, size=${size:.2f}")

    # Mark to market
    mtm_prices = {sym: float(df["close"].iloc[-1]) for sym, df in prices.items() if len(df)}
    eq = state.mark_to_market(mtm_prices)
    current_dd = (state.peak_equity - eq) / state.peak_equity if state.peak_equity > 0 else 0
    state.save()

    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] MTM equity: ${eq:.2f}")
    print(f"Peak: ${state.peak_equity:.2f}")
    print(f"Max DD: {state.max_dd:.2%}")
    print(f"Positions: {len(state.positions)}")

    if current_dd > CONFIG.max_drawdown_pct:
        print(f"Flattening: DD {current_dd:.2%} > {CONFIG.max_drawdown_pct:.2%}")
        state.flatten_all(mtm_prices)
        state.halt(f"live_drawdown_flatten_{pd.Timestamp.now().isoformat()}")
        state.save()


if __name__ == "__main__":
    main()
