"""
Single source of tunable configuration for the live paper trader and replay.

Keeping these in one module (instead of scattered module-level constants) makes
the live trader and the backtest reproducible and reviewable: change a threshold
here, not in three files.
"""

import os
import subprocess
from portfolio_engine import EngineConfig


# ── Hardware (parallelism) ────────────────────────────────────────────────
# Two numbers, detected automatically if you don't set them:
#   PHYSICAL_CORES = real cores (hyperthreads/SMT excluded) — use for
#                    memory-bandwidth-bound work (feature pipelines, model train).
#   LOGICAL_CORES  = total threads incl. SMT — use for PURE CPU-bound work
#                    (e.g. the DEX per-coin selection Pool, which is compute-only).
#
# To PIN a value (after a CPU upgrade), uncomment + set the matching line. The
# TRADING_BOT_CORES env var still overrides PHYSICAL_CORES (back-compat).
#
# This host (Ryzen 5 5600X): 6 physical cores × 2 threads = 12 logical.
# PHYSICAL_CORES = 6
# LOGICAL_CORES  = 12

def _detect_logical_cores() -> int:
    return max(1, os.cpu_count() or 1)

def _detect_physical_cores() -> int:
    # Try lscpu (Core(s) per socket × Socket(s)); fall back to logical//2.
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        cores = socks = None
        for line in out.splitlines():
            if line.startswith("Core(s) per socket:"):
                cores = int(line.split(":")[1])
            elif line.startswith("Socket(s):"):
                socks = int(line.split(":")[1])
        if cores and socks:
            return max(1, cores * socks)
    except Exception:
        pass
    return max(1, (_detect_logical_cores() // 2))

LOGICAL_CORES = _detect_logical_cores()
PHYSICAL_CORES = int(os.environ.get("TRADING_BOT_CORES") or _detect_physical_cores())
# Leave one physical core free for the OS + collector daemon.
N_JOBS = max(1, PHYSICAL_CORES - 1)
# Pure-CPU worker count (walk-forward replays, etc.): leave ONE thread free for
# the system + control code, so n-1 of logical.
N_WORKERS_CPU = max(1, LOGICAL_CORES - 1)

# ── Risk / cost (shared engine) ───────────────────────────────────────────
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
    enable_vol_target=False,   # regime-gated vol target (off by default)
    vol_lookback=20,
    target_vol=0.15,
    min_vol_scale=0.25,
    max_vol_scale=1.5,
)

# ── Regime selection (live trader) ────────────────────────────────────────
USE_IMPROVED_REGIME = True     # ADX + vol + Kaufman ER + hysteresis
USE_MA_REGIME = True           # friend's MA crossover (short>long) as regime filter
USE_HURST_REGIME = False       # optional persistence filter (no material change)
HURST_WINDOW = 60
HURST_THRESHOLD = 0.5

# ── Active rules per regime (best from verification) ──────────────────────
TREND_RULE = "rei"             # REI-trend strongest on recent OOS / walk-forward
CHOP_RULE = "cci"             # CCI is the chop-primary per 2026-07-14 decision:
                              # best isolated entry edge (+14.1 entry contr, 33.6% win%),
                              # +29.2% vs d40 +7.9% in rei+cci head-to-head. Held until
                              # MTF 4h data (≈Oct 2026) achieves stat significance.
SECONDARY_CHOP_RULE = "ma30_ema"  # fill-in when cci is silent (chop days w/ no signal);
                              # rei+cci head-to-head showed cci already fires every
                              # meaningful chop day, so this is rarely triggered
                              # (0/8 WF slices changed; inert safety net on 68-coin set).
                              # TREND NEEDS NO FILL-IN: measured REI silent on only
                              # 1/63 trend days (1.6%) universe-wide vs CCI 0/97 — so
                              # the asymmetry is intentional, not an oversight. Going
                              # flat on the rare silent-trend day is harmless.

# ── Vol target (paranoia mode) ────────────────────────────────────────────
USE_PARANOID_VOL_TARGET = False
VOL_TARGET = 0.15

# ── Trailing exits (optional, off by default) ─────────────────────────────
USE_CHANDELIER_TRAILING = False
USE_ATR_TRAILING = True        # 14/2.0, trend-gated (walk-forward: +effSR, lower DD)
ATR_PERIOD = 14
ATR_MULT = 2.0
