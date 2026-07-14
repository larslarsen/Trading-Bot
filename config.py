"""
Single source of tunable configuration for the live paper trader and replay.

Keeping these in one module (instead of scattered module-level constants) makes
the live trader and the backtest reproducible and reviewable: change a threshold
here, not in three files.
"""

import os
from portfolio_engine import EngineConfig


# ── Hardware (parallelism) ────────────────────────────────────────────────
# Number of PHYSICAL CPU cores on the host. Drives n_jobs for the model
# trainer / walk-forward / grid workers. Set via TRADING_BOT_CORES env var so
# you can change it (e.g. after upgrading your CPU) without editing this file:
#   export TRADING_BOT_CORES=8
# Logical-core / hyperthread counts are NOT used: crypto feature pipelines are
# memory-bandwidth bound, so oversubscribing to SMT threads just adds contention.
# Fallback: half of os.cpu_count() (≈ physical cores on most consumer chips),
# or 4 if detection fails.
def _detect_physical_cores() -> int:
    guess = (os.cpu_count() or 8) // 2
    return max(1, guess)


N_PHYSICAL_CORES = int(os.environ.get("TRADING_BOT_CORES") or _detect_physical_cores())
# Leave one core free for the system + the collector daemon.
N_JOBS = max(1, N_PHYSICAL_CORES - 1)

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
                              # meaningful chop day, so this is rarely triggered.

# ── Vol target (paranoia mode) ────────────────────────────────────────────
USE_PARANOID_VOL_TARGET = False
VOL_TARGET = 0.15

# ── Trailing exits (optional, off by default) ─────────────────────────────
USE_CHANDELIER_TRAILING = False
USE_ATR_TRAILING = True        # 14/2.0, trend-gated (walk-forward: +effSR, lower DD)
ATR_PERIOD = 14
ATR_MULT = 2.0
