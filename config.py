"""
Single source of tunable configuration for the live paper trader and replay.

Keeping these in one module (instead of scattered module-level constants) makes
the live trader and the backtest reproducible and reviewable: change a threshold
here, not in three files.
"""

from portfolio_engine import EngineConfig

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
CHOP_RULE = "williams_r"       # Williams %R best in chop

# ── Vol target (paranoia mode) ────────────────────────────────────────────
USE_PARANOID_VOL_TARGET = False
VOL_TARGET = 0.15

# ── Trailing exits (optional, off by default) ─────────────────────────────
USE_CHANDELIER_TRAILING = False
USE_ATR_TRAILING = True        # 14/2.0, trend-gated (walk-forward: +effSR, lower DD)
ATR_PERIOD = 14
ATR_MULT = 2.0
