# Configuration Documentation

There is **no single config file**. Configuration lives in three places, in
order of precedence for a given concern:

1. `config.py` — risk parameters + core-count detection (**single source of
   truth for risk**).
2. Module-level constants in the daemons — scheduling/network tuning.
3. `requirements.txt` — dependency versions.

Environment variable: `TRADING_BOT_CORES` pins `PHYSICAL_CORES` (overrides
auto-detection).

## `config.py` — `EngineConfig`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `initial_capital` | 10000.0 | Starting equity (USD). |
| `max_daily_loss_pct` | 0.03 | Halt if daily pnl < −3% of peak. |
| `max_drawdown_pct` | 0.20 | Halt if peak-to-trough DD > 20%. |
| `max_positions` | 5 | Hard cap on concurrent positions. |
| `max_position_pct` | 0.20 | Max notional per position as fraction of capital. |
| `min_equity_to_trade` | 100.0 | Equity floor; below this, halt. |
| `flash_crash_bars` | 5 | Window length for flash-crash detection. |
| `flash_crash_pct` | 0.50 | Move > 50% in window → halt (unless > `extreme_move_pct`). |
| `extreme_move_pct` | 0.90 | Move > 90% treated as data artifact, window cleared (no halt). |
| `cost_bps` | 8e-4 (0.08%) | Per-side trading cost. |
| `slippage_bps` | 5e-4 (0.05%) | Per-side slippage. |
| `enable_vol_target` | False | Regime-gated vol target (off by default). |
| `vol_lookback` / `target_vol` / `min_vol_scale` / `max_vol_scale` | 20 / 0.15 / 0.25 / 1.5 | Vol-target params (used only if enabled). |

The live trader derives its sizing from this config:
`MAX_POSITIONS = CONFIG.max_positions`,
`SIZE_USD = CONFIG.initial_capital * CONFIG.max_position_pct` (= 2000 at
defaults). The confidence gate `CONFIDENCE_THRESHOLD = 0.60` is a
trader-specific constant in `cex_ml_xgb_5m.py`.

### Core-count detection

`N_JOBS = max(1, PHYSICAL_CORES - 1)` (training parallelism, leaves one core
free). `N_WORKERS_CPU = max(1, LOGICAL_CORES - 1)` (pure-CPU pools). Override
`PHYSICAL_CORES` with `TRADING_BOT_CORES`. **Do not cap cores below the hardware
reality** — the bot pins conservatively on purpose.

## Daemon scheduling constants

| Module | Constant | Default | Meaning |
|--------|----------|---------|---------|
| `collector_daemon` | `CYCLE_INTERVAL` | 900s | Seconds between full sweeps. |
| | `PAUSE` | 0.30s | Base delay between ccxt calls. |
| | `MAX_FETCHES_PER_MIN` | 150 | Global rate ceiling. |
| | `MAX_CYCLE_SECONDS` | 1500s | Abort a cycle past this (self-throttle). |
| | `LIMIT` | 500 | Bars per ccxt call. |
| | `ENABLE` | `{'mexc':True,'blofin':True,'kraken':False}` | Which exchanges to collect. |
| `data_poller` | `MICRO_INTERVAL` | 600s | DEX breadth poll cadence. |
| | `DEX_FWD_INTERVAL` | 300s | DEX 5m forward snap cadence. |
| | `UNIVERSE_INTERVAL` | 7×86400s | DEX universe rebuild cadence. |
| `cex_ml_xgb_5m` | `POLL_SEC` | 300s | Trader ranking cadence. |
| | `MEM_GUARD_MB` | 6144 | Skip-build throttle above this RSS. |
| | `CONFIDENCE_THRESHOLD` | 0.60 | Min confidence to take a directional signal. |

## Systemd unit knobs

Units live in `~/.config/systemd/user/`. Relevant hardening:

| Unit | `MemoryHigh` | `MemoryMax` | `TimeoutStopSec` |
|------|-------------|-------------|-----------------|
| `trading-bot-ml-multi` | 6G | 8G | 30 |
| `trading-bot-data` | 4G | 6G | 30 |
| `collector-daemon` | — | 512M | — |

`MemoryMax` is the hard ceiling; the in-code guards (`mem_guard_abort(4096)`,
`MEM_GUARD_MB=6144`) fire first so the process can abort gracefully rather than
be SIGKILLed.

## Changing configuration safely

- **Risk params:** edit `config.EngineConfig` fields. They apply to both live
  and backtest because the engine is shared.
- **Universe:** edit `screener_ml_multi.txt` (one pair per line, `#` comments)
  to restrict which models the live trader trades; omit the file to trade all
  discovered models.
- **Feature set:** never edit `canonical_features.CANONICAL` without retraining
  every model via `retrain_all.py` afterward — the serving dimension must match.
