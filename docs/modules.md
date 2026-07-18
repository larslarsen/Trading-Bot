# Module Documentation

This page documents each production module. Research/experiment scripts
(`exp_*`, `<rule>_*`, `oos_*`, `wf_*`, `eval_*`) are intentionally omitted —
they are not part of the runtime path.

## Runtime daemons

### `collector_daemon.py`
Forward OHLCV collector for MEXC + BloFin. Builds a job list of
`(exchange, symbol, ccxt_symbol, timeframe)` and appends new bars forward each
cycle (15-min cadence, 25-min self-throttle budget). Safety properties:
incremental fetch (only bars newer than the last stored timestamp), a global
rate gate (`MAX_FETCHES_PER_MIN`), a lock file preventing double-poll, and a
per-cycle error count. `fetch_forward()` writes atomically; the REST call is
retried with exponential backoff + jitter. Handles SIGTERM gracefully.

### `data_poller.py`
Unified poller running several daemon threads: CEX 5m sweep, DEX micro, DEX
forward, DEX universe rebuild (weekly), extra CEX top-up, on-chain top-up.
State (cursor, last-universe) is persisted atomically. Each worker is wrapped
so one transient error cannot kill the others; `systemd` restarts the process
if a worker dies.

### `cex_ml_xgb_5m.py` — the live ML paper trader
Loads every discovered `models/<sym>_xgb.json` once, fails fast on feature
mismatch, then loops every 300s: build features per pair (cached by bar
timestamp), predict LONG/SHORT/FLAT + confidence, rank by absolute strength,
hold the top-N above `CONFIDENCE_THRESHOLD`. Performs daily-bar cadence
(`_maybe_start_daily_bar`), circuit-breaker checks, mark-to-market, and
drawdown-flatten. SIGTERM handler saves state and exits.

### `dex_ohlcv_sampler.py` / `backfill_dex_history_gt.py`
DEX live sampler and historical backfill (GeckoTerminal pool-level, no key).
Deployed as `trading-bot-dex-sampler` and `trading-bot-dex-backfill`.

## Core libraries

### `pipeline.py`
The shared feature/label/validation builder used by **both** training and
serving (so the served input matches the trained input). Key exports:
`fetch_data`, `add_resampled_features`, `load_macro_data`, `add_macro_signals`,
`derive_features`, `detect_regime`, `triple_barrier_labels`,
`walk_forward_splits`, `cost_aware_filter`, `ALL_FEATURES`. The 5m screener's
feature set (temporals, technicals, cross-asset, micro, DEX, regime) is
defined here; `canonical_features.resolve` then freezes it to 98 columns.

### `canonical_features.py`
Freezes the 98-feature contract (`CANONICAL`, `N_FEATURES`, `CROSS_ASSETS`).
`resolve(df, features)` guarantees every pair emits exactly those 98 columns in
the same order — missing columns zero-filled, extras dropped. This is what
makes one serving block valid for all trained models. **Editing `CANONICAL`
requires retraining every model.**

### `model_trainer.py`
`build_symbol_features(symbol)` → `(df, feats)` (the canonical feature frame);
`train_and_save(symbol)` trains an XGBoost multiclass model and writes
`models/<sym>_xgb.json` (+ meta/metrics). Path helpers
(`model_out_path`, `meta_out_path`, `metrics_out_path`) centralize naming.

### `quality_gate.py`
Liquidity / data-presence gating. `gated_universe()` returns pairs that pass
the CEX data-coverage gate; `dex_gated_universe()` the DEX equivalent.
`save_gate` / `save_dex_gate` persist the resulting universes
(`universe_gated.json`, `universe_dex_gated.json`).

### `portfolio_engine.py`
`PortfolioEngine` — pure execution core: position lifecycle, cash accounting,
mark-to-market, and risk. `EngineConfig` holds all tunable risk/cost params.
Circuit breakers: `check_circuit_breakers()` (equity floor, daily-loss,
max-drawdown, flash-crash, data-artifact guard) and `start_daily_bar()`
(rolls the daily window — **must be called once/day by the live trader**).
Contains no file IO and no network access; persistence lives in subclasses.

### `order_manager_multi.py`
`MultiPositionState(PortfolioEngine)` — adds JSON persistence + trade journal.
`save()` and `_log_trade()` are both atomic (temp file + `os.replace`) and
lock-guarded. `load()` restores state on startup. This is the live state of the
ML trader.

### `config.py`
`EngineConfig` import + auto-detected core counts. `N_JOBS` and
`N_WORKERS_CPU` are derived from `lscpu` (or fall back to logical//2), minus
one physical core for headroom. This is the single source of truth for risk
parameters and parallelism; the live trader reads it.

### `reliability.py`
Shared crash-safety primitives: `atomic_write_csv/json/text`,
`retry_call` / `retryable` (exponential backoff + full jitter), and `safe_log`
(fsnyc'd, never raises). Imported by the daemons.

### `mem_guard.py`
`rss_mb()` reads RSS from `/proc/self/statm`; `guard(limit_mb)` aborts the
process if RSS exceeds the cap (re-run is safe). Used by the ML trader as an
in-code OOM backstop beneath the systemd `MemoryMax`.

## Feature feeds (optional, zero-filled when missing)

- `multi_asset_features` — cross-asset (BTC/ETH/DOGE) series.
- `micro_features` — Bybit order-flow + funding rate.
- `onchain_features` — per-chain network metrics.
- `dex_features` — DEX-wide breadth (tokens up, gainers fraction, …).
- `equities_regime` — equities-regime proxy.

If any feed is unavailable, `build_features()` drops those columns and
`canonical_features.resolve` zero-fills them — the model input dimension is
never shifted.
