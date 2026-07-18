# Architecture Overview

## 1. What the system does

The bot turns free, public OHLCV data into a continuously-ranked set of
directional crypto signals and trades the strongest ones in a simulated
(paper) book. There are three logical stages:

1. **Collect** — append new bars forward, forever, so lookback never runs out.
2. **Train / serve** — per-pair XGBoost models predict LONG / SHORT / FLAT on a
   shared 98-feature block; the live bot ranks all pairs and holds the top-N.
3. **Risk** — a single `PortfolioEngine` enforces sizing, drawdown, daily-loss,
   flash-crash and equity-floor breakers identically in live and backtest.

## 2. Deployment topology

Five `systemd --user` services run the production system. Each is
`Restart=always` (or `on-failure`) so a crash self-heals.

| Service | Script | Role |
|---------|--------|------|
| `collector-daemon` | `collector_daemon.py` | Incremental forward OHLCV for MEXC + BloFin across 5m/1h/4h/1d/8h. Rate-gated, lock-guarded, self-throttling per cycle. |
| `trading-bot-data` | `data_poller.py` | Unified poller: CEX 5m sweep + DEX micro/forward/universe + extra CEX top-up + on-chain top-up. Multi-threaded workers. |
| `trading-bot-dex-sampler` | `dex_ohlcv_sampler.py` | Live DEX 1m/5m sampler (GeckoTerminal). |
| `trading-bot-ml-multi` | `cex_ml_xgb_5m.py` | **The live ML paper trader.** Loads models, ranks signals, executes top-N. |
| `trading-bot-dex-backfill` | `backfill_dex_history_gt.py` | One-off DEX historical backfill. |

Plus Hermes-managed cron jobs (research/retrain cadence): `weekly_model_retrain`,
`weekly_liquidity_screen`, `dex_scout_poll` (active); `donchian_paper_trader`
and DEX paper trader (paused).

## 3. Data flow

```
                ┌─────────────── network (ccxt / GeckoTerminal / curl) ───────────────┐
                ▼                                                                       │
   collector_daemon.py ──► data/<SYM>_<tf>_<ex>_max.csv   (incremental append, atomic)
   data_poller.py     ──► data/cex/<SYM>_5m.csv, data/dex_micro/, data/*_5m_max.csv
                │
                ▼
   model_trainer.build_symbol_features(sym)  ── reads CSV, builds 98-feature frame
                │
                ▼
   canonical_features.resolve(df)  ── freezes exactly the 98-feature block (order matters)
                │
                ▼
   XGBoost model  ──► models/<sym>_xgb.json   (BTC alias: latest_xgb.json)
                │                                        │
                │ (same pipeline at serve time)          │ loaded once at startup
                ▼                                        ▼
   cex_ml_xgb_5m.py: build_features(pair) → predict_pair() → ranked → top-N
                │
                ▼
   order_manager_multi.MultiPositionState  (subclass of PortfolioEngine)
     • open/close/flatten positions
     • check_circuit_breakers()  (daily-loss, max-DD, flash-crash, equity-floor)
     • start_daily_bar()  ── MUST be called once/day to reset breakers (see §5)
     • save()  ── atomic, locked JSON state
     • _log_trade()  ── atomic, locked trade journal
```

## 4. The canonical-feature contract (why it exists)

`canonical_features.py` freezes **one** 98-column feature list shared by the
trainer and the serving bot. Three models (BTC/ETH/DOGE) were historically
trained with divergent code → 85/81/75 features → the server fed them a
different block → silently flat/misaligned predictions. `resolve(df)` zeroes
missing columns and drops extras so every pair presents the identical
98-dim input in the same order. **Changing `CANONICAL` changes the model input
dimension — you must retrain all models afterward.**

## 5. Critical runtime invariants

These are load-bearing; breaking them causes silent or dangerous behavior:

- **Daily-bar cadence.** `cex_ml_xgb_5m.py` must call
  `state.start_daily_bar(ref_price)` exactly once per UTC day. This resets the
  daily-loss tally and rolls the flash-crash window. Without it the "daily"
  loss limit is *cumulative* (halts permanently after any 3% drawdown) and the
  flash-crash breaker never fires. The live loop does this via
  `_maybe_start_daily_bar()`.
- **Atomic state/journal writes.** `order_manager_multi` writes state and the
  trade journal via temp-file + `os.replace` under an `fcntl` lock, so a kill
  mid-write cannot corrupt them.
- **Single source of truth.** Live sizing/positions come from `config.CONFIG`
  (`MAX_POSITIONS`, `initial_capital * max_position_pct`). The trader does not
  hardcode risk parameters.
- **Fail-fast model load.** `load_models()` aborts the whole process if any
  model's `n_features_in_` / `feature_names_in_` disagree with `CANONICAL`. A
  stale model is never served silently.

## 6. Process / resource model

- The ML trader is single-process, inference-inline (models loaded once).
  No model server, no per-pair process.
- Memory is bounded two ways: in-code guards (`mem_guard_abort(4096)` hard
  abort, `MEM_GUARD_MB=6144` throttle) **and** systemd `MemoryHigh`/`MemoryMax`
  on the units (trader 6G/8G, data poller 4G/6G, collector 512M). The in-code
  guards fire first so the process can abort gracefully instead of being
  SIGKILLed.
- On `systemd stop`, SIGTERM is handled: the trader finishes its cycle, saves
  state, and exits. `TimeoutStopSec=30` then SIGKILLs only a hung cycle.

## 7. Out of scope (intentionally)

Real order execution, a live model-serving HTTP API, and the legacy
`serve_ml_xgb_legacy.py` (FastAPI) path. Those are optional and not part of the
production deployment.
