# Developer Guide

## Mental model

- **Train and serve share one pipeline.** `pipeline.py` + `canonical_features`
  build the exact feature block used for both training and live inference. If
  you change a feature, change it in `pipeline`/`canonical_features` — never
  fork it in the trader.
- **The 98-feature contract is sacred.** `canonical_features.CANONICAL` is the
  input dimension every model expects. Edit it ⇒ retrain everything.
- **Risk lives in one engine.** `portfolio_engine.PortfolioEngine` is used by
  both live and backtest replay. Don't special-case risk in the trader.
- **Crashes are normal.** Daemons wrap every loop; `systemd` restarts them.
  Write code that is safe to kill at any point — that's why writes are atomic.

## Conventions

- **Imports:** `sys.path.insert(0, REPO)` then `import <sibling>`. No package
  install; the repo root is the package.
- **Style:** KISS/DRY, match surrounding code. Functions over classes unless
  stateful. Type hints on public functions (see `docs/api.md`).
- **No secrets.** All data is public/free. Never add API keys; if a broker
  adapter is ever added, read credentials from the environment, never the repo.
- **Comments:** explain *why*, not *what*. Especially: rate limits, the
  feature contract, and circuit-breaker semantics.
- **Safety:** every network call goes through `reliability.retry_call` (with
  jitter) or an existing rate gate. Every stateful file write goes through
  `reliability.atomic_write_*` or the locked `save()` in
  `order_manager_multi`.

## Day-to-day workflow

```bash
# activate
. .venv/bin/activate

# run the test suite (fast subset)
python -m pytest tests/test_portfolio_engine.py tests/test_collector_daemon.py -q

# run the whole suite
python -m pytest -q

# one-pass live trader (does not start the service)
python cex_ml_xgb_5m.py --once

# retrain all models on the canonical feature set
python retrain_all.py
```

## Adding a feature

1. Add the column in `pipeline.py` (or the relevant feed module) **and** add its
   name to `canonical_features.CANONICAL` if it should reach the model.
2. If you touched `CANONICAL`, bump the implied version mentally and re-run
   `retrain_all.py` before serving.
3. Add/extend a test in `tests/` covering the behavior.

## Adding a data source

- Backfills are one-shot scripts (`backfill_*.py`) that write CSVs to `data/`.
- The collector/daemon reads those CSVs; it does not call the network at serve
  time for history. Keep that split: **acquire → CSV**, then **serve from CSV**.
- Respect rate limits. Use `reliability.retry_call` for REST; use the existing
  `RateGate` in the collector for bulk sweeps.

## Testing

- Tests live in `tests/` and use `pytest`. Fixtures mock network/exchange
  calls (see `tests/test_collector_daemon.py` `FakeEx`).
- `tests/test_prod_readiness.py` pins the production-critical invariants:
  the live trader calls `start_daily_bar` (so circuit breakers work), and the
  trade journal write is atomic.
- Keep new behavior covered. The CI workflow (`.github/workflows/ci.yml`) runs
  `pytest` on every push/PR.

## Common gotchas

- **Don't edit `CANONICAL` without retraining** — silent flat predictions.
- **Don't hardcode risk params in the trader** — use `config.CONFIG`.
- **Don't remove the daily-bar call** in the trader — breakers go dead.
- **Don't make a state write non-atomic** — a kill mid-write corrupts state and
  the next start can double-open or lose track.
- **Don't cap `N_JOBS` to throttle thermals** — the box pins conservatively on
  purpose; capping caused CPU cooking before.
