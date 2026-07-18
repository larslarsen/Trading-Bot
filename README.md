# Trading Bot

Multi-pair 5-minute ML paper-trading system for crypto (CEX + DEX). It
collects OHLCV forward continuously, trains per-pair XGBoost directional
models (long / short / flat) on a single canonical 113-feature block, and runs
a live "rank-the-signals" paper trader that holds the top-N strongest
directional bets through a shared, backtest-identical risk engine.

> **Paper trading only.** No real orders are placed. The execution engine is
> fully simulated (cash/position accounting + mark-to-market). Swapping in a
> broker adapter is a deliberate, separate step that is out of scope here.

## Quick start

```bash
python -m venv .venv && . .venv/bin/python -m pip install -r requirements.txt
python -m pytest -q            # 190+ tests
# run the live ML paper trader (one pass, then exit):
python cex_ml_xgb_5m.py --once
```

## Documentation index

| Document | Purpose |
|----------|---------|
| [Architecture](docs/architecture.md) | System topology, data flow, deployment model |
| [Packages](docs/packages.md) | Directory / package layout |
| [Modules](docs/modules.md) | What each module does |
| [API](docs/api.md) | Public functions & classes |
| [Configuration](docs/configuration.md) | Every tunable + where it lives |
| [Installation](docs/installation.md) | From zero to running daemons |
| [Developer guide](docs/developer-guide.md) | Workflow, conventions, testing |
| [Troubleshooting](docs/troubleshooting.md) | Symptoms → causes → fixes |
| [Contributing](docs/contributing.md) | How to propose changes |

## Repository at a glance

```
collector_daemon.py     forward OHLCV collector (MEXC + BloFin), incremental
data_poller.py          unified data poller (CEX 5m + DEX micro/forward/universe)
cex_ml_xgb_5m.py        LIVE ML paper trader (ranks models, holds top-N)
pipeline.py             feature build + labels + walk-forward (shared by train+serve)
canonical_features.py    the frozen 113-feature contract (trainer + server agree)
portfolio_engine.py     risk engine: positions, cash, MTM, circuit breakers
order_manager_multi.py  live state persistence + trade journal (atomic)
config.py               EngineConfig + core-count detection (single source of truth)
reliability.py          atomic writes, retry+backoff, safe logging
model_trainer.py        per-pair model training + save
quality_gate.py         liquidity / data-presence gating of the universe
```

See [Architecture](docs/architecture.md) for how these fit together.
