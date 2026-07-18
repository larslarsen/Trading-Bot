# Packages & Directory Layout

The repo is a **flat package** (no `src/` subtree): every daemon, library, and
research script lives at the top level and imports siblings via
`sys.path.insert(0, REPO)` + `import <module>`. There is no installed
distribution; the daemons run directly from the repo with the local `.venv`.

## Top-level layout

```
trading-bot/
├── README.md                  entrypoint + doc index
├── requirements.txt           pinned runtime + analysis + optional + dev deps
├── cex_ml_xgb_5m.py           LIVE ML paper trader (systemd: trading-bot-ml-multi)
├── collector_daemon.py        forward OHLCV collector (systemd: collector-daemon)
├── data_poller.py             unified data poller (systemd: trading-bot-data)
├── dex_ohlcv_sampler.py       DEX live sampler (systemd: trading-bot-dex-sampler)
├── backfill_dex_history_gt.py DEX history backfill (systemd: trading-bot-dex-backfill)
├── pipeline.py                shared feature/label/walk-forward builder
├── canonical_features.py      frozen 98-feature contract
├── model_trainer.py           per-pair model train + save
├── quality_gate.py            universe liquidity/data gating
├── portfolio_engine.py        risk engine (positions, cash, MTM, breakers)
├── order_manager_multi.py     live persistence + trade journal
├── config.py                  EngineConfig + core-count detection
├── reliability.py             atomic writes, retry+backoff, safe logging
├── mem_guard.py               RSS guard (abort if over cap)
├── canonical_features.py      feature contract
├── multi_asset_features.py    cross-asset (BTC/ETH/DOGE) features
├── micro_features.py          Bybit micro-structure (order flow, funding)
├── onchain_features.py        on-chain network metrics
├── dex_features.py            DEX-wide breadth features
├── dex_*.py / backfill_*.py   DEX + CEX data acquisition
├── <rule>_*.py, exp_*.py      research: rule variants + regime experiments
├── eval_*.py, *_v2.py         OOS / paper verification scripts
├── tests/                     pytest suite (engine, order, collector, prod-readiness)
├── models/                    trained XGBoost models (<sym>_xgb.json) + meta/metrics
├── data/                      OHLCV CSVs (the system's source of truth)
├── logs/                      daemon logs (rotated via deploy/logrotate_trading_bot.conf)
├── docs/                      this documentation set
├── deploy/                    deployment artifacts (logrotate conf)
├── archive/                   retired experiments (not imported by active code)
└── .github/workflows/ci.yml   CI: install requirements.txt + run pytest
```

## Logical groupings

| Group | Modules | Notes |
|-------|---------|-------|
| **Runtime / daemons** | `collector_daemon`, `data_poller`, `dex_ohlcv_sampler`, `backfill_dex_history_gt`, `cex_ml_xgb_5m` | Deployed as systemd services. Never die; every loop is wrapped. |
| **Core libraries** | `pipeline`, `canonical_features`, `model_trainer`, `quality_gate`, `portfolio_engine`, `order_manager_multi`, `config`, `reliability`, `mem_guard` | Imported by both training and serving. The "single source of truth" layer. |
| **Feature feeds** | `multi_asset_features`, `micro_features`, `onchain_features`, `dex_features`, `equities_regime` | Optional exogenous feeds; missing feeds are zero-filled, not fatal. |
| **Data acquisition** | `backfill_*`, `dex_*`, `build_*`, `derive_*`, `collect_all_data` | One-shot backfills + derives. Not always-on. |
| **Research / experiments** | `exp_*`, `<rule>_*`, `oos_*`, `eval_*`, `*_v2`, `wf_*` | Throwaway/verification. Live path does NOT depend on them. |
| **Tests** | `tests/` | 190+ tests; the only CI gate. |

## Data dictionary (on-disk)

| Path | Contents |
|------|----------|
| `data/<SYM>USDT_5m_max.csv` | canonical 5m bars for a CEX pair (single source of truth the trader reads) |
| `data/<SYM>_<tf>_<ex>_max.csv` | collector output per (exchange, symbol, timeframe) |
| `data/dex_micro/<TOKEN>.csv` | DEX micro-poller breadth snapshots |
| `data/funding/*.csv` | Bybit 8h funding rates |
| `models/<sym>_xgb.json` | trained XGBoost model (`latest_xgb.json` = BTC) |
| `execution_state_ml_multi.json` | live trader position/equity state (atomic) |
| `trade_journal_ml_multi.json` | append-only trade log (atomic) |
| `data/.poller_state.json` | poller cursor / last-universe timestamp (atomic) |
| `run/collector_daemon.lock` | flock guard preventing two collector instances |

## Environment

- Python 3.12+ in `.venv` (created from `requirements.txt`).
- No `pyproject.toml` / `setup.py`; the repo *is* the package.
- User-level systemd (`--user`) owns the services; logs go to
  `logs/*.log` (append) and are rotated by the deploy logrotate snippet.
