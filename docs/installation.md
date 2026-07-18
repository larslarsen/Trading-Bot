# Installation Guide

## 1. Prerequisites

- Linux (the daemons use `systemd --user`, `fcntl`, `/proc/self/statm`).
- Python 3.12+.
- `git`, `curl`, `logrotate` (for log rotation), and `lscpu` (core detection).
- Network egress to public exchange REST endpoints (ccxt / GeckoTerminal). No
  API keys are required — all data sources are free/public.

## 2. Clone & venv

```bash
git clone <repo-url> trading-bot
cd trading-bot
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` is split into sections:

- **[runtime]** — daemons + pipeline + cron (numpy, pandas, ccxt, xgboost,
  scikit-learn, scipy, requests).
- **[analysis]** — backtesting/research scripts (vectorbt, statsmodels,
  matplotlib).
- **[optional]** — single-script deps (lightgbm, catboost, optuna,
  hmmlearn, pyarrow, gdown, yfinance, fastapi, uvicorn, pydantic, joblib).
- **[dev]** — pytest.

Install only `[runtime]` + `[dev]` for a minimal production host.

## 3. Verify the install

```bash
python -m pytest -q          # 190+ tests should pass
python -c "import collector_daemon, data_poller, cex_ml_xgb_5m, config"
```

## 4. Seed data (first run)

The daemons *append forward* from whatever already exists, so you need at least
a starting history. Backfills (one-shot) populate it:

```bash
# CEX 5m for the screened universe
python backfill_cex_all.py
python backfill_cex_others.py --venue bybit
python backfill_cex_others.py --venue okx
# DEX history
python backfill_dex_history_gt.py --tf day --sleep 4.0
# Train the models the live trader will load
python retrain_all.py
```

After this, `data/` holds the CSVs and `models/` holds `<sym>_xgb.json`.

## 5. Install systemd user services

Copy the unit files (already present under `~/.config/systemd/user/`) and
enable them:

```bash
systemctl --user daemon-reload
systemctl --user enable --now collector-daemon trading-bot-data \
    trading-bot-dex-sampler trading-bot-ml-multi trading-bot-dex-backfill
```

Enable linger so the services survive logout:

```bash
sudo loginctl enable-linger "$USER"
```

## 6. Install log rotation

```bash
sudo cp deploy/logrotate_trading_bot.conf /etc/logrotate.d/trading_bot
# dry-run to validate:
sudo logrotate -d /etc/logrotate.d/trading_bot
```

## 7. Smoke test the live trader

Run one ranking pass without starting the service:

```bash
python cex_ml_xgb_5m.py --once
```

It should load models, print a `ranked:` line, and exit. Watch the journal:

```bash
journalctl --user -u trading-bot-ml-multi -f
```

## 8. Upgrade

```bash
git pull
. .venv/bin/activate
pip install -r requirements.txt
systemctl --user daemon-reload
systemctl --user restart trading-bot-ml-multi trading-bot-data collector-daemon
```

If `canonical_features.CANONICAL` changed upstream, re-run `python retrain_all.py`
before restarting the trader.
