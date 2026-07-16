# Trading Bot Registry

Single source of truth for every bot/trader in this repo. When you add a bot,
add a row. Naming rule: `<venue>_<strategy>_<tf>` so logs/models are unambiguous.

## Paper traders (live paper, no real money)
| Bot file                | Registry name              | Venue | Strategy            | TF   | Model / Rule        | Data source                     | Journal / log file              |
|-------------------------|----------------------------|-------|---------------------|------|---------------------|---------------------------------|---------------------------------|
| cex_ml_xgb_5m.py        | cex_ml_xgb_5m             | CEX   | ML XGBoost multi-pair (ranked) | 5m | models/<pair>_xgb.json (inline, per-pair registry) | btc_5m.csv + data/<SYM>USDT_5m_max.csv + multi-asset + DEX breadth | logs/cex_ml_xgb_5m.log |
| cex_multi_screen_1d.py  | cex_multi_screen_1d        | CEX   | Multi-coin screen   | 1d   | rule-based screen   | MEXC 1d klines                  | logs/cex_multi_screen_1d.log + trade_journal_multi.json |
| dex_screen_1d.py        | dex_screen_1d              | DEX   | Retail-alt screen (rule) | 1d | rule-based        | dex_data/<TOK>_1d_max.csv       | logs/dex_screen_1d.log + trade_journal_dex.json |
| dex_ml_xgb_1d.py        | dex_ml_xgb_1d              | DEX   | ML XGBoost pooled cross-token (ranked) | 1d | models/dex_xgb.json (ONE pooled model, all tokens) | dex_data/<TOK>_1d_max.csv | logs/dex_ml_xgb_1d.log + trade_journal_dex_ml.json |
| cex_ml_xgb_1d.py        | cex_ml_xgb_1d              | CEX   | ML XGBoost pooled cross-symbol (ranked) | 1d | models/cex_1d_xgb.json (ONE pooled model, all symbols) | data/<SYM>_1d_max.csv | logs/cex_ml_xgb_1d.log + trade_journal_cex_1d_ml.json |
| cex_donchian_1d.py      | cex_donchian_1d            | CEX   | Donchian 40         | 1d   | rule (canonical)    | data/<SYM>_1d_max.csv (CEX)     | (paused)                        |

Legacy (superseded, kept for tests/reference, NOT run):
- cex_ml_xgb_5m_single_legacy.py (was paper_trader.py) -- old single-pair BTC bot, needed a model_server.
- serve_ml_xgb_legacy.py (was model_server.py) -- old FastAPI signal server; cex_ml_xgb_5m.py now does inline inference.

## Model pipeline (produces the ML models the bots consume)
| File                | Registry name        | What it does                                              | Output                          |
|---------------------|----------------------|-----------------------------------------------------------|---------------------------------|
| model_trainer.py    | train_ml_xgb         | Trains XGBoost 3-class per CEX pair on fetch_data() (+multi-asset+DEX breadth) | models/latest_xgb.json (BTC) |
|                     |                      | `--symbol DOGE` trains DOGE -> models/doge_xgb.json       | models/<sym>_xgb.json           |
| train_dex_ml.py     | train_dex_ml         | ONE pooled cross-token XGBoost over ALL dex_data tokens (per-token features, chronological split) | models/dex_xgb.json |
| train_cex_1d_ml.py  | train_cex_1d_ml      | ONE pooled cross-symbol XGBoost over ALL CEX 1d symbols (per-symbol features, chronological split) | models/cex_1d_xgb.json |
| walk_forward_validate.py | eval_wf_xgb      | OOS walk-forward of the ML approach                       | models/walk_forward_report.json |

## Data backfill (free sources only: GeckoTerminal/DEX, CDD/Binance/Blockchain.com, BloFin)
| File                      | Registry name        | What it pulls                                  | Output pattern                  |
|---------------------------|----------------------|------------------------------------------------|---------------------------------|
| backfill_dex_mtf.py       | backfill_dex_ohlc    | DEX 1h/4h/1d OHLCV (GeckoTerminal, free)       | data/<SYM>_{1h,4h,1d}_dex_max.csv |
| dex_forward_collector.py  | collect_dex_5m       | DEX 5m live poll (DexScreener, free)           | data/<SYM>_5m_dex_max.csv       |
| download_cex_history.py   | backfill_cex_1d      | CEX 1d history (CDD/Binance, free)             | data/<SYM>USDT_1d_max.csv       |
| backfill_cex_history.py   | merge_cex_1d         | merge/extend CEX 1d history                    | data/<SYM>USDT_1d_max.csv       |
| backfill_cex_5m.py       | backfill_cex_5m      | CEX 5m DEEP history (Binance data-api mirror, free, no key; MEXC fallback) -> data/<SYM>USDT_5m_max.csv | data/<SYM>USDT_5m_max.csv |

## Naming conventions (enforced)
- Data files: `<SYM>_<tf>_<venue>_max.csv`  (venue = dex | cex | blofin)
- Model files: `models/<sym>_xgb.json` ; BTC default = `models/latest_xgb.json`
- Journal files: `trade_journal_<venue>_<strategy>.json`
- Log files: named by registry (`logs/<registry_name>.log`).
  - Data poller (systemd): `logs/data_poller.log` (all 4 workers, tagged).
  - Paper traders (cron): `logs/cex_multi_screen_1d.log`, `logs/dex_screen_1d.log`.
  - model_server (cex_ml_xgb_5m bot's signal source): `logs/model_server.log`.

## Data collection (systemd-managed, NOT cron)
| Service file | Registry name | What it does | Coverage |
|--------------|---------------|-------------|----------|
| trading-bot-data.service (user systemd) | collect_all | ONE process, Restart=always, all free data all the time | see below |

`data_poller.py` runs 4 concurrent worker threads and replaces all the old
scattered data tools (ad-hoc background processes + the weekly cron DEX
rebuild). systemd restarts it on crash (Restart=always, RestartSec=5) and
starts it on login (enabled).
- CEX worker: pulls 5m for all 459 USDT pairs (Binance klines mirror, no key)
  -> data/cex/<SYM>_5m.csv, then derives 1h/4h/1d locally (derive_cex_tf).
  Resumable cursor in data/.poller_state.json.
- DEX micro worker: DexScreener + CoinGecko breadth poll -> data/dex_micro/
  every 600s.
- DEX forward worker: DexScreener 5m snapshot -> data/<SYM>_5m_dex_max.csv
  every 300s.
- DEX universe worker: build_dex_universe + backfill_dex_history weekly, so
  the 426-token universe stays current.
Logs: logs/data_poller.log. Commands:
  systemctl --user status trading-bot-data; systemctl --user restart trading-bot-data

## Cron jobs (active) -- trading bots only (data is systemd now)
- weekly_model_retrain (train_ml_xgb): Sundays 00:00
- cex_multi_screen_1d.py: daily 00:05 UTC -> logs/cex_multi_screen_1d.log
- dex_screen_1d.py: daily 00:07 UTC -> logs/dex_screen_1d.log
- dex_ml_xgb_1d.py: daily 00:09 UTC -> logs/dex_ml_xgb_1d.log
- cex_ml_xgb_1d.py: daily 00:11 UTC -> logs/cex_ml_xgb_1d.log
- (PAUSED) cex_donchian_1d.py
Rule: 1d bots = cron (staggered); 5m bots = systemd daemons.
The one daemon trading bot: cex_ml_xgb_5m (systemd trading-bot-ml-multi.service).
NOTE: a future dex_ml_xgb_5m would ALSO be a daemon -- waits on DEX 5m depth.
