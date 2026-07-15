# Trading Bot Registry

Single source of truth for every bot/trader in this repo. When you add a bot,
add a row. Naming rule: `<venue>_<strategy>_<tf>` so logs/models are unambiguous.

## Paper traders (live paper, no real money)
| Bot file                | Registry name              | Venue | Strategy            | TF   | Model / Rule        | Data source                     | Journal / log file              |
|-------------------------|----------------------------|-------|---------------------|------|---------------------|---------------------------------|---------------------------------|
| paper_trader.py         | cex_ml_xgb_5m              | CEX   | ML XGBoost (BTC)    | 5m   | models/latest_xgb.json (via model_server) | btc_5m.csv + macro + multi-asset | trade_journal.json              |
| paper_trader_multi.py   | cex_multi_screen_1d        | CEX   | Multi-coin screen   | 1d   | rule-based screen   | MEXC 1d klines                  | (cron log) + trade_journal_multi.json |
| paper_trader_dex.py     | dex_screen_1d              | DEX   | Retail-alt screen   | 1d   | rule-based          | data/<SYM>_1d_max.csv (DEX)     | trade_journal_dex.json          |
| paper_trader_donchian.py| cex_donchian_1d            | CEX   | Donchian 40         | 1d   | rule (canonical)    | data/<SYM>_1d_max.csv (CEX)     | (cron log)                      |

## Model pipeline (produces the ML model the cex_ml_xgb_5m bot consumes)
| File                | Registry name        | What it does                                              | Output                          |
|---------------------|----------------------|-----------------------------------------------------------|---------------------------------|
| model_trainer.py    | train_ml_xgb         | Trains XGBoost 3-class on fetch_data() (+multi-asset)     | models/latest_xgb.json (BTC)    |
|                     |                      | `--symbol DOGE` trains DOGE -> models/doge_xgb.json       | models/<sym>_xgb.json           |
| walk_forward_validate.py | eval_wf_xgb      | OOS walk-forward of the ML approach                       | models/walk_forward_report.json |
| model_server.py     | serve_ml_xgb         | FastAPI serving latest_xgb.json -> /signal for cex_ml_xgb_5m | http://127.0.0.1:8080         |

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
- Log files: stdout -> `/tmp/<registry_name>.log` when run under cron

## Cron jobs (active)
- weekly_model_retrain (train_ml_xgb): Sundays 00:00
- dex_scout_poll: every 5m
- weekly_liquidity_screen: Saturdays 00:00
- (PAUSED) donchian_paper_trader (cex_donchian_1d): daily 14:00
- (PAUSED) DEX paper trader (dex_screen_1d): daily 07:05
