# API Documentation

Public API surface of the production modules. Signatures are the contract —
changing them (or the 113-feature block in `canonical_features`) requires
retraining models and/or updating callers.

## `pipeline`

```python
fetch_data(symbol: str | None = None) -> pd.DataFrame
    # Load a pair's 5m history. symbol=None -> the BTC root file.

add_resampled_features(df) -> pd.DataFrame
    # Add 1h/4h resampled momentum/technical columns (joined, not mutated in place).

load_macro_data(df_index) -> pd.DataFrame
    # Build macro proxies (SPY/GLD/TLT/UUP/VIX) forward-filled to df_index.

add_macro_signals(df, macro=None) -> pd.DataFrame

derive_features(df) -> pd.DataFrame

detect_regime(df) -> pd.DataFrame
    # Appends regime_high_vol / regime_trending columns.

triple_barrier_labels(df, horizon=HORIZON_BARS) -> pd.DataFrame
    # Symmetric TP/SL labeling; adds 'label' when coverage suffices.

walk_forward_splits(df, folds=FOLDS) -> list[dict]
    # Expanding-window folds; each dict has 'train_idx'/'val_idx'/'test_idx'.

cost_aware_filter(probs, prev_pos, lam=None, cost=None) -> int
    # Bysik-style cost-aware action filter.

ALL_FEATURES : tuple[str, ...]   # raw candidate feature names
```

## `canonical_features`

```python
CANONICAL : list[str]        # the frozen 113-feature contract
N_FEATURES : int             # == len(CANONICAL)
CROSS_ASSETS : list[str]     # ["ETHUSDT", "DOGEUSDT"] (locked)

resolve(df, features) -> (pd.DataFrame, list[str])
    # Return (df, features) where df has exactly CANONICAL columns in order;
    # missing -> zero-filled, extras dropped. Valid for any trained model.
```

## `model_trainer`

```python
build_symbol_features(symbol) -> (pd.DataFrame, list[str] | None)
    # The canonical (df, feats) pair used by training AND serving.

train_and_save(symbol=None) -> bool | None
    # Train XGBoost on symbol and write models/<sym>_xgb.json (+ meta/metrics).

model_out_path(sym_tag) / meta_out_path(sym_tag) / metrics_out_path(sym_tag) -> Path
```

## `quality_gate`

```python
gated_universe() -> list[str]          # CEX pairs passing the data-coverage gate
dex_gated_universe() -> list[str]      # DEX equivalent
save_gate(pairs, path=...) / save_dex_gate(pairs, path=...)
```

## `portfolio_engine`

```python
@dataclass EngineConfig:
    initial_capital, max_daily_loss_pct, max_drawdown_pct, max_positions,
    max_position_pct, min_equity_to_trade, flash_crash_bars, flash_crash_pct,
    extreme_move_pct, cost_bps, slippage_bps, enable_vol_target, vol_lookback,
    target_vol, min_vol_scale, max_vol_scale

class PortfolioEngine:
    open_position(symbol, fill_price, size_usd) -> Position | None
    close_position(symbol, exit_price) -> float | None   # returns pnl
    flatten_all(prices: dict) -> None
    position_value(prices=None) -> float
    mark_to_market(prices: dict) -> float
    start_daily_bar(reference_price: float) -> None      # reset daily pnl + roll flash window
    check_circuit_breakers() -> (bool, str | None)        # (ok, reason); halts on trip
    halt(reason: str) -> None
    vol_scale(...) -> float
    to_state_dict() -> dict
    load_state_dict(data: dict) -> None
```

## `order_manager_multi`

```python
class MultiPositionState(PortfolioEngine):
    def __init__(self, initial_capital=10000.0,
                 state_file=execution_state_multi.json,
                 journal_file=trade_journal.json)
    save()                 # atomic + locked
    _log_trade(trade)      # atomic + locked journal append
    reset_daily()          # zero daily pnl + flash window (ops use)
```

## `config`

```python
CONFIG : EngineConfig                 # the live risk parameters
N_JOBS : int                          # physical cores - 1 (training parallelism)
N_WORKERS_CPU : int                   # logical cores - 1 (pure-CPU pools)
# env override: TRADING_BOT_CORES pins PHYSICAL_CORES
```

## `reliability`

```python
atomic_write_csv(path, df, **kwargs)
atomic_write_json(path, obj, indent=2)
atomic_write_text(path, text)
retry_call(fn, *, tries=4, base=1.0, cap=30.0, jitter=0.3,
           sleep=time.sleep, exceptions=(Exception,), on_retry=None) -> Any
retryable(*, tries, base, cap, jitter, exceptions)        # decorator form
safe_log(path, msg, *, stamp=True)                        # fsync'd, never raises
```

## `mem_guard`

```python
rss_mb() -> float                # RSS in MiB via /proc/self/statm (0.0 if unavailable)
guard(limit_mb: int) -> None     # sys.exit(0) if RSS > limit_mb (limit 0 disables)
```

## `cex_ml_xgb_5m` (live trader entrypoints)

```python
discover_models() -> dict[str, Path]     # models/<sym>_xgb.json -> pair
load_screener() -> list[str] | None      # screener_ml_multi.txt (None = all)
active_pairs() -> dict[str, Path]        # screener-restricted models
load_models() -> dict[str, XGBClassifier]  # fails fast on feature mismatch
build_features(symbol) -> (pd.DataFrame, pd.Series) | (None, None)
predict_pair(model, fvec) -> (signal, conf, strength)
run_once(models, state)                  # one ranking + execution pass
main()  # argparse: --once for a single pass
```
