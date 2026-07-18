# Troubleshooting Guide

Symptom → likely cause → fix. All logs are under `logs/` (or
`journalctl --user -u <service>`).

## The live trader (`trading-bot-ml-multi`)

### Trader won't start / exits immediately
- **Missing or stale model.** `load_models()` fails fast if any
  `models/<sym>_xgb.json` has `n_features_in_` / `feature_names_in_` ≠
  `CANONICAL`. → Re-run `python retrain_all.py`.
- **No models discovered.** `discover_models()` found nothing in `models/`. →
  Train first (`retrain_all.py`).
- **No screener file → trades ALL models.** If `screener_ml_multi.txt` is
  absent, the trader trades every discovered model. To restrict, create the
  file with one pair per line.

### Trader halts and stays flat
- **Drawdown / daily-loss / flash-crash / equity-floor breaker tripped.**
  `state.halted` is `True` and `halt_reason` is set in
  `execution_state_ml_multi.json`. → Inspect `trade_journal_ml_multi.json`,
  confirm the halt is legitimate, then clear it by restarting after removing the
  halt flag (or resetting state if intentional). **A halted state must be
  acknowledged, not silently bypassed.**

### Predictions look flat / no trades
- **Feature mismatch (silent).** If `predict_pair` sees missing
  `feature_names_in_`, it returns `FLAT`. → Check the model was trained on the
  current `CANONICAL`; retrain.
- **Confidence below `CONFIDENCE_THRESHOLD` (0.60).** Expected in choppy
  regimes — not a bug.

### High memory / OOM-ish behavior
- **In-code guard fires first.** `mem_guard_abort(4096)` hard-aborts;
  `MEM_GUARD_MB=6144` skips a build cycle. systemd `MemoryMax=8G` is the backstop.
  → If you see `MEM THROTTLE` lines, the box is memory-constrained; the guard is
  working as designed. If it hard-aborts, systemd restarts it (re-run safe).

## The collectors / poller

### `collector-daemon` not appending new bars
- **Lock held by a dead instance.** `run/collector_daemon.lock` (fcntl). →
  `systemctl --user restart collector-daemon`; if stuck, remove the lock file
  (only when no instance is running).
- **Rate-limited (429/503).** The `RateGate` + retry-with-jitter handle
  transient errors; persistent 403 means the exchange is blocking the IP. →
  Back off; check `collector_daemon.log` for `ERR` lines and the per-cycle
  error count.

### Data files look truncated / corrupt
- **Kill mid-write.** Should not happen — writes are atomic (temp +
  `os.replace`). If a CSV is genuinely corrupt, delete it and let the
  collector/poller rebuild it from the last good timestamp (incremental fetch
  resumes). **Never hand-edit the CSVs while a daemon runs.**

### `data_poller` worker died
- The main loop exits for `systemd` restart if a worker thread dies. → Check
  `logs/data_poller.log` for the offending token/symbol; the per-token context
  is now logged (was previously swallowed).

## Tests / CI

### `pytest` collection is slow (~1 min)
Expected — modules import heavy deps at load. Run a targeted file during
development.

### A test fails after a feature change
- Most likely you changed `CANONICAL` without retraining, or broke the
  daily-bar invariant. `tests/test_prod_readiness.py` pins the latter.

## Log rotation

### Log files grow without bound
- Install the logrotate snippet (`deploy/logrotate_trading_bot.conf` →
  `/etc/logrotate.d/trading_bot`). Validate with `sudo logrotate -d
  /etc/logrotate.d/trading_bot`. Uses `copytruncate` so open file descriptors
  keep writing.

## Process introspection

```bash
# daemon status
systemctl --user status trading-bot-ml-multi

# live tail
journalctl --user -u trading-bot-ml-multi -f

# is the process actually alive + its memory?
systemctl --user show trading-bot-ml-multi -p MemoryCurrent -p SubState

# restart cleanly (SIGTERM handled: finishes cycle, saves state, exits)
systemctl --user restart trading-bot-ml-multi
```

## "It doubled / log is huge"
Historical: a per-pair double-load and unbounded log growth were fixed
(atomic writes, daily-bar wiring, logrotate). If you still see duplication,
confirm the running service is the **current code** (`systemctl --user restart`)
— an old in-memory process keeps running the pre-fix version until restarted.
