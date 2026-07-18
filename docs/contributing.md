# Contributing Guide

Thanks for improving the bot. This is a research-grade trading system where
**evidence and reproducibility** matter more than velocity. Keep changes
literature-backed and verified.

## Ground rules

1. **One source of truth.** Risk → `config.py`. Features →
   `pipeline.py` + `canonical_features.py`. Execution math →
   `portfolio_engine.py`. Don't fork these into the trader or a script.
2. **The 113-feature contract is load-bearing.** Changing
   `canonical_features.CANONICAL` changes the model input dimension. If you
   change it, you MUST retrain every model (`retrain_all.py`) and say so in the
   PR.
3. **Crash safety is non-negotiable.** Every stateful write goes through
   `reliability.atomic_write_*` or the locked `save()` in
   `order_manager_multi`. Every network call goes through
   `reliability.retry_call` (with jitter) or an existing rate gate.
4. **No secrets, no API keys.** All data is public/free. Credentials (if a
   broker adapter is ever added) come from the environment only.
5. **Evidence over claims.** A change that affects trading behavior needs a
   test or a reproducible measurement. "It looks better" is not sufficient.

## Workflow

```bash
git checkout -b <topic>
. .venv/bin/activate
# make changes, then:
python -m pytest -q
# for runtime changes, also smoke the affected daemon:
python cex_ml_xgb_5m.py --once      # if you touched the trader
python -c "import collector_daemon, data_poller"  # if you touched a daemon
git commit -m "concise: what + why"
git push -u origin <topic>
# open a PR
```

## Commit messages

- Imperative, one line: `fix(trader): call start_daily_bar so breakers reset`.
- Why, not what — the diff shows what; the message explains the reason and any
  risk (e.g. "requires retrain").

## What to test

- New behavior → new test in `tests/`. Mock network/exchange (see
  `tests/test_collector_daemon.py::FakeEx`).
- `tests/test_prod_readiness.py` pins production invariants (daily-bar wiring,
  atomic journal). Don't weaken it.
- Keep the full suite green; CI runs `pytest` on every push/PR.

## Code style

- Match surrounding code (it's terse, typed on public functions, comment-driven
  on *why*).
- Prefer functions over classes unless stateful.
- No `eval`/`exec`/`pickle` of untrusted input; no `shell=True`.
- Keep research/experiment scripts in `archive/` or clearly separated; don't
  let them import into the runtime path.

## Review checklist (for the maintainer)

- [ ] Risk params still flow from `config.CONFIG`?
- [ ] `CANONICAL` unchanged, or retrain planned + stated?
- [ ] Writes atomic? Network calls retried with jitter?
- [ ] Tests added/updated? Full suite green?
- [ ] Docs updated if behavior/API changed?
- [ ] Running service restarted to pick up the new code?
