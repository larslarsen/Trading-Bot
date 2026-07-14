# Deployment & Operations Setup

This repo runs two long-lived processes and one weekly job. All scheduling is
**system-level** (OS cron + systemd user service), NOT the chat-assistant cron.
Read this before deploying on a fresh machine or after a CPU upgrade.

---

## 1. Live paper trader — system cron

The live trader (`paper_trader_multi.py`) runs once per day, **just after the
UTC daily candle closes**, so signals are computed on a *completed* daily bar
(no lookahead — see `docs/multitimeframe_research.md`, QuantPedia D1H1).

- Schedule: **`5 0 * * *`**  →  00:05 UTC = **17:05 PDT** local.
- The assistant installs this as a real user crontab (`crontab -l` to view).
- `fetch_latest()` deliberately reads `raw[-2]` (the last *closed* MEXC 1d
  bar), never `raw[-1]` (the still-forming bar). Do not change this back.

Install / verify:

```bash
crontab -l
# expect:
# 5 0 * * * cd /home/lars/trading-bot && /home/lars/trading-bot/.venv/bin/python \
#   /home/lars/trading-bot/paper_trader_multi.py >> /home/lars/trading-bot/logs/paper_trader_cron.log 2>&1

# ensure the cron daemon is running
systemctl status cron        # should be active (enabled)
```

> The chat-assistant cron job `donchian_paper_trader` (id `22226284b0dc`) is
> **paused** — it used to drive the trader at 21:00 UTC (lookahead-prone). Do
> not re-enable it; the system crontab above is authoritative.

---

## 2. OHLCV forward collector — systemd user service

`collector_daemon.py` continuously appends 1h/4h/1d (MEXC+BloFin) + 8h
(BloFin only) so the intraday lookback compounds. It is a **systemd user
service** (`collector-daemon.service`), already enabled and running.

Status:

```bash
systemctl --user status collector-daemon
journalctl --user -u collector-daemon -f      # live logs
```

### ⚠ REQUIRED: enable-linger (survives logout)

The service is `enabled`, but `Linger=no` for user `lars` — meaning it
**stops when you log out / the session ends**. Fix once per machine:

```bash
sudo loginctl enable-linger lars      # run as root (or your user w/ sudo)
loginctl show-user lars -p Linger     # expect Linger=yes
```

After this, the collector runs 24/7 even with no active login session.

Restart after config/edits:

```bash
systemctl --user restart collector-daemon
```

---

## 3. Live universe screening — daily re-screen + weekly CoinGecko refresh

The live universe (`backtest_output/screen_liqu_idio_*.csv`) is **rebuilt every
day**, not hand-run. Two layers:

1. **Daily re-screen (local, inside the trader).** `paper_trader_multi.py` calls
   `screen_liquidity_idiosyncratic.run_screen()` at startup *before* it reads the
   screen. `run_screen()` is local-only (reads `data/universe_broad.csv` +
   local `*_1d_max.csv`, no network) and writes a **timestamped** CSV. The trader
   then reads the latest one. This means:
   - new listings are picked up automatically the next day,
   - delistings / coins that fall out of the top-idio-vol 20% are dropped,
   - it piggybacks on the existing `5 0 * * *` trader cron (no new scheduler),
   - a screen failure is caught and falls back to the last saved CSV (the
     trading run never aborts on a screen error).

2. **Weekly CoinGecko broad-universe refresh (network, rate-limited).** The
   screen only re-ranks within `universe_broad.csv`; to discover *brand-new*
   coins you must refresh that file. It hits the CoinGecko API (rate-limited),
   so it runs **weekly**, not daily:

   ```
   37 11 * * 0  cd /home/lars/trading-bot && /home/lars/trading-bot/.venv/bin/python \
     fetch_coingecko_universe.py >> /home/lars/trading-bot/logs/coingecko_fetch.log 2>&1
   ```
   (Sun 11:37 UTC.) Verify with `crontab -l`.

### Dropoff policy B (live positions)

When a coin with an **open position** drops off the daily screen, the trader
**force-closes it that day** (logged `CLOSE <sym> (dropped off live screen —
policy B)`). Rationale: without this, an unmonitored open position would
*strand* — the trader only watches screen-listed coins, so no future signal
would ever close it. The A/B dropoff test (`test_dropoff_policy.py`) showed
riding-to-exit (policy A) leaves this tail risk open; force-close (B) bounds it.
Positions still on the screen are managed normally to their signal exit.

---

## 4. CPU cores — `TRADING_BOT_CORES`

The model trainer / walk-forward / grid workers size their parallelism from a
single setting in `config.py`:

```python
N_PHYSICAL_CORES = int(os.environ.get("TRADING_BOT_CORES") or _detect_physical_cores())
N_JOBS = max(1, N_PHYSICAL_CORES - 1)   # leaves one core for OS + collector
```

- We use **physical** cores, not logical/SMT threads — crypto feature
  pipelines are memory-bandwidth bound, so oversubscribing to hyperthreads
  just adds contention.
- **After a CPU upgrade** (e.g. new Ryzen), set the env var — no code edit:

```bash
export TRADING_BOT_CORES=8      # in your shell profile / service env
```

- If unset, it falls back to `os.cpu_count() // 2` (≈ physical cores on most
  consumer chips; current host = Ryzen 5 5600X → 6 physical → N_JOBS = 5).

Verify the resolved value:

```bash
python -c "import config; print(config.N_PHYSICAL_CORES, config.N_JOBS)"
```

---

## 5. First-deploy checklist

- [ ] `python -m venv .venv && .venv/bin/pip install -r requirements.txt`
- [ ] `crontab -l` shows the `5 0 * * *` paper-trader line
- [ ] `systemctl --user status collector-daemon` active
- [ ] `sudo loginctl enable-linger lars` done (Linger=yes)
- [ ] `export TRADING_BOT_CORES=<your physical cores>` in profile
- [ ] Hermes cron job `22226284b0dc` left **paused**
