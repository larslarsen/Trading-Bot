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

## 3. CPU cores — `TRADING_BOT_CORES`

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

## 4. First-deploy checklist

- [ ] `python -m venv .venv && .venv/bin/pip install -r requirements.txt`
- [ ] `crontab -l` shows the `5 0 * * *` paper-trader line
- [ ] `systemctl --user status collector-daemon` active
- [ ] `sudo loginctl enable-linger lars` done (Linger=yes)
- [ ] `export TRADING_BOT_CORES=<your physical cores>` in profile
- [ ] Hermes cron job `22226284b0dc` left **paused**
