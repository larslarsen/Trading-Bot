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

## 4. DEX universe (GeckoTerminal, free) + MTF poller requirement

DEX data comes from **GeckoTerminal's free public API** (no key). Two scripts:

- `build_dex_universe.py` — ranks top DEX pools by 24h volume across chains
  (eth, base, bsc, arbitrum, polygon, solana), dedupe to the highest-volume
  pool per token, writes `dex_universe.csv`. Data-driven, **low-cap-inclusive**
  (retail alts rank by volume). Re-runnable.
- `backfill_dex_history.py` — reads `dex_universe.csv`, fetches daily OHLC per
  token -> `dex_data/<SYM>_1d_max.csv` (same format as CEX `data/`). Free tier
  caps ~181 bars (~6mo) per pool + 429-rate-limits; `fetch_ohlcv` retries with
  exponential backoff, and the run is resume-safe (skips already-deep files).

Daily cron (rebuild universe + backfill), so DEX stays current:

```
17 3 * * *  cd /home/lars/trading-bot && source .venv/bin/activate && \
  python -u build_dex_universe.py --top-per-network 200 \
    --networks eth,base,bsc,arbitrum,polygon,solana --sleep 1.0 >> /tmp/dex_univ_cron.log 2>&1 && \
  python -u backfill_dex_history.py --sleep 2.5 >> /tmp/dex_fill_cron.log 2>&1
```

(03:17 UTC, no collision with the 00:05 trader or Sun-11:37 CoinGecko jobs.)

> Both `dex_universe.csv` and `dex_data/` are **gitignored** — they are local
> caches, regenerable by re-running the scripts. Never commit them.

### MTF / short-timeframe requirement (NOT yet built — design only)

The daily cron is correct for the **daily** paper trader. When we move to
**MTF (1h/4h) or short-timeframe (1m/5m) DEX trading**, universe freshness must
tighten, because a hot low-cap can 10x within minutes and a daily refresh would
miss the entire move. Two distinct concerns:

1. **Universe membership** (what tokens exist, top-by-volume): for MTF, refresh
   on a tighter schedule (e.g. every 1–4h via cron, or called at MTF session
   start) — reuses `build_dex_universe.py`.
2. **Per-trade liquidity / rug sanity** (is *this* token tradeable RIGHT NOW):
   must be a **lightweight probe inside the poller's entry path**, not a full
   universe rebuild. Before acting on a signal, the bot calls a
   `dex_liquidity_ok(symbol)` check (one GeckoTerminal pool call for current
   24h volume + that the pool isn't dead/rugged). This is the "constant update"
   for short timeframes — cheap (one request per candidate), API-friendly
   (doesn't hammer the free API with full rebuilds), and always-fresh at the
   decision point even if the cached universe is stale.

**Do NOT** rebuild the full 458-token universe constantly — that melts the free
API. Pattern: *stale universe for candidacy + fresh per-token validation at the
decision point.* Write `dex_liquidity_ok()` + a `refresh_dex_universe()` hook
when the MTF DEX bot is actually built (MTF 4h depth matures ~Oct 2026; free
DEX history is only ~6mo, so MTF DEX backtesting also needs paid CoinGecko
first).

---

## 5. CPU cores — `TRADING_BOT_CORES`

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

## 6. First-deploy checklist

- [ ] `python -m venv .venv && .venv/bin/pip install -r requirements.txt`
- [ ] `crontab -l` shows the `5 0 * * *` paper-trader line
- [ ] `systemctl --user status collector-daemon` active
- [ ] `sudo loginctl enable-linger lars` done (Linger=yes)
- [ ] `export TRADING_BOT_CORES=<your physical cores>` in profile
- [ ] Hermes cron job `22226284b0dc` left **paused**
