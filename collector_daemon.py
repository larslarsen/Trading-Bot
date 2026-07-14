#!/usr/bin/env python3
"""
collector_daemon.py -- Hermes-independent multi-timeframe OHLCV forward collector.

Runs forever (intended as a `systemctl --user` service). For each configured
(exchange, symbol, timeframe) it APPENDS new bars forward (incremental: from the
last stored timestamp + 1 ms) to  data/<SYM>_<tf>_<exchange>_max.csv .

WHY incremental + WHY a daemon
  * Daily 1d history already exists. 4h / 1h / 8h CANNOT be resampled from 1d,
    so the only way to build their lookback is to collect forward continuously
    and never let the window stop growing. A daemon that appends every few
    minutes guarantees no gaps and a compounding lookback -- exactly the user's
    concern ("they don't run out of lookback and leave gaps").

SAFETY (so we never repeat the July-9 API-storm wedge)
  * Incremental fetch: each request asks only for bars newer than what we have.
    Steady-state work per (symbol,tf) is a handful of new bars, NOT a 1000-bar
    backfill.
  * Per-fetch PAUSE + a global rate ceiling (MAX_FETCHES_PER_MIN).
  * Per-cycle time budget (MAX_CYCLE_SECONDS): a cycle is cut short if it runs
    too long, so the loop self-throttles and can never wedge the machine. The
    initial BloFin backfill (494 symbols with no file yet) is therefore spread
    safely across several bounded cycles instead of one storm.
  * Lock file prevents two instances double-polling (that would double the load).
  * Every (symbol,tf) failure is caught and skipped; the daemon never dies.

Run:
  python collector_daemon.py            # forever loop
  python collector_daemon.py --once     # one cycle, then exit (for testing)
"""
from __future__ import annotations
import sys, os, time, fcntl, datetime as dt
import pandas as pd
import ccxt
from pathlib import Path

# ------------------------------- CONFIG --------------------------------------
REPO            = Path('/home/lars/trading-bot')
ROOT            = REPO / 'data'
SCREEN_DIR      = REPO / 'backtest_output'
BLOFIN_LIST     = REPO / 'data' / 'blofin_swap_pairs.txt'
LOG_PATH        = REPO / 'logs' / 'collector_daemon.log'
LOCK_PATH       = REPO / 'run' / 'collector_daemon.lock'
VENV_PY         = REPO / '.venv' / 'bin' / 'python'

# Timeframes to collect. 1h/4h are the intraday TFs that cannot be resampled
# from 1d and are the entire reason for this daemon. 1d is maintained too (cheap,
# closes any gaps in the daily series). 8h exists only on BloFin.
# 5m added for forward accumulation (deep 5m backfill is infeasible: BloFin
# ignores `since`; MEXC 5m over 2yr would be ~5hrs of calls). So 5m compounds
# from today and becomes testable in a few months.
TIMEFRAMES       = ['5m', '1h', '4h', '1d']
BLOFIN_EXTRA_TF  = ['8h']

CYCLE_INTERVAL   = 900      # seconds between full sweeps (15 min)
PAUSE            = 0.30     # base seconds between ccxt calls
MAX_FETCHES_PER_MIN = 150   # global rate ceiling (backstop to ccxt's own limiter)
MAX_CYCLE_SECONDS  = 1500   # abort a cycle past 25 min (self-throttle)
LIMIT            = 500      # max bars per ccxt call (initial backfill size)

# Which exchanges to collect. Kraken is OFF by default: its 787-symbol universe
# caused the original storm and its 1d already exists. Flip to True to add it.
ENABLE = {'mexc': True, 'blofin': True, 'kraken': False}
# ----------------------------------------------------------------------------

_ex_cache: dict = {}


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, 'a') as f:
            f.write(line + "\n")
    except Exception:
        pass


def get_ex(name: str):
    if name not in _ex_cache:
        ex = getattr(ccxt, name)({'enableRateLimit': True})
        ex.load_markets()
        _ex_cache[name] = ex
    return _ex_cache[name]


def map_symbols(ex, bare_list):
    """Map bare stems (e.g. BTCUSDT) to ccxt symbols.

    Handles spot-style (BASE/QUOTE) and derivative-style (BASE/QUOTE:SETTLE)
    exchanges like BloFin, where BTCUSDT lives at 'BTC/USDT:USDT'.
    """
    out = {}
    for sym in bare_list:
        matched = False
        for quote in ('USDT', 'USDC', 'USD'):
            if not sym.endswith(quote):
                continue
            base = sym[:-len(quote)]
            for cand in (f"{base}/{quote}", f"{base}/{quote}:{quote}"):
                if cand in ex.markets:
                    out[sym] = cand
                    matched = True
                    break
            if matched:
                break
    return out


def build_jobs():
    jobs = []
    if ENABLE['mexc']:
        sf = sorted(SCREEN_DIR.glob('screen_liqu_idio_*.csv'))
        if sf:
            stems = pd.read_csv(sf[-1])['stem'].astype(str).tolist()
            m = map_symbols(get_ex('mexc'), stems)
            for sym, cs in m.items():
                for tf in TIMEFRAMES:
                    jobs.append(('mexc', sym, cs, tf))
            log(f"MEXC: {len(m)} symbols x {len(TIMEFRAMES)} tf = {len(m)*len(TIMEFRAMES)} jobs")
    if ENABLE['blofin'] and BLOFIN_LIST.exists():
        pairs = [l.strip() for l in BLOFIN_LIST.read_text().split() if l.strip()]
        m = map_symbols(get_ex('blofin'), pairs)
        for sym, cs in m.items():
            for tf in TIMEFRAMES + BLOFIN_EXTRA_TF:
                jobs.append(('blofin', sym, cs, tf))
        log(f"BloFin: {len(m)} symbols x {len(TIMEFRAMES)+len(BLOFIN_EXTRA_TF)} tf = "
            f"{len(m)*(len(TIMEFRAMES)+len(BLOFIN_EXTRA_TF))} jobs")
    if ENABLE['kraken']:
        log("Kraken enabled=True but mapping not implemented here; skipping (set False).")
    return jobs


def fetch_forward(ex, ccxt_sym, tf, fname):
    """Append new bars to fname. Returns count of newly added bars."""
    existing = pd.read_csv(fname) if fname.exists() else None
    if existing is not None and len(existing):
        # CSV round-trip leaves 'ts' as str; coerce so concat/sort types match.
        existing['ts'] = pd.to_datetime(existing['ts'], utc=True)
    since = None
    if existing is not None and len(existing):
        last_ms = int(existing['ts'].max().value // 10**6) + 1
        since = last_ms
    bars = ex.fetchOHLCV(ccxt_sym, tf, since=since, limit=LIMIT)
    if not bars:
        return 0
    df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    if existing is not None and len(existing):
        df = pd.concat([existing, df])
    df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    df.to_csv(fname, index=False)
    return max(0, len(df) - (len(existing) if existing is not None else 0))


class RateGate:
    def __init__(self, per_min):
        self.per_min = per_min
        self.times: list[float] = []

    def tick(self):
        now = time.time()
        self.times = [t for t in self.times if now - t < 60]
        if len(self.times) >= self.per_min:
            sleep_for = 60 - (now - self.times[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.times.append(time.time())


def run_cycle(gate: RateGate, jobs: list, start_index: int = 0) -> tuple[int, int]:
    """Run one cycle over jobs (rotating start). Returns (new_bars, jobs_done)."""
    t0 = time.time()
    total_new = 0
    done = 0
    n = len(jobs)
    for i in range(n):
        idx = (start_index + i) % n
        ex_name, sym, ccxt_sym, tf = jobs[idx]
        if time.time() - t0 > MAX_CYCLE_SECONDS:
            log(f"cycle budget hit ({MAX_CYCLE_SECONDS}s); deferred {n - i} jobs to next cycle")
            break
        ex = get_ex(ex_name)
        fname = ROOT / f"{sym}_{tf}_{ex_name}_max.csv"
        try:
            gate.tick()
            added = fetch_forward(ex, ccxt_sym, tf, fname)
            total_new += added
            done += 1
            if done % 50 == 0:
                log(f"  ...progress {done}/{n} jobs, +{total_new} bars so far")
            time.sleep(PAUSE)
        except Exception as e:
            log(f"ERR {ex_name} {sym} {tf}: {str(e)[:80]}")
            time.sleep(PAUSE * 3)
    return total_new, done


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lockf = open(LOCK_PATH, 'w')
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another instance holds the lock; exiting")
        sys.exit(0)

    once = '--once' in sys.argv
    log(f"collector daemon start (once={once}, venv={VENV_PY.exists()})")
    jobs = build_jobs()
    log(f"TOTAL jobs: {len(jobs)}; cycle_interval={CYCLE_INTERVAL}s; budget={MAX_CYCLE_SECONDS}s")
    gate = RateGate(MAX_FETCHES_PER_MIN)
    cycle_count = 0
    if once:
        n, done = run_cycle(gate, jobs, 0)
        log(f"--once cycle: +{n} new bars over {done} jobs")
        return
    while True:
        cycle_start = time.time()
        try:
            n, done = run_cycle(gate, jobs, cycle_count)
        except Exception as e:
            log(f"cycle crashed (recovered): {str(e)[:120]}")
            n, done = 0, 0
        cycle_count += 1
        elapsed = time.time() - cycle_start
        log(f"cycle #{cycle_count}: +{n} new bars over {done} jobs ({elapsed:.0f}s)")
        sleep_for = max(5, CYCLE_INTERVAL - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
