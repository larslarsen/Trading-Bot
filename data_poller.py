#!/usr/bin/env python3
"""Unified data poller -- the ONLY data-collection process.

Systemd-managed (Restart=always) so it auto-restarts on crash. Replaces the
scattered ad-hoc background processes and cron data jobs with one process
that covers ALL free data, all the time, via concurrent worker threads:

  * CEX thread  -> data/cex/<SYM>_5m.csv (Binance klines mirror, no key),
    then derive 1h/4h/1d locally (derive_cex_tf). Resumable: a cursor file
    tracks progress; restart continues where it left off. Runs continuously
    (wraps and refreshes all symbols forever).
  * DEX micro thread -> data/dex_micro/<TOKEN>.csv (DexScreener + CoinGecko
    backup) every MICRO_INTERVAL seconds.
  * DEX forward thread -> data/<SYM>_5m_dex_max.csv (DexScreener snapshot)
    every DEX_FWD_INTERVAL seconds (reuses dex_forward_collector.snapshot).
  * DEX universe rebuild -> build_dex_universe + backfill_dex_history, weekly,
    so the 426-token universe stays current.

Each worker is a self-contained loop wrapped in try/except; a transient error
in one does not kill the others. State persisted to data/.poller_state.json.

Usage:
  python data_poller.py            # run all worker threads forever
  python data_poller.py --once     # run one pass of each worker then exit
"""
import argparse
import json
import sys
import time
import threading
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import dex_micro_poller as micro
import backfill_cex_all as cex
import derive_cex_tf as derive

REPO = Path(__file__).parent
STATE = REPO / "data" / ".poller_state.json"

MICRO_INTERVAL = 600        # DEX breadth poll every 10 min
DEX_FWD_INTERVAL = 300      # DEX 5m forward snap every 5 min
UNIVERSE_INTERVAL = 7 * 86400  # DEX universe rebuild weekly
CEX_BATCH_PER_PASS = 20     # symbols per CEX pass before yielding to DEX ticks
DEX_FWD_SLEEP = 0.15


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"cex_cursor": 0, "last_universe": 0.0}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=2))


def cex_5m_path(sym):
    """Canonical 5m path for a symbol -- the SAME file every bot + the trainer
    reads (single source of truth, one place). BTC is the root btc_5m.csv;
    all other pairs follow the data/<SYM>USDT_5m_max.csv convention that
    pipeline.fetch_data() resolves. The poller writes HERE and nowhere else."""
    if sym == "BTCUSDT":
        return REPO / "btc_5m.csv"
    stem = sym.replace("USDT", "")
    return REPO / "data" / f"{stem}USDT_5m_max.csv"


def cex_worker(s, once):
    syms = cex.get_syms()
    n = len(syms)
    while True:
        cursor = int(s.get("cex_cursor", 0)) % n
        end = min(cursor + CEX_BATCH_PER_PASS, n)
        for i in range(cursor, end):
            sym = syms[i]
            path = cex_5m_path(sym)
            last = cex.existing_last_ms(path)
            now_ms = int(time.time() * 1000)
            if last is not None and last >= cex.floor_ts(now_ms, "5m"):
                continue  # already complete
            start = 1262304000000 if last is None else cex.floor_ts(last + 1, "5m")
            try:
                rows = cex.pull(sym, "5m", start)
                if rows:
                    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                    if path.exists():
                        old = pd.read_csv(path)
                        old["ts"] = pd.to_datetime(old["ts"], utc=True)
                        df = pd.concat([old, df]).drop_duplicates(subset=["ts"]).sort_values("ts")
                    df.to_csv(path, index=False)
                    derive.derive_sym(sym, ["1h", "4h", "1d"])
            except Exception as e:
                print(f"  [cex {sym}] error: {e}", flush=True)
        s["cex_cursor"] = end % n
        save_state(s)
        print(f"  [cex] cursor -> {s['cex_cursor']}/{n}", flush=True)
        if once:
            return
        # yield briefly so DEX threads get scheduler time (they sleep anyway)
        time.sleep(1)


def micro_worker(once):
    while True:
        try:
            toks = micro.tokens()
            done = 0
            for tok in toks:
                try:
                    rec, err = micro.poll_token(tok)
                    if rec is None:
                        continue
                    row = {"ts": pd.Timestamp.now("UTC"), **rec}
                    df = pd.DataFrame([row])
                    p = micro.OUT / f"{tok}.csv"
                    if p.exists():
                        old = pd.read_csv(p, parse_dates=["ts"])
                        df = pd.concat([old, df]).drop_duplicates(subset=["ts"]).sort_values("ts")
                    df.to_csv(p, index=False)
                    done += 1
                except Exception:
                    pass
            print(f"  [micro] updated {done}/{len(toks)} tokens", flush=True)
        except Exception as e:
            print(f"  [micro] error: {e}", flush=True)
        if once:
            return
        time.sleep(MICRO_INTERVAL)


def dex_fwd_worker(once):
    while True:
        try:
            import dex_forward_collector as dfc
            pairs = dfc.resolve_pairs(force=False)
            if pairs:
                dfc.snapshot(pairs)
                print(f"  [dex_fwd] snapped {len(pairs)} pairs", flush=True)
        except Exception as e:
            print(f"  [dex_fwd] error: {e}", flush=True)
        if once:
            return
        time.sleep(DEX_FWD_INTERVAL)


def universe_worker(s, once):
    while True:
        now = time.time()
        if now - float(s.get("last_universe", 0.0)) >= UNIVERSE_INTERVAL:
            try:
                import build_dex_universe as bdu
                bdu.build(networks=["eth", "base", "bsc", "arbitrum", "polygon", "solana"],
                          top_per_network=200, sleep=1.0)
                try:
                    import backfill_dex_history as bdh
                    bdh.main()
                except Exception as e:
                    print(f"  [universe] history error: {e}", flush=True)
                s["last_universe"] = now
                save_state(s)
                print("  [universe] rebuilt", flush=True)
            except Exception as e:
                print(f"  [universe] error: {e}", flush=True)
        if once:
            return
        time.sleep(3600)  # check hourly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="one pass of each worker then exit")
    args = ap.parse_args()
    s = load_state()
    print("data_poller: starting workers (CEX 5m sweep + DEX micro + DEX forward + DEX universe)", flush=True)
    threads = [
        threading.Thread(target=cex_worker, args=(s, args.once), daemon=True),
        threading.Thread(target=micro_worker, args=(args.once,), daemon=True),
        threading.Thread(target=dex_fwd_worker, args=(args.once,), daemon=True),
        threading.Thread(target=universe_worker, args=(s, args.once), daemon=True),
    ]
    for t in threads:
        t.start()
    if args.once:
        for t in threads:
            t.join()
        print("data_poller: one-pass done", flush=True)
        return
    # keep main alive so systemd sees the process; restart handled by systemd
    while True:
        time.sleep(60)
        if not all(t.is_alive() for t in threads):
            print("data_poller: a worker died; exiting for systemd restart", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
