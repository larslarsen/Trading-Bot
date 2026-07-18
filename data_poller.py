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
import reliability as rel
import dex_micro_poller as micro
import backfill_cex_all as cex
import derive_cex_tf as derive

REPO = Path(__file__).parent
STATE = REPO / "data" / ".poller_state.json"
# Guards the shared `s` dict (cex_cursor / last_universe) mutated + persisted by
# multiple worker threads. Without it, concurrent writes interleave and one
# worker's update can be clobbered by another's save_state().
_STATE_LOCK = threading.Lock()

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
            # corrupt state file -> start fresh rather than crash the poller
            print(f"  [state] corrupt .poller_state.json; resetting", flush=True)
    return {"cex_cursor": 0, "last_universe": 0.0}


def save_state(s):
    # atomic: a kill mid-write cannot leave a half-written state file that
    # load_state would then choke on next start.
    rel.atomic_write_json(STATE, s)


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
    # backfill_cex_all exposes SYMS (csv string), not get_syms(); its other
    # pull_* helpers don't exist, so top up the canonical 5m file per symbol
    # using the working bybit/okx/mexc kline functions (resumable: they only
    # append bars newer than what tgt already holds).
    import backfill_cex_others as bco
    import backfill_funding_mexc as bfm
    syms = [x.strip() for x in __import__("backfill_cex_all").SYMS.split(",") if x.strip()]
    n = len(syms)
    while True:
        cursor = int(s.get("cex_cursor", 0)) % n
        end = min(cursor + CEX_BATCH_PER_PASS, n)
        for i in range(cursor, end):
            sym = syms[i]
            path = cex_5m_path(sym)
            last = None
            if path.exists():
                d = pd.read_csv(path)
                if len(d) and "ts" in d.columns:
                    last = int(pd.to_datetime(d["ts"], utc=True).max().timestamp() * 1000) + 1
            start = last or int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)
            try:
                _, err = bco.bybit_klines(sym, start, path)
                if err:
                    _, err = bco.okx_klines(sym.replace("USDT", "-USDT"),
                                            int(time.time() * 1000), path)
                if not err:
                    derive.derive_sym(sym, ["1h", "4h", "1d"])
                    print(f"  [cex {sym}] topped -> {len(pd.read_csv(path))} rows", flush=True)
            except Exception as e:
                print(f"  [cex {sym}] error: {e}", flush=True)
        s["cex_cursor"] = end % n
        with _STATE_LOCK:
            save_state(s)
        print(f"  [cex] cursor -> {s['cex_cursor']}/{n}", flush=True)
        if once:
            return
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
                    row = {"ts": pd.Timestamp.now("UTC"), "token": tok, **rec}
                    df = pd.DataFrame([row])
                    p = micro.OUT / f"{micro.safe_name(tok)}.csv"
                    if p.exists():
                        old = pd.read_csv(p, parse_dates=["ts"])
                        df = pd.concat([old, df]).drop_duplicates(subset=["ts"]).sort_values("ts")
                    rel.atomic_write_csv(p, df, index=False)  # crash-safe append
                    done += 1
                except Exception as e:
                    # log the offending token so a bad one is visible, not swallowed
                    print(f"  [micro] {tok} error: {type(e).__name__}: {e}", flush=True)
            print(f"  [micro] updated {done}/{len(toks)} tokens", flush=True)
        except Exception as e:
            print(f"  [micro] error: {type(e).__name__}: {e}", flush=True)
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
                with _STATE_LOCK:
                    save_state(s)
                print("  [universe] rebuilt", flush=True)
            except Exception as e:
                print(f"  [universe] error: {e}", flush=True)
        if once:
            return
        time.sleep(3600)  # check hourly


def cex_extra_topup_worker(once):
    """Top-up the extra CEX venues (Bybit/OKX/MEXC 5m) + Bybit funding rates
    that the one-shot backfills produced, so they stay current ('all the data,
    all the time'). Reuses backfill_cex_others + backfill_funding_mexc funcs."""
    import backfill_cex_others as bco
    import backfill_funding_mexc as bfm
    DATADIR = REPO / "data"
    EXTRA = {
        "bybit": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                  "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT"],
        "okx": ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT",
                "ADA-USDT", "DOGE-USDT", "AVAX-USDT", "LINK-USDT", "MATIC-USDT"],
        "mexc": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                 "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT"],
    }
    FUND = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT"]
    while True:
        try:
            # 5m top-up per venue
            for venue, syms in EXTRA.items():
                for sym in syms:
                    try:
                        tgt = DATADIR / f"{sym}_5m_max.csv"
                        last = None
                        if tgt.exists():
                            d = pd.read_csv(tgt)
                            if len(d) and "ts" in d.columns:
                                last = int(pd.to_datetime(d["ts"], utc=True).max().timestamp() * 1000) + 1
                        start = last or int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)
                        if venue == "bybit":
                            # bybit_klines(sym, start_ms, tgt, limit) appends
                            # pages directly to tgt; returns (result, err).
                            _, err = bco.bybit_klines(sym, start, tgt)
                        elif venue == "okx":
                            # okx_klines(sym, after_ms, tgt, limit) walks BACKWARD
                            # from now until oldest <= after_ms. Pass the cursor
                            # (last+1) as after_ms so it backfills from the cursor
                            # to now. Passing `int(time.time()*1000)` (now) made it
                            # a permanent no-op (oldest <= now immediately).
                            _, err = bco.okx_klines(sym.replace("-", ""), start, tgt)
                        else:
                            # mexc_klines(sym, start_ms, end_ms=None) RETURNS rows.
                            rows, err = bfm.mexc_klines(sym, start)
                            if err or not rows:
                                continue
                            clean = []
                            for r in rows:
                                try:
                                    clean.append([int(float(r[0])), float(r[1]),
                                                  float(r[2]), float(r[3]),
                                                  float(r[4]), float(r[5])])
                                except Exception:
                                    pass
                            if not clean:
                                continue
                            df = pd.DataFrame(clean, columns=["ts", "open", "high",
                                                              "low", "close", "volume"])
                            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                            if tgt.exists():
                                old = pd.read_csv(tgt, parse_dates=["ts"]).set_index("ts")
                                df = pd.concat([old, df.set_index("ts")]).sort_index()
                                df = df[~df.index.duplicated(keep="last")]
                            else:
                                df = df.set_index("ts")
                            rel.atomic_write_csv(tgt, df.reset_index(), index=False)
                        if err:
                            continue
                        if sym == syms[0]:
                            print(f"  [extra {venue}] topped {sym} -> {len(pd.read_csv(tgt))} rows", flush=True)
                    except Exception as e:
                        print(f"  [extra {venue} {sym}] err {str(e)[:120]}", flush=True)
            # funding top-up
            fdir = DATADIR / "funding"
            fdir.mkdir(parents=True, exist_ok=True)
            end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
            for sym in FUND:
                try:
                    tgt = fdir / f"{sym}_funding.csv"
                    cursor = int(pd.Timestamp("2019-01-01", tz="UTC").timestamp() * 1000)
                    if tgt.exists():
                        o = pd.read_csv(tgt, parse_dates=["ts"])
                        if len(o):
                            cursor = int(o["ts"].max().timestamp() * 1000) + 1
                            if cursor > end_ms - 8 * 3600 * 1000:  # already fresh
                                continue
                    rows, err = bfm.bybit_funding(sym, cursor, end_ms, limit=200, win_ms=10 * 86400 * 1000)
                    if err or not rows:
                        continue
                    df = pd.DataFrame(rows, columns=["ts", "funding_rate"])
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                    if tgt.exists():
                        old = pd.read_csv(tgt, parse_dates=["ts"]).set_index("ts")
                        df = pd.concat([old, df.set_index("ts")]).sort_index()
                        df = df[~df.index.duplicated(keep="last")]
                    else:
                        df = df.set_index("ts")
                    df["interval_hours"] = 8
                    rel.atomic_write_csv(tgt, df.reset_index(), index=False)
                except Exception as e:
                    print(f"  [funding {sym}] err {str(e)[:120]}", flush=True)
            print("  [extra] top-up pass done", flush=True)
        except Exception as e:
            print(f"  [extra] worker error: {e}", flush=True)
        if once:
            return
        time.sleep(3600)  # top-up hourly


def onchain_topup_worker(once):
    """Top-up on-chain feature CSVs (BTC/ETH + 5 new SonarX chains) so they stay
    current. Reuses backfill_onchain.list_days + daily_features."""
    import backfill_onchain as bon
    CHAINS = ["btc", "eth", "base", "arbitrum", "aptos", "provenance", "xrp"]
    while True:
        try:
            for chain in CHAINS:
                try:
                    days = bon.list_days(chain)
                    if not days:
                        continue
                    tgt = bon.OUT / f"{chain}_features_daily.csv"
                    done = set()
                    if tgt.exists():
                        old = pd.read_csv(tgt)
                        done = set(old["date"].astype(str))
                    fresh = [d for d in days if d not in done]
                    if not fresh:
                        continue
                    rows = []
                    for d in fresh:  # all missing days; done-set prevents redo next pass
                        try:
                            f = bon.daily_features(chain, d)
                            if f:
                                rows.append(f)
                        except Exception as e:
                            # log the specific day so a bad fetch is visible
                            print(f"  [onchain {chain}] {d} error: {type(e).__name__}: {e}", flush=True)
                    if rows:
                        out = pd.DataFrame(rows).set_index("date").sort_index()
                        if tgt.exists():
                            o = pd.read_csv(tgt).set_index("date")
                            out = pd.concat([o, out]).drop_duplicates()
                        rel.atomic_write_csv(tgt, out)
                        print(f"  [onchain {chain}] +{len(out)} days", flush=True)
                except Exception as e:
                    print(f"  [onchain {chain}] err {e!r}"[:160], flush=True)
            print("  [onchain] top-up pass done", flush=True)
        except Exception as e:
            print(f"  [onchain] worker error: {e}", flush=True)
        if once:
            return
        time.sleep(6 * 3600)  # on-chain top-up every 6h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="one pass of each worker then exit")
    args = ap.parse_args()
    s = load_state()
    print("data_poller: starting workers (CEX 5m sweep + DEX micro + DEX forward + "
          "DEX universe + extra CEX top-up + on-chain top-up)", flush=True)
    threads = [
        threading.Thread(target=cex_worker, args=(s, args.once), daemon=True),
        threading.Thread(target=micro_worker, args=(args.once,), daemon=True),
        threading.Thread(target=dex_fwd_worker, args=(args.once,), daemon=True),
        threading.Thread(target=universe_worker, args=(s, args.once), daemon=True),
        threading.Thread(target=cex_extra_topup_worker, args=(args.once,), daemon=True),
        threading.Thread(target=onchain_topup_worker, args=(args.once,), daemon=True),
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
