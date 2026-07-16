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
                # Bulk CDN first (static ZIPs, no rate limit) for deep history,
                # then REST pull() tops up the last live bars the ZIPs lack.
                rows = cex.pull_bulk(sym, "5m", start)
                bulk_last = rows[-1][0] if rows else (last if last else start)
                live = cex.pull(sym, "5m", cex.floor_ts(bulk_last + 1, "5m"))
                if live:
                    rows = rows + live
                if rows:
                    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume",
                                                     "close_time", "qav", "trades", "tbav", "tqav", "ignore"])
                    df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                    if path.exists():
                        old = pd.read_csv(path)
                        old["ts"] = cex.parse_ts(old["ts"])
                        df = pd.concat([old, df]).drop_duplicates(subset=["ts"]).sort_values("ts")
                    df.to_csv(path, index=False)
                    derive.derive_sym(sym, ["1h", "4h", "1d"])
                    print(f"  [cex {sym}] {len(rows)} new bars -> {path.name}", flush=True)
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
                    row = {"ts": pd.Timestamp.now("UTC"), "token": tok, **rec}
                    df = pd.DataFrame([row])
                    p = micro.OUT / f"{micro.safe_name(tok)}.csv"
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
                            d = pd.read_csv(tgt, parse_dates=["ts"])
                            if len(d):
                                last = int(d["ts"].max().timestamp() * 1000) + 1
                        start = last or int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)
                        if venue == "bybit":
                            rows, err = bco.bybit_klines(sym, start, 1000)
                            # bybit returns 7 fields [ts,o,h,l,c,v,close_time]; keep 6
                            rows = [r[:6] for r in rows] if rows else rows
                        elif venue == "okx":
                            # okx_klines walks backward from after_ms; use now to grab recent
                            rows, err = bco.okx_klines(sym.replace("-", ""), int(time.time() * 1000), 200)
                        else:
                            rows, err = bfm.mexc_klines(sym, start)
                        if err or not rows:
                            continue
                        # coerce each row to [ts(int-ms), o,h,l,c,v(float)]
                        clean = []
                        for r in rows:
                            try:
                                clean.append([int(float(r[0])), float(r[1]), float(r[2]),
                                              float(r[3]), float(r[4]), float(r[5])])
                            except Exception:
                                pass
                        if not clean:
                            continue
                        df = pd.DataFrame(clean, columns=["ts", "open", "high", "low", "close", "volume"])
                        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                        if tgt.exists():
                            old = pd.read_csv(tgt, parse_dates=["ts"]).set_index("ts")
                            df = pd.concat([old, df.set_index("ts")]).sort_index()
                            df = df[~df.index.duplicated(keep="last")]
                        else:
                            df = df.set_index("ts")
                        df.to_csv(tgt)
                        if sym == syms[0]:
                            print(f"  [extra {venue}] topped {sym} -> {len(df)} rows", flush=True)
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
                    df.to_csv(tgt)
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
                    for d in fresh[-30:]:  # only recent gap (rest is history)
                        try:
                            f = bon.daily_features(chain, d)
                            if f:
                                rows.append(f)
                        except Exception:
                            pass
                    if rows:
                        out = pd.DataFrame(rows).set_index("date").sort_index()
                        if tgt.exists():
                            o = pd.read_csv(tgt).set_index("date")
                            out = pd.concat([o, out]).drop_duplicates()
                        out.to_csv(tgt)
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
