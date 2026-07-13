#!/usr/bin/env python3
"""
Live data feed + microstructure poller.
Bybit CCXT pulls OHLCV, funding, OI; REST fallbacks cover Bybit v5 gaps.
Adds public taker trade history + orderbook snapshot polling.
Saves to CSV for later feature engineering.
"""
import os
import sys
import time
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import ccxt

# ── CONFIG ──────────────────────────────────────────────────────────
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
DERIV_SYMBOL = "BTC/USDT:USDT"
OUT_DIR = Path(__file__).parent

OHLCV_FILE = OUT_DIR / "live_history.csv"
FUND_FILE = OUT_DIR / "funding_history.csv"
OI_FILE = OUT_DIR / "oi_history.csv"
INS_FILE = OUT_DIR / "liquidations_history.csv"
TRADE_AGG_FILE = OUT_DIR / "trade_agg_5m.csv"
ORDERBOOK_FILE = OUT_DIR / "orderbook_5m.csv"

FETCH_LIMIT = 500
POLL_SEC = 270

BYBIT_REST = "https://api.bybit.com/v5/market"


# ── HELPERS ─────────────────────────────────────────────────────────
def ts_ms_to_dt(ts):
    if ts is None:
        return None
    if ts > 1e12:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def floor_to_5m(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def _coerce_num(v, default=None):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


# ── IO ──────────────────────────────────────────────────────────────
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["ts"])
    if "ts" in df.columns:
        df["ts"] = _to_utc(df["ts"])
        df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def append_csv(path: Path, new: pd.DataFrame, expected_cols: list[str] | None = None):
    expected_cols = expected_cols or ["ts"] + [c for c in new.columns if c != "ts"]
    old = read_csv(path)
    merged = pd.concat([old, new], ignore_index=True) if not old.empty else new.copy()
    merged = merged.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    for c in expected_cols:
        if c not in merged.columns:
            merged[c] = None
    merged[expected_cols].to_csv(path, index=False)


dump_csv = lambda path, df: df.to_csv(path, index=False)


# ── FETCHERS ────────────────────────────────────────────────────────
def fetch_ohlcv(ex):
    bars = ex.fetch_ohlcv("BTC/USDT:USDT", timeframe=TIMEFRAME, limit=FETCH_LIMIT)
    rows = []
    for b in bars:
        rows.append({
            "ts": ts_ms_to_dt(b[0]),
            "open": _coerce_num(b[1]),
            "high": _coerce_num(b[2]),
            "low": _coerce_num(b[3]),
            "close": _coerce_num(b[4]),
            "volume": _coerce_num(b[5]),
        })
    return pd.DataFrame(rows)


def fetch_funding(ex):
    try:
        fr = ex.fetchFundingRate(DERIV_SYMBOL)
        row = {
            "ts": ts_ms_to_dt(fr.get("timestamp")),
            "funding_rate": _coerce_num(fr.get("fundingRate")),
            "index_price": _coerce_num(fr.get("indexPrice")),
            "mark_price": _coerce_num(fr.get("markPrice")),
        }
        return pd.DataFrame([row])
    except Exception:
        try:
            r = requests.get(f"{BYBIT_REST}/funding/prev-rate", params={
                "category": "linear", "symbol": "BTCUSDT", "limit": 1
            }, timeout=10)
            item = r.json().get("result", {}).get("list", [{}])[0]
            return pd.DataFrame([{
                "ts": ts_ms_to_dt(int(item.get("fundingRateTimestamp") or 0) or None),
                "funding_rate": _coerce_num(item.get("fundingRate")),
                "index_price": None,
                "mark_price": None,
            }])
        except Exception:
            return pd.DataFrame(columns=["ts", "funding_rate", "index_price", "mark_price"])


def fetch_open_interest(ex):
    try:
        oi = ex.fetchOpenInterest(DERIV_SYMBOL)
        info = oi.get("info", {}) if isinstance(oi, dict) else {}
        row = {
            "ts": ts_ms_to_dt(oi.get("timestamp")),
            "open_interest": _coerce_num(
                oi.get("openInterestAmount")
                or info.get("openInterest")
                or info.get("singleOpenInterest")
            ),
            "open_interest_value": _coerce_num(oi.get("openInterestValue")),
        }
        return pd.DataFrame([row])
    except Exception:
        return pd.DataFrame(columns=["ts", "open_interest", "open_interest_value"])


def fetch_insurance():
    try:
        r = requests.get(f"{BYBIT_REST}/insurance", params={"category": "linear"}, timeout=10)
        d = r.json()
        if d.get("retCode") == 0:
            upd = d["result"]["updatedTime"]
            btc = next((x for x in d["result"].get("list", []) if x.get("coin") == "BTC"), {})
            return pd.DataFrame([{
                "ts": datetime.fromtimestamp(upd / 1000, tz=timezone.utc),
                "insurance_balance": _coerce_num(btc.get("balance")),
                "symbols": btc.get("symbols"),
            }])
    except Exception:
        pass
    return pd.DataFrame(columns=["ts", "insurance_balance", "symbols"])


def fetch_recent_trades():
    """
    Fetch recent taker trades from Bybit linear market.
    Returns individual trades with exact timestamps for tick-level storage.
    """
    try:
        params = {
            "category": "linear",
            "symbol": "BTCUSDT",
            "limit": 200,
        }
        r = requests.get(f"{BYBIT_REST}/recent-trade", params=params, timeout=15)
        d = r.json()
        items = d.get("result", {}).get("list", [])
        if not items:
            return pd.DataFrame(columns=["ts", "side", "size", "trade_id"])
        rows = []
        seen = set()
        for it in items:
            t = ts_ms_to_dt(int(it.get("time") or 0) or None)
            tid = str(it.get("execId") or it.get("trade_id") or "")
            if t is None or tid in seen:
                continue
            seen.add(tid)
            side = (it.get("side") or "").lower()
            size = _coerce_num(it.get("size"), default=0.0)
            rows.append({"ts": t, "side": side, "size": size, "trade_id": tid})
        return pd.DataFrame(rows)
    except Exception as e:
        print("[micro] recent-trade error:", e)
        return pd.DataFrame(columns=["ts", "side", "size", "trade_id"])


def fetch_orderbook_snapshot():
    """
    Snapshot best bid/ask from Bybit linear orderbook.
    """
    try:
        params = {"category": "linear", "symbol": "BTCUSDT", "limit": 1}
        r = requests.get(f"{BYBIT_REST}/orderbook", params=params, timeout=10)
        d = r.json()
        b = d.get("result", {}).get("b", [[None, None]])[0]
        a = d.get("result", {}).get("a", [[None, None]])[0]
        best_bid, best_bid_size = b
        best_ask, best_ask_size = a
        bid = _coerce_num(best_bid)
        ask = _coerce_num(best_ask)
        row = {
            "ts": datetime.now(timezone.utc).replace(second=0, microsecond=0),
            "best_bid": bid,
            "best_bid_size": _coerce_num(best_bid_size),
            "best_ask": ask,
            "best_ask_size": _coerce_num(best_ask_size),
        }
        if bid is not None and ask is not None and ask > 0 and bid > 0:
            row["spread"] = ask - bid
            row["mid_price"] = (ask + bid) / 2
            row["imbalance"] = row["best_bid_size"] / (row["best_bid_size"] + row["best_ask_size"] + 1e-10)
        else:
            row["spread"] = None
            row["mid_price"] = None
            row["imbalance"] = None
        return pd.DataFrame([row])
    except Exception as e:
        print("[micro] orderbook error:", e)
        return pd.DataFrame(columns=["ts", "best_bid", "best_bid_size", "best_ask", "best_ask_size", "spread", "mid_price", "imbalance"])


# ── MAIN LOOP ───────────────────────────────────────────────────────
def poll_once():
    ex = ccxt.bybit({"enableRateLimit": True})
    ohlcv = fetch_ohlcv(ex)
    append_csv(OHLCV_FILE, ohlcv)
    print(f"  OHLCV: {len(ohlcv)} bars, latest={ohlcv.iloc[-1]['close']}")

    fr = fetch_funding(ex)
    if not fr.empty:
        append_csv(FUND_FILE, fr)
        print(f"  Funding: {fr.iloc[0]['funding_rate']} mark={fr.iloc[0]['mark_price']}")

    oi = fetch_open_interest(ex)
    if not oi.empty:
        append_csv(OI_FILE, oi)
        print(f"  OI: {oi.iloc[0]['open_interest']}")

    ins = fetch_insurance()
    if not ins.empty:
        append_csv(INS_FILE, ins)
        print(f"  Insurance: {ins.iloc[0]['insurance_balance']}")

    trades = fetch_recent_trades()
    if not trades.empty:
        append_csv(TRADE_AGG_FILE, trades, ["ts", "taker_buy_vol", "taker_sell_vol", "trade_count"])
        print(f"  Trades: {trades['trade_count'].sum()} trades, taker_buy={trades.iloc[-1]['taker_buy_vol']}")

    ob = fetch_orderbook_snapshot()
    if not ob.empty:
        append_csv(ORDERBOOK_FILE, ob, ["ts", "best_bid", "best_bid_size", "best_ask", "best_ask_size", "spread", "mid_price", "imbalance"])
        print(f"  OB: spread={ob.iloc[0]['spread']}, imbalance={ob.iloc[0]['imbalance']}")


def run():
    print("Live microstructure poller")
    print(f"  symbol={SYMBOL} deriv={DERIV_SYMBOL} timeframe={TIMEFRAME}")
    print(f"  poll every {POLL_SEC}s\n")
    while True:
        try:
            poll_once()
        except Exception as e:
            print("Error:", e)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    run()
