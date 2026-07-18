#!/usr/bin/env python3
"""
Multi-pair 5m ML paper trader.

One daemon that trades ALL our single-pair XGBoost models, ranked by signal
strength -- NOT all pairs all the time. For each configured pair it loads the
pair's 5m history, builds features with the EXACT pipeline model_trainer uses
(resampled + macro + multi-asset + micro + DEX breadth + derive + regime),
loads that pair's model, predicts a signal + confidence, then trades the
top-N pairs by signal strength via a single multi-position book.

This is the "rank ML trades like a screener" bot: instead of holding every
pair, it holds the N strongest directional signals (N = CONFIG.max_positions).

Inference is inline (models loaded once at startup) so there is one process to
supervise, not one server + one trader per pair.

Usage:
  python cex_ml_xgb_5m.py            # loop forever, poll every 300s
  python cex_ml_xgb_5m.py --once    # one ranking + execution pass
"""
import argparse
import gc
import json
import math
import signal
import sys
import time
import warnings
from datetime import datetime, timezone, date
from functools import lru_cache
from pathlib import Path
import fcntl

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import xgboost as xgb

from config import CONFIG
from pipeline import (
    fetch_data, add_resampled_features, load_macro_data, add_macro_signals,
    derive_features, detect_regime, ALL_FEATURES, MULTI_ASSET_FILE, USE_MULTI_ASSET,
    ROOT, DATA_FILE,
)
from multi_asset_features import add_multi_asset_features
from order_manager_multi import MultiPositionState
from mem_guard import rss_mb, guard as mem_guard_abort

# ── cached CSV reads ──────────────────────────────────────────────────────
# fetch_data() does a full pd.read_csv of the multi-million-row 5m files on
# EVERY call. price_for() calls it once per ranked pair per poll, so a single
# cycle re-reads BTC's 1.4M-row CSV a dozen+ times -> RSS spikes to ~22 GB and
# the kernel OOM-killer fires. Cache each CSV per file-mtime: re-read only when
# the poller appends a new bar (mtime changes), so live freshness is preserved.
import os as _os
_FETCH_CACHE = {}  # path_str -> (mtime_ns, DataFrame)

def _path_for(symbol):
    """Resolve the CSV fetch_data() would load, without calling it."""
    if symbol is None or str(symbol).upper() in ("BTC", "BTC/USDT"):
        return ROOT / DATA_FILE
    stem = str(symbol).upper().replace("/", "").replace("USDT", "")
    candidates = [
        ROOT / "data" / f"{stem}USDT_5m_max.csv",
        ROOT / "data" / f"{stem}USDT_5m_blofin_max.csv",
        ROOT / "data" / f"{stem}USDC_5m_blofin_max.csv",
        ROOT / f"{stem.lower()}_5m.csv",
    ]
    return next((p for p in candidates if p.exists()), None)

def _cached_fetch(symbol):
    path = _path_for(symbol)
    if path is None:
        return fetch_data(symbol)  # let fetch_data raise its own FileNotFoundError
    key = str(path)
    mtime = _os.stat(path).st_mtime_ns
    if key in _FETCH_CACHE and _FETCH_CACHE[key][0] == mtime:
        return _FETCH_CACHE[key][1]
    df = fetch_data(symbol)
    # Cap the persistent cache to a rolling window: the model only needs recent
    # history, and holding every full CSV per pair is what OOM-killed us.
    if len(df) > MAX_BARS:
        df = df.iloc[-MAX_BARS:]
    _FETCH_CACHE[key] = (mtime, df)
    return df

# Live daemon: THROTTLE (skip a pass), never hard-exit, so trading keeps running.
MEM_GUARD_MB = 6144  # above this RSS, skip the heavy feature build for one cycle

# Rolling window retained per symbol. The model only predicts on the latest
# bar's feature vector; indicators (resampled 1h/4h, MAs, regime) need history
# but ~5000 5m bars (~17d) is far more than enough. Capping BOTH caches to this
# window bounds persistent RSS to ~32 * 5000 * ~120 cols (~150 MB) instead of
# holding every full CSV + full resolved frame per pair (which OOM-killed us
# at 8 GB RAM + 7.6 GB swap every cycle).
MAX_BARS = 5000

REPO = Path(__file__).parent
MODELS_DIR = REPO / "models"
# Screener: which models are ACTIVE at any given time. The bot auto-discovers
# every models/<sym>_xgb.json, but only trades the pairs listed here. Edit this
# file to reshuffle the universe without retraining. One pair per line, '#'
# comments. Missing models are skipped at load.
SCREENER_FILE = REPO / "screener_ml_multi.txt"

POLL_SEC = 300
STATE_FILE = REPO / "execution_state_ml_multi.json"
JOURNAL_FILE = REPO / "trade_journal_ml_multi.json"

# Screener / sizing now sourced from config.CONFIG (single source of truth) so
# the live trader and the backtest replay share identical risk parameters.
# Defaults evaluate to the historical values: 5 positions, $2000 = 20% of $10k.
MAX_POSITIONS = CONFIG.max_positions
SIZE_USD = CONFIG.initial_capital * CONFIG.max_position_pct
# Minimum confidence to take a directional (LONG/SHORT) signal. Trader-specific
# gate; kept here with the live-trader knobs.
CONFIDENCE_THRESHOLD = 0.60
FEE_BP = 0.60
SLIPPAGE_BP = 5

# ── graceful shutdown ───────────────────────────────────────────────────────
_SHUTDOWN = False

def _on_signal(signum, frame):
    global _SHUTDOWN
    _SHUTDOWN = True
    print(f"[{datetime.now(timezone.utc).isoformat()}] received "
          f"{signal.Signals(signum).name}; finishing cycle then saving state + exiting")

# ── daily-bar cadence ───────────────────────────────────────────────────────
# Circuit breakers (daily loss, flash-crash) are reset/rolled by
# start_daily_bar(), which the live loop MUST call once per UTC day. Without it
# the "daily" loss limit is cumulative (halts permanently after any 3% DD) and
# the flash-crash window never populates. Call it on day rollover only.
_last_daily_day: list = [None]

def _maybe_start_daily_bar(state, ref_price):
    today = date.today()  # local date is fine; rollover granularity is what matters
    if _last_daily_day[0] != today:
        if ref_price and ref_price > 0:
            state.start_daily_bar(ref_price)
        _last_daily_day[0] = today

# ── model auto-discovery + screener ────────────────────────────────────────
def discover_models():
    """Return {pair: model_path} for every 5m single-pair model on disk.

    Conventions:
      - latest_xgb.json      -> BTCUSDT (BTC 5m model, serving-bot alias)
      - <SYM>USDT_xgb.json   -> <SYM>USDT (e.g. doge_xgb.json -> DOGEUSDT)
    Foreign bots' models (cex_1d_xgb.json, dex_xgb.json) are EXCLUDED -- they
    are not 5m single-pair models for this multi-pair screener bot.
    """
    EXCLUDE = {"cex_1d_xgb.json", "dex_xgb.json"}  # other bots' models
    found = {}
    for p in MODELS_DIR.glob("*_xgb.json"):
        if p.name in EXCLUDE:
            continue
        if p.name == "latest_xgb.json":
            found["BTCUSDT"] = p
            continue
        sym = p.stem.replace("_xgb", "").upper()
        pair = sym if sym.endswith("USDT") else f"{sym}USDT"
        found[pair] = p
    return found

def load_screener():
    """Pairs the bot should trade, from SCREENER_FILE (one per line)."""
    if not SCREENER_FILE.exists():
        return None  # None => trade ALL discovered models
    pairs = []
    for line in SCREENER_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pairs.append(line.upper())
    return pairs

def active_pairs():
    """{pair: model_path} restricted to the screener (or all if no screener)."""
    allm = discover_models()
    screen = load_screener()
    if screen is None:
        return allm
    return {p: allm[p] for p in screen if p in allm}


def load_models():
    """Load every ACTIVE pair's model once. Returns {pair: XGBClassifier}.

    FAILS FAST AND LOUD: every model must expose n_features_in_ == canonical
    N_FEATURES AND feature_names_in_ == canonical CANONICAL. A mismatch means a
    stale/wrong model was trained or a code drift changed the feature set;
    serving a mismatched block silently ranks FLAT (or worse, garbage), so we
    abort the whole process instead of degrading quietly."""
    from canonical_features import N_FEATURES, CANONICAL
    models = {}
    for pair, path in active_pairs().items():
        if not path.exists():
            raise RuntimeError(
                f"[ml_multi] FATAL: missing model for {pair} ({path.name})")
        m = xgb.XGBClassifier()
        m.load_model(str(path))
        nf = getattr(m, "n_features_in_", None)
        if nf != N_FEATURES:
            raise RuntimeError(
                f"[ml_multi] FATAL: {pair} model {path.name} has n_features_in_="
                f"{nf}, expected canonical {N_FEATURES}. Retrain with "
                f"retrain_all.py before starting the bot.")
        exp = getattr(m, "feature_names_in_", None)
        if exp is not None and list(exp) != list(CANONICAL):
            raise RuntimeError(
                f"[ml_multi] FATAL: {pair} model {path.name} feature names do "
                f"not match canonical CANONICAL. Retrain with retrain_all.py.")
        models[pair] = m
        print(f"[ml_multi] loaded {pair} model: n_features={nf} (canonical OK)")
    if not models:
        raise RuntimeError("[ml_multi] FATAL: no models loaded; refusing to run")
    return models


def fetch_symbol(pair):
    """Map a pair key to the symbol string pipeline.fetch_data accepts.
    BTCUSDT -> 'BTC' (root btc_5m.csv); everything else -> the pair itself."""
    return "BTC" if pair == "BTCUSDT" else pair


# ── feature caching ────────────────────────────────────────────────────────
# Global features (symbol-independent) are loaded ONCE. Per-symbol base frames
# + per-symbol exogenous feeds are cached and invalidated when a new 5m bar
# closes (so each poll reuses the prior build unless the data advanced).
_GLOBAL_CACHE = {"ready": False, "macro": None, "dex": None, "multi": None}
_SYM_CACHE = {}  # pair -> {"bar_ts": ts, "df": built_frame}


def _load_globals():
    if _GLOBAL_CACHE["ready"]:
        return
    # macro (static daily CSVs, symbol-independent). Load over a WIDE index so
    # the 20-period MA signals compute; add_macro_signals forwards to the 5m
    # index. A 2-row stub made every macro feature NaN (MA needs >=20 pts).
    try:
        _GLOBAL_CACHE["macro"] = load_macro_data(
            pd.date_range("2000-01-01", periods=8000, freq="D", tz="UTC"))
    except Exception:
        _GLOBAL_CACHE["macro"] = None
    # DEX breadth (global)
    try:
        from dex_features import add_dex_features
        _GLOBAL_CACHE["dex"] = add_dex_features
    except Exception:
        _GLOBAL_CACHE["dex"] = None
    # multi-asset cross features (global)
    if USE_MULTI_ASSET and Path(MULTI_ASSET_FILE).exists():
        _GLOBAL_CACHE["multi"] = MULTI_ASSET_FILE
    else:
        _GLOBAL_CACHE["multi"] = None
    _GLOBAL_CACHE["ready"] = True


def build_features(symbol):
    """Reproduce model_trainer.train_and_save's feature build for `symbol`.
    Cached: global features loaded once; per-symbol frame reused until a new
    5m bar closes."""
    fsym = fetch_symbol(symbol)
    _load_globals()

    # per-symbol cache key = timestamp of the last bar in the raw 5m series
    raw = _cached_fetch(fsym)
    if raw.empty:
        return None, None
    # Drop any all-NaN / NaT rows (the poller can append a malformed trailing
    # row with no timestamp). A NaT in the index makes int(index[-1].timestamp())
    # raise -> the whole pair's prediction dies every poll. Strip before keying.
    raw = raw[~raw.index.isna()]
    if raw.empty:
        return None, None
    bar_ts = int(raw.index[-1].timestamp())

    if symbol in _SYM_CACHE and _SYM_CACHE[symbol]["bar_ts"] == bar_ts:
        df = _SYM_CACHE[symbol]["df"]
    else:
        df = raw.copy()
        tf = add_resampled_features(df)
        df = df.join(tf, how="left")
        if _GLOBAL_CACHE["macro"] is not None:
            df = add_macro_signals(df, _GLOBAL_CACHE["macro"])
        multi_cols = []
        if _GLOBAL_CACHE["multi"] is not None:
            df, multi_cols = add_multi_asset_features(df, _GLOBAL_CACHE["multi"])
        # micro (Bybit CEX)
        try:
            from micro_features import load_micro
            micro = load_micro(df.index)
            if not micro.empty and micro.notna().any().any():
                df = df.join(micro, how="left")
        except Exception:
            pass
        # deep-history Bybit funding rate
        try:
            from micro_features import load_funding
            if "funding_rate" in df.columns:
                df = df.drop(columns=["funding_rate"])
            fund = load_funding(symbol, df.index)
            if fund is not None and not fund.empty:
                df = df.join(fund, how="left")
        except Exception:
            pass
        # on-chain network metrics
        try:
            from onchain_features import load_onchain
            oc = load_onchain(df.index, symbol)
            if oc is not None and not oc.empty:
                df = df.join(oc, how="left")
        except Exception:
            pass
        # DEX-wide breadth
        dex_cols = []
        if _GLOBAL_CACHE["dex"] is not None:
            try:
                df, dex_cols = _GLOBAL_CACHE["dex"](df)
            except Exception:
                dex_cols = []
        df = derive_features(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.sort_index()
        df = detect_regime(df)
        features = [f for f in ALL_FEATURES if f in df.columns] + multi_cols + dex_cols + ["regime_high_vol", "regime_trending"]
        extra = ["funding_rate"] + [c for c in df.columns if c.startswith("oc_")]
        for cand in extra:
            if cand in df.columns:
                features.append(cand)
        # CANONICAL: present exactly the frozen, shared 113-feature block so the
        # serving input matches what the retrained models expect (identical
        # dims + order). Zero-fills optional columns a pair lacks.
        from canonical_features import resolve
        df, features = resolve(df, features)
        # Bound the cached resolved frame to the rolling window so memory stays
        # flat across cycles (the full 517k-row frame per pair was the OOM cause).
        if len(df) > MAX_BARS:
            df = df.iloc[-MAX_BARS:]
        _SYM_CACHE[symbol] = {"bar_ts": bar_ts, "df": df, "features": features}

    df = _SYM_CACHE[symbol]["df"]
    features = _SYM_CACHE[symbol]["features"]
    fvec = df[features].tail(1).copy()
    last_bar = df.tail(1).iloc[0]
    return fvec, last_bar


def predict_pair(model, fvec):
    """Predict a signal for one pair. Returns (signal, confidence, strength)."""
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and len(fvec.columns) != int(expected):
        return "FLAT", 0.0, 0.0
    exp_names = getattr(model, "feature_names_in_", None)
    if exp_names is not None:
        missing = [c for c in exp_names if c not in fvec.columns]
        if missing:
            return "FLAT", 0.0, 0.0
        fvec = fvec[list(exp_names)]
    X = np.nan_to_num(fvec.values, nan=0.0, posinf=0.0, neginf=0.0)
    probs = model.predict_proba(X)[0]
    cls = int(model.predict(X)[0])
    signal_map = {0: "SHORT", 1: "LONG", 2: "FLAT"}
    signal = signal_map.get(cls, "FLAT")
    conf = float(max(probs))
    # strength: directional edge scaled by confidence (FLAT = 0)
    if signal == "LONG":
        strength = conf
    elif signal == "SHORT":
        strength = -conf
    else:
        strength = 0.0
    return signal, conf, strength


def price_for(symbol):
    """Last close for a pair from its 5m history.

    Returns None (never NaN) if the series is empty or the last close is
    non-finite. A NaN close (the poller can append a malformed trailing bar)
    must NOT be returned: the caller's `if not px or px <= 0` guard does not
    catch NaN (NaN is truthy and NaN<=0 is False), so a NaN would slip through
    into open_position -> NaN shares -> NaN equity -> corrupted state.
    """
    df = _cached_fetch(fetch_symbol(symbol))
    if df is None or df.empty:
        return None
    last = df["close"].iloc[-1]
    if not isinstance(last, (int, float)) or not math.isfinite(float(last)):
        return None
    return float(last)


def run_once(models, state):
    # Hard in-cycle backstop: abort BEFORE a single poll can exceed the cap.
    # The between-cycle guard only checks between cycles, so it can't catch a
    # single-cycle allocation spike -- that's exactly what OOM-killed us.
    mem_guard_abort(4096)
    # Roll the daily risk window ONCE per UTC day so the daily-loss limit resets
    # and the flash-crash window populates. Without this the breakers are dead.
    ref_pair = "BTCUSDT" if "BTCUSDT" in models else next(iter(models), None)
    ref_price = price_for(ref_pair) if ref_pair else None
    _maybe_start_daily_bar(state, ref_price)
    # 1) gather ranked signals
    ranked = []
    for pair, model in models.items():
        try:
            fvec, _ = build_features(pair)
            if fvec is None:
                continue
            signal, conf, strength = predict_pair(model, fvec)
            ranked.append({"pair": pair, "signal": signal, "conf": conf, "strength": strength})
        except Exception as e:
            print(f"[{datetime.now(timezone.utc).isoformat()}] {pair} predict error: {e}")
    ranked.sort(key=lambda r: -abs(r["strength"]))
    print(f"[{datetime.now(timezone.utc).isoformat()}] ranked: " +
          ", ".join(f"{r['pair']}={r['signal']}({r['conf']:.2f})" for r in ranked))

    # 2) decide desired book = top-N LONG signals above threshold.
    # LONG-ONLY: the live strategy is long-only, so we only OPEN on a LONG
    # prediction. A SHORT model output means "don't be long" -> it is excluded
    # here (and any existing long in that pair gets closed by step 3). Opening
    # a SHORT prediction as a LONG (the prior behavior) is directionally wrong.
    desired = [r for r in ranked
               if r["signal"] == "LONG" and r["conf"] >= CONFIDENCE_THRESHOLD][:MAX_POSITIONS]
    desired_pairs = {r["pair"]: r for r in desired}

    # 3) close positions no longer desired / no longer signaled
    for sym in list(state.positions.keys()):
        if sym not in desired_pairs:
            px = price_for(sym)
            if px and math.isfinite(px):
                state.close_position(sym, px)
                print(f"CLOSE {sym} @ {px:.6f} (not in top-{MAX_POSITIONS})")

    # 4) open top-N desired (respect max positions + circuit breakers)
    for r in desired:
        sym = r["pair"]
        if sym in state.positions:
            continue
        if len(state.positions) >= MAX_POSITIONS:
            break
        px = price_for(sym)
        if not (px and math.isfinite(px)) or px <= 0:
            continue
        ok, reason = state.check_circuit_breakers()
        if not ok:
            print(f"CIRCUIT BREAKER: {reason}, staying flat")
            break
        size = SIZE_USD
        pos = state.open_position(sym, px, size)
        if pos:
            print(f"OPEN {sym} @ {px:.6f}, size=${size:.2f} ({r['signal']} conf={r['conf']:.2f})")

    # 5) mark to market + save
    try:
        mtm = {r["pair"]: price_for(r["pair"]) for r in ranked}
        eq = state.mark_to_market({k: v for k, v in mtm.items() if v})
        dd = (state.peak_equity - eq) / state.peak_equity if state.peak_equity > 0 else 0
        print(f"MTM equity: ${eq:.2f}  Peak: ${state.peak_equity:.2f}  DD: {dd:.2%}  Positions: {len(state.positions)}")
        if dd > 0.20:
            print(f"Flattening: DD {dd:.2%} > 20%")
            state.flatten_all({k: v for k, v in mtm.items() if v})
            state.halt(f"live_drawdown_flatten_{pd.Timestamp.now().isoformat()}")
    except Exception as e:
        print(f"MTM error: {e}")
    state.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    # Single-instance guard: the systemd service and a manual `--once` run must
    # not both write execution_state_ml_multi.json / trade_journal_ml_multi.json
    # concurrently (that corrupts state). Mirror the collector's flock pattern.
    _lock_path = REPO / "run" / "ml_multi.lock"
    _lock_path.parent.mkdir(parents=True, exist_ok=True)
    _lockf = open(_lock_path, "w")
    try:
        fcntl.flock(_lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[ml_multi] another instance holds {_lock_path}; exiting")
        sys.exit(0)
    models = load_models()
    if not models:
        print("[ml_multi] no models loaded, exiting")
        return
    # register graceful-shutdown handlers (systemd Stop sends SIGTERM)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    state = MultiPositionState(initial_capital=CONFIG.initial_capital,
                                state_file=STATE_FILE, journal_file=JOURNAL_FILE)
    discovered = discover_models()
    active = active_pairs()
    print(f"[ml_multi] discovered models: {sorted(discovered.keys())}")
    print(f"[ml_multi] screener active: {sorted(active.keys())}  (screener={SCREENER_FILE.name} exists={SCREENER_FILE.exists()})")
    print(f"[ml_multi] starting. pairs={list(models.keys())} max_positions={MAX_POSITIONS}")
    if args.once:
        run_once(models, state)
        return
    while True:
        if _SHUTDOWN:
            print(f"[{datetime.now(timezone.utc).isoformat()}] shutdown requested; saving state + exiting")
            break
        # Memory throttle: if RSS is climbing, skip the heavy build this cycle
        # (log + reclaim + extra cooldown) instead of risking an OOM kill.
        rss = rss_mb()
        if rss > MEM_GUARD_MB:
            print(f"[{datetime.now(timezone.utc).isoformat()}] MEM THROTTLE: "
                  f"RSS={rss:.0f}MB > cap={MEM_GUARD_MB}MB — skipping build, gc + cooldown")
            gc.collect()
            time.sleep(POLL_SEC + 60)
            continue
        try:
            run_once(models, state)
        except Exception as e:
            print(f"[{datetime.now(timezone.utc).isoformat()}] error: {e}")
        gc.collect()
        # interruptible sleep so a stop is honored promptly
        slept = 0.0
        while slept < POLL_SEC and not _SHUTDOWN:
            time.sleep(min(5.0, POLL_SEC - slept))
            slept += 5.0
    state.save()
    print(f"[{datetime.now(timezone.utc).isoformat()}] exited cleanly")


if __name__ == "__main__":
    main()
