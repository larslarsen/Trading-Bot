#!/usr/bin/env python3
"""
DEX ML paper trader -- one pooled XGBoost model, ranked like a screener.

Loads models/dex_xgb.json (trained by train_dex_ml.py on all DEX tokens),
scores every token in dex_data/, ranks by signal strength, and holds the top-N
directional signals via a multi-position book. This is the ML counterpart to
the rule-based dex_screen_1d.py -- kept as a SEPARATE bot (separate model,
state, journal, log) so ML vs rule can be scored head-to-head.

Registry: dex_ml_xgb_1d  |  state: execution_state_dex_ml.json
Usage:
  python dex_ml_xgb_1d.py           # one screen + execution pass (cron-style)
  python dex_ml_xgb_1d.py --loop 86400
"""
import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline import derive_features
from order_manager_multi import MultiPositionState

REPO = Path(__file__).parent
DEX_DIR = REPO / "dex_data"
MODEL = REPO / "models" / "dex_xgb.json"
META = REPO / "models" / "dex_ml_meta.json"
STATE_FILE = REPO / "execution_state_dex_ml.json"
JOURNAL_FILE = REPO / "trade_journal_dex_ml.json"

MAX_POSITIONS = 5
CONFIDENCE_THRESHOLD = 0.45   # DEX model is weak; lower bar than CEX
SIZE_USD = 2000.0
MIN_BARS = 60


def load_model():
    if not MODEL.exists():
        raise SystemExit(f"no DEX model at {MODEL}; run train_dex_ml.py first")
    m = xgb.XGBClassifier()
    m.load_model(str(MODEL))
    features = json.loads(META.read_text())["features"] if META.exists() else None
    return m, features


def score_token(path, model, features):
    """Return (token, signal, conf, strength, last_close) or None."""
    df = pd.read_csv(path, parse_dates=["ts"]).dropna(subset=["close", "high", "low", "volume"])
    if len(df) < MIN_BARS:
        return None
    df = df.sort_values("ts").set_index("ts")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")]
    df = derive_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats = features or [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        return None
    row = df[feats].tail(1)
    if row.isna().all(axis=1).iloc[0]:
        return None
    X = np.nan_to_num(row.values, nan=0.0, posinf=0.0, neginf=0.0)
    probs = model.predict_proba(X)[0]
    cls = int(model.predict(X)[0])
    signal = {0: "SHORT", 1: "LONG", 2: "FLAT"}.get(cls, "FLAT")
    conf = float(max(probs))
    strength = conf if signal == "LONG" else (-conf if signal == "SHORT" else 0.0)
    token = path.stem.replace("_1d_max", "")
    return token, signal, conf, strength, float(df["close"].iloc[-1])


def run_once(model, features, state):
    scored = []
    prices = {}
    for p in sorted(DEX_DIR.glob("*_1d_max.csv")):
        r = score_token(p, model, features)
        if r:
            token, sig, conf, strength, px = r
            scored.append({"token": token, "signal": sig, "conf": conf, "strength": strength})
            prices[token] = px
    scored.sort(key=lambda r: -abs(r["strength"]))
    top = [r for r in scored if r["signal"] != "FLAT" and r["conf"] >= CONFIDENCE_THRESHOLD][:MAX_POSITIONS]
    print(f"[{datetime.now(timezone.utc).isoformat()}] scored {len(scored)} tokens; "
          f"top {len(top)}: " + ", ".join(f"{r['token']}={r['signal']}({r['conf']:.2f})" for r in top))
    desired = {r["token"] for r in top}

    # close positions no longer in the top screen
    for sym in list(state.positions.keys()):
        if sym not in desired and sym in prices:
            state.close_position(sym, prices[sym])
            print(f"CLOSE {sym} @ {prices[sym]:.8f}")

    # open top-N
    for r in top:
        sym = r["token"]
        if sym in state.positions or len(state.positions) >= MAX_POSITIONS:
            continue
        ok, reason = state.check_circuit_breakers()
        if not ok:
            print(f"CIRCUIT BREAKER: {reason}")
            break
        if prices.get(sym, 0) > 0 and state.open_position(sym, prices[sym], SIZE_USD):
            print(f"OPEN {sym} @ {prices[sym]:.8f} ({r['signal']} conf={r['conf']:.2f})")

    eq = state.mark_to_market(prices)
    dd = (state.peak_equity - eq) / state.peak_equity if state.peak_equity > 0 else 0
    print(f"MTM equity: ${eq:.2f}  Peak: ${state.peak_equity:.2f}  DD: {dd:.2%}  Positions: {len(state.positions)}")
    if dd > 0.20:
        print(f"Flattening: DD {dd:.2%} > 20%")
        state.flatten_all(prices)
        state.halt(f"live_drawdown_flatten_{pd.Timestamp.now().isoformat()}")
    state.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes; 0 = one pass")
    args = ap.parse_args()
    model, features = load_model()
    state = MultiPositionState(initial_capital=10000.0, state_file=STATE_FILE, journal_file=JOURNAL_FILE)
    print(f"=== dex_ml_xgb_1d run: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC} ===", flush=True)
    print(f"START dex_ml_xgb_1d.py (features={len(features) if features else '?'})", flush=True)
    if args.loop <= 0:
        run_once(model, features, state)
        return
    while True:
        try:
            run_once(model, features, state)
        except Exception as e:
            print(f"[{datetime.now(timezone.utc).isoformat()}] error: {e}", flush=True)
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
