#!/usr/bin/env python3
"""
Inference server for BTC/USDT ML trading bot.
Endpoints:
  /health
  /signal
  /dashboard
  /refresh
  /logout
"""
import json
import traceback
import threading
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import xgboost as xgb

from micro_features import load_micro
from data_feed import fetch_latest, HISTORY_CSV, LOCAL_CSV
from order_manager import STATE_FILE, TRADE_JOURNAL
from pipeline import (
    add_resampled_features,
    load_macro_data,
    add_macro_signals,
    derive_features,
    detect_regime,
    USE_MULTI_ASSET,
    MULTI_ASSET_FILE,
    ALL_FEATURES,
)
from multi_asset_features import add_multi_asset_features

app = FastAPI(title="BTC ML Bot Inference")

latest_model = None
model_loaded_at = None
MODEL_GLOB = str(Path(__file__).parent / 'models' / 'latest_xgb.json')
# Guards the model globals: load_latest_model() writes them; get_signal() reads
# and predicts against them. FastAPI runs handlers in a threadpool, so without
# this a concurrent /refresh could swap latest_model mid-predict (F1).
_model_lock = threading.Lock()


class SignalResponse(BaseModel):
    timestamp: str
    signal: str
    confidence: float
    probabilities: list[float]
    regime: str | None = None
    model_version: str | None = None
    error: str | None = None


def _latest_model_path():
    paths = sorted(Path(__file__).parent.glob('models/latest_xgb.json')) + sorted(Path(__file__).parent.glob('model_*.pkl'))
    if paths:
        return paths[-1]
    return None


def load_latest_model():
    global latest_model, model_loaded_at
    with _model_lock:
        p = _latest_model_path()
        if p is None:
            print('[model_server] no model path found')
            return False
        print(f'[model_server] trying model path: {p}')
        try:
            if str(p).endswith('.json'):
                latest_model = xgb.XGBClassifier()
                latest_model.load_model(str(p))
            else:
                latest_model = joblib.load(str(p))
            model_loaded_at = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
            print(f'[model_server] model loaded successfully, features={getattr(latest_model, "n_features_in_", None)}')
            return True
        except Exception as e:
            latest_model = None
            model_loaded_at = None
            print(f'[model_server] load failed: {e}')
            return False


def _safe_read_csv(path):
    """Copy a CSV to a private temp file, then read it.

    The live history file is rewritten by data_feed (a separate process) using
    an atomic tmp+rename, but reading the live path directly could still catch a
    frame mid-rename on some filesystems. Copying a closed file first gives a
    byte-consistent snapshot (F2). Raises FileNotFoundError if the file is
    missing (mirrors the original pd.read_csv behaviour).
    """
    if not path.exists():
        raise FileNotFoundError(path)
    tmp = tempfile.mktemp(suffix='.csv')
    shutil.copyfile(path, tmp)
    try:
        return pd.read_csv(tmp, parse_dates=['ts'])
    finally:
        try:
            Path(tmp).unlink()
        except OSError:
            pass


def compute_features_from_history():
    try:
        # Load BTC local history and any live history appended by data_feed.
        # Read a private snapshot (not the live path) to avoid a torn frame if
        # data_feed rewrites the file concurrently (F2).
        local = _safe_read_csv(LOCAL_CSV)
        hist_raw = _safe_read_csv(HISTORY_CSV) if HISTORY_CSV.exists() else None
        hist = hist_raw if hist_raw is not None else pd.DataFrame(columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        combined = pd.concat([local, hist], ignore_index=True).drop_duplicates('ts').sort_values('ts')
        if combined.empty or combined.shape[0] < 60:
            raise RuntimeError('not enough history for feature computation')
        # Keep last ~60 bars so rolling features have enough lookback
        buf = 80
        if combined.shape[0] > buf:
            combined = combined.iloc[-buf:].copy()
        combined.set_index('ts', inplace=True)
        df = combined.rename(columns={c: c.lower() for c in combined.columns})
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'ts'

        idx_name = df.index.name or 'timestamp'
        resampled = add_resampled_features(df)
        df = df.join(resampled, how='left')

        macro = load_macro_data(df.index)
        df = add_macro_signals(df, macro)

        # Drop macro-derived columns that are entirely NaN; they come from missing daily files
        macro_cols = [c for c in df.columns if any(x in c for x in ['spy_', 'gld_', 'tlt_', 'uup_', 'vix_', 'btc_spy_ratio', 'btc_gld_ratio', 'btc_uup_ratio'])]
        keep_macro = [c for c in macro_cols if df[c].notna().any()]
        drop_macro = [c for c in macro_cols if c not in keep_macro]
        if drop_macro:
            print(f'[model_server] dropping all-NaN macro columns: {drop_macro}')
        df.drop(columns=drop_macro, inplace=True, errors='ignore')

        if USE_MULTI_ASSET and Path(MULTI_ASSET_FILE).exists():
            df = add_multi_asset_features(df, MULTI_ASSET_FILE)

        df = derive_features(df)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.sort_index()
        df = detect_regime(df)

        feature_cols = [f for f in ALL_FEATURES if f in df.columns]
        for col in ['regime_high_vol', 'regime_trending']:
            if col not in df.columns:
                df[col] = 0
                feature_cols.append(col)

        # Exclude features that are entirely NaN from the required subset
        required = [c for c in feature_cols if df[c].notna().any()]
        missing_required = [c for c in feature_cols if c not in required]
        if missing_required:
            print(f'[model_server] excluding all-NaN features: {missing_required}')
        feature_cols = required

        df = df.dropna(subset=feature_cols)
        if df.empty:
            raise RuntimeError('no rows after feature dropna')

        fvec = df[feature_cols].tail(1).copy()
        last_bar = df.tail(1).iloc[0]
        return fvec, last_bar
    except Exception as e:
        raise RuntimeError(f'Feature computation failed: {e}')


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': latest_model is not None,
        'model_loaded_at': model_loaded_at,
        'model_n_features': getattr(latest_model, 'n_features_in_', None) if latest_model else None,
    }


@app.get('/dashboard')
def dashboard():
    state_path = STATE_FILE
    journal_path = TRADE_JOURNAL
    state = {}
    trades = []
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            state = {}
    if journal_path.exists():
        try:
            raw = journal_path.read_text().strip()
            if raw:
                trades = json.loads(raw)
                if isinstance(trades, dict):
                    trades = [trades]
        except Exception:
            trades = []
    equity = float(state.get('equity', 1000.0))
    peak_equity = float(state.get('peak_equity', 1000.0))
    daily_pnl = float(state.get('daily_pnl', 0.0))
    halted = bool(state.get('halted', False))
    halt_reason = state.get('halt_reason')
    drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
    recent = trades[-20:] if isinstance(trades, list) else []
    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'equity': round(equity, 2),
        'peak_equity': round(peak_equity, 2),
        'daily_pnl': round(daily_pnl, 2),
        'drawdown_pct': round(drawdown * 100, 2),
        'halted': halted,
        'halt_reason': halt_reason,
        'trade_count': len(trades) if isinstance(trades, list) else 0,
        'recent_trades': recent,
    }


@app.get('/logout')
def logout():
    return {'status': 'logged_out'}


@app.get('/signal', response_model=SignalResponse)
def get_signal() -> SignalResponse:
    try:
        print('[model_server] /signal hit')
        # Ensure loaded (load_latest_model() takes _model_lock internally; do NOT
        # hold the lock here to avoid re-entrant deadlock on threading.Lock).
        if latest_model is None:
            load_latest_model()
        # Snapshot the model reference under lock so a concurrent /refresh cannot
        # swap latest_model mid-predict (F1).
        with _model_lock:
            model = latest_model
        if model is None:
            return SignalResponse(timestamp=datetime.now(timezone.utc).isoformat(), signal='FLAT', confidence=0.0, probabilities=[0.0, 0.0, 1.0], error='No model loaded')

        fvec, last_bar = compute_features_from_history()
        print(f'[model_server] feature vector shape={fvec.shape}, cols={list(fvec.columns)}')
        print(f'[model_server] last_bar head={dict(last_bar.head(5)) if hasattr(last_bar, "head") else str(last_bar)[:200]}')

        expected = getattr(model, 'n_features_in_', None)
        # F3: never silently truncate (keeps wrong first-N cols) or zero-pad
        # (injects fake features). Align by name when the model exposes them,
        # otherwise fail loudly on any count mismatch.
        if expected is not None and len(fvec.columns) != int(expected):
            return SignalResponse(timestamp=datetime.now(timezone.utc).isoformat(), signal='FLAT', confidence=0.0, probabilities=[0.0, 0.0, 1.0], error=f'feature count mismatch: got {len(fvec.columns)}, expected {expected}')
        exp_names = getattr(model, 'feature_names_in_', None)
        if exp_names is not None:
            missing = [c for c in exp_names if c not in fvec.columns]
            if missing:
                return SignalResponse(timestamp=datetime.now(timezone.utc).isoformat(), signal='FLAT', confidence=0.0, probabilities=[0.0, 0.0, 1.0], error=f'missing model features: {missing}')
            fvec = fvec[list(exp_names)]  # explicit named order

        print('[model_server] predicting...')
        X = np.nan_to_num(fvec.values, nan=0.0, posinf=0.0, neginf=0.0)
        # Predict under lock so a concurrent reload cannot swap the model between
        # predict_proba and predict (F1).
        with _model_lock:
            probs = model.predict_proba(X)[0]
            cls = int(model.predict(X)[0])
        print(f'[model_server] prediction cls={cls} probs={probs.tolist()}')
        signal_map = {0: 'SHORT', 1: 'LONG', 2: 'FLAT'}
        final_signal = signal_map.get(cls, 'FLAT')

        regime_str = None
        if isinstance(last_bar, pd.Series) and 'regime' in last_bar:
            regime_str = last_bar['regime']

        return SignalResponse(
            timestamp=last_bar.name.isoformat() if hasattr(last_bar, 'name') else datetime.now(timezone.utc).isoformat(),
            signal=final_signal,
            confidence=float(max(probs)),
            probabilities=probs.tolist(),
            regime=regime_str,
            model_version=model_loaded_at,
            error=None,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return SignalResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            signal='FLAT', confidence=0.0,
            probabilities=[0.0, 0.0, 1.0],
            error=str(e),
        )


@app.get('/refresh')
def refresh_model():
    ok = load_latest_model()
    return {'reloaded': ok, 'loaded_at': model_loaded_at}


if __name__ == '__main__':
    load_latest_model()
    print(f'Starting inference server. Model loaded: {latest_model is not None}')
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8080)
