#!/usr/bin/env python3
"""
Equities/ETF macro data collector.

Downloads key risk-on/risk-off and rates proxies via Yahoo Finance.
Stores daily CSV files for later merging into the main feature pipeline.

Used as exogenous signal source for BTC crypto trading.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

OUT_DIR = Path(__file__).parent / 'data'
OUT_DIR.mkdir(exist_ok=True)

TICKERS = {
    # risk on / off
    'SPY': 'SPY',       # S&P 500 ETF
    'QQQ': 'QQQ',       # Nasdaq 100 ETF
    'IWM': 'IWM',       # Russell 2000 ETF
    # rates / credit
    'TLT': 'TLT',       # 20+ Year Treasury ETF
    'HYG': 'HYG',       # High Yield Corporate Bond ETF
    'LQD': 'LQD',       # Investment Grade Corporate Bond ETF
    # vol / momentum
    'VIX': '^VIX',      # Volatility Index
    # dollar / FX
    'UUP': 'UUP',       # US Dollar Index ETF
    'DXY': 'DX-Y.NYB',  # US Dollar Index futures
    # gold / inflation hedge
    'GLD': 'GLD',       # Gold ETF
}


def fetch(ticker: str, period: str = 'max') -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval='1d', progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = 'date'
        lower = {c.lower(): c for c in df.columns}
        def pick(name):
            real = lower.get(name.lower())
            return df[real] if real in df.columns else None
        close = pick('close')
        volume = pick('volume')
        if close is None:
            return pd.DataFrame()
        out = pd.DataFrame({f'{ticker}_close': close})
        if volume is not None:
            out[f'{ticker}_volume'] = volume
        return out
    except Exception as e:
        print(f'  Equities error {ticker}: {e}')
        return pd.DataFrame()


def collect():
    day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    print(f'Equities collector run at {day}')
    frames = []
    for name, ticker in TICKERS.items():
        df = fetch(ticker)
        if not df.empty:
            frames.append(df)
            print(f'  {name}: {len(df)} days, latest={df.index[-1].date()}')
        time.sleep(0.3)
    if frames:
        merged = pd.concat(frames, axis=1)
        merged = merged.loc[~merged.index.duplicated(keep='last')]
        out = OUT_DIR / f'equities_daily_{day}.csv'
        merged.to_csv(out, index=True)
        print(f'Saved {len(merged)} rows to {out.name}')
    else:
        print('No equities data collected')


if __name__ == '__main__':
    collect()
