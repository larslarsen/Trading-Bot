#!/usr/bin/env python3
"""Screen the DEX universe with our proven multi-asset liquidity/idiosyncratic-vol
screener (screen_liquidity_idiosyncratic.run_screen), adapted for DEX.

Our CEX screener expects a `tier` column (large/mid/tail by size) and reads OHLCV
from data/*. DEX tokens have no CEX tier and their bars live in dex_data/. This
wrapper:
  - assigns a size tier by vol24h terciles (large/mid/tail) -- mirrors how CEX
    tiers work (by ADV/size), so the per-tier idio-vol ranking runs correctly;
  - points run_screen at dex_data/ (ROOT) + a temp universe-with-tier CSV;
  - writes backtest_output/screen_dex_idio_<date>.csv (the DEX screened universe).

NOTE: DEX `volume` column is a trade-count proxy (GeckoTerminal free gives no
notional USD volume), so the ADV/liquidity rank is by trade-frequency. The
idio_vol (price volatility) signal is correct. For real USD-liquidity ranking,
re-pull volumeUSD into the bars (later improvement).

Usage:
    python screen_dex_idio.py
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd
import screen_liquidity_idiosyncratic as s

DEX = Path("dex_data")
UNI = Path("dex_universe.csv")
OUT = Path("backtest_output")


def main():
    if not UNI.exists():
        print(f"ERROR: {UNI} missing. Run build_dex_universe.py first.")
        return
    u = pd.read_csv(UNI).dropna(subset=["symbol"])
    u["vol24h"] = pd.to_numeric(u["vol24h"], errors="coerce").fillna(0)
    q1, q2 = u["vol24h"].quantile([0.333, 0.667])
    u["tier"] = u["vol24h"].apply(
        lambda v: "large" if v >= q2 else ("mid" if v >= q1 else "tail"))
    tmp = Path(tempfile.mkdtemp()) / "dex_uni_tier.csv"
    u[["symbol", "tier", "vol24h"]].to_csv(tmp, index=False)

    s.ROOT = DEX  # point the screener at DEX bars
    out = OUT / f"screen_dex_idio_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
    s.run_screen(universe_csv=str(tmp), out_csv=str(out))
    print(f"\nDEX idio-vol screen -> {out}")


if __name__ == "__main__":
    main()
