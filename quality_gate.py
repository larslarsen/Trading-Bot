#!/usr/bin/env python3
"""
Quality gate for the CEX 5m training/universe selection.

Literature anchor: Bartolucci (2020, R. Soc. Open Sci., PMC7481708) "A model
of the optimal selection of crypto assets" -- crypto assets are characterised
by TWO selection features:
  1. SECURITY   -- network/chain security & asset maturity (longevity, not a
                  dead/exit-scam token).
  2. LIQUIDITY  -- tradability; thin-volume assets are excluded because they
                  carry uninformative microstructure noise that degrades a
                  shared (multi-asset / pooled) model via negative transfer
                  (Liu, Liang & Gitter, AAAI 2019 -- negative transfer is
                  prevalent in multi-task models).

We operationalise those two features as concrete, falsifiable gates on the
5m history we already collect. A pair PASSES iff it meets ALL thresholds.
Tune the thresholds; the comparison script tells you if a threshold helps
or hurts OOS (don't hand-pick -- let the panel decide).

Liquidity is measured as QUOTE volume (volume * close) so it is comparable
across coins (base-unit volume is NOT -- 1 BTC != 1 DOGE).

Usage:
  from quality_gate import gated_universe
  pairs = gated_universe()                      # all data/*USDT_5m_max.csv
  pairs = gated_universe(min_history_days=730)  # stricter maturity
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent
DATA = REPO / "data"
DEX_DIR = DATA / "dex"
MICRO_DIR = REPO / "dex_data" / "micro"  # DexScreener per-token liquidity_usd
# DEX-native relaxed thresholds (DEX tokens are young + thin by nature):
DEFAULT_DEX_MIN_BARS = 100          # >=100 1m bars collected (~100 min of signal)
DEFAULT_DEX_MIN_LIQUIDITY_USD = 100_000.0  # Bartolucci liquidity, DEX-native ($ pool)
DEFAULT_DEX_FALLBACK_TOP = 20        # if gate yields too few, use top-N by liquidity
DEFAULT_MIN_CANDIDATES = 10          # below this -> fall back to full top-N set

# ── Defaults (literature-anchored, falsifiable) ──────────────────────────
# SECURITY / maturity: >= 2 years of 5m history (Bartolucci longevity proxy).
DEFAULT_MIN_HISTORY_DAYS = 730
# LIQUIDITY (quote volume = volume*close, USD-comparable across coins):
# mean 5m quote volume over the history must clear a floor.
DEFAULT_MIN_AVG_QUOTE_VOLUME = 100_000.0
# LIQUIDITY (still-alive): recent 30d quote volume must be > 0 (not dead).
DEFAULT_MIN_RECENT_QUOTE_VOLUME = 10_000.0
# Tradability: require a minimum bar count so features aren't fit on noise.
DEFAULT_MIN_BARS = 50_000
# Delisted / migrated pairs we know are dead (e.g. MATICUSDT -> POLUSDT).
DELISTED = {"MATICUSDT"}
# Stablecoins -- flat price, not trend-tradeable assets; exclude from universe.
STABLES = {"FDUSDUSDT", "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "DAIUSDT",
           "USDPUSDT", "PYUSDUSDT", "GUSDUSDT", "USTCUSDT", "FRAXUSDT"}


def _pair_from_file(p: Path) -> str:
    return p.stem.replace("_5m_max", "")


def gated_universe(
    data_dir: Path = DATA,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    min_avg_quote_volume: float = DEFAULT_MIN_AVG_QUOTE_VOLUME,
    min_recent_quote_volume: float = DEFAULT_MIN_RECENT_QUOTE_VOLUME,
    min_bars: int = DEFAULT_MIN_BARS,
    exclude: set[str] = DELISTED,
    verbose: bool = False,
) -> list[str]:
    """Return the list of pairs that pass the quality gate.

    Liquidity = QUOTE volume (volume * close), comparable across coins."""
    # Only Binance canonical files (XUSDT_5m_max.csv); skip venue-suffixed
    # duplicates like BTC-USDT_5m_max.csv (OKX/Bybit) to avoid double-counting.
    files = sorted(p for p in data_dir.glob("*USDT_5m_max.csv")
                   if "-" not in p.stem)
    passed, reasons = [], {}
    for f in files:
        pair = _pair_from_file(f)
        if pair in exclude or pair in STABLES:
            reasons[pair] = "delisted" if pair in exclude else "stable"
            continue
        try:
            # avoid parse_dates (slow on 1M-row files); ts stays ISO string,
            # which sorts lexicographically so first/last give the span.
            d = pd.read_csv(f, usecols=["ts", "close", "volume"],
                            dtype={"ts": "string"})
        except Exception as e:
            reasons[pair] = f"read_err:{e!r}"[:40]
            continue
        n = len(d)
        if n < min_bars:
            reasons[pair] = f"bars:{n}<{min_bars}"
            continue
        # SECURITY: history span (ISO strings sort lexicographically)
        ts = pd.to_datetime(d["ts"], errors="coerce")
        ts_valid = ts.dropna()
        if ts_valid.empty:
            reasons[pair] = "no_valid_ts"
            continue
        # SECURITY: history span (ISO strings sort lexicographically; use the
        # last VALID ts so a malformed trailing row with no timestamp can't
        # NaT the whole pair -- same strip the serving bot applies on load).
        span_days = (ts_valid.iloc[-1] - ts_valid.iloc[0]).days
        if span_days < min_history_days:
            reasons[pair] = f"history:{span_days}d<{min_history_days}"
            continue
        close = pd.to_numeric(d["close"], errors="coerce").fillna(0.0)
        vol = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0)
        quote = close * vol
        avg_qv = float(quote.mean())
        if avg_qv < min_avg_quote_volume:
            reasons[pair] = f"avgqv:{avg_qv:.0f}<{min_avg_quote_volume:.0f}"
            continue
        # LIQUIDITY-alive: last 30d must still trade
        cutoff = ts_valid.iloc[-1] - pd.Timedelta(days=30)
        cutoff_s = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        recent_qv = float(quote[d["ts"] >= cutoff_s].mean())
        if recent_qv < min_recent_quote_volume:
            reasons[pair] = f"recentqv:{recent_qv:.0f}<{min_recent_quote_volume:.0f}"
            continue
        passed.append(pair)
    if verbose:
        print(f"[gate] {len(passed)}/{len(files)} passed; rejected={len(reasons)}")
        for k, v in sorted(reasons.items())[:12]:
            print(f"  skip {k}: {v}")
    return passed


def dex_gated_universe(
    dex_dir: Path = DEX_DIR,
    min_bars: int = DEFAULT_DEX_MIN_BARS,
    min_liquidity_usd: float = DEFAULT_DEX_MIN_LIQUIDITY_USD,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
    fallback_top: int = DEFAULT_DEX_FALLBACK_TOP,
    verbose: bool = False,
) -> list[str]:
    """Literature selection for the DEX universe (Bartolucci two features),
    with DEX-native relaxed thresholds + a fallback to the full top-N.

    DEX tokens are young and thin by nature, so the CEX absolute-age gate
    does not apply. Instead:
      SECURITY  -> >= min_bars of 1m data collected (not pure noise).
      LIQUIDITY -> live pool liquidity_usd (from DexScreener micro poller)
                   clears a floor (Bartolucci's liquidity feature, DEX-native).
    If the gated set is smaller than min_candidates, FALL BACK to the top-N
    tokens by live liquidity (the full set we have 1m data for) -- per the
    standing directive: when there isn't enough data to screen, use all 20.
    """
    # live liquidity ranking (reuse the sampler's rank_tokens if importable)
    liq_rank = {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dex_ohlcv_sampler", REPO / "dex_ohlcv_sampler.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ranked = mod.rank_tokens(fallback_top * 4)  # wider net for the gate
        liq_rank = {t: i for i, t in enumerate(ranked)}
    except Exception:
        liq_rank = {}
    # fallback order: by live liquidity rank, else by filename
    def _order(tok):
        return liq_rank.get(tok, 10_000)

    files = sorted(dex_dir.glob("*_1m_max.csv"))
    passed, reasons = [], {}
    for f in files:
        tok = f.stem.replace("_1m_max", "").replace("_1m", "")
        try:
            d = pd.read_csv(f, usecols=["ts", "volume"], dtype={"ts": "string"})
        except Exception as e:
            reasons[tok] = f"read_err:{e!r}"[:40]
            continue
        n = len(d)
        if n < min_bars:
            reasons[tok] = f"bars:{n}<{min_bars}"
            continue
        liq = liq_rank.get(tok, 0.0)
        # liquidity_usd may be 0 if micro poller hasn't scraped this token;
        # only enforce the floor when we actually have a liquidity reading.
        if liq > 0 and liq < min_liquidity_usd:
            reasons[tok] = f"liq:{liq:.0f}<{min_liquidity_usd:.0f}"
            continue
        passed.append(tok)
    passed.sort(key=_order)

    # FALLBACK: not enough screened candidates -> use full top-N by liquidity
    if len(passed) < min_candidates:
        if verbose:
            print(f"[gate:dex] only {len(passed)} passed (<{min_candidates}); "
                  f"falling back to top-{fallback_top} by liquidity")
        fb = sorted(liq_rank.keys(), key=lambda t: liq_rank[t])[:fallback_top]
        # include any gated passes first, then fill from fallback
        seen = set(passed)
        for t in fb:
            if t not in seen:
                passed.append(t)
                seen.add(t)
        reasons = {k: f"{v} (pre-fallback)" for k, v in reasons.items()}
    if verbose:
        print(f"[gate:dex] {len(passed)} tokens selected (incl. fallback)")
        for k, v in sorted(reasons.items())[:12]:
            print(f"  skip {k}: {v}")
    return passed


def save_gate(pairs, path: Path = REPO / "universe_gated.json"):
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(pairs),
        "venue": "cex",
        "thresholds": {
            "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
            "min_avg_quote_volume": DEFAULT_MIN_AVG_QUOTE_VOLUME,
            "min_recent_quote_volume": DEFAULT_MIN_RECENT_QUOTE_VOLUME,
            "min_bars": DEFAULT_MIN_BARS,
        },
        "pairs": pairs,
    }
    path.write_text(json.dumps(meta, indent=2))
    return path


def save_dex_gate(pairs, path: Path = REPO / "universe_dex_gated.json"):
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_tokens": len(pairs),
        "venue": "dex",
        "thresholds": {
            "min_bars": DEFAULT_DEX_MIN_BARS,
            "min_liquidity_usd": DEFAULT_DEX_MIN_LIQUIDITY_USD,
            "min_candidates": DEFAULT_MIN_CANDIDATES,
            "fallback_top": DEFAULT_DEX_FALLBACK_TOP,
        },
        "pairs": pairs,
    }
    path.write_text(json.dumps(meta, indent=2))
    return path


if __name__ == "__main__":
    ps = gated_universe(verbose=True)
    p = save_gate(ps)
    print(f"Gated universe: {len(ps)} pairs -> {p}")
