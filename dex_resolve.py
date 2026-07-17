#!/usr/bin/env python3
"""Shared DEX (chain, pool) resolver for free GeckoTerminal OHLCV collection.

Why this exists: the old `backfill_dex_history_gt.resolve_top_pool` called
DexScreener `/search` with no strict matching and mapped by chainId, which
returned "no pairs" for 99% of our universe (Robinhood-tokenized tickers,
garbage names, raw contract addresses). The WORKING resolver was already in
`dex_micro_poller.resolve_address` (it produced all 607 dex_data files) but it
returns the `robinhood` chain for most of our universe -- and GeckoTerminal has
no "robinhood" network slug, so those tokens cannot be pulled.

This module resolves a ticker to the highest-liquidity REAL-CHAIN pool
(ethereum/bsc/solana/base/arbitrum/polygon/avax/optimism) and returns the
GeckoTerminal (network, pool_address) needed for the still-free pool-level
OHLCV endpoint: /networks/{net}/pools/{pool}/ohlcv/{tf}.

Used by both backfill_dex_history_gt.py (deep history) and dex_ohlcv_sampler.py
(live 1m/5m). Single source of truth -> DRY.
"""
import time
from pathlib import Path

import requests

REPO = Path(__file__).parent
UA = {"User-Agent": "Mozilla/5.0 (research)"}
DEX_API = "https://api.dexscreener.com/latest/dex"
GT_API = "https://api.geckoterminal.com/api/v2"

# DexScreener chainId -> GeckoTerminal network slug
NETMAP = {
    "ethereum": "eth", "bsc": "bsc", "solana": "solana", "base": "base",
    "arbitrum": "arbitrum", "polygon": "polygon", "avax": "avax",
    "optimism": "optimism", "fantom": "fantom", "cronos": "cronos",
}
# Chains we explicitly cannot pull from GeckoTerminal (tokenized stocks, etc.)
EXCLUDE_CHAINS = ("robinhood",)


def fetch_json(url, params=None, timeout=20, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r.json(), None
            if r.status_code == 429:
                time.sleep(min(2 ** i * 3, 30))
                continue
            return None, f"HTTP {r.status_code}"
        except Exception as e:
            time.sleep(min(2 ** i * 2, 15))
            continue
    return None, "exhausted retries"


def safe_name(tok):
    """Filesystem-safe token name (mirrors dex_micro_poller.safe_name)."""
    return (tok.replace("$", "_").replace(" ", "").replace("/", "_")
            .replace("\\", "_"))


def real_top_pool(tok, exclude_chains=EXCLUDE_CHAINS, resolve_tries=4):
    """Resolve tok -> (gtnet, pool_address, liquidity) for the highest-LIQ
    REAL-CHAIN pool. Prefers an exact base-symbol match; falls back to the
    top-liquidity real-chain pair so breadth is preserved.

    DexScreener's free API returns HTTP 200 with an EMPTY `pairs` list when
    our (shared VPN) IP is over its rate limit -- NOT a 429. So an empty
    result is ambiguous: it may be a genuine no-match OR a silent throttle.
    We treat empty `pairs` as retryable ('throttled') and back off, so a
    transient throttle doesn't permanently drop a collectable token.
    Returns (tuple, None) or (None, reason)."""
    for attempt in range(resolve_tries):
        d, err = fetch_json(f"{DEX_API}/search", params={"q": tok})
        if err:
            if "429" in str(err):
                time.sleep(min(2 ** attempt * 4, 30))
                continue
            return None, err
        if not d:
            time.sleep(2)
            continue
        pairs = d.get("pairs") or []
        if not pairs:
            # ambiguous: likely throttle (200-empty). back off + retry.
            if attempt < resolve_tries - 1:
                time.sleep(min(2 ** attempt * 4, 30))
                continue
            return None, "no pairs"
        norm = lambda s: (s or "").upper().lstrip("$").strip()
        q = norm(tok)
        best = None
        exact = None
        for p in pairs:
            ch = (p.get("chainId") or "").lower()
            if ch in exclude_chains or ch not in NETMAP:
                continue
            pa = p.get("pairAddress")
            if not pa:
                continue
            liq = float(p.get("liquidity", {}).get("usd") or 0)
            base = norm(p.get("baseToken", {}).get("symbol", ""))
            cand = (NETMAP[ch], pa, liq)
            if base == q and (exact is None or liq > exact[2]):
                exact = cand
            if best is None or liq > best[2]:
                best = cand
        pick = exact or best
        if not pick:
            return None, "no real-chain pool"
        return (pick[0], pick[1], pick[2]), None
    return None, "throttled"


def gt_pool_ohlcv(net, pool, tf="day", page=1, limit=1000, aggregate=1):
    """Free pool-level OHLCV. Returns (ohlcv_list, None), (None, 'throttled')
    on 429, or (None, err) on other failures."""
    url = (f"{GT_API}/networks/{net}/pools/{pool}/ohlcv/{tf}"
           f"?limit={limit}&page={page}&aggregate={aggregate}")
    d, err = fetch_json(url)
    if err:
        return None, ("throttled" if "429" in str(err) else err)
    if "data" not in d:
        return None, "no data"
    return d["data"].get("attributes", {}).get("ohlcv_list") or [], None
