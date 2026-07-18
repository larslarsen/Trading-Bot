#!/usr/bin/env python3
"""Shared DEX (chain, pool) resolver for free GeckoTerminal OHLCV collection.

Resolves a ticker to the highest-liquidity REAL-CHAIN pool and returns the
GeckoTerminal (network, pool_address) for the still-free pool-level OHLCV
endpoint: /networks/{net}/pools/{pool}/ohlcv/{tf}.

Dual-egress: this box has two usable egress IPs --
  * CLEAN  -> 192.168.1.100 (enp6s0)  -> clean ISP IP (fast, but GT free cap)
  * VPN    -> 10.0.129.5   (azirevpn) -> shared VPN IP (slower, throttled more)
We bind the source address to force a given egress, and on a 429 we fall back
to the OTHER egress, so both IPs get used and throughput ~doubles.

Used by backfill_dex_history_gt.py (deep history) and dex_ohlcv_sampler.py
(live 1m/5m). Single source of truth -> DRY.
"""
import socket
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

from net_bypass import session as _bypass_session

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

# Egress bypass: net_bypass binds to the LOCAL interface (BYPASS_VPN_IFACE,
# e.g. enp6s0 -> 73.220.x.x) via SO_BINDTODEVICE, dodging the throttled VPN
# exit (37.46.x.x). This replaces the old IP-literal source_address bind
# (192.168.1.100) which silently failed and fell back to the VPN. When
# BYPASS_VPN_IFACE is unset, net_bypass falls back to the default route.
# The `egress` param is accepted for API compat but the interface is chosen
# by the env var; passing egress only narrows which net_bypass does.


def _session():
    s = _bypass_session(local=True)
    s.headers.update(UA)
    return s


_SESSION = _session()


def fetch_json(url, params=None, timeout=20, tries=3, egress=None):
    """GET url as JSON via the local-interface bypass. Returns (dict, None)
    or (None, reason). On HTTP 429 (throttle) we retry up to `tries` times
    with brief backoff; net_bypass already routes us off the throttled VPN IP."""
    last_err = "exhausted retries"
    for cyc in range(tries):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json(), None
            if r.status_code == 429:
                last_err = "HTTP 429"
            else:
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = f"err:{e}"
        if cyc < tries - 1:
            time.sleep(min(2 ** cyc * 2, 10))
    return None, last_err


def safe_name(tok):
    """Filesystem-safe token name (mirrors dex_micro_poller.safe_name)."""
    return (tok.replace("$", "_").replace(" ", "").replace("/", "_")
            .replace("\\", "_"))


def real_top_pool(tok, exclude_chains=EXCLUDE_CHAINS, resolve_tries=4, egress=None):
    """Resolve tok -> (gtnet, pool_address, liquidity) for the highest-LIQ
    REAL-CHAIN pool.

    egress: pin DexScreener request to one source IP ('192.168.1.100' clean
    or '10.0.129.5' vpn); None races both. Empty `pairs` is treated as
    retryable (silent throttle), not a permanent miss."""
    for attempt in range(resolve_tries):
        d, err = fetch_json(f"{DEX_API}/search", params={"q": tok}, egress=egress)
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
    """Free GeckoTerminal POOL-LEVEL OHLCV. Races both egresses (clean first,
    VPN on 429). Returns (rows, None) or (None, reason)."""
    url = f"{GT_API}/networks/{net}/pools/{pool}/ohlcv/{tf}"
    d, err = fetch_json(url, params={
        "page": page, "limit": limit, "aggregate": aggregate,
        "currency": "usd",
    })
    if err:
        return None, err
    try:
        attrs = d["data"]["attributes"]
        rows = attrs.get("ohlcv_list") or []
        out = [[r[0], float(r[1]), float(r[2]), float(r[3]),
                float(r[4]), float(r[5])] for r in rows]
        return out, None
    except Exception as e:
        return None, f"parse:{e}"


if __name__ == "__main__":
    import sys
    tok = sys.argv[1] if len(sys.argv) > 1 else "AAVE"
    r = real_top_pool(tok)
    print("real_top_pool:", r)
    if r[0]:
        net, pool, liq = r[0]
        bars, e = gt_pool_ohlcv(net, pool, "day", 1, 5)
        print("gt_pool_ohlcv bars:", len(bars) if bars else 0, "err:", e)
