#!/usr/bin/env python3
"""Shared memory guard: abort a long-running backfill safely if its resident
set size approaches an OOM-kill level. Imported by the backfill scripts so the
RSS-cap logic lives in exactly one place.

The guard reads RSS from /proc/self/statm (Linux only); on any failure it
reports 0 MiB, which means "no cap enforced" when limit_mb is also 0.
"""
import os
import sys


def rss_mb() -> float:
    """Resident set size in MiB via /proc/self/statm. 0.0 if unavailable."""
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return 0.0


def guard(limit_mb: int) -> None:
    """Abort the process if RSS exceeds `limit_mb`.

    A hard cap guarantees a backfill can never climb to an OOM-kill level;
    re-running is always safe (caches/output persist). Limit of 0 disables.
    """
    if not limit_mb:
        return
    rss = rss_mb()
    if rss > limit_mb:
        print(f"MEMORY GUARD TRIPPED: RSS={rss:.0f}MB > cap={limit_mb}MB "
              f"— aborting to prevent OOM kill (re-run is safe)")
        sys.exit(0)
