#!/usr/bin/env python3
"""Shared reliability primitives for the trading-bot data layer.

Single home for the patterns every daemon/poller needs but used to hand-roll
(or skip) inconsistently:

  * atomic_write_csv / atomic_write_json -- write to a sibling .tmp file then
    os.replace() it over the target. os.replace is atomic on POSIX, so a kill
    (systemd stop, OOM) mid-write can NEVER leave a half-written CSV that the
    next run would read as gospel. This is the #1 crash-safety fix for the
    forever-append daemons.
  * retry_call / retryable -- exponential backoff with FULL jitter (not just
    capped sleep) so a fleet of workers retrying against the same 429/503 wall
    does not synchronize into a retry storm (the July-9 API-storm wedge).
  * safe_log -- line logger that never raises (logging failures must not take
    down the caller) and flushes.

Everything here fails soft: a helper raising an exception is the caller's
problem, but the helpers themselves will not mask their own internal errors in
a way that hides data loss. Atomic writes still raise on real I/O failure so
callers can decide what to do.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple, Type, Union

import pandas as pd


# ── Atomic writes ──────────────────────────────────────────────────────────
def atomic_write_csv(path: Union[str, Path], df: "pd.DataFrame", **kwargs) -> None:
    """Write `df` to `path` atomically: temp file in the same dir, then
    os.replace(). The original file is untouched until the new bytes are fully
    on disk, so a crash mid-write cannot corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        df.to_csv(tmp, **kwargs)
        os.replace(tmp, path)  # atomic on same filesystem
    finally:
        # best-effort cleanup of a stray tmp left by a hard kill
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def atomic_write_json(path: Union[str, Path], obj: Any, indent: int = 2) -> None:
    """Write JSON atomically (same .tmp + os.replace pattern as CSV)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(obj, indent=indent))
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    """Write raw text atomically (for manifests, state dumps, etc.)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


# ── Retry / backoff ─────────────────────────────────────────────────────────
def retry_call(
    fn: Callable[[], Any],
    *,
    tries: int = 4,
    base: float = 1.0,
    cap: float = 30.0,
    jitter: float = 0.3,
    sleep: Callable[[float], None] = time.sleep,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    on_retry: "Callable[[int, BaseException], None] | None" = None,
) -> Any:
    """Call `fn` with exponential backoff + full jitter.

    Sleep after attempt i (0-indexed, before retry) is:
        sleep = min(cap, base * 2**i) * (1 + uniform(-jitter, +jitter))
    Jitter decorrelates a fleet of workers so they don't all retry in lockstep
    against a rate-limited endpoint. Re-raises the last exception after
    `tries` attempts. `on_retry(attempt, err)` is called before each sleep.
    """
    last: BaseException | None = None
    for i in range(tries):
        try:
            return fn()
        except exceptions as e:  # noqa: PERF203
            last = e
            if i == tries - 1:
                break
            raw = min(cap, base * (2 ** i))
            delay = raw * (1.0 + random.uniform(-jitter, jitter))
            delay = max(0.0, delay)
            if on_retry:
                try:
                    on_retry(i + 1, e)
                except Exception:
                    pass
            sleep(delay)
    assert last is not None
    raise last


def retryable(
    *,
    tries: int = 4,
    base: float = 1.0,
    cap: float = 30.0,
    jitter: float = 0.3,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
):
    """Decorator form of retry_call for wrapping functions directly."""

    def deco(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return retry_call(
                lambda: fn(*args, **kwargs),
                tries=tries, base=base, cap=cap, jitter=jitter,
                exceptions=exceptions,
            )
        return wrapper
    return deco


# ── Safe logging ─────────────────────────────────────────────────────────────
def safe_log(path: Union[str, Path, None], msg: str, *, stamp=True) -> None:
    """Append a line to a log file, never raising. Flushes so a crash right
    after still has the line on disk. Returns nothing; logging failure is
    swallowed silently because a logger must not take down the process."""
    if path is None:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}" if stamp else msg)
        with open(p, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass
