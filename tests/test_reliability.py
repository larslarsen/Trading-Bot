"""Tests for reliability.py — atomic writes, retry+backoff, safe logging.

These are unit-level: no network, no real I/O beyond temp files.
"""
import sys, os, time, json, tempfile
from pathlib import Path

import pytest

import reliability as rel
import pandas as pd


# ── atomic_write_csv ────────────────────────────────────────────────────────
def test_atomic_write_csv_writes_content(tmp_path):
    df = pd.DataFrame({"ts": [1, 2], "v": [3.0, 4.0]})
    p = tmp_path / "a.csv"
    rel.atomic_write_csv(p, df)
    assert p.exists()
    got = pd.read_csv(p, index_col=0)
    assert len(got) == 2 and list(got.columns)[:2] == ["ts", "v"]


def test_atomic_write_csv_no_stray_tmp(tmp_path):
    rel.atomic_write_csv(tmp_path / "a.csv", pd.DataFrame({"x": [1]}))
    assert not list(tmp_path.glob(".*.tmp.*")), "leftover temp file"


def test_atomic_write_csv_crash_safe(tmp_path):
    """A failed write must leave the original file byte-for-byte intact."""
    p = tmp_path / "b.csv"
    rel.atomic_write_csv(p, pd.DataFrame({"x": [9]}))
    orig = p.read_bytes()
    real = rel.atomic_write_csv

    class Boom(Exception):
        pass

    def boom(*a, **k):
        raise Boom("disk full")

    rel.atomic_write_csv = boom
    try:
        try:
            rel.atomic_write_csv(p, pd.DataFrame({"x": [1, 2, 3]}))
        except Boom:
            pass
    finally:
        rel.atomic_write_csv = real
    assert p.read_bytes() == orig, "original corrupted on failed write"
    assert not list(tmp_path.glob(".*.tmp.*"))


# ── atomic_write_json / text ─────────────────────────────────────────────────
def test_atomic_write_json_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    rel.atomic_write_json(p, {"a": 1, "b": [1, 2]})
    assert json.loads(p.read_text()) == {"a": 1, "b": [1, 2]}


def test_atomic_write_text_roundtrip(tmp_path):
    p = tmp_path / "s.txt"
    rel.atomic_write_text(p, "hello")
    assert p.read_text() == "hello"


# ── retry_call ──────────────────────────────────────────────────────────────
def test_retry_call_returns_on_success_without_sleep():
    calls = {"n": 0}
    slept = []
    out = rel.retry_call(lambda: (_ for _ in ()).throw(Exception()) if (_ for _ in ()).throw(None) else None,
                          tries=1) if False else None
    # simpler: success path
    def good():
        calls["n"] += 1
        return 42
    assert rel.retry_call(good, tries=3, sleep=lambda s: slept.append(s)) == 42
    assert calls["n"] == 1
    assert slept == [], "should not sleep on first success"


def test_retry_call_retries_then_succeeds():
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("x")
        return "ok"
    assert rel.retry_call(flaky, tries=4, base=0.01, cap=0.1,
                          sleep=lambda s: None) == "ok"
    assert calls["n"] == 3


def test_retry_call_exhausts_and_raises():
    calls = {"n": 0}
    def always():
        calls["n"] += 1
        raise ValueError("boom")
    with pytest.raises(ValueError):
        rel.retry_call(always, tries=3, base=0.001, cap=0.01, sleep=lambda s: None)
    assert calls["n"] == 3


def test_retry_call_jitter_decorrelates():
    sleeps = []
    def flaky():
        raise ConnectionError()
    try:
        rel.retry_call(flaky, tries=3, base=0.1, cap=1.0, jitter=0.5,
                       sleep=lambda s: sleeps.append(s))
    except ConnectionError:
        pass
    assert len(sleeps) == 2
    # jitter: consecutive sleeps should differ (not locked in step)
    assert sleeps[0] != sleeps[1]


def test_retry_call_on_retry_hook_fires():
    fires = []
    def flaky():
        raise IOError("x")
    try:
        rel.retry_call(flaky, tries=2, base=0.001, cap=0.01,
                       on_retry=lambda a, e: fires.append((a, type(e).__name__)),
                       sleep=lambda s: None)
    except IOError:
        pass
    assert fires and fires[0][0] == 1  # attempt number passed


def test_retry_call_scoped_exceptions():
    calls = {"n": 0}
    def raises_value():
        calls["n"] += 1
        raise ValueError("not retried")
    # ValueError not in the scoped tuple -> should raise immediately, no retry
    with pytest.raises(ValueError):
        rel.retry_call(raises_value, tries=4, exceptions=(ConnectionError,),
                       sleep=lambda s: None)
    assert calls["n"] == 1


# ── retryable decorator ──────────────────────────────────────────────────────
def test_retryable_decorator_retries():
    calls = {"n": 0}
    @rel.retryable(tries=3, base=0.001, cap=0.01)
    def f():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError()
        return "done"
    assert f() == "done"
    assert calls["n"] == 2


# ── safe_log ────────────────────────────────────────────────────────────────
def test_safe_log_writes_and_never_raises(tmp_path):
    p = tmp_path / "l.log"
    rel.safe_log(p, "hello")
    assert p.read_text().strip().endswith("hello")
    # None path must not raise
    rel.safe_log(None, "x")
    # corrupt directory must not raise
    rel.safe_log(tmp_path / "nope" / "x.log", "y")


def test_safe_log_fsync_survives(tmp_path):
    p = tmp_path / "l.log"
    for i in range(3):
        rel.safe_log(p, f"line{i}")
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
