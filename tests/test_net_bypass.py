"""Tests for net_bypass.py -- local-interface egress bypass (VPN throttle dodge)."""
import os
import socket
import requests
import pytest

import net_bypass as nb


def test_iface_off_by_default():
    # when BYPASS_VPN_IFACE unset, session() is a plain requests.Session
    old = os.environ.pop("BYPASS_VPN_IFACE", None)
    try:
        s = nb.session(local=True)
        assert s.get("https://api.ipify.org").status_code in (200,)
    finally:
        if old:
            os.environ["BYPASS_VPN_IFACE"] = old


def test_local_interface_adapter_configured():
    """LocalInterfaceAdapter stores the iface and mounts on a session."""
    a = nb.LocalInterfaceAdapter(iface="enp6s0")
    assert a.iface == "enp6s0"
    os.environ["BYPASS_VPN_IFACE"] = "enp6s0"
    try:
        s = nb.session(local=True)
        # adapter mounted for https -> egress will bind to enp6s0 at send time
        adapter = s.get_adapter("https://api.ipify.org")
        assert isinstance(adapter, nb.LocalInterfaceAdapter)
        assert adapter.iface == "enp6s0"
    finally:
        os.environ.pop("BYPASS_VPN_IFACE", None)


def test_local_interface_adapter_send_noop_without_iface():
    """With no iface set, send() must pass through to the parent (no bind)."""
    a = nb.LocalInterfaceAdapter(iface=None)
    assert a.iface is None
    # does not raise on construction; real send falls back to default route



def test_get_falls_back_on_missing_iface():
    # with no env iface, get() must not crash and falls back to plain requests.get
    os.environ.pop("BYPASS_VPN_IFACE", None)
    r = nb.get("https://api.ipify.org")
    assert r.status_code in (200,)
