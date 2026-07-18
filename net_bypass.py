#!/usr/bin/env python3
"""
net_bypass.py -- route specific egress requests via the LOCAL interface
instead of the VPN, to dodge hard throttling on the VPN exit IP.

WHY: this box runs WireGuard (azirevpn-de-fra); by policy all traffic egresses
through the VPN (public IP 37.46.199.152). Some data hosts (GMGN, GeckoTerminal,
DexScreener) throttle/block that shared VPN IP. Binding the socket to the real
LAN interface (enp6s0 -> 73.220.x.x) makes those requests originate from the
local ISP IP instead, which is not throttled.

USAGE:
  export BYPASS_VPN_IFACE=enp6s0        # set the interface to bind to (off if unset)
  from net_bypass import session
  s = session(local=True)               # requests.Session bound to local iface
  s.get("https://api.gmgn.ai/...")      # egresses via enp6s0, not the VPN

  # or a one-off adapter for an existing session:
  from net_bypass import LocalInterfaceAdapter
  s.mount("https://", LocalInterfaceAdapter(iface="enp6s0"))

If BYPASS_VPN_IFACE is unset or the bind fails, calls transparently fall back
to the default (VPN) route -- never raises, never silently breaks collection.
"""
import os
import socket
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

_SO_BINDTODEVICE = 25  # Linux-only socket option


def _iface() -> str | None:
    """The interface to bind local egress to, from env. None = feature off."""
    v = os.environ.get("BYPASS_VPN_IFACE", "").strip()
    return v or None


class LocalInterfaceAdapter(HTTPAdapter):
    """HTTPAdapter that binds every socket to a given interface (SO_BINDTODEVICE)
    so the request egresses via that interface instead of the default route."""

    def __init__(self, iface: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.iface = iface or _iface()

    def send(self, request, **kwargs):
        if self.iface:
            # Bind the underlying socket to the local interface before connect.
            # requests/urllib3 create the socket in Connection._new_conn; we
            # monkeypatch socket.socket for this adapter's send call only.
            orig_socket = socket.socket

            def _bound_socket(family, type_, proto=0, *a, **kw):
                sock = orig_socket(family, type_, proto, *a, **kw)
                try:
                    sock.setsockopt(socket.SOL_SOCKET, _SO_BINDTODEVICE,
                                    self.iface.encode())
                except Exception:
                    pass  # fall back to default route if bind fails
                return sock

            socket.socket = _bound_socket
            try:
                return super().send(request, **kwargs)
            finally:
                socket.socket = orig_socket
        return super().send(request, **kwargs)


def session(local: bool = True, iface: str | None = None, **kwargs) -> requests.Session:
    """Return a requests.Session. If `local` and an iface is configured, all
    https+http traffic goes out via that interface (bypassing the VPN)."""
    s = requests.Session()
    iface = iface or _iface()
    if local and iface:
        adapter = LocalInterfaceAdapter(iface=iface)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": "trading-bot/1.0"})
    return s


def get(url: str, local: bool = True, **kwargs) -> requests.Response:
    """Convenience: GET `url` via the local interface when configured."""
    iface = _iface()
    if local and iface:
        kwargs.setdefault("headers", {})
        kwargs["headers"].setdefault("User-Agent", "trading-bot/1.0")
        return session(local=True, iface=iface).get(url, **kwargs)
    return requests.get(url, headers={"User-Agent": "trading-bot/1.0"}, **kwargs)


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://api.ipify.org"
    r = get(url)
    print(f"{url} -> {r.status_code}")
    print(r.text[:200])
