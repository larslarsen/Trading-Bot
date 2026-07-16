#!/usr/bin/env python3
"""
On-chain BTC/ETH history puller (AWS Public Blockchain -- FREE, no key, anonymous S3).

Source: s3://aws-public-blockchain  (v1.0/btc/{blocks,transactions}/, v1.0/eth/...)
Each day = one snappy.parquet. We derive ML-ready on-chain FEATURES per day:
  - tx count, total fees (BTC) / gas-used (ETH)
  - large-transfer count/volume (whale flow proxy): outputs/values above a threshold
  - active-address proxy (unique senders+receivers)
  - avg fee per tx, median tx value
These attach to BTC/ETH PRICE series as exogenous features (network stress,
whale activity) for the ML bots.

Usage: python backfill_onchain.py [--chain btc] [--start 2009-01-03]
Writes: data/onchain/<CHAIN>_features_daily.csv  (indexed by date)
"""
import argparse
import io
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).parent
OUT = REPO / "data" / "onchain"
BUCKET = "https://aws-public-blockchain.s3.amazonaws.com"
WHALE_BTC = 100.0      # BTC
WHALE_ETH = 1000.0     # ETH


def list_days(chain):
    prefix = f"v1.0/{chain}/transactions/"
    out = []
    cont = None
    while True:
        url = f"{BUCKET}/?prefix={prefix}&max-keys=500"
        if cont:
            url += f"&continuation-token={cont}"
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as r:
            xml = r.read().decode()
        keys = re.findall(r"<Key>([^<]+)</Key>", xml)
        for k in keys:
            m = re.search(r"date=(\d{4}-\d{2}-\d{2})/", k)
            if m:
                out.append(m.group(1))
        nxt = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if nxt:
            cont = nxt.group(1)
        else:
            break
    return sorted(set(out))


def fetch_parquet(url):
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as r:
        return pq.read_table(io.BytesIO(r.read()))


def daily_features(chain, day):
    # list the actual parquet key for this day
    import urllib.request, re as _re
    lst = urllib.request.urlopen(f"{BUCKET}/?prefix=v1.0/{chain}/transactions/date={day}/",
                                 timeout=30).read().decode()
    key = _re.search(r"<Key>([^<]+)</Key>", lst)
    if not key:
        return None
    t = fetch_parquet(f"{BUCKET}/{key.group(1)}")
    df = t.to_pandas()
    n = len(df)
    if n == 0:
        return None
    cols = {c.lower(): c for c in df.columns}
    feats = {"date": day, "chain_tx_count": n}
    # fees / gas
    if "fee" in cols:
        feats["chain_total_fee"] = float(df[cols["fee"]].sum())
        feats["chain_avg_fee"] = float(df[cols["fee"]].mean())
    if "gasused" in cols or "gas" in cols:
        gcol = cols.get("gasused") or cols.get("gas")
        if gcol:
            feats["chain_total_gas"] = float(df[gcol].sum())
    # value: BTC 'output_value' is already in BTC; ETH 'value' is in wei (1e18)
    vcol = cols.get("output_value") or cols.get("value")
    if vcol is not None:
        vals = df[vcol].astype(float)
        if chain == "eth":
            vals = vals / 1e18
        thr = WHALE_BTC if chain == "btc" else WHALE_ETH
        feats["chain_total_value"] = float(vals.sum())
        feats["chain_whale_tx"] = int((vals >= thr).sum())
        feats["chain_whale_value"] = float(vals[vals >= thr].sum())
        feats["chain_med_value"] = float(vals.median()) if n else 0.0
    # counts
    if "output_count" in cols:
        feats["chain_total_outputs"] = int(df[cols["output_count"]].sum())
    if "input_count" in cols:
        feats["chain_total_inputs"] = int(df[cols["input_count"]].sum())
    if "is_coinbase" in cols:
        feats["chain_coinbase_tx"] = int(df[cols["is_coinbase"]].sum())
    # unique addresses (senders/receivers) -- ETH has 'from'/'to'; BTC uses outputs list
    for role in ("from", "to", "sender", "receiver"):
        if role in cols:
            feats[f"chain_uniq_{role}"] = int(df[cols[role]].nunique())
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="btc")
    ap.add_argument("--start", default="2009-01-03")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    days = [d for d in list_days(args.chain) if d >= args.start]
    print(f"on-chain {args.chain}: {len(days)} days to process", flush=True)
    rows = []
    for i, day in enumerate(days):
        try:
            f = daily_features(args.chain, day)
            if f:
                rows.append(f)
        except Exception as e:
            print(f"  {day}: ERR {e!r}"[:160], flush=True)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(days)} days", flush=True)
    if rows:
        out = pd.DataFrame(rows).set_index("date").sort_index()
        tgt = OUT / f"{args.chain}_features_daily.csv"
        if tgt.exists():
            old = pd.read_csv(tgt).set_index("date")
            out = pd.concat([old, out]).drop_duplicates()
        out.to_csv(tgt)
        print(f"DONE {args.chain}: {len(out)} days -> {tgt}", flush=True)


if __name__ == "__main__":
    main()
