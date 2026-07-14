#!/usr/bin/env python3
"""shred_coverage_probe.py — measure what the shred intent feed is actually worth.

Joins intent_capture (pre-block shred intents) to grpc_capture (executed trades)
by exact tx signature over the last two closed hours. Reports:
  1. coverage: % of executed fresh-classic BUYS that were visible as intents first
  2. lead time: how much earlier the shred path saw them (the latency budget a
     shred-side trigger could harvest; also bounds the cost of the 50ms drain)
  3. competing-tip distribution among those intents (calibrates the tip ladder)
  4. cluster contention: when >=2 buy intents hit the same mint within 500ms,
     what jito tip would have outbid 90% of the visible competition
Read-only; no live changes.
"""
import glob
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/the-distribution-will-manifest")
FRESH = 3_000_000_000

intent_files = sorted(glob.glob(str(ROOT / "shred_bot/intent_capture/intent-*.jsonl.gz")))[-2:]
print("intent files:", [Path(f).name for f in intent_files])

intents = {}
n_int = 0
for path in intent_files:
    with gzip.open(path, "rt") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("type") not in ("buy", "buy_quote", "buy_sol_in"):
                continue
            sig = r.get("first_sig")
            if not sig or "recv_ns" not in r:
                continue
            n_int += 1
            # keep the EARLIEST sighting per sig
            if sig not in intents or r["recv_ns"] < intents[sig][0]:
                intents[sig] = (r["recv_ns"], r.get("jito_tip_lam", 0) or 0,
                                r.get("priority_fee_micro", 0) or 0,
                                r.get("mint", ""), r.get("cu_limit", 0))
print(f"buy intents: {n_int:,} ({len(intents):,} unique sigs)")

lo_ns = min(v[0] for v in intents.values())
hi_ns = max(v[0] for v in intents.values())

cap_files = sorted(glob.glob(str(ROOT / "grpc_capture/*.jsonl*")))[-5:]
first_seen = {}
tot = {"all": 0, "fresh": 0}
match = {"all": 0, "fresh": 0}
leads = []
fresh_by_mint = {}
for path in cap_files:
    op = gzip.open if path.endswith(".gz") else open
    try:
        fh = op(path, "rt")
    except OSError:
        continue
    with fh:
        for line in fh:
            if "vsol" not in line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            mint = r.get("mint")
            if not mint or "vsol" not in r or not r.get("is_buy"):
                continue
            try:
                vsol = float(r["vsol"]); rsol = float(r["rsol"]); t = float(r["t"])
            except (KeyError, TypeError, ValueError):
                continue
            if t * 1e9 < lo_ns - 5e9 or t * 1e9 > hi_ns + 5e9:
                continue
            classic = abs(vsol - 30_000_000_000 - rsol) < 50_000_000
            if not classic:
                continue
            if mint not in first_seen:
                first_seen[mint] = rsol
            fresh = first_seen[mint] < FRESH
            tot["all"] += 1
            if fresh:
                tot["fresh"] += 1
            sig = r.get("sig")
            hit = intents.get(sig)
            if hit:
                match["all"] += 1
                lead_ms = (t - hit[0] / 1e9) * 1e3
                if fresh:
                    match["fresh"] += 1
                    leads.append(lead_ms)
                    fresh_by_mint.setdefault(mint, []).append((hit[0], hit[1]))

print(f"\nexecuted classic buys in window: {tot['all']:,} (fresh-launch: {tot['fresh']:,})")
print(f"coverage (intent seen first by sig): all {match['all']/max(tot['all'],1):.1%}  "
      f"fresh {match['fresh']/max(tot['fresh'],1):.1%}")
leads = np.array(leads)
if len(leads):
    q = np.percentile(leads, [10, 25, 50, 75, 90, 99])
    print(f"lead time (gRPC receipt - shred receipt) ms: p10={q[0]:.0f} p25={q[1]:.0f} "
          f"p50={q[2]:.0f} p75={q[3]:.0f} p90={q[4]:.0f} p99={q[5]:.0f}  "
          f"(negative = shreds later than gRPC)  n={len(leads):,}")
    print(f"share with lead > 50ms (drain-cadence-material): {(leads>50).mean():.1%}; "
          f"> 200ms: {(leads>200).mean():.1%}; > 400ms (full slot): {(leads>400).mean():.1%}")

tips = np.array([v[1] for v in intents.values()])
prio = np.array([v[2] for v in intents.values()])
print(f"\nintent jito tips: nonzero {np.mean(tips>0):.1%}; nonzero quantiles (lam): "
      f"{[int(x) for x in np.percentile(tips[tips>0],[50,75,90,99])] if (tips>0).any() else 'n/a'}")
print(f"priority fee micro: nonzero {np.mean(prio>0):.1%}; nonzero quantiles: "
      f"{[int(x) for x in np.percentile(prio[prio>0],[50,75,90,99])] if (prio>0).any() else 'n/a'}")

clustered_tips = []
cluster_mints = 0
for mint, recs in fresh_by_mint.items():
    recs.sort()
    best = 1
    for i in range(len(recs)):
        c = [r for r in recs if abs(r[0] - recs[i][0]) <= 500_000_000]
        best = max(best, len(c))
        if len(c) >= 2:
            clustered_tips.extend(t for _, t in c)
    if best >= 2:
        cluster_mints += 1
ct = np.array(clustered_tips)
print(f"\nfresh mints with >=2 buy intents in any 500ms window: {cluster_mints} "
      f"of {len(fresh_by_mint)}")
if len(ct):
    print(f"tips inside those clusters: nonzero {np.mean(ct>0):.1%}; "
          f"p50/p90/p99 (lam): {[int(x) for x in np.percentile(ct,[50,90,99])]}  "
          f"-> outbid-90% tip = {int(np.percentile(ct,90)):,} lam = {np.percentile(ct,90)/1e9:.6f} SOL")
