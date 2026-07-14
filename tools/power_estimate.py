#!/usr/bin/env python3
"""power_estimate.py — what sample is missing for meaningful conclusions, and when.

Q1 (strategy positive): n_pat needed so the 90%/95% CI lower bound of the
   deduped mean clears zero, given the OBSERVED pattern-level mean/std.
Q2 (policy choice): n_pat needed to resolve a delta of ~0.10 between two
   exit policies (paired comparison, observed per-pattern diff std).
Rate: observed distinct-pattern arrivals in the live era -> ETA dates.
"""
import json, time
from pathlib import Path
import numpy as np

ROOT = Path("/root/the-distribution-will-manifest")
cut = 1781047096.0

# live era pattern stats
by = {}
first_seen = {}
with open(ROOT / "bot_data/shadow_run.jsonl") as f:
    for ln in f:
        if "position_close" not in ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if (r.get("t") or 0) < cut or r.get("live_policy_net") is None:
            continue
        # pattern key: entry score rounded (need the fire's score; position_close lacks it
        # -> approximate pattern by joining later; use net-based key fallback)
        by.setdefault(r["mint"], []).append(float(r["live_policy_net"]))
# join mints to scores via entry_decision rows
score = {}
with open(ROOT / "bot_data/shadow_run.jsonl") as f:
    for ln in f:
        if "entry_decision" not in ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if (r.get("t") or 0) < cut or float(r.get("score", 0)) < 0.50:
            continue
        score[r["mint"]] = round(float(r["score"]), 6)
        first_seen.setdefault(round(float(r["score"]), 6), r["t"])
pat = {}
for m, nets in by.items():
    k = score.get(m)
    if k is None:
        continue
    pat.setdefault(k, []).extend(nets)
pm = np.array([np.mean(v) for v in pat.values()])
mu, sd, n = pm.mean(), pm.std(ddof=1), len(pm)
print(f"LIVE era pattern-level: n_pat={n}  mean={mu:+.3f}  std={sd:.3f}")

import math
z90, z95 = 1.645, 1.96
for conf, z in (("90%", z90), ("95%", z95)):
    need = math.ceil((z * sd / max(mu, 1e-9)) ** 2)
    print(f"  Q1 strategy>0 at {conf}: need n_pat ~ {need} (have {n})")
# Q2: resolve delta=0.10 between policies, assume paired diff std ~ 0.5*sd (correlated payoffs)
sd_d = 0.5 * sd
for conf, z in (("90%", z90), ("95%", z95)):
    need = math.ceil((z * sd_d / 0.10) ** 2)
    print(f"  Q2 policy delta 0.10 at {conf} (paired, sd_diff~{sd_d:.2f}): need n_pat ~ {need}")

# arrival rate of NEW distinct patterns
ts = sorted(first_seen.values())
if len(ts) > 3:
    hours = (ts[-1] - ts[0]) / 3600
    rate = (len(ts) - 1) / hours * 24
    print(f"\npattern arrivals: {len(ts)} distinct over {hours:.0f}h -> ~{rate:.1f} NEW patterns/day")
    now = time.time()
    for tgt in (30, 50, 100, 150):
        if tgt <= n:
            continue
        eta_days = (tgt - n) / max(rate, 1e-9)
        eta = time.strftime("%b %d", time.gmtime(now + eta_days * 86400))
        print(f"  n_pat={tgt}: ~{eta_days:.1f} days -> ~{eta}")
