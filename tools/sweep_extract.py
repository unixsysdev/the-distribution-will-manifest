#!/usr/bin/env python3
"""sweep_extract.py — ONE-pass extraction for a K/V challenger cell.

Records, per joint-trigger-ready mint (live gates, honest population): the
decision slot/reserves, the 22 K+V entry features, the K-anchored peak_ret
label, AND the forward trades (ts, slot, vsol, vtok, is_buy, tip) so the same
file feeds BOTH the cross-day AUC comparison and the execution-adjusted sim.

K/V from env (no hardcoded assertion). Output: data/sweep_k{K}v{V}.pkl
Run: K_TRIGGER=7 V_TRIGGER=0.5 PYTHONPATH=. python tools/sweep_extract.py
"""
import glob
import gzip
import json
import os
import pickle
import time
from pathlib import Path

K = os.environ["K_TRIGGER"]
V = os.environ["V_TRIGGER"]
from feature_accum import TokenState, ENTRY_FEATURE_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRESH = 3_000_000_000
HORIZON = 300.0
MAX_FWD = 400


def is_classic(vsol, rsol):
    return abs(vsol - 30_000_000_000 - rsol) < 50_000_000


def build_sig_tip():
    m = {}
    for path in sorted(glob.glob(str(ROOT / "shred_bot/intent_capture/intent-*.jsonl*"))):
        op = gzip.open if path.endswith(".gz") else open
        try:
            fh = op(path, "rt")
        except OSError:
            continue
        with fh:
            for ln in fh:
                if '"first_sig"' not in ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if r.get("first_sig"):
                    m[r["first_sig"]] = float(r.get("jito_tip_lam") or 0.0)
    return m


def main():
    print(f"K={K} V={V}; building sig->tip ...", flush=True)
    sig_tip = build_sig_tip()
    print(f"  {len(sig_tip):,} tipped sigs", flush=True)

    states, first_seen, ready, feats_at, fwd = {}, {}, {}, {}, {}
    t0 = time.time()
    n = 0
    max_ts = 0.0
    for path in sorted(glob.glob(str(ROOT / "grpc_capture/*.jsonl*"))):
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
                if not mint or "vsol" not in r:
                    continue
                try:
                    vsol = float(r["vsol"]); vtok = float(r["vtok"]); rsol = float(r["rsol"])
                    sol = float(r["sol"]) / 1e9; ts = float(r["ev_ts"]); slot = int(r["slot"])
                except (KeyError, TypeError, ValueError):
                    continue
                if vsol <= 0 or vtok <= 0 or not is_classic(vsol, rsol):
                    continue
                n += 1
                if n % 1_000_000 == 0:
                    print(f"  .. {n/1e6:.0f}M | ready={len(ready)} | {time.time()-t0:.0f}s", flush=True)
                if ts > max_ts:
                    max_ts = ts
                if mint not in first_seen:
                    first_seen[mint] = 1e18 if rsol >= FRESH else rsol
                    if first_seen[mint] >= FRESH:
                        continue
                if first_seen[mint] >= FRESH:
                    continue
                if mint in ready:
                    lst = fwd[mint]
                    if ts - ready[mint]["ts"] <= HORIZON and len(lst) < MAX_FWD:
                        tip = sig_tip.get(r.get("sig"))
                        lst.append((ts, slot, vsol, vtok, bool(r.get("is_buy")),
                                    float(tip) if tip is not None else None))
                    continue
                is_buy = bool(r.get("is_buy")); user = r.get("user", "")
                st = states.get(mint)
                if st is None:
                    states[mint] = TokenState(vsol, vtok, sol, is_buy, user, ts)
                    continue
                st.update(vsol, vtok, sol, is_buy, user, ts)
                if st.k_fired and st.v_fired:
                    ready[mint] = {"ts": ts, "slot": slot, "vsol": vsol, "vtok": vtok}
                    feats_at[mint] = list(st.combined_entry_features())
                    fwd[mint] = []
                    del states[mint]

    out = {}
    for m in ready:
        if ready[m]["ts"] > max_ts - HORIZON - 10 or not fwd.get(m):
            continue
        dmid = ready[m]["vsol"] / ready[m]["vtok"]
        peak = max((vs / vt for (_t, _s, vs, vt, _b, _tp) in fwd[m]), default=dmid) / dmid - 1.0
        out[m] = {"decision": ready[m], "feats": feats_at[m], "peak_ret": peak, "fwd": fwd[m]}
    fn = ROOT / f"data/sweep_k{K}v{str(V).replace('.','')}.pkl"
    pickle.dump({"K": int(K), "V": float(V), "names": list(ENTRY_FEATURE_NAMES), "mints": out},
                open(fn, "wb"))
    br = sum(1 for d in out.values() if d["peak_ret"] >= 0.5) / max(len(out), 1)
    print(f"streamed {n:,} in {(time.time()-t0)/60:.1f}min; ready+labeled={len(out)} "
          f"peak>=50% base {br:.3f}; wrote {fn}")


if __name__ == "__main__":
    main()
