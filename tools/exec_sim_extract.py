#!/usr/bin/env python3
"""exec_sim_extract.py — forward trades WITH slot + tip, for the execution simulator.

For every joint-trigger-ready mint (live gates), records the decision slot/reserves
and the forward trades as (ts, slot, vsol, vtok, is_buy, tip). slot has FULL
coverage (every capture row has it); tip is joined from the shred intent stream
by signature (~49% coverage, logged). This feeds tools/exec_sim.py which scores
fills under the old fixed-latency model AND a slot-aware + tip-rank landing model.

Run: K_TRIGGER=3 V_TRIGGER=0.3 PYTHONPATH=. python tools/exec_sim_extract.py
"""
import glob
import gzip
import json
import os
import pickle
import time
from pathlib import Path

assert os.environ.get("K_TRIGGER") == "3" and os.environ.get("V_TRIGGER") == "0.3"
from feature_accum import TokenState  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRESH = 3_000_000_000
HORIZON = 300.0
MAX_FWD = 400


def is_classic(vsol, rsol):
    return abs(vsol - 30_000_000_000 - rsol) < 50_000_000


def build_sig_tip():
    """signature -> jito_tip_lam from the shred intent capture (buys)."""
    m = {}
    files = sorted(glob.glob(str(ROOT / "shred_bot/intent_capture/intent-*.jsonl*")))
    for path in files:
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
                sig = r.get("first_sig")
                if sig:
                    m[sig] = float(r.get("jito_tip_lam") or 0.0)
    return m


def main():
    print("building sig->tip map from shred intents ...", flush=True)
    sig_tip = build_sig_tip()
    print(f"  {len(sig_tip):,} signatures with tips", flush=True)

    states, first_seen, ready = {}, {}, {}
    fwd = {}
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
                    print(f"  .. {n/1e6:.0f}M classic | ready={len(ready)} | {time.time()-t0:.0f}s", flush=True)
                if ts > max_ts:
                    max_ts = ts
                if mint not in first_seen:
                    first_seen[mint] = 1e18 if rsol >= FRESH else rsol
                    if first_seen[mint] >= FRESH:
                        continue
                if first_seen[mint] >= FRESH:
                    continue
                rd = ready.get(mint)
                if rd is not None:
                    lst = fwd[mint]
                    if ts - rd["ts"] <= HORIZON and len(lst) < MAX_FWD:
                        tip = sig_tip.get(r.get("sig"))
                        lst.append((ts, slot, vsol, vtok, bool(r.get("is_buy")),
                                    (float(tip) if tip is not None else None)))
                    continue
                is_buy = bool(r.get("is_buy")); user = r.get("user", "")
                st = states.get(mint)
                if st is None:
                    states[mint] = TokenState(vsol, vtok, sol, is_buy, user, ts)
                    continue
                st.update(vsol, vtok, sol, is_buy, user, ts)
                if st.k_fired and st.v_fired:
                    ready[mint] = {"ts": ts, "slot": slot, "vsol": vsol, "vtok": vtok}
                    fwd[mint] = []
                    del states[mint]

    out = {m: {"decision": ready[m], "fwd": fwd[m]}
           for m in ready if ready[m]["ts"] <= max_ts - HORIZON - 10 and fwd.get(m)}
    cov = [t for d in out.values() for (_, _, _, _, b, t) in d["fwd"] if b]
    known = sum(1 for t in cov if t is not None)
    pickle.dump(out, open(ROOT / "data/exec_sim_fwd_k3v03.pkl", "wb"))
    print(f"streamed {n:,} classic in {(time.time()-t0)/60:.1f}min; ready+labeled mints={len(out)}")
    print(f"forward BUY tip coverage: {known:,}/{len(cov):,} ({known/max(len(cov),1):.0%})")
    print(f"wrote data/exec_sim_fwd_k3v03.pkl")


if __name__ == "__main__":
    main()
