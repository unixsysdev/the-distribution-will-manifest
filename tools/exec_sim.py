#!/usr/bin/env python3
"""exec_sim.py — execution simulator: OLD fixed-latency vs NEW slot-aware+tip-rank,
PLUS realized-slippage measurement and slippage-cap learning.

For every fire (deployed 22-feat model, score>=0.50), over the observed forward
path (level_tp_50 / stale, q=0.1, 250bps, 0.0015/tx, horizon 300s):

  Model A (incumbent): land at forward offset {0,1,2}. The blind lat gradient.
  Model B (slot+tip):  land in the slot after the decision; within it,
     higher-jito-tip competitors land before us (full slot coverage, tip-rank
     where shred tip known ~49%). Sweeps our tip {100k,1M,5M} lam.

  SLIPPAGE: realized entry slippage = landing_mid/decision_mid - 1 (Model B,
  1M-lam reference tip). Reports its distribution, the OUTCOME bucketed by
  entry slippage (do high-slip fills rocket or dump?), and a cap sweep (skip
  = revert at ~no cost with atomic bundles) to learn the accept/revert cut.

Saves data/exec_sim_result.json for accumulation + post-arming comparison.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

Q_SOL = 0.1
COST_BPS = 250.0
FEE_TX = 0.0015
TP = 0.50
OUR_TIPS = [100_000, 1_000_000, 5_000_000]
REF_TIP = 1_000_000   # reference tip for the slippage study


def buy_tokens(vs, vt, dsol):
    return vt - (vs * vt) / (vs + dsol)


def sell_sol(vs, vt, dtok):
    return vs - (vs * vt) / (vt + dtok)


def walk(entry_vs, entry_vt, path_from):
    q = Q_SOL * 1e9
    tok = buy_tokens(entry_vs, entry_vt, q)
    e_mid = entry_vs / entry_vt
    xvs, xvt = entry_vs, entry_vt
    for (_ts, _slot, vs, vt, _b, _t) in path_from:
        ret = (vs / vt) / e_mid - 1.0
        xvs, xvt = vs, vt
        if ret >= TP:
            break
    return sell_sol(xvs, xvt, tok) / q - 1.0 - COST_BPS / 1e4 - (FEE_TX * 2.0) / Q_SOL


def model_a(f, lat):
    if not f:
        return None
    j = min(lat, len(f) - 1)
    return walk(f[j][2], f[j][3], f[j + 1:])


def land_index_b(dec, f, our_tip):
    """index in f where our buy lands under the slot-aware + tip-rank model."""
    dslot = dec["slot"]
    i = 0
    while i < len(f) and f[i][1] <= dslot:   # skip rest of decision slot
        i += 1
    if i >= len(f):
        return None
    land_slot = f[i][1]
    while i < len(f) and f[i][1] == land_slot:
        tip = f[i][5]
        if tip is not None and tip > our_tip:   # higher-tip competitor lands first
            i += 1
        else:
            break
    return i if i < len(f) else None


def main():
    fwd = pickle.load(open(ROOT / "data/exec_sim_fwd_k3v03.pkl", "rb"))
    clf = pickle.load(open(ROOT / "bot_artifacts_k3v03_final/entry_model.pkl", "rb"))
    spec = json.loads((ROOT / "bot_artifacts_k3v03_final/model_spec.json").read_text())
    feats = spec["entry"]["features"]
    lm = pd.read_parquet(ROOT / "data/live_matched_k3v03_all2.parquet")
    lm["score"] = clf.predict_proba(lm[feats].values)[:, 1]
    fires = sorted(set(lm[lm.score >= 0.50].mint) & set(fwd))
    print(f"fires with forward data: {len(fires)}\n")

    rows = []
    for m in fires:
        dec, f = fwd[m]["decision"], fwd[m]["fwd"]
        rec = {"mint": m, "contest": sum(1 for (_t, s, _v, _w, b, _tp) in f if b and s <= dec["slot"] + 1)}
        for lat in (0, 1, 2):
            rec[f"A_lat{lat}"] = model_a(f, lat)
        for tip in OUR_TIPS:
            li = land_index_b(dec, f, tip)
            rec[f"B_tip{int(tip/1000)}k"] = walk(f[li][2], f[li][3], f[li + 1:]) if li is not None else None
        # realized entry slippage at the reference tip
        li = land_index_b(dec, f, REF_TIP)
        if li is not None:
            dmid = dec["vsol"] / dec["vtok"]
            lmid = f[li][2] / f[li][3]
            rec["entry_slip"] = lmid / dmid - 1.0
            rec["net_ref"] = rec[f"B_tip{int(REF_TIP/1000)}k"]
        rows.append(rec)
    df = pd.DataFrame(rows)

    def summ(col):
        v = df[col].dropna()
        return f"n={len(v):3d} mean={v.mean():+.3f} med={v.median():+.3f} win={(v>0).mean():.0%}" if len(v) else "n=0"

    print("=== Model A (incumbent fixed-latency) ===")
    for lat in (0, 1, 2):
        print(f"  lat={lat}:  {summ(f'A_lat{lat}')}")
    print("=== Model B (slot-aware + tip-rank) ===")
    for tip in OUR_TIPS:
        print(f"  our_tip={int(tip/1000)}k:  {summ(f'B_tip{int(tip/1000)}k')}")

    s = df.dropna(subset=["entry_slip"])
    sl = s.entry_slip
    print(f"\n=== REALIZED ENTRY SLIPPAGE (Model B @ {int(REF_TIP/1000)}k tip, n={len(s)}) ===")
    print(f"  distribution: p10={sl.quantile(.1):+.1%} p50={sl.median():+.1%} "
          f"p90={sl.quantile(.9):+.1%} p99={sl.quantile(.99):+.1%}  (positive = price ran up before fill)")
    s = s.assign(slipbin=pd.cut(sl, [-1, 0, .02, .05, .10, .25, 10]))
    print("\n  outcome by entry-slip bucket (does high slip rocket or dump?):")
    g = s.groupby("slipbin", observed=True).agg(n=("net_ref", "size"), mean_net=("net_ref", "mean"),
                                                win=("net_ref", lambda x: (x > 0).mean()))
    print(g.to_string(float_format=lambda v: f"{v:+.3f}"))

    print("\n  slippage-cap sweep (skip=revert at ~0 cost; total net over all fires):")
    print(f"  {'cap':>8s} {'take_rate':>10s} {'mean_taken':>11s} {'TOTAL_net':>10s}")
    base_total = s.net_ref.sum()
    for cap in (10.0, 0.25, 0.15, 0.10, 0.05, 0.02):
        taken = s[s.entry_slip <= cap]
        capn = "none" if cap > 1 else f"{cap:.0%}"
        print(f"  {capn:>8s} {len(taken)/len(s):10.0%} {taken.net_ref.mean():+11.3f} {taken.net_ref.sum():+10.2f}")
    print(f"  read: if a tighter cap RAISES TOTAL_net it is removing net-negative high-slip fills;"
          f"\n  if TOTAL_net only falls, slippage is not the discriminator and a loose cap is right.")

    out = ROOT / "data/exec_sim_result.json"
    out.write_text(df.to_json(orient="records"))
    print(f"\nsaved {out} ({len(df)} fires)")


if __name__ == "__main__":
    main()
