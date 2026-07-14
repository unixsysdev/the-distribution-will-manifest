"""Offline A/B for the fill-anchored exit fix (2026-06-12), on the exec_sim
forward data. Both arms enter at the SAME realistic slipped landing price; they
differ only in where the +50% take-profit / -30% stop is measured from:

  BROKEN (current live): ret = mid/DECISION_mid - 1   (mirage anchor)
  FIX  (fill-anchored) : ret = mid/LANDING_mid  - 1   (our real fill)

This is the exact change shipped to shadow_harness. If FIX beats BROKEN (and is
positive), the live fix reproduces exec_sim's positive Model B logic."""
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
Q = 0.1 * 1e9
COST_BPS, FEE_TX, TP, STOP = 250.0, 0.0015, 0.50, -0.30
REF_TIP = 1_000_000


def buy_tokens(vs, vt, d): return vt - (vs * vt) / (vs + d)
def sell_sol(vs, vt, dt): return vs - (vs * vt) / (vt + dt)


def land_index_b(dec, f, our_tip):
    dslot = dec["slot"]; i = 0
    while i < len(f) and f[i][1] <= dslot: i += 1
    if i >= len(f): return None
    ls = f[i][1]
    while i < len(f) and f[i][1] == ls:
        tip = f[i][5]
        if tip is not None and tip > our_tip: i += 1
        else: break
    return i if i < len(f) else None


def walk(entry_vs, entry_vt, anchor_mid, path):
    """Enter at entry reserves (pay the slipped price); TP/stop measured vs
    anchor_mid. Returns net fraction."""
    tok = buy_tokens(entry_vs, entry_vt, Q)
    xvs, xvt = entry_vs, entry_vt
    for (_ts, _slot, vs, vt, _b, _t) in path:
        xvs, xvt = vs, vt
        ret = (vs / vt) / anchor_mid - 1.0
        if ret >= TP or ret <= STOP:
            break
    return sell_sol(xvs, xvt, tok) / Q - 1.0 - COST_BPS / 1e4 - (FEE_TX * 2) / (Q / 1e9)


def main():
    fwd = pickle.load(open(ROOT / "data/exec_sim_fwd_k3v03.pkl", "rb"))
    clf = pickle.load(open(ROOT / "bot_artifacts_k3v03_final/entry_model.pkl", "rb"))
    spec = json.loads((ROOT / "bot_artifacts_k3v03_final/model_spec.json").read_text())
    feats = spec["entry"]["features"]
    lm = pd.read_parquet(ROOT / "data/live_matched_k3v03_all2.parquet")
    lm["score"] = clf.predict_proba(lm[feats].values)[:, 1]
    fires = sorted(set(lm[lm.score >= 0.50].mint) & set(fwd))

    broken, fixed = [], []
    for m in fires:
        dec, f = fwd[m]["decision"], fwd[m]["fwd"]
        li = land_index_b(dec, f, REF_TIP)
        if li is None: continue
        land_vs, land_vt = f[li][2], f[li][3]
        dmid = dec["vsol"] / dec["vtok"]
        lmid = land_vs / land_vt
        path = f[li + 1:]
        broken.append(walk(land_vs, land_vt, dmid, path))   # TP from decision (mirage)
        fixed.append(walk(land_vs, land_vt, lmid, path))     # TP from fill (the fix)

    b, x = np.array(broken), np.array(fixed)
    def s(a): return f"n={len(a):3d} mean={a.mean():+.3f} med={np.median(a):+.3f} win={(a>0).mean():.0%} total={a.sum():+.2f}"
    print("entry = realistic slipped landing for BOTH; only the TP/stop anchor differs\n")
    print("BROKEN (TP from DECISION/mirage, = current live):", s(b))
    print("FIXED  (TP from FILL/landing,    = the fix)     :", s(x))
    print(f"\ndelta (fix - broken): mean {x.mean()-b.mean():+.3f}/fire  total {x.sum()-b.sum():+.2f}")


if __name__ == "__main__":
    main()
