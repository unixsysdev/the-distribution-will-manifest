"""The structural question: live we can only FILL the tokens that don't graduate
before our buy lands. The graduating rockets (high vsK at decision) revert. If
those were exec_sim's winners, our realizable (catchable-only) edge is below the
+0.087/fire the sim showed assuming-we-fill-everything.

Split the 157 fires by decision vsK (live separator: fills 50-52, grad-reverts
59-70 SOL) and compare the fill-anchored net of each subset."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

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
        if f[i][5] is not None and f[i][5] > our_tip: i += 1
        else: break
    return i if i < len(f) else None

def walk(evs, evt, anchor, path):
    tok = buy_tokens(evs, evt, Q); xvs, xvt = evs, evt
    for (_ts, _s, vs, vt, _b, _t) in path:
        xvs, xvt = vs, vt
        r = (vs / vt) / anchor - 1.0
        if r >= TP or r <= STOP: break
    return sell_sol(xvs, xvt, tok) / Q - 1.0 - COST_BPS / 1e4 - (FEE_TX * 2) / (Q / 1e9)

def main():
    fwd = pickle.load(open(ROOT / "data/exec_sim_fwd_k3v03.pkl", "rb"))
    clf = pickle.load(open(ROOT / "bot_artifacts_k3v03_final/entry_model.pkl", "rb"))
    spec = json.loads((ROOT / "bot_artifacts_k3v03_final/model_spec.json").read_text())
    feats = spec["entry"]["features"]
    lm = pd.read_parquet(ROOT / "data/live_matched_k3v03_all2.parquet")
    lm["score"] = clf.predict_proba(lm[feats].values)[:, 1]
    fires = sorted(set(lm[lm.score >= 0.50].mint) & set(fwd))

    rows = []
    for m in fires:
        dec, f = fwd[m]["decision"], fwd[m]["fwd"]
        li = land_index_b(dec, f, REF_TIP)
        if li is None: continue
        lvs, lvt = f[li][2], f[li][3]
        net = walk(lvs, lvt, lvs / lvt, f[li + 1:])
        # peak vsol reached in the path = proxy for "graduated" (curve ran to completion)
        peak_vs = max((x[2] for x in f), default=dec["vsol"]) / 1e9
        rows.append((dec["vsol"] / 1e9, peak_vs, net))
    df = pd.DataFrame(rows, columns=["vsK", "peak_vs", "net"])
    print(f"fires={len(df)}  decision vsK: p50={df.vsK.median():.0f} p90={df.vsK.quantile(.9):.0f}")
    def s(d): return f"n={len(d):3d} mean_net={d.net.mean():+.3f} med={d.net.median():+.3f} win={(d.net>0).mean():.0%}"
    print("\nsplit by decision vsK (live: fills<=52, grad-reverts>=59):")
    for lo, hi, lab in [(0,55,"CATCHABLE vsK<55"),(55,80,"WOULD-GRAD 55-80"),(80,1e9,"vsK>=80")]:
        d = df[(df.vsK>=lo)&(df.vsK<hi)]
        if len(d): print(f"  {lab:18}: {s(d)}")
    print("\nsplit by peak vsol reached (>=110 ~ graduated):")
    for lo, hi, lab in [(0,90,"never near grad"),(90,110,"approached"),(110,1e9,"GRADUATED ~peak>=110")]:
        d = df[(df.peak_vs>=lo)&(df.peak_vs<hi)]
        if len(d): print(f"  {lab:22}: {s(d)}")

if __name__ == "__main__":
    main()
