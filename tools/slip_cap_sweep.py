"""Under the FILL-ANCHORED exit, sweep the buy slippage cap and compare to a
post-fill slip-kill. For each exec_sim fire: enter at the realistic slipped
landing, TP/stop measured from the FILL (the shipped fix). If entry_slip > cap,
the buy reverts instead (net = -REVERT_COST, like the phantom). Also models a
slip-kill: fills but exits immediately at ~fill (net ~ -fees) if slip > kill."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
Q = 0.1 * 1e9
COST_BPS, FEE_TX, TP, STOP = 250.0, 0.0015, 0.50, -0.30
REF_TIP = 1_000_000
REVERT_COST = -0.006   # ~0.0003 SOL fee on a 0.05 bet, as a fraction of the 0.1-notional q
KILL_NET = -0.07       # fill then immediate market exit at ~fill: lose round-trip cost+slip (~ -0.0035/0.05)

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
        dmid, lmid = dec["vsol"] / dec["vtok"], lvs / lvt
        slip = lmid / dmid - 1.0
        net = walk(lvs, lvt, lmid, f[li + 1:])   # fill-anchored net if we DO hold
        rows.append((slip, net))
    s = np.array([r[0] for r in rows]); n = np.array([r[1] for r in rows])
    N = len(s)
    print(f"fires={N}  slip p50={np.median(s):+.0%} p90={np.percentile(s,90):+.0%}\n")
    print("BUY-CAP sweep (slip>cap -> revert): fill-anchored exit on kept fills")
    print(f"  {'cap(x)':>7} {'kept':>5} {'mean/fire':>10} {'total':>8} {'win%':>6}")
    for cap in (0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 99.0):
        kept = s <= cap
        net_all = np.where(kept, n, REVERT_COST)
        print(f"  {(1+cap):>6.2f}x {kept.sum():>5} {net_all.mean():>+10.3f} {net_all.sum():>+8.2f} {(net_all>0).mean():>5.0%}")
    print("\nSLIP-KILL sweep (slip>kill -> fill then immediate exit at ~fill):")
    print(f"  {'kill(x)':>7} {'mean/fire':>10} {'total':>8} {'win%':>6}")
    for kill in (0.4, 0.5, 1.0, 2.0, 99.0):
        net_all = np.where(s <= kill, n, KILL_NET)
        print(f"  {(1+kill):>6.2f}x {net_all.mean():>+10.3f} {net_all.sum():>+8.2f} {(net_all>0).mean():>5.0%}")

if __name__ == "__main__":
    main()
