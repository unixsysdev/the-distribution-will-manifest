"""tip_model.py — model the Jito tip by ACTIVITY (competition) from captured buy
intents, to replace the fixed base*2/base*4 + p90*1.25/p90*2.0 multipliers.
Activity = # of competing buy intents on the SAME mint in the prior 2s window
(what the bot sees at decision). Reports the competing tip/priority-fee
distribution per activity bucket -> the tip to bid to be above competitors."""
import gzip, glob, json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("/root/the-distribution-will-manifest")
files = sorted(glob.glob(str(ROOT / "shred_bot/intent_capture/*.jsonl.gz")))[-2:]
buys = []
for f in files:
    with gzip.open(f, "rt") as fh:
        for ln in fh:
            try: r = json.loads(ln)
            except Exception: continue
            if not r.get("is_buy"): continue
            buys.append((int(r.get("recv_ns", 0)), r.get("mint"), int(r.get("slot", 0)),
                         int(r.get("priority_fee_micro", 0) or 0), int(r.get("jito_tip_lam", 0) or 0)))
print(f"buy intents: {len(buys):,} from {len(files)} files")
jt = np.array([b[4] for b in buys]); pf = np.array([b[3] for b in buys])
print(f"jito_tip_lam>0 coverage: {(jt > 0).mean():.1%} ({int((jt > 0).sum()):,})   priority_fee>0: {(pf > 0).mean():.1%}")

def pcts(a, label):
    a = a[a > 0]
    if len(a) < 5: print(f"  {label}: too sparse (n={len(a)})"); return
    print(f"  {label} (n>0={len(a):,}): p25={np.percentile(a,25):.0f} p50={np.percentile(a,50):.0f} "
          f"p75={np.percentile(a,75):.0f} p90={np.percentile(a,90):.0f} p99={np.percentile(a,99):.0f}")
print("OVERALL:")
pcts(jt, "jito_tip_lam"); pcts(pf, "priority_fee_micro")

# contention: competitors on same mint within prior 2s
bym = defaultdict(list)
for b in buys: bym[b[1]].append(b)
rows = []
for m, lst in bym.items():
    lst.sort()
    rns = [x[0] for x in lst]; j = 0
    for i in range(len(lst)):
        while rns[i] - rns[j] > 2_000_000_000: j += 1
        rows.append((i - j, lst[i][4], lst[i][3]))
rows = np.array(rows)
c = rows[:, 0]
print(f"\ncontention (competing buys same mint, prior 2s): mean={c.mean():.2f} "
      f"p50={np.percentile(c,50):.0f} p90={np.percentile(c,90):.0f} max={c.max():.0f}")
for thr in [1, 2, 5, 10]:
    print(f"  >= {thr} competitors: {(c >= thr).mean():.1%}")

print("\n=== competing JITO TIP (lam) by activity bucket -> bid above to land ===")
for lo, hi, lab in [(0, 1, "calm (0)"), (1, 3, "low (1-2)"), (3, 8, "med (3-7)"), (8, 1e9, "hot (8+)")]:
    mask = (c >= lo) & (c < hi)
    sj = rows[mask, 1]; sj = sj[sj > 0]
    sp = rows[mask, 2]; sp = sp[sp > 0]
    js = (f"jito_tip p75={np.percentile(sj,75):.0f} p90={np.percentile(sj,90):.0f} p95={np.percentile(sj,95):.0f}"
          if len(sj) > 10 else f"jito_tip sparse(n={len(sj)})")
    ps = (f"prio_fee p90={np.percentile(sp,90):.0f}" if len(sp) > 10 else "prio sparse")
    print(f"  {lab:10s} n={int(mask.sum()):>8,}  {js}  | {ps}")
print("\nMODEL: tip(activity) = p90/p95 of competing jito tips at the observed contention")
print("(replaces base*2/base*4). Caveat: jito_tip coverage is partial -> these are LOWER bounds.")
