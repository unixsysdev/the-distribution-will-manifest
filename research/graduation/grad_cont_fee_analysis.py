#!/usr/bin/env python3
"""What does it COST to land a buy on a graduated (PumpSwap) coin? -> can we compete + stay profitable?

Scans recent grpc_capture for PumpSwap.BuyEvent rows (each carries priority_fee_micro, cu, jito_tip_lam)
and reports the priority-fee + Jito-tip distribution in SOL. Compares the cost-to-compete against the
per-trade gross profit (bet * net_frac). Tips/priority are FIXED SOL per tx -> they hurt SMALL bets most.

priority_fee_sol = priority_fee_micro(micro-lamports/CU) * cu / 1e6 / 1e9
jito_tip_sol     = jito_tip_lam / 1e9

Usage: ./venv/bin/python grad_cont_fee_analysis.py [n_files] [bet_sol] [net_frac]
"""
import json, sys, gzip, glob
import numpy as np

STDIN = (len(sys.argv) > 1 and sys.argv[1] == "--stdin")
a = sys.argv[2:] if STDIN else sys.argv[1:]
BET = float(a[0]) if len(a) > 0 else 0.5
NETF = float(a[1]) if len(a) > 1 else 0.45   # gross net/trade as fraction of position
CAP = 1500000

def lines():
    if STDIN:
        for ln in sys.stdin:
            yield ln
    else:
        for fn in sorted(glob.glob("/root/the-distribution-will-manifest/grpc_capture/*.jsonl.gz"))[-8:]:
            for ln in gzip.open(fn, "rt"):
                yield ln

prio = []; tip = []; tipped = 0; nbuy = 0; prio_pos = 0
for ln in lines():
    if "PumpSwap.BuyEvent" not in ln:
        continue
    try: r = json.loads(ln)
    except Exception: continue
    if r.get("event") != "PumpSwap.BuyEvent":
        continue
    nbuy += 1
    pm = float(r.get("priority_fee_micro") or 0); cu = float(r.get("cu") or 0)
    prio.append(pm * cu / 1e6 / 1e9)
    if pm > 0: prio_pos += 1
    jt = r.get("jito_tip_lam")
    if jt:
        tip.append(float(jt) / 1e9); tipped += 1
    if nbuy >= CAP:
        break

prio = np.array(prio); tip = np.array(tip) if tip else np.array([0.0])
total_p50 = np.quantile(prio, .5) + (np.quantile(tip, .5) if len(tip) else 0)
print(f"source={'stdin' if STDIN else 'files'}  PumpSwap buys sampled={nbuy}")
print(f"  pay priority>0: {prio_pos/max(nbuy,1):.2f}   pay jito tip>0: {tipped/max(nbuy,1):.2f}")
def q(a, name, unit="SOL"):
    print(f"  {name:18s} p50={np.quantile(a,.5):.6f} p75={np.quantile(a,.75):.6f} "
          f"p90={np.quantile(a,.9):.6f} p99={np.quantile(a,.99):.6f} {unit}")
q(prio, "priority_fee_sol")
if tipped: q(tip, "jito_tip_sol(tippers)")
# cost to BEAT THE FIELD ~ p90 priority + p90 tip(among tippers); round trip = buy+sell ~2x
beat = np.quantile(prio, .9) + (np.quantile(tip, .9) if tipped else 0.0)
typ = np.quantile(prio, .75) + (np.quantile(tip, .75) if tipped else 0.0)
gross = BET * NETF
print(f"\n  --- compete & profit?  (bet={BET} SOL, gross net/trade=+{gross:.3f} SOL) ---")
for lbl, c in (("typical (p75)", typ), ("beat-the-field (p90)", beat)):
    rt = 2 * c   # buy + sell both pay
    print(f"  {lbl:22s} entry cost~{c:.5f} SOL  round-trip~{rt:.5f} SOL  "
          f"= {100*rt/gross:.1f}% of gross profit  -> net~{gross-rt:+.4f} SOL/trade")
print(f"  (fixed-SOL costs: at bet={BET} round-trip p90 is {100*2*beat/BET:.2f}% of position; "
      f"halve the bet -> double that %.)")
