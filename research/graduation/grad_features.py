"""Compare decision-time features of graduate-before-fill reverts vs catchable fills."""
import json
ROOT = "/root/the-distribution-will-manifest"
ARM = 1781221200
OUT = {"H7M1Ldg": "FILL -0.0098", "2E1KarF": "FILL -0.0054", "EgrpNax8": "REV programid",
       "XTLLGEK": "REV grad", "1o6aEh6": "REV grad", "HPLF4Vw": "REV grad"}

def short(m):
    if not m:
        return None
    for k in OUT:
        if m.startswith(k):
            return k
    return None

print("{:9} {:14} {:6} {:8} {:7} {:9} {:8} {:9}".format(
    "fire", "outcome", "score", "cum_buy", "n_ready", "trades/s", "win_ret", "vsK_SOL"))
seen = set()
for ln in open(ROOT + "/bot_data/shadow_run.jsonl"):
    try:
        r = json.loads(ln)
    except Exception:
        continue
    if r.get("t", 0) < ARM or r.get("kind") != "entry_decision" or not r.get("fire"):
        continue
    k = short(r.get("mint"))
    if not k or k in seen:
        continue
    seen.add(k)
    f = r.get("features", {})
    tps = f.get("trades_per_sec")
    wr = f.get("win_ret")
    vsK = r.get("vsK")
    print("{:9} {:14} {:<6} {:<8} {:<7} {:<9} {:<8} {:<9}".format(
        k, OUT[k], round(r["score"], 3), round(r.get("cum_buy_sol", 0), 1),
        r.get("n_at_ready"),
        round(tps, 2) if tps is not None else "?",
        round(wr, 2) if wr is not None else "?",
        round(vsK / 1e9, 1) if vsK else "?"))
