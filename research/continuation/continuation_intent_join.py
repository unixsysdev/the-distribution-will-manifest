"""Join CAUSAL intent features (shred capture) onto each 2x cross in cont_rich_panel. For each
cross (mint, cross_t) aggregate intents with the same mint in [cross_t - W, cross_t] (strictly
causal, no lookahead). Mint is the '...pump'-suffixed account in ix_accounts (robust across the
v1/v2/sol_in layouts). Writes cont_rich_intent_panel.jsonl (panel + int_* columns on crosses).
"""
import argparse, glob, gzip, json, sys
from collections import defaultdict
import numpy as np

ROOT = "/root/the-distribution-will-manifest"
ap = argparse.ArgumentParser()
ap.add_argument("--panel", default=f"{ROOT}/bot_data/cont_rich_panel.jsonl")
ap.add_argument("--out", default=f"{ROOT}/bot_data/cont_rich_intent_panel.jsonl")
ap.add_argument("--window", type=float, default=3.0)
args = ap.parse_args()
W = args.window
BUY = {"buy", "buy_quote", "buy_sol_in"}; SELL = {"sell"}

recs = [json.loads(l) for l in open(args.panel)]
cross_t = {r["mint"]: r["t"] for r in recs if r["kind"] == "cross"}
mints = set(cross_t)
if not mints:
    print("no crosses in panel"); sys.exit(0)
tmin = min(cross_t.values()) - W - 5; tmax = max(cross_t.values()) + 5


def openf(f):
    return gzip.open(f, "rt") if f.endswith(".gz") else open(f)


def mint_of(accs):
    for a in accs:
        if a.endswith("pump"):
            return a
    return None


by_mint = defaultdict(list); nseen = 0
for f in sorted(glob.glob(f"{ROOT}/shred_bot/intent_capture/intent-*.jsonl*")):
    try:
        for line in openf(f):
            try: e = json.loads(line)
            except Exception: continue
            t = e.get("type")
            if t not in BUY and t not in SELL:
                continue
            rn = e.get("recv_ns")
            if not rn:
                continue
            rs = rn / 1e9
            if rs < tmin or rs > tmax:
                continue
            m = mint_of(e.get("ix_accounts", []))
            if m is None or m not in mints:
                continue
            by_mint[m].append((rs, 1 if t in BUY else 0, e.get("priority_fee_micro", 0) or 0,
                               e.get("jito_tip_lam", 0) or 0, e.get("signer", "")))
            nseen += 1
    except Exception:
        continue
for m in by_mint:
    by_mint[m].sort()
print(f"indexed {nseen} causal intents for {len(by_mint)}/{len(mints)} cross mints", flush=True)


def feats(m, ct):
    win = [x for x in by_mint.get(m, []) if ct - W <= x[0] <= ct]
    if not win:
        return {"int_n": 0, "int_buy": 0, "int_sell": 0, "int_buy_frac": 0.0, "int_uniq": 0,
                "int_prio_p90": 0.0, "int_tip_rate": 0.0, "int_tip_p90": 0.0}
    nb = sum(x[1] for x in win); prios = [x[2] for x in win]; tips = [x[3] for x in win]
    return {"int_n": len(win), "int_buy": nb, "int_sell": len(win) - nb, "int_buy_frac": nb / len(win),
            "int_uniq": len(set(x[4] for x in win)), "int_prio_p90": float(np.percentile(prios, 90)),
            "int_tip_rate": sum(1 for x in tips if x > 0) / len(tips), "int_tip_p90": float(np.percentile(tips, 90))}


matched = 0
with open(args.out, "w") as o:
    for r in recs:
        if r["kind"] == "cross":
            fe = feats(r["mint"], r["t"]); r = dict(r, **fe)
            if fe["int_n"] > 0:
                matched += 1
        o.write(json.dumps(r) + "\n")
print(f"wrote {args.out}: {matched}/{len(cross_t)} crosses had >=1 causal intent", flush=True)
