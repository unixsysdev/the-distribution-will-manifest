"""Ad-hoc: dump full detail for the night's live fires (decision, submit, recon)."""
import json, time, sys
ROOT = "/root/the-distribution-will-manifest"
PREFIX = {"H7M1Ldg": "fire1", "2E1KarF": "fire2", "EgrpNax8": "phantom"}

def short(m):
    if not m:
        return None
    for k, v in PREFIX.items():
        if m.startswith(k):
            return v
    return None

def t(x):
    return time.strftime("%H:%M:%S", time.localtime(x))

dec = {}
for ln in open(ROOT + "/bot_data/shadow_run.jsonl"):
    try:
        r = json.loads(ln)
    except Exception:
        continue
    k = short(r.get("mint"))
    if not k:
        continue
    if r.get("kind") == "entry_decision" and r.get("fire"):
        evs, evt = r.get("entry_vs"), r.get("entry_vt")
        dec.setdefault(k, {}).update(
            score=round(r["score"], 4), cum_buy=round(r.get("cum_buy_sol", 0), 2),
            dec_mid=(evs / evt if evs and evt else None), fire_t=t(r["t"]))
    if r.get("kind") == "front_run_tip_bump":
        dec.setdefault(k, {})["tip_bump"] = dict(
            tier=r.get("tier"), override=r.get("override_tip_lam"), p90=r.get("p90_visible_tip_lam"))

print("=== ENTRY DECISIONS ===")
for k in ("fire1", "fire2", "phantom"):
    print(" ", k, dec.get(k))

print("\n=== BROKER SUBMISSIONS ===")
for ln in open(ROOT + "/logs/broker_jito.jsonl"):
    try:
        r = json.loads(ln)
    except Exception:
        continue
    k = short(r.get("mint"))
    if not k:
        continue
    print("  {} {} {}/{} slot={} bh_slot={} maxcost={} tip={} asm_ms={} err={}".format(
        k, t(r["t"]), r.get("op"), r.get("status"), r.get("slot"), r.get("bh_slot"),
        r.get("max_sol_cost_lam"), r.get("tip_lam"), r.get("asm_ms"), str(r.get("err", ""))[:70]))

print("\n=== RECON (landed/fill/failed/retry) ===")
for ln in open(ROOT + "/logs/broker_recon.jsonl"):
    try:
        r = json.loads(ln)
    except Exception:
        continue
    k = short(r.get("mint"))
    if not k:
        continue
    if r.get("kind") in ("landed", "fill", "failed", "sell_will_retry", "sell_retry"):
        print("  {} {} {} landed_slot={} slot_gap={} act_tok={} act_sol={} fee={} reason={} src={}".format(
            k, r.get("kind"), r.get("op"), r.get("landed_slot"), r.get("slot_gap"),
            r.get("actual_tok_delta"), r.get("actual_sol_delta_lam"), r.get("fee_lam"),
            str(r.get("reason", ""))[:40], r.get("source")))
