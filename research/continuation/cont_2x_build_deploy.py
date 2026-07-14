#!/usr/bin/env python3
"""Build the deploy panel + reputation seed USING THE SHARED cont_aug_features module
(the bot's exact serving code). Diffing this panel's SHRED/REP columns against the
inline-built cont_2x_aug2_panel.jsonl proves parity (shared module == offline).

Single chrono pass over the x2 intent archive; mint resolution + SHRED + as-of REP all
via cont_aug_features. Emits cont_2x_deploy_panel.jsonl + cont2x_shredrep_seed.json.
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, sys
from pathlib import Path

from . import cont_aug_features as caf

ROOT = os.getenv("PUMPFUN_ROOT", str(Path(__file__).resolve().parents[2]))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default=f"{ROOT}/bot_data/cont_2x_panel_*.jsonl")
    ap.add_argument("--intent-glob", default="/mnt/storagebox/backup/archive/intent_capture/x2/intent-*.jsonl.gz")
    ap.add_argument("--since", default="20260618")
    ap.add_argument("--out", default=f"{ROOT}/bot_data/cont_2x_deploy_panel.jsonl")
    ap.add_argument("--seed", default=f"{ROOT}/bot_data/cont2x_shredrep_seed.json")
    ap.add_argument("--progress-every", type=int, default=5000000)
    return ap.parse_args()


def main():
    a = parse_args()
    cross = {}
    for fn in sorted(glob.glob(a.panels)):
        for ln in open(fn):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            m = r.get("mint")
            if m and m not in cross:
                cross[m] = {"cross_slot": int(r.get("cross_slot") or 0), "row": r, "acc": caf.ShredAccum()}
    sys.stderr.write(f"[deploy-build] {len(cross)} crosses\n"); sys.stderr.flush()

    files = [f for f in sorted(glob.glob(a.intent_glob))
             if os.path.basename(f).split("intent-")[1][:8] >= a.since]
    n_seen = n_used = 0
    for fi, fn in enumerate(files):
        try:
            fh = gzip.open(fn, "rt", errors="replace")
        except Exception as e:
            sys.stderr.write(f"[deploy-build] open fail {fn}: {e}\n"); continue
        for ln in fh:
            n_seen += 1
            if (n_seen % a.progress_every) == 0:
                sys.stderr.write(f"[deploy-build] seen={n_seen} used={n_used} f={fi+1}/{len(files)}\n"); sys.stderr.flush()
            if '"type":"buy' not in ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            mint = caf.mint_of(r)
            if not mint:
                continue
            mm = cross.get(mint)
            if mm is None:
                continue
            slot = int(r.get("slot") or 0)
            if slot > mm["cross_slot"]:
                continue
            mm["acc"].add(slot, r.get("signer"), r.get("priority_fee_micro"), r.get("jito_tip_lam"))
            n_used += 1
        fh.close()
    sys.stderr.write(f"[deploy-build] intent pass done seen={n_seen} used={n_used}\n"); sys.stderr.flush()

    order = sorted(cross.values(), key=lambda mm: (mm["cross_slot"], mm["row"].get("cross_t", 0)))
    rep = caf.ShredRep()
    out = open(a.out, "w"); n_emit = 0
    for mm in order:
        signers = mm["acc"].signer_list()
        row = dict(mm["row"])
        row.update(mm["acc"].features(mm["cross_slot"]))
        row.update(rep.features(signers))
        out.write(json.dumps(row, separators=(",", ":")) + "\n"); n_emit += 1
        rep.update(signers, int(mm["row"].get("y", 0)))
    out.close()
    rep.save(a.seed)
    sys.stderr.write(f"[deploy-build] DONE emitted={n_emit} signers={len(rep.rep)} out={a.out} seed={a.seed}\n"); sys.stderr.flush()


if __name__ == "__main__":
    main()
