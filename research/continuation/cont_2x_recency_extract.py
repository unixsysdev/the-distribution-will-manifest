#!/usr/bin/env python3
"""Per-cross shred RECENCY extraction: for each 2x cross, the gap to the most-recent buy
intent + buy counts at multiple windows (1s/5s/20s/40s, slot-based, ~0.4s/slot). Joins to
the deploy panel for outcomes -> graded recency curve. Single chrono pass over the intent archive."""
from __future__ import annotations
import argparse, glob, gzip, json, os, sys
from pathlib import Path

from . import cont_aug_features as caf

ROOT = os.getenv("PUMPFUN_ROOT", str(Path(__file__).resolve().parents[2]))

WIN = {"1s": 3, "5s": 13, "20s": 50, "40s": 100}   # seconds -> slots (~0.4s/slot)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default=f"{ROOT}/bot_data/cont_2x_panel_*.jsonl")
    ap.add_argument("--intent-glob", default="/mnt/storagebox/backup/archive/intent_capture/x2/intent-*.jsonl.gz")
    ap.add_argument("--since", default="20260618")
    ap.add_argument("--out", default=f"{ROOT}/bot_data/cont_2x_recency.jsonl")
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
            r = json.loads(ln); m = r.get("mint")
            if m and m not in cross:
                cross[m] = {"cs": int(r.get("cross_slot") or 0), "sc": {}}   # sc = slot->count of buys
    sys.stderr.write(f"[recency] {len(cross)} crosses\n"); sys.stderr.flush()
    files = [f for f in sorted(glob.glob(a.intent_glob))
             if os.path.basename(f).split("intent-")[1][:8] >= a.since]
    seen = used = 0
    for fi, fn in enumerate(files):
        try: fh = gzip.open(fn, "rt", errors="replace")
        except Exception: continue
        for ln in fh:
            seen += 1
            if seen % a.progress_every == 0:
                sys.stderr.write(f"[recency] seen={seen} used={used} f={fi+1}/{len(files)}\n"); sys.stderr.flush()
            if '"type":"buy' not in ln:
                continue
            try: r = json.loads(ln)
            except Exception: continue
            mint = caf.mint_of(r)
            if not mint: continue
            mm = cross.get(mint)
            if mm is None: continue
            slot = int(r.get("slot") or 0)
            if slot > mm["cs"]: continue
            mm["sc"][slot] = mm["sc"].get(slot, 0) + 1
            used += 1
        fh.close()
    out = open(a.out, "w"); ne = 0
    for mint, mm in cross.items():
        cs = mm["cs"]; sc = mm["sc"]
        row = {"mint": mint, "n_total": sum(sc.values())}
        if sc:
            row["last_gap_s"] = round((cs - max(sc)) * 0.4, 2)
        else:
            row["last_gap_s"] = -1.0
        for nm, w in WIN.items():
            row[f"n_{nm}"] = sum(sc.get(s, 0) for s in range(cs - w, cs + 1))
        out.write(json.dumps(row, separators=(",", ":")) + "\n"); ne += 1
    out.close()
    sys.stderr.write(f"[recency] DONE rows={ne} seen={seen} used={used} out={a.out}\n"); sys.stderr.flush()

if __name__ == "__main__":
    main()
