#!/usr/bin/env python3
"""Augment the 2x-continuation panel with FULL-COVERAGE shred features + as-of
shred-signer (whale/bot) REPUTATION, then emit a joined panel for retraining.

Sources (all on sol, cheap — no firehose re-decode):
  - crosses: bot_data/cont_2x_panel_*.jsonl  (mint, cross_slot, cross_t, y, RICH...)
  - shred intents: /mnt/storagebox/backup/archive/intent_capture/x2/intent-*.jsonl.gz

Mint resolution (settled empirically): buy -> mint field / accs[2];
buy_sol_in & buy_quote -> accs[0] (96%/79% membership in dense known-mint set).
Signer = the `signer` field (fee payer = operating wallet/bot).

Causality: every shred record is slot-aligned (slot <= cross_slot) — clock-skew
immune (cross_slot is chain-canonical; intent recv_ns is x2's wall clock). Wallet
reputation is as-of: a cross's signer reputations come ONLY from prior crosses'
outcomes (processed in cross_slot order, updated AFTER scoring). Leakage-free,
and matches how a live ring-based reputation would accrue.
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, sys
from collections import defaultdict
from pathlib import Path

BUY_TYPES = {"buy", "buy_sol_in", "buy_quote"}
ROOT = os.getenv("PUMPFUN_ROOT", str(Path(__file__).resolve().parents[2]))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default=f"{ROOT}/bot_data/cont_2x_panel_*.jsonl")
    ap.add_argument("--intent-glob", default="/mnt/storagebox/backup/archive/intent_capture/x2/intent-*.jsonl.gz")
    ap.add_argument("--since", default="20260618")
    ap.add_argument("--out", default=f"{ROOT}/bot_data/cont_2x_aug2_panel.jsonl")
    ap.add_argument("--progress-every", type=int, default=5000000)
    return ap.parse_args()


def pctile(a, q):
    if not a:
        return 0.0
    s = sorted(a); return float(s[min(len(s) - 1, int(q / 100.0 * len(s)))])


class M:
    __slots__ = ("cross_slot", "row", "signers", "n", "slotcount", "prios", "ntip", "tipmax")
    def __init__(self, cross_slot, row):
        self.cross_slot = cross_slot; self.row = row
        self.signers = set(); self.n = 0
        self.slotcount = defaultdict(int); self.prios = []
        self.ntip = 0; self.tipmax = 0


def main():
    a = parse_args()
    # 1) load crosses
    cross = {}
    for fn in sorted(glob.glob(a.panels)):
        for ln in open(fn):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            m = r.get("mint")
            if m and m not in cross:   # one cross per mint; keep first
                cross[m] = M(int(r.get("cross_slot") or 0), r)
    sys.stderr.write(f"[shred-rep] {len(cross)} crosses loaded\n"); sys.stderr.flush()

    # 2) single chrono pass over intent files; accumulate buy intents (slot<=cross_slot) for cross mints
    files = [f for f in sorted(glob.glob(a.intent_glob))
             if os.path.basename(f).split("intent-")[1][:8] >= a.since]
    sys.stderr.write(f"[shred-rep] {len(files)} intent files since {a.since}\n"); sys.stderr.flush()
    n_seen = n_used = 0
    for fi, fn in enumerate(files):
        try:
            fh = gzip.open(fn, "rt", errors="replace")
        except Exception as e:
            sys.stderr.write(f"[shred-rep] open fail {fn}: {e}\n"); continue
        for ln in fh:
            n_seen += 1
            if (n_seen % a.progress_every) == 0:
                sys.stderr.write(f"[shred-rep] seen={n_seen} used={n_used} f={fi+1}/{len(files)}\n"); sys.stderr.flush()
            if '"type":"buy' not in ln:    # fast-path: buy / buy_sol_in / buy_quote
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            t = r.get("type")
            if t not in BUY_TYPES:
                continue
            accs = r.get("ix_accounts") or []
            if t == "buy":
                mint = r.get("mint") or (accs[2] if len(accs) > 2 else None)
            else:  # buy_sol_in, buy_quote
                mint = accs[0] if accs else None
            if not mint:
                continue
            mm = cross.get(mint)
            if mm is None:
                continue
            slot = int(r.get("slot") or 0)
            if slot > mm.cross_slot:     # causal: only intents at/before the cross
                continue
            mm.signers.add(r.get("signer"))
            mm.n += 1
            mm.slotcount[slot] += 1
            if len(mm.prios) < 5000:
                mm.prios.append(int(r.get("priority_fee_micro") or 0))
            tip = int(r.get("jito_tip_lam") or 0)
            if tip > 0:
                mm.ntip += 1
                if tip > mm.tipmax:
                    mm.tipmax = tip
            n_used += 1
        fh.close()
    sys.stderr.write(f"[shred-rep] intent pass done: seen={n_seen} used={n_used}\n"); sys.stderr.flush()

    # 3) shred-window flow features per cross
    def flow(mm):
        cs = mm.cross_slot
        n5 = sum(mm.slotcount.get(s, 0) for s in range(cs - 4, cs + 1))
        return {
            "shred_nbuy": mm.n, "shred_nbuy_5slot": n5,
            "shred_uniq_signers": len(mm.signers),
            "shred_prio_p90": pctile(mm.prios, 90), "shred_prio_max": max(mm.prios) if mm.prios else 0,
            "shred_tip_rate": (mm.ntip / mm.n) if mm.n else 0.0, "shred_tip_max": mm.tipmax,
            "shred_nslots": len(mm.slotcount),
            "shred_maxperslot": max(mm.slotcount.values()) if mm.slotcount else 0,
        }

    # 4) as-of shred-signer reputation (sorted by cross_slot) + WALLET-IDENTITY-PERMUTATION null.
    # pi is a fixed random bijection over all signers; rep_shuf_* looks up pi(signer)'s as-of
    # record instead of the signer's own -> same set sizes & rep-value distribution, but the
    # cross->its-own-wallets link is broken. If the model's lift survives rep_shuf it's an
    # artifact; if it collapses to comp-shuffle it is real wallet skill.
    import random as _random
    allsig = sorted({s for mm in cross.values() for s in mm.signers if s})
    _perm = allsig[:]; _random.Random(1234).shuffle(_perm)
    pi = dict(zip(allsig, _perm))
    sys.stderr.write(f"[shred-rep] {len(allsig)} distinct signers; building real + permuted rep\n"); sys.stderr.flush()

    order = sorted(cross.values(), key=lambda mm: (mm.cross_slot, mm.row.get("cross_t", 0)))
    rep = {}   # signer -> [seen, win]   (real as-of map; used for both real and permuted lookups)

    def repfeats(sset, keyfn, pfx):
        known = []; nsmart = 0
        for s in sset:
            rc = rep.get(keyfn(s))
            if rc and rc[0] > 0:
                v = (rc[1] + 1.0) / (rc[0] + 2.0)
                known.append(v)
                if rc[0] >= 3 and v >= 0.6:
                    nsmart += 1
        return {
            f"{pfx}mean": (sum(known) / len(known)) if known else 0.5,
            f"{pfx}max": max(known) if known else 0.0,
            f"{pfx}nknown": len(known),
            f"{pfx}frac_known": (len(known) / len(sset)) if sset else 0.0,
            f"{pfx}frachigh": (sum(1 for r in known if r > 0.5) / len(known)) if known else 0.0,
            f"{pfx}nsmart": nsmart,
        }

    n_emitted = 0
    out = open(a.out, "w")
    for mm in order:
        sset = [s for s in mm.signers if s]
        row = dict(mm.row); row.update(flow(mm))
        row.update(repfeats(sset, lambda s: s, "rep_"))
        row.update(repfeats(sset, lambda s: pi.get(s, s), "rep_shuf_"))
        out.write(json.dumps(row, separators=(",", ":")) + "\n"); n_emitted += 1
        # update reputation AFTER scoring (as-of)
        y = int(mm.row.get("y", 0))
        for s in sset:
            rc = rep.get(s)
            if rc is None:
                rep[s] = [1, y]
            else:
                rc[0] += 1; rc[1] += y
    out.close()
    # save the matured as-of reputation map as the LIVE seed (signer -> [seen, win]).
    # cont_aug_features.ShredRep.load() consumes this; the bot warm-starts from it.
    seed_path = os.path.join(os.path.dirname(a.out), "cont2x_shredrep_seed.json")
    json.dump({s: list(v) for s, v in rep.items()}, open(seed_path, "w"))
    sys.stderr.write(f"[shred-rep] DONE emitted={n_emitted} distinct_signers={len(rep)} out={a.out} seed={seed_path}\n"); sys.stderr.flush()


if __name__ == "__main__":
    main()
