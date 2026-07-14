"""cont_aug_features.py — SHARED feature code for the augmented 2x-continuation model.

ONE code path for offline panel build (cont_2x_shred_rep.py), the replay-parity harness,
and the live bot (continuation_live_rep_bot.py) => train==live by construction.

Computes the slot-based SHRED-window flow features and the as-of shred-SIGNER reputation
that cont2x_deploy_model expects. Mirrors the original cont_2x_shred_rep.py logic exactly.
Identities are shred `signer` (fee payer / operating wallet), NOT TradeEvent buyers.
"""
from __future__ import annotations
import json, os

SHRED_FEATS = ["shred_nbuy", "shred_nbuy_5slot", "shred_uniq_signers", "shred_prio_p90",
               "shred_prio_max", "shred_tip_rate", "shred_tip_max", "shred_nslots", "shred_maxperslot"]
REP_FEATS = ["rep_mean", "rep_max", "rep_nknown", "rep_frac_known", "rep_frachigh", "rep_nsmart"]
BUY_TYPES = {"buy", "buy_sol_in", "buy_quote"}
PRIO_CAP = 5000


def _pctile(a, q):
    if not a:
        return 0.0
    s = sorted(a)
    return float(s[min(len(s) - 1, int(q / 100.0 * len(s)))])


def mint_of(rec):
    """Resolve the mint of a shred intent record (matches the recorder + offline resolver):
    buy -> mint field / accs[2]; buy_sol_in / buy_quote -> accs[0]. None if not a buy."""
    t = rec.get("type")
    if t not in BUY_TYPES:
        return None
    if t == "buy":
        m = rec.get("mint")
        if m:
            return m
        accs = rec.get("ix_accounts") or []
        return accs[2] if len(accs) > 2 else None
    m = rec.get("mint")            # post recorder-fix, buy_sol_in/buy_quote carry mint=accs[0]
    if m:
        return m
    accs = rec.get("ix_accounts") or []
    return accs[0] if accs else None


class ShredAccum:
    """Incremental accumulator of a mint's buy intents (slot <= cross_slot). Bounded memory."""
    __slots__ = ("signers", "n", "slotcount", "prios", "ntip", "tipmax")

    def __init__(self):
        self.signers = set(); self.n = 0; self.slotcount = {}
        self.prios = []; self.ntip = 0; self.tipmax = 0

    def add(self, slot, signer, prio, tip):
        self.signers.add(signer); self.n += 1
        self.slotcount[slot] = self.slotcount.get(slot, 0) + 1
        if len(self.prios) < PRIO_CAP:
            self.prios.append(int(prio or 0))
        t = int(tip or 0)
        if t > 0:
            self.ntip += 1
            if t > self.tipmax:
                self.tipmax = t

    def features(self, cross_slot):
        cs = cross_slot
        n5 = sum(self.slotcount.get(s, 0) for s in range(cs - 4, cs + 1))
        return {
            "shred_nbuy": self.n, "shred_nbuy_5slot": n5,
            "shred_uniq_signers": len(self.signers),
            "shred_prio_p90": _pctile(self.prios, 90), "shred_prio_max": max(self.prios) if self.prios else 0,
            "shred_tip_rate": (self.ntip / self.n) if self.n else 0.0, "shred_tip_max": self.tipmax,
            "shred_nslots": len(self.slotcount),
            "shred_maxperslot": max(self.slotcount.values()) if self.slotcount else 0,
        }

    def signer_list(self):
        return [s for s in self.signers if s]


def shred_features_from_ring(ring_records, cross_slot):
    """JSONL/intent-record schema (type / signer / ix_accounts). Buys only."""
    acc = ShredAccum()
    for r in ring_records:
        if mint_of(r) is None:           # buy-type only
            continue
        sl = int(r.get("slot") or 0)
        if sl > cross_slot:
            continue
        acc.add(sl, r.get("signer"), r.get("priority_fee_micro"), r.get("jito_tip_lam"))
    return acc.features(cross_slot), acc.signer_list()


def shred_features_from_ring_records(records, cross_slot):
    """LIVE SHM-ring schema (intent_ring._bytes_to_intent): is_buy / user(=signer after the
    ring-write fix) / mint / slot / priority_fee_micro / jito_tip_lam. Buys only."""
    acc = ShredAccum()
    for r in records:
        if not r.get("is_buy"):
            continue
        sl = int(r.get("slot") or 0)
        if sl > cross_slot:
            continue
        acc.add(sl, r.get("user"), r.get("priority_fee_micro"), r.get("jito_tip_lam"))
    return acc.features(cross_slot), acc.signer_list()


class ShredRep:
    """As-of shred-signer reputation. Laplace (win+1)/(seen+2). features() BEFORE update()."""
    def __init__(self):
        self.rep = {}

    def features(self, signers):
        sset = [s for s in signers if s]
        known = []; nsmart = 0
        for s in sset:
            rc = self.rep.get(s)
            if rc and rc[0] > 0:
                v = (rc[1] + 1.0) / (rc[0] + 2.0)
                known.append(v)
                if rc[0] >= 3 and v >= 0.6:
                    nsmart += 1
        return {
            "rep_mean": (sum(known) / len(known)) if known else 0.5,
            "rep_max": max(known) if known else 0.0,
            "rep_nknown": len(known),
            "rep_frac_known": (len(known) / len(sset)) if sset else 0.0,
            "rep_frachigh": (sum(1 for r in known if r > 0.5) / len(known)) if known else 0.0,
            "rep_nsmart": nsmart,
        }

    def update(self, signers, y):
        y = int(y)
        for s in signers:
            if not s:
                continue
            rc = self.rep.get(s)
            if rc is None:
                self.rep[s] = [1, y]
            else:
                rc[0] += 1; rc[1] += y

    def save(self, path):
        json.dump(self.rep, open(path, "w"))

    @classmethod
    def load(cls, path):
        o = cls()
        if path and os.path.exists(path):
            o.rep = {s: list(v) for s, v in json.load(open(path)).items()}
        return o
