"""continuation_reputation.py — the buyer/creator reputation map for the continuation strategy.

SHARED between the offline causal rep-build (continuation_reputation_join) and the LIVE bot, so
the live features are computed by the EXACT same code the model trained on (no offline/live skew).

  offline: rep = Reputation(); time-sorted sweep -> rep.add_launch(creator) at a create,
           rep.features(buyers, creator) at a cross, rep.update(buyers, creator, y) at a resolve;
           rep.save(seed) at the end (the matured map = the live seed).
  live:    rep = Reputation.load(seed); rep.features(buyers, creator) at a cross;
           rep.update(buyers, creator, y) when a position resolves (keeps it current).

Causality is the CALLER's job (only call update() for coins resolved before the cross you score).
The dominant feature is buy_rep_mean (avg continuation track-record of the bots buying the coin);
creator features are minor and seed-based live.
"""
import json
from collections import defaultdict

REP_FEATURES = ["cre_n_launch", "cre_n_2x_res", "cre_winrate",
                "buy_n", "buy_known_frac", "buy_rep_mean", "buy_rep_max"]


class Reputation:
    def __init__(self):
        self.sig = defaultdict(lambda: [0, 0])    # signer  -> [sum_y, n_resolved_coins_bought]
        self.cre = defaultdict(lambda: [0, 0, 0])  # creator -> [n_launch, sum_y_2x, n_2x_resolved]

    def add_launch(self, creator):
        if creator:
            self.cre[creator][0] += 1

    def update(self, buyers, creator, y):
        """A coin resolved (y=1 continued). Push y onto its buyers' + creator's track records."""
        yi = 1 if y else 0
        for sg in buyers:
            r = self.sig[sg]; r[0] += yi; r[1] += 1
        if creator:
            r = self.cre[creator]; r[1] += yi; r[2] += 1

    def features(self, buyers, creator):
        reps = []
        for sg in buyers:
            r = self.sig.get(sg)
            if r and r[1] > 0:
                reps.append(r[0] / r[1])
        c = self.cre.get(creator) if creator else None
        n2x = c[2] if c else 0
        nb = len(buyers)
        return {
            "cre_n_launch": (c[0] if c else 0),
            "cre_n_2x_res": n2x,
            "cre_winrate": (c[1] / n2x) if n2x > 0 else 0.5,
            "buy_n": nb,
            "buy_known_frac": (len(reps) / nb) if nb else 0.0,
            "buy_rep_mean": (sum(reps) / len(reps)) if reps else 0.5,
            "buy_rep_max": max(reps) if reps else 0.5,
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"sig": {k: v for k, v in self.sig.items() if v[1] > 0},
                       "cre": {k: v for k, v in self.cre.items() if v[0] > 0 or v[2] > 0}}, f)

    @classmethod
    def load(cls, path):
        r = cls(); d = json.load(open(path))
        for sg, v in d.get("sig", {}).items():
            r.sig[sg] = v
        for c, v in d.get("cre", {}).items():
            r.cre[c] = v
        return r

    def stats(self):
        return {"signers": len(self.sig), "creators": len(self.cre),
                "signers_with_history": sum(1 for v in self.sig.values() if v[1] > 0)}


if __name__ == "__main__":
    rep = Reputation()
    # bot A always picks continuers, bot B always picks losers; creator C mixed
    rep.add_launch("C")
    rep.update(["A", "B"], "C", 1)      # coin1: A,B bought; continued
    rep.add_launch("C")
    rep.update(["A", "B"], "C", 0)      # coin2: A,B bought; failed
    rep.add_launch("C")
    rep.update(["A"], "C", 1)           # coin3: only A; continued
    f = rep.features(["A", "B", "Z"], "C")   # Z = unknown bot
    assert abs(f["buy_rep_mean"] - ((2/3 + 1/2) / 2)) < 1e-9, f["buy_rep_mean"]  # A=2/3, B=1/2, Z unknown
    assert f["buy_known_frac"] == 2/3 and f["buy_n"] == 3
    assert f["cre_n_launch"] == 3 and f["cre_n_2x_res"] == 3
    assert abs(f["cre_winrate"] - 2/3) < 1e-9
    rep.save("/tmp/_rep_seed_test.json")
    r2 = Reputation.load("/tmp/_rep_seed_test.json")
    assert r2.features(["A", "B", "Z"], "C") == f, "save/load round-trip mismatch"
    print("OK: reputation module self-test passed", rep.stats())
