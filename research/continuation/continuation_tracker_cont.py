"""Continuous (any-multiple) continuation tracker (2026-06-14). Instead of fixed
milestones, it triggers a candidate at ANY point with recent upward momentum (with a
cooldown to limit overlapping near-duplicates), using ANCHOR-FREE rolling features:
  r_mom  = mid / mid[-LAG] - 1            (recent momentum)
  r_dd   = min(recent mids) / mid - 1     (local drawdown)
  r_tps  = trades/sec over last WIN
  r_uniq = unique buyers last WIN
  r_bf   = buy fraction last WIN
  mult_ctx = mid / launch                 (soft context: where in the trajectory)
fill = next trade after trigger; target +TP / -STOP from fill. The outcome event carries
the features + t0 (trigger time, for OOS day-split) + mint (for the BY-COIN train/test split).
"""
from __future__ import annotations
from collections import deque

LAG = 10; WIN = 20; COOLDOWN = 30; TP = 0.50; STOP = 0.30; STALE_SEC = 1800; MOM_GATE = 0.0


class _S:
    __slots__ = ("p0", "mids", "tss", "users", "buys", "last_t", "last_trig", "cands", "ntc")

    def __init__(self, mid, ts):
        self.p0 = mid
        self.mids = deque(maxlen=WIN + 1); self.tss = deque(maxlen=WIN + 1)
        self.users = deque(maxlen=WIN + 1); self.buys = deque(maxlen=WIN + 1)
        self.last_t = ts; self.last_trig = -10 ** 9; self.cands = []; self.ntc = 0


class ContinuousTracker:
    def __init__(self, tp=TP, stop=STOP, cooldown=COOLDOWN, mom_gate=MOM_GATE, lag=LAG):
        self.tp = tp; self.stop = stop; self.cooldown = cooldown; self.mom_gate = mom_gate; self.lag = lag
        self.state: dict[str, _S] = {}

    def update(self, mint, vsol, vtok, is_buy, ts, user=""):
        if vtok <= 0 or vsol <= 0:
            return ()
        mid = vsol / vtok
        s = self.state.get(mint)
        if s is None:
            s = self.state[mint] = _S(mid, ts)
        s.ntc += 1; s.last_t = ts
        s.mids.append(mid); s.tss.append(ts); s.users.append(user); s.buys.append(1 if is_buy else 0)
        out = []
        # progress in-flight candidates
        for c in s.cands:
            if c["phase"] == "await_fill":
                c["fill"] = mid; c["tp"] = mid * (1 + self.tp); c["stop"] = mid * (1 - self.stop)
                c["phase"] = "await_outcome"
            elif c["phase"] == "await_outcome":
                if mid >= c["tp"]:
                    c["phase"] = "done"
                    out.append({"kind": "outcome", "mint": mint, "t": ts, "t0": c["t0"],
                                "y": 1, "ret": mid / c["fill"] - 1.0, **c["feat"]})
                elif mid <= c["stop"]:
                    c["phase"] = "done"
                    out.append({"kind": "outcome", "mint": mint, "t": ts, "t0": c["t0"],
                                "y": 0, "ret": mid / c["fill"] - 1.0, **c["feat"]})
        if s.cands:
            s.cands = [c for c in s.cands if c["phase"] != "done"]
        # maybe trigger a new candidate
        if len(s.mids) > self.lag and (s.ntc - s.last_trig) >= self.cooldown:
            ref = s.mids[-(self.lag + 1)]
            r_mom = (mid / ref - 1.0) if ref else 0.0
            if r_mom > self.mom_gate:
                span = (s.tss[-1] - s.tss[0]) if len(s.tss) >= 2 else 0.0
                r_tps = (len(s.tss) - 1) / max(span, 0.1) if len(s.tss) >= 2 else 0.0
                feat = {"r_mom": r_mom, "r_dd": min(s.mids) / mid - 1.0, "r_tps": r_tps,
                        "r_uniq": len(set(s.users)), "r_bf": sum(s.buys) / len(s.buys),
                        "mult_ctx": mid / s.p0}
                s.cands.append({"phase": "await_fill", "feat": feat, "fill": None, "t0": ts})
                s.last_trig = s.ntc
        return out

    def prune(self, now):
        dead = [m for m, s in self.state.items() if (now - s.last_t) > STALE_SEC and not s.cands]
        for m in dead:
            del self.state[m]
        return len(dead)


if __name__ == "__main__":
    t = ContinuousTracker(cooldown=2, lag=2)
    ev = []
    # sustained uptrend -> candidates trigger and win (+50% reachable)
    seq = [1.0, 1.1, 1.3, 1.5, 1.6, 2.0, 2.5, 3.0]
    for i, m in enumerate(seq):
        ev += t.update("UP", m * 1e9, 1e9, 1, float(i))
    outs = [e for e in ev if e["kind"] == "outcome"]
    print(f"uptrend: {len(outs)} candidate outcomes, wins={sum(e['y'] for e in outs)}")
    assert len(outs) >= 1 and any(e["y"] == 1 for e in outs), "uptrend should trigger and win some"
    # a pump then dump -> a candidate that loses
    t2 = ContinuousTracker(cooldown=2, lag=2)
    ev2 = []
    for i, m in enumerate([1.0, 1.2, 1.5, 1.8, 2.0, 1.3, 1.0, 0.8]):
        ev2 += t2.update("DUMP", m * 1e9, 1e9, 1, float(i))
    outs2 = [e for e in ev2 if e["kind"] == "outcome"]
    print(f"pump-dump: {len(outs2)} outcomes, wins={sum(e['y'] for e in outs2)}")
    assert any(e["y"] == 0 for e in outs2), "a candidate should lose on the dump"
    sample = outs[0]
    print("feature keys:", [k for k in sample if k in ("r_mom","r_dd","r_tps","r_uniq","r_bf","mult_ctx")])
    print("OK: continuous tracker self-test passed")
