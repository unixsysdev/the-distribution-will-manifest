"""Multi-milestone continuation tracker (2026-06-14). Generalizes continuation_tracker
to fire at EACH of several launch-multiples (1.5x, 2x, 3x, 4x, 5x) so we can measure
whether continuation edge exists at milestones beyond 2x (the STATIC multi-anchor model).

Each (coin, milestone) is an INDEPENDENT entry candidate:
  features  = launch-anchored path to that milestone (dd, buy_frac, ntr, recent, tps, uniq)
  fill      = next trade after the milestone crossing (gap-0)
  target    = +0.5x / -0.3x from that fill  (the same validated geometry as the 2x model)

A single coin can fire at many milestones as it climbs; a single trade can cross several
at once (a 1x->3x jump fires 1.5/2/3 together) and each progresses independently.

EXECUTABILITY GUARD (2026-06-14): exit_latency>0 models that a sell cannot land in the same
slot we SEE the +0.5x/-0.3x trigger -- it lands ~exit_latency later, so the realized return
is measured at the first trade >= trigger_t + exit_latency (whatever price is there), not at
the instantaneous trigger. This kills the non-executable same-slot captures. exit_latency=0
reproduces the original instantaneous-exit behavior.
"""
from __future__ import annotations
from collections import deque

ENTRY_MULTS = (1.5, 2.0, 3.0, 4.0, 5.0)
TP = 0.50
STOP = 0.30
RECENT_LAG = 10
STALE_SEC = 1800


class _MS:
    """Per-milestone sub-state."""
    __slots__ = ("phase", "fill", "tp", "stop", "cross_t", "exit_at", "trig_y")

    def __init__(self):
        self.phase = "pre"            # pre -> await_fill -> await_outcome -> (await_exit) -> done
        self.fill = self.tp = self.stop = self.cross_t = None
        self.exit_at = self.trig_y = None


class _S:
    __slots__ = ("p0", "mx", "mn", "nb", "nt", "recent", "tswin", "usrwin", "last_t", "ms")

    def __init__(self, mid, ts, mults):
        self.p0 = mid; self.mx = mid; self.mn = mid
        self.nb = 0; self.nt = 0
        self.recent = deque(maxlen=RECENT_LAG + 1)
        self.tswin = deque(maxlen=21)
        self.usrwin = deque(maxlen=21)
        self.last_t = ts
        self.ms = {k: _MS() for k in mults}


class MultiMilestoneTracker:
    """update() returns a list of event dicts, each tagged with `mult`."""

    def __init__(self, mults=ENTRY_MULTS, tp=TP, stop=STOP, exit_latency=0.0):
        self.mults = tuple(sorted(mults)); self.tp = tp; self.stop = stop
        self.exit_latency = exit_latency
        self.state: dict[str, _S] = {}

    def update(self, mint, vsol, vtok, is_buy, ts, user=""):
        if vtok <= 0 or vsol <= 0:
            return ()
        mid = vsol / vtok
        s = self.state.get(mint)
        if s is None:
            s = self.state[mint] = _S(mid, ts, self.mults)
        s.last_t = ts; s.nt += 1
        if is_buy:
            s.nb += 1
        if mid > s.mx: s.mx = mid
        if mid < s.mn: s.mn = mid
        s.recent.append(mid); s.tswin.append(ts); s.usrwin.append(user)
        out = []
        for k in self.mults:
            ms = s.ms[k]
            if ms.phase == "pre":
                if mid >= s.p0 * k:
                    ref = s.recent[0] if len(s.recent) > 0 else mid
                    span = (s.tswin[-1] - s.tswin[0]) if len(s.tswin) >= 2 else 0.0
                    tps = (len(s.tswin) - 1) / max(span, 0.1) if len(s.tswin) >= 2 else 0.0
                    out.append({"kind": "cross", "mint": mint, "mult": k, "t": ts,
                                "dd": s.mn / s.p0 - 1.0, "buy_frac": s.nb / s.nt, "ntr": s.nt,
                                "recent": (mid / ref - 1.0) if ref else 0.0,
                                "tps": tps, "uniq": len(set(s.usrwin)), "cross_mid": mid})
                    ms.phase = "await_fill"; ms.cross_t = ts
            elif ms.phase == "await_fill":
                ms.fill = mid; ms.tp = mid * (1 + self.tp); ms.stop = mid * (1 - self.stop)
                ms.phase = "await_outcome"
                out.append({"kind": "fill", "mint": mint, "mult": k, "t": ts, "fill_mid": mid})
            elif ms.phase == "await_outcome":
                hit = 1 if mid >= ms.tp else (0 if mid <= ms.stop else None)
                if hit is not None:
                    if self.exit_latency <= 0:                 # instantaneous exit (original)
                        ms.phase = "done"
                        out.append({"kind": "outcome", "mint": mint, "mult": k, "t": ts,
                                    "y": hit, "ret": mid / ms.fill - 1.0})
                    else:                                      # guard: sell lands exit_latency later
                        ms.phase = "await_exit"; ms.exit_at = ts + self.exit_latency; ms.trig_y = hit
            elif ms.phase == "await_exit":                     # realize at 1st trade >= trigger+latency
                if ts >= ms.exit_at:
                    ms.phase = "done"
                    out.append({"kind": "outcome", "mint": mint, "mult": k, "t": ts,
                                "y": ms.trig_y, "ret": mid / ms.fill - 1.0, "guarded": True})
        return out

    def prune(self, now):
        dead = [m for m, s in self.state.items()
                if (now - s.last_t) > STALE_SEC or all(x.phase == "done" for x in s.ms.values())]
        for m in dead:
            del self.state[m]
        return len(dead)


if __name__ == "__main__":
    t = MultiMilestoneTracker()
    ev = []
    # monotonic runner 1->8x: should fire crosses at all 5 mults and win the lower ones
    for i, m in enumerate([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0]):
        ev += t.update("RUN", m * 1e9, 1e9, 1, float(i))
    crosses = sorted(e["mult"] for e in ev if e["kind"] == "cross")
    wins = sorted(e["mult"] for e in ev if e["kind"] == "outcome" and e["y"] == 1)
    print("crosses fired at:", crosses)
    print("milestones that won (+0.5x from fill):", wins)
    assert crosses == [1.5, 2.0, 3.0, 4.0, 5.0], f"should cross all 5: {crosses}"
    assert set(wins) >= {1.5, 2.0, 3.0}, f"lower milestones should win on a 1->8x run: {wins}"
    # a dumper: crosses 1.5x then dies -> loss at 1.5x
    ev2 = []
    for i, m in enumerate([1.0, 1.5, 1.6, 1.0]):   # cross 1.5x, fill 1.6, drop to 1.0 (<1.6*0.7=1.12)
        ev2 += t.update("DUMP", m * 1e9, 1e9, 1, float(i))
    assert any(e["kind"] == "outcome" and e["mult"] == 1.5 and e["y"] == 0 for e in ev2), "dumper should lose at 1.5x"
    print("OK: multi-milestone tracker self-test passed")
    # guard self-test: a same-slot spike that reverts must NOT be counted as a win under the guard
    tg = MultiMilestoneTracker(mults=(2.0,), exit_latency=0.4)
    evg = []
    seq = [(1.0, 0.0), (2.0, 0.10), (2.0, 0.11), (3.2, 0.12), (1.4, 0.60)]  # cross@.10 fill@.11 tp-spike@.12 revert@.60
    for m, tt in seq:
        evg += tg.update("SPIKE", m * 1e9, 1e9, 1, tt)
    oc = [e for e in evg if e["kind"] == "outcome"]
    assert oc and oc[0]["ret"] < 0, f"guarded same-slot spike that reverts should realize a LOSS, got {oc}"
    print("OK: executability guard self-test passed (same-slot spike -> realized loss)")
