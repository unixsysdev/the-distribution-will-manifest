"""Continuation tracker — pure, testable state machine for the dry-run continuation
shadow (2026-06-13). Mirrors tools_local panel exactly: per mint track p0/max/min/
counts/recent; on the FIRST cross of ENTRY_MULT x launch emit a 'cross' with features
(dd, buy_frac, ntr, recent); the NEXT trade is the would-be gap-0 fill; then track to
+TP / -STOP from the FILL and emit the realized outcome. No I/O, no model — the shadow
service wraps it, logs events, and (optionally) scores them.

Feeds offline parity: dd = min(mid)/p0 - 1 (cumulative drawdown to cross),
buy_frac = n_buy/n_trades, ntr = n_trades to cross, recent = mid/mid[-10] - 1.
"""
from __future__ import annotations
from collections import deque

ENTRY_MULT = 2.0   # enter at 2x from launch
TP = 0.50          # +50% from fill = win
STOP = 0.30        # -30% from fill = loss
RECENT_LAG = 10    # trades back for momentum
STALE_SEC = 1800   # prune a mint with no update for 30 min


class _S:
    __slots__ = ("p0", "mx", "mn", "nb", "nt", "recent", "phase",
                 "fill", "tp", "stop", "cross_t", "last_t", "tswin", "usrwin")

    def __init__(self, mid, ts):
        self.p0 = mid; self.mx = mid; self.mn = mid
        self.nb = 0; self.nt = 0
        self.recent = deque(maxlen=RECENT_LAG + 1)
        self.tswin = deque(maxlen=21)    # last ~20 trade timestamps (competition: tps)
        self.usrwin = deque(maxlen=21)   # last ~20 buyers (competition: uniq)
        self.phase = "pre"          # pre -> await_fill -> await_outcome -> done
        self.fill = self.tp = self.stop = self.cross_t = None
        self.last_t = ts


class ContinuationTracker:
    """update() returns a list of event dicts (possibly empty)."""

    def __init__(self, entry_mult=ENTRY_MULT, tp=TP, stop=STOP):
        self.entry_mult = entry_mult; self.tp = tp; self.stop = stop
        self.state: dict[str, _S] = {}

    def update(self, mint: str, vsol: float, vtok: float, is_buy: bool, ts: float, user: str = ""):
        if vtok <= 0 or vsol <= 0:
            return ()
        mid = vsol / vtok
        s = self.state.get(mint)
        if s is None:
            s = self.state[mint] = _S(mid, ts)
        s.last_t = ts
        s.nt += 1
        if is_buy:
            s.nb += 1
        if mid > s.mx:
            s.mx = mid
        if mid < s.mn:
            s.mn = mid
        s.recent.append(mid)
        s.tswin.append(ts); s.usrwin.append(user)
        out = []

        if s.phase == "pre":
            if mid >= s.p0 * self.entry_mult:
                ref = s.recent[0] if len(s.recent) > 0 else mid
                span = (s.tswin[-1] - s.tswin[0]) if len(s.tswin) >= 2 else 0.0
                tps = (len(s.tswin) - 1) / max(span, 0.1) if len(s.tswin) >= 2 else 0.0
                feats = {
                    "dd": s.mn / s.p0 - 1.0,
                    "buy_frac": s.nb / s.nt,
                    "ntr": s.nt,
                    "recent": (mid / ref - 1.0) if ref else 0.0,
                    "tps": tps,                       # trade rate over recent window (live competition)
                    "uniq": len(set(s.usrwin)),       # unique recent buyers (live competition)
                    "cross_mid": mid,
                }
                s.phase = "await_fill"; s.cross_t = ts
                out.append({"kind": "cross", "mint": mint, "t": ts, **feats})
        elif s.phase == "await_fill":
            # this is the NEXT trade after the cross = the would-be gap-0 fill
            s.fill = mid; s.tp = mid * (1 + self.tp); s.stop = mid * (1 - self.stop)
            s.phase = "await_outcome"
            out.append({"kind": "fill", "mint": mint, "t": ts, "fill_mid": mid,
                        "slip_vs_cross": None})  # slip filled in by shadow vs the cross event
        elif s.phase == "await_outcome":
            if mid >= s.tp:
                s.phase = "done"
                out.append({"kind": "outcome", "mint": mint, "t": ts, "y": 1,
                            "exit_mid": mid, "ret": mid / s.fill - 1.0})
            elif mid <= s.stop:
                s.phase = "done"
                out.append({"kind": "outcome", "mint": mint, "t": ts, "y": 0,
                            "exit_mid": mid, "ret": mid / s.fill - 1.0})
        return out

    def prune(self, now: float):
        """Drop done/stale mints to bound memory. Call periodically."""
        dead = [m for m, s in self.state.items()
                if s.phase == "done" or (now - s.last_t) > STALE_SEC]
        for m in dead:
            del self.state[m]
        return len(dead)


if __name__ == "__main__":
    # self-test: a runner (2x then +50% from fill) and a dumper (2x then -30%)
    t = ContinuationTracker()
    ev = []
    # runner: p0=1.0, climbs to 2.0 (cross), next trade 2.1 (fill), then 3.2 (>2.1*1.5=3.15 win)
    for i, (m, b) in enumerate([(1.0,1),(1.2,1),(1.0,0),(2.0,1),(2.1,1),(2.5,1),(3.2,1)]):
        ev += t.update("RUN", m*1e9, 1e9, b, float(i))
    # dumper: p0=1.0 -> 2.0 cross -> 2.05 fill -> 1.3 (<2.05*0.7=1.435 loss)
    for i, (m, b) in enumerate([(1.0,1),(1.5,1),(2.0,1),(2.05,1),(1.3,0)]):
        ev += t.update("DUMP", m*1e9, 1e9, b, float(i))
    import json
    for e in ev:
        print(json.dumps(e))
    kinds = [(e["mint"], e["kind"], e.get("y")) for e in ev]
    assert ("RUN","cross",None) in kinds and ("RUN","outcome",1) in kinds, "runner should win"
    assert ("DUMP","outcome",0) in kinds, "dumper should lose"
    print("OK: tracker self-test passed")
