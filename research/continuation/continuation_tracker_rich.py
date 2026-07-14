"""continuation_tracker_rich.py (2026-06-14) — 2x continuation tracker with a RICH causal
feature set at the cross, for a better filter/model. Every feature is computed STRICTLY from the
launch->cross path (the crossing trade is observable at decision time = no lookahead). update()
takes the full trade economics (real_sol, sol_amount) so it can build volume / mcap / flow.

Feature families (all at the first 2x-from-launch cross):
  base-6   : dd, buy_frac, ntr, recent, tps, uniq            (the current model)
  timing   : t_to_2x, log_t_to_2x, accel, last_gap
  size/vol : mcap_sol, vol_sol, sol_per_trade, max_buy_sol, whale_frac, net_flow
  micro    : n_buyers, n_sellers, bs_ratio, signer_conc
  shape    : up_frac, max_runup
Intent features are joined SEPARATELY (causal, from the shred capture) so each family's lift is
measured on its own. exit_latency>0 keeps the executability guard available (default 0).
"""
from __future__ import annotations
from collections import deque, Counter
import math

TP = 0.50
STOP = 0.30
RECENT_LAG = 10
STALE_SEC = 1800

BASE6 = ["dd", "buy_frac", "ntr", "recent", "tps", "uniq"]
RICH = BASE6 + ["t_to_2x", "log_t_to_2x", "accel", "last_gap", "mcap_sol", "vol_sol",
                "sol_per_trade", "max_buy_sol", "whale_frac", "net_flow", "n_buyers", "n_sellers",
                "bs_ratio", "signer_conc", "up_frac", "max_runup"]


class _S:
    __slots__ = ("p0", "mx", "mn", "nb", "nt", "recent", "tswin", "usrwin", "t0", "last_t",
                 "last_gap", "prev_mid", "vol_sol", "buy_sol", "sell_sol", "max_buy_sol",
                 "buyers", "sellers", "sigc", "n_up", "phase", "fill", "tp", "stop", "cross_t",
                 "exit_at", "trig_y")

    def __init__(self, mid, ts):
        self.p0 = mid; self.mx = mid; self.mn = mid; self.nb = 0; self.nt = 0
        self.recent = deque(maxlen=RECENT_LAG + 1); self.tswin = deque(maxlen=21); self.usrwin = deque(maxlen=21)
        self.t0 = ts; self.last_t = ts; self.last_gap = 0.0; self.prev_mid = mid
        self.vol_sol = 0.0; self.buy_sol = 0.0; self.sell_sol = 0.0; self.max_buy_sol = 0.0
        self.buyers = set(); self.sellers = set(); self.sigc = Counter(); self.n_up = 0
        self.phase = "pre"; self.fill = self.tp = self.stop = self.cross_t = None
        self.exit_at = self.trig_y = None


class RichTracker:
    def __init__(self, k=2.0, tp=TP, stop=STOP, exit_latency=0.0):
        self.k = k; self.tp = tp; self.stop = stop; self.exit_latency = exit_latency
        self.state: dict[str, _S] = {}

    def update(self, mint, vsol, vtok, real_sol, sol_amount, is_buy, ts, user=""):
        if vtok <= 0 or vsol <= 0:
            return ()
        mid = vsol / vtok; sol = sol_amount / 1e9
        s = self.state.get(mint)
        if s is None:
            s = self.state[mint] = _S(mid, ts)
        s.nt += 1
        s.last_gap = (ts - s.last_t) if s.nt > 1 else 0.0
        s.last_t = ts
        if is_buy:
            s.nb += 1; s.buy_sol += sol; s.buyers.add(user)
            if sol > s.max_buy_sol:
                s.max_buy_sol = sol
        else:
            s.sell_sol += sol; s.sellers.add(user)
        s.vol_sol += sol; s.sigc[user] += 1
        if mid > s.prev_mid:
            s.n_up += 1
        s.prev_mid = mid
        if mid > s.mx: s.mx = mid
        if mid < s.mn: s.mn = mid
        s.recent.append(mid); s.tswin.append(ts); s.usrwin.append(user)
        out = []
        if s.phase == "pre":
            if mid >= s.p0 * self.k:
                out.append({"kind": "cross", "mint": mint, "t": ts, "cross_mid": mid,
                            **self._features(s, mid, real_sol, ts)})
                s.phase = "await_fill"; s.cross_t = ts
        elif s.phase == "await_fill":
            s.fill = mid; s.tp = mid * (1 + self.tp); s.stop = mid * (1 - self.stop)
            s.phase = "await_outcome"
            out.append({"kind": "fill", "mint": mint, "t": ts, "fill_mid": mid})
        elif s.phase == "await_outcome":
            hit = 1 if mid >= s.tp else (0 if mid <= s.stop else None)
            if hit is not None:
                if self.exit_latency <= 0:
                    s.phase = "done"
                    out.append({"kind": "outcome", "mint": mint, "t": ts, "y": hit, "ret": mid / s.fill - 1.0})
                else:
                    s.phase = "await_exit"; s.exit_at = ts + self.exit_latency; s.trig_y = hit
        elif s.phase == "await_exit":
            if ts >= s.exit_at:
                s.phase = "done"
                out.append({"kind": "outcome", "mint": mint, "t": ts, "y": s.trig_y, "ret": mid / s.fill - 1.0})
        return out

    def _features(self, s, mid, real_sol, ts):
        span = (s.tswin[-1] - s.tswin[0]) if len(s.tswin) >= 2 else 0.0
        tps = (len(s.tswin) - 1) / max(span, 0.1) if len(s.tswin) >= 2 else 0.0
        ref = s.recent[0] if len(s.recent) > 0 else mid
        t_to = max(ts - s.t0, 1e-3)
        overall_tps = s.nt / t_to
        vol = max(s.vol_sol, 1e-9)
        return {
            "dd": s.mn / s.p0 - 1.0, "buy_frac": s.nb / s.nt, "ntr": s.nt,
            "recent": (mid / ref - 1.0) if ref else 0.0, "tps": tps, "uniq": len(set(s.usrwin)),
            "t_to_2x": t_to, "log_t_to_2x": math.log(t_to), "accel": tps / max(overall_tps, 1e-9),
            "last_gap": s.last_gap, "mcap_sol": real_sol / 1e9, "vol_sol": s.vol_sol,
            "sol_per_trade": s.vol_sol / s.nt, "max_buy_sol": s.max_buy_sol, "whale_frac": s.max_buy_sol / vol,
            "net_flow": (s.buy_sol - s.sell_sol) / vol, "n_buyers": len(s.buyers), "n_sellers": len(s.sellers),
            "bs_ratio": len(s.buyers) / (len(s.sellers) + 1), "signer_conc": (max(s.sigc.values()) / s.nt) if s.sigc else 0.0,
            "up_frac": s.n_up / s.nt, "max_runup": s.mx / s.p0 - 1.0,
        }

    def prune(self, now):
        dead = [m for m, s in self.state.items() if (now - s.last_t) > STALE_SEC or s.phase == "done"]
        for m in dead:
            del self.state[m]
        return len(dead)


if __name__ == "__main__":
    t = RichTracker()
    ev = []
    # 1->2.5x monotonic climb, then +0.5x more -> a win; check the rich features populate
    seq = [(1.0, 0.0, 1.0, 1), (1.3, 0.5, 2.0, 1), (1.7, 1.1, 0.5, 1), (2.0, 1.6, 3.0, 1),
           (2.2, 2.0, 1.0, 1), (3.5, 3.0, 2.0, 1)]   # fill@2.2 -> tp=3.3; 3.5 clears it -> win
    users = ["a", "b", "a", "c", "b", "d"]
    for (m, tt, solv, isb), u in zip(seq, users):
        ev += t.update("RUN", m * 1e9, 1e9, int((m - 1) * 1e9), int(solv * 1e9), isb, tt, u)
    cross = next((e for e in ev if e["kind"] == "cross"), None)
    assert cross is not None, "should fire a 2x cross"
    for f in RICH:
        assert f in cross, f"missing feature {f}"
    assert cross["t_to_2x"] > 0 and cross["mcap_sol"] > 0 and cross["n_buyers"] >= 1
    win = any(e["kind"] == "outcome" and e["y"] == 1 for e in ev)
    print("cross features:", {k: round(cross[k], 3) for k in RICH})
    print("won:", win)
    assert win, "monotonic runner should win"
    print("OK: rich tracker self-test passed (%d features)" % len(RICH))
