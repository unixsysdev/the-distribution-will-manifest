"""Paper book — per-mint position manager that implements the EXACT policy from
pumpfun_causal_revalidate_V.stressed_ret(): scale-out-into-strength once ret>0 (cap 8
slices), precision death-cut when P(recover)<0.10 in drawdown, AMM impact compounded
across exit slices, per-transaction fee. NO real execution.

Closed-loop test: feed the OOS V path snapshots through this paper book and verify
per-bet net P&L equals the analytical stressed_ret per-bet net to floating-point
precision. That validates the live-pipeline policy wiring.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


def buy_tokens(vs, vt, dsol): return vt - (vs*vt)/(vs+dsol)
def sell_sol(vs, vt, dtok):  return vs - (vs*vt)/(vt+dtok)


@dataclass
class PaperPosition:
    mint: str
    vsK: float           # entry virtual SOL (V trigger reserves)
    vtK: float           # entry virtual token reserves
    vsC: float           # terminal virtual SOL (for hold-to-end exit)
    vtC: float
    snaps_vs: list[float] = field(default_factory=list)  # forward snap reserves (post-trigger)
    snaps_vt: list[float] = field(default_factory=list)
    snaps_fwd: list[int]  = field(default_factory=list)
    snaps_ret_vs_midV: list[float] = field(default_factory=list)   # for finding ip (profit)
    snaps_t: list[float] = field(default_factory=list)              # wall-clock per snap; lets us
                                                                     # replay time-based policies
                                                                     # (h_time_spaced gap_s, etc) at
                                                                     # close-time to get live-policy net
    p_rec_at: dict[int, float] = field(default_factory=dict)        # fwd -> recovery score
    closed: bool = False
    net_return: float = float("nan")
    kind: str = ""        # "rider" | "cut" | "hold"


class PaperBook:
    """Mirrors stressed_ret(entry_lat, fee_tx) exactly. Default = PLAUSIBLE settings."""
    def __init__(self, q_sol: float = 1.0, cost_bps: float = 250.0,
                 fee_per_tx_sol: float = 0.0015, max_slices: int = 8,
                 entry_lat_snaps: int = 1, c_death: float = 0.10):
        self.q_lam   = q_sol * 1e9
        self.cost    = cost_bps / 1e4
        self.fee_lam = fee_per_tx_sol * 1e9
        self.max_slices = max_slices
        self.entry_lat = entry_lat_snaps
        self.c_death = c_death
        self.positions: dict[str, PaperPosition] = {}

    def open(self, mint: str, vsK: float, vtK: float, vsC: float, vtC: float) -> None:
        """Token's V-trigger entry. Subsequent snapshots added via add_snapshot."""
        self.positions[mint] = PaperPosition(mint=mint, vsK=vsK, vtK=vtK, vsC=vsC, vtC=vtC)

    def add_snapshot(self, mint: str, fwd: int, vs: float, vt: float,
                     ret_vs_midV: float, p_rec: float | None = None,
                     t: float | None = None) -> None:
        p = self.positions.get(mint)
        if p is None or p.closed: return
        p.snaps_vs.append(vs); p.snaps_vt.append(vt); p.snaps_fwd.append(fwd)
        p.snaps_ret_vs_midV.append(ret_vs_midV)
        # t = wall-clock seconds at which this snap was observed. Optional for
        # back-compat with old call sites (would just leave snaps_t shorter
        # than snaps_vs). When present, lets the harness compute live-policy
        # net at close time by replaying time-based policies.
        if t is not None: p.snaps_t.append(float(t))
        if p_rec is not None: p.p_rec_at[fwd] = p_rec

    def close_all(self) -> None:
        """Apply scale-out/death-cut/hold decisions to every open position. Returns nothing;
        results stored in p.net_return + p.kind."""
        for p in self.positions.values():
            if p.closed: continue
            self._close_one(p)

    def _close_one(self, p: PaperPosition) -> None:
        # Replicate stressed_ret exactly.
        # entry_lat=1 means we fill at snaps[0] reserves; pool for exit = snaps[1:].
        n = len(p.snaps_vs)
        if self.entry_lat == 0:
            vse, vte, start = p.vsK, p.vtK, 0
        else:
            if n == 0:
                # No snapshots at all -> hold to terminal
                vse, vte, start = p.vsK, p.vtK, 0
                pos = buy_tokens(vse, vte, self.q_lam)
                proceeds = sell_sol(p.vsC, p.vtC, pos)
                net = proceeds/self.q_lam - 1.0 - self.cost - (self.fee_lam * 2)/self.q_lam
                p.net_return = net; p.kind = "hold"; p.closed = True
                return
            ei = min(self.entry_lat - 1, n - 1)
            vse, vte, start = p.snaps_vs[ei], p.snaps_vt[ei], self.entry_lat
        if vse <= 0 or vte <= 0:
            p.net_return = float("nan"); p.kind = "skip"; p.closed = True
            return
        pos = buy_tokens(vse, vte, self.q_lam)
        emk = vse/vte
        pool_vs = p.snaps_vs[start:]; pool_vt = p.snaps_vt[start:]; pool_fwd = p.snaps_fwd[start:]
        if len(pool_vs) == 0:
            # entry-lat ate the pool -> hold to terminal
            net = sell_sol(p.vsC, p.vtC, pos)/self.q_lam - 1.0 - self.cost - (self.fee_lam * 2)/self.q_lam
            p.net_return = net; p.kind = "hold"; p.closed = True
            return
        # find ip (first my_ret > 0) and idth (first my_ret < 0 with p_rec < c_death)
        my_ret = [pool_vs[i]/pool_vt[i]/emk - 1.0 for i in range(len(pool_vs))]
        ip = next((i for i in range(len(pool_vs)) if my_ret[i] > 0), None)
        idth = next((i for i in range(len(pool_vs)) if my_ret[i] < 0 and
                     p.p_rec_at.get(pool_fwd[i], 1.0) < self.c_death), None)
        if ip is not None and (idth is None or ip <= idth):
            sched = list(range(ip, len(pool_vs))); kind = "rider"
        elif idth is not None:
            sched = [idth]; kind = "cut"
        else:
            net = sell_sol(p.vsC, p.vtC, pos)/self.q_lam - 1.0 - self.cost - (self.fee_lam * 2)/self.q_lam
            p.net_return = net; p.kind = "hold"; p.closed = True
            return
        # cap slices
        if len(sched) > self.max_slices:
            idx = np.linspace(0, len(sched)-1, self.max_slices).round().astype(int)
            sched = [sched[j] for j in sorted(set(idx))]
        nsl = len(sched); tsl = pos/nsl
        proc = 0.0; ovs = 0.0; ovt = 0.0
        for s in sched:
            vs, vt = pool_vs[s], pool_vt[s]
            got = sell_sol(max(vs - ovs, 1.0), vt + ovt, tsl)
            proc += got; ovs += got; ovt += tsl
        net = proc/self.q_lam - 1.0 - self.cost - (self.fee_lam * (1 + nsl))/self.q_lam
        p.net_return = net; p.kind = kind; p.closed = True

    def returns(self) -> np.ndarray:
        return np.array([p.net_return for p in self.positions.values() if p.closed], dtype=float)

    def kinds(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(p.kind for p in self.positions.values() if p.closed))
