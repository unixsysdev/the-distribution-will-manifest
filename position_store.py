"""Atomic position event log for restart recovery.

Append-only JSONL of `open`, `snap`, `close` events. On startup, `replay` rebuilds
the PaperBook state. Unclosed positions are returned so the bot can decide
(force-close on restart for paper, chain-reconcile for live).

Concurrency: single-writer. Line-buffered for crash safety (at most one event lost
on hard crash). No fsync per write — paper mode tolerates that; live mode should
upgrade to per-write fsync via `fsync_every_event=True`.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Iterable


class PositionStore:
    def __init__(self, path: str | Path, fsync_every_event: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a", buffering=1)   # line buffered
        self.fsync_every_event = fsync_every_event

    def _write(self, rec: dict) -> None:
        try:
            self.fh.write(json.dumps(rec, default=str) + "\n")
            if self.fsync_every_event:
                self.fh.flush()
                os.fsync(self.fh.fileno())
        except Exception:
            # crash-loop avoidance: never raise from logging
            pass

    def record_open(self, mint: str, vsK: float, vtK: float, vsC: float, vtC: float,
                    score: float | None = None, opened_t: float | None = None) -> None:
        self._write({"kind": "open", "t": opened_t or time.time(), "mint": mint,
                     "vsK": vsK, "vtK": vtK, "vsC": vsC, "vtC": vtC,
                     "entry_score": score})

    def record_snap(self, mint: str, fwd: int, vs: float, vt: float,
                    ret: float, p_rec: float | None = None) -> None:
        self._write({"kind": "snap", "t": time.time(), "mint": mint, "fwd": fwd,
                     "vs": vs, "vt": vt, "ret": ret, "p_rec": p_rec})

    def record_close(self, mint: str, net_return: float, exit_kind: str,
                     reason: str, closed_t: float | None = None,
                     exit_ret: float | None = None,
                     live_policy_net: float | None = None,
                     live_policy_name: str | None = None) -> None:
        rec = {"kind": "close", "t": closed_t or time.time(), "mint": mint,
               "net_return": net_return, "exit_kind": exit_kind, "reason": reason}
        if exit_ret is not None:
            rec["exit_ret"] = exit_ret
        # live_policy_net = fractional realized P&L under whatever exit policy was
        # ACTIVE at close time, computed via offline replay of the live policy
        # against this position's snap timeline. Distinct from net_return which
        # is PaperBook's GREEN analytical reference.
        if live_policy_net is not None:
            rec["live_policy_net"] = live_policy_net
        if live_policy_name is not None:
            rec["live_policy_name"] = live_policy_name
        self._write(rec)

    def replay(self, book, force_close_on_restart: bool = True,
               restart_reason: str = "restart") -> set[str]:
        """Rebuild PaperBook state from the log. Returns the set of mints that were
        OPEN at the time of the last shutdown (i.e. had an `open` with no matching
        `close`). If force_close_on_restart, those positions are additionally closed
        in-place using snapshots seen so far and a `close` event is appended with the
        given reason — but the returned set still names them so the caller can log /
        avoid re-entering them.
        """
        from paper_book import PaperPosition
        if not self.path.exists():
            return set()
        # Read and replay in order
        opened: dict[str, dict] = {}
        closed: dict[str, dict] = {}
        snaps_by_mint: dict[str, list[dict]] = {}
        with open(self.path) as f:
            for ln in f:
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                k = rec.get("kind"); m = rec.get("mint")
                if not k or not m: continue
                if k == "open":
                    opened[m] = rec
                    snaps_by_mint.setdefault(m, [])
                    closed.pop(m, None)
                elif k == "snap":
                    snaps_by_mint.setdefault(m, []).append(rec)
                elif k == "close":
                    closed[m] = rec
        # Rebuild each opened position
        open_mints: set[str] = set()
        for m, op in opened.items():
            pos = PaperPosition(mint=m,
                                vsK=float(op["vsK"]), vtK=float(op["vtK"]),
                                vsC=float(op["vsC"]), vtC=float(op["vtC"]))
            for s in snaps_by_mint.get(m, []):
                pos.snaps_vs.append(float(s["vs"]))
                pos.snaps_vt.append(float(s["vt"]))
                pos.snaps_fwd.append(int(s["fwd"]))
                pos.snaps_ret_vs_midV.append(float(s["ret"]))
                p_rec = s.get("p_rec")
                if p_rec is not None:
                    pos.p_rec_at[int(s["fwd"])] = float(p_rec)
                # also keep vsC/vtC up to date as we replay snapshots
                pos.vsC = float(s["vs"])
                pos.vtC = float(s["vt"])
            if m in closed:
                cl = closed[m]
                pos.closed = True
                pos.net_return = float(cl.get("net_return", float("nan")))
                pos.kind = str(cl.get("exit_kind", ""))
            else:
                open_mints.add(m)
            book.positions[m] = pos
        # Force-close any still-open at restart (but still return their names)
        if force_close_on_restart and open_mints:
            for m in list(open_mints):
                pos = book.positions[m]
                if not pos.closed:
                    book._close_one(pos)
                    last_ret = pos.snaps_ret_vs_midV[-1] if pos.snaps_ret_vs_midV else None
                    self.record_close(m, pos.net_return, pos.kind, restart_reason,
                                      exit_ret=last_ret)
        return open_mints

    def close(self) -> None:
        try: self.fh.flush(); self.fh.close()
        except Exception: pass


if __name__ == "__main__":
    # Smoke test
    import tempfile
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from paper_book import PaperBook
    tmpd = tempfile.mkdtemp()
    p = Path(tmpd) / "positions.jsonl"
    st = PositionStore(p)
    st.record_open("aaaa", vsK=30e9, vtK=1.07e12, vsC=30e9, vtC=1.07e12, score=0.92)
    st.record_snap("aaaa", fwd=1, vs=31e9, vt=1.06e12, ret=0.01, p_rec=0.6)
    st.record_snap("aaaa", fwd=2, vs=32e9, vt=1.04e12, ret=0.03, p_rec=None)
    st.record_open("bbbb", vsK=30e9, vtK=1.07e12, vsC=30e9, vtC=1.07e12, score=0.55)
    st.record_close("bbbb", net_return=-0.02, exit_kind="hold", reason="stale")
    st.close()
    print(f"wrote events to {p}")
    book = PaperBook(q_sol=1.0, cost_bps=250.0, fee_per_tx_sol=0.0015,
                     max_slices=8, entry_lat_snaps=1, c_death=0.10)
    st2 = PositionStore(p)
    open_mints = st2.replay(book, force_close_on_restart=True, restart_reason="smoke_restart")
    st2.close()
    print(f"after replay: {len(book.positions)} positions in book")
    for m, pos in book.positions.items():
        print(f"  {m}  closed={pos.closed}  kind={pos.kind}  net={pos.net_return:+.4f}  "
              f"snaps={len(pos.snaps_vs)}")
    print(f"open at restart (force-closed): {open_mints}")
