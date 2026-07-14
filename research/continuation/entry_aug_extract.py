#!/usr/bin/env python3
"""Augmented ENTRY-selection feature extractor (Phase 1) — streams the recent
filtered grpc_capture jsonl on sol and emits one row per fresh classic-curve mint.

Why this exists: the frozen May snapshot the offline thread exhausted had ONLY
trade economics. The capture pipeline was widened 2026-06-09 to also record, per
tx: failed / fee_lam / cu / cu_limit / priority_fee_micro / jito_tip_lam / route /
n_inner_ix / n_keys, PLUS per-slot BlockMeta congestion and CreateEvent. This
extractor builds the proven 11-feature K-window baseline AND a NEW execution-
competition + congestion feature block over the same early window, with strictly
causal forward-peak / forward-terminal labels (frozen at decision, label from
forward trades only). The trainer then tests whether competition+congestion add
incremental AUC + top-decile lift over the baseline.

Memory-bounded: per-mint state is evicted once the label horizon closes, so only
mints inside the active forward window live in RAM (~hundreds at any time).

Causal discipline: every feature is computed from trades at-or-before the K-th
trade (the decision). Labels use only trades strictly after the decision.
"""
from __future__ import annotations
import argparse, glob, gzip, io, json, os, sys, time
from collections import deque
from pathlib import Path

PUMP_INIT_VSOL = 30_000_000_000           # classic curve initial virtual SOL (lamports)
KNOWN_ROUTER_HINT = True                    # route field already resolved by capture
ROOT = os.getenv("PUMPFUN_ROOT", str(Path(__file__).resolve().parents[2]))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dirs", nargs="+", default=[
        f"{ROOT}/grpc_capture",
        "/mnt/storagebox/backup/archive/grpc_capture",
    ])
    ap.add_argument("--since", default="20260609", help="YYYYMMDD min file date (extras began 06-09)")
    ap.add_argument("--until", default="99999999", help="YYYYMMDD max file date")
    ap.add_argument("--k", type=int, default=10, help="decision = first K trades (FolioBot Finding 5 used 10)")
    ap.add_argument("--label-horizon-s", type=float, default=1800.0, help="forward window for peak/terminal")
    ap.add_argument("--min-window-n", type=int, default=10, help="drop mints that never reach this many trades")
    ap.add_argument("--fresh-rsol-max-lam", type=int, default=8_000_000_000,
                    help="first-observed real_sol_reserves cap to count as a fresh launch")
    ap.add_argument("--out", default=f"{ROOT}/bot_data/entry_aug_panel.jsonl")
    ap.add_argument("--max-files", type=int, default=0, help="0 = all")
    ap.add_argument("--progress-every", type=int, default=20000)
    return ap.parse_args()


def list_files(dirs, since, until):
    seen = {}
    for d in dirs:
        for fn in glob.glob(os.path.join(d, "capture_*.jsonl*")):
            base = os.path.basename(fn)
            # capture_YYYYMMDDTHHMMSSZ.jsonl[.gz]
            try:
                stamp = base.split("_", 1)[1][:8]
            except Exception:
                continue
            if stamp < since or stamp > until:
                continue
            # prefer the local (non-storagebox) copy when basename collides; prefer plain .jsonl over .gz
            key = base.replace(".gz", "")
            prev = seen.get(key)
            if prev is None or (prev.endswith(".gz") and not fn.endswith(".gz")):
                seen[key] = fn
    return [seen[k] for k in sorted(seen)]


def open_any(fn):
    if fn.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(fn, "rb"), encoding="utf-8", errors="replace")
    return open(fn, "r", encoding="utf-8", errors="replace")


class Mint:
    __slots__ = ("first_ts", "first_slot", "mid0", "n", "mids", "users", "user_sol",
                 "n_buy", "net_sol", "tot_sol", "entry_sol", "win_dup", "win_ddown",
                 "last_ts", "decided", "midK", "trig_ts", "trig_slot", "n_at_trig",
                 # competition aggregates over the decision window (buys)
                 "fees", "prios", "cus", "culimits", "tips_present", "tips_lam",
                 "inners", "nkeys", "route_n", "win_slots",
                 # forward
                 "run_max", "term_mid", "n_fwd", "row_feats")

    def __init__(self, e, mid):
        self.first_ts = e["t"]; self.first_slot = e.get("slot") or 0
        self.mid0 = mid; self.n = 1
        self.mids = [mid]
        u = e.get("user"); s = e["sol"] / 1e9; isb = e["is_buy"]
        self.users = {u}; self.user_sol = {u: s}
        self.n_buy = 1 if isb else 0
        self.net_sol = s if isb else -s
        self.tot_sol = s; self.entry_sol = s
        self.win_dup = 0.0; self.win_ddown = 0.0
        self.last_ts = e["t"]
        self.decided = False; self.midK = 0.0; self.trig_ts = 0.0; self.trig_slot = 0; self.n_at_trig = 0
        self.fees = []; self.prios = []; self.cus = []; self.culimits = []
        self.tips_present = 0; self.tips_lam = []; self.inners = []; self.nkeys = []; self.route_n = 0
        self.win_slots = []
        self._acc_cmp(e)
        self.run_max = 0.0; self.term_mid = mid; self.n_fwd = 0; self.row_feats = None

    def _acc_cmp(self, e):
        # competition fields accumulate over the pre-decision window (all trades incl sells;
        # they all reflect contention to interact with this mint)
        self.fees.append(int(e.get("fee_lam") or 0))
        self.prios.append(int(e.get("priority_fee_micro") or 0))
        self.cus.append(int(e.get("cu") or 0))
        self.culimits.append(int(e.get("cu_limit") or 0))
        tl = e.get("jito_tip_lam")
        if tl:
            self.tips_present += 1; self.tips_lam.append(int(tl))
        self.inners.append(int(e.get("n_inner_ix") or 0))
        self.nkeys.append(int(e.get("n_keys") or 0))
        if e.get("route"):
            self.route_n += 1
        sl = e.get("slot")
        if sl:
            self.win_slots.append(int(sl))

    def add_trade(self, e, mid):
        if self.decided:
            self.n_fwd += 1; self.last_ts = e["t"]; self.term_mid = mid
            if self.midK > 0:
                r = mid / self.midK - 1.0
                if r > self.run_max:
                    self.run_max = r
            return
        self.n += 1; self.mids.append(mid); self.last_ts = e["t"]
        u = e.get("user"); s = e["sol"] / 1e9; isb = e["is_buy"]
        self.users.add(u); self.user_sol[u] = self.user_sol.get(u, 0.0) + s
        if isb:
            self.n_buy += 1; self.net_sol += s
        else:
            self.net_sol -= s
        self.tot_sol += s
        rr = mid / self.mid0 - 1.0 if self.mid0 > 0 else 0.0
        if rr > self.win_dup: self.win_dup = rr
        if rr < self.win_ddown: self.win_ddown = rr
        self._acc_cmp(e)

    def baseline_feats(self):
        mids = self.mids
        d = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
        sa = sum(abs(x) for x in d)
        dir_eff = abs(sum(d)) / sa if sa > 0 else 0.0
        win_ret = mids[-1] / self.mid0 - 1.0 if self.mid0 > 0 else 0.0
        span = max(1e-6, self.last_ts - self.first_ts)
        sas = max(self.user_sol.values()) / self.tot_sol if self.tot_sol > 0 else 0.0
        return {
            "win_ret": win_ret, "dir_eff": dir_eff, "buy_frac": self.n_buy / self.n,
            "uniq": len(self.users), "net_sol": self.net_sol, "tot_sol": self.tot_sol,
            "single_actor_share": sas, "trades_per_sec": self.n / span,
            "entry_sol": self.entry_sol, "win_drawup": self.win_dup, "win_drawdown": self.win_ddown,
        }

    def competition_feats(self, slot_exec):
        def stats(a, pfx):
            if not a:
                return {f"{pfx}_mean": 0.0, f"{pfx}_max": 0.0}
            return {f"{pfx}_mean": sum(a) / len(a), f"{pfx}_max": max(a)}
        nb = max(1, self.n)
        f = {}
        f.update(stats(self.fees, "cmp_fee"))
        f.update(stats(self.prios, "cmp_prio"))
        f.update(stats(self.cus, "cmp_cu"))
        f.update(stats(self.culimits, "cmp_culimit"))
        f.update(stats(self.inners, "cmp_inner"))
        f.update(stats(self.nkeys, "cmp_nkeys"))
        # priority-fee escalation across the window (fee war ramp)
        prio_nz = [p for p in self.prios if p > 0]
        f["cmp_prio_escal"] = (prio_nz[-1] / prio_nz[0]) if len(prio_nz) >= 2 and prio_nz[0] > 0 else 1.0
        f["cmp_tip_rate"] = self.tips_present / nb
        f["cmp_tip_mean_lam"] = (sum(self.tips_lam) / len(self.tips_lam)) if self.tips_lam else 0.0
        f["cmp_route_frac"] = self.route_n / nb
        f["cmp_distinct_buyers"] = len(self.users)
        # block congestion over the window slots (firehose-grade, BlockMeta-sourced)
        execs = [slot_exec[s] for s in self.win_slots if s in slot_exec]
        f["cong_exec_mean"] = (sum(execs) / len(execs)) if execs else 0.0
        f["cong_exec_max"] = max(execs) if execs else 0.0
        f["cong_slot_span"] = (max(self.win_slots) - min(self.win_slots)) if self.win_slots else 0
        return f


def main():
    a = parse_args()
    files = list_files(a.capture_dirs, a.since, a.until)
    if a.max_files:
        files = files[:a.max_files]
    sys.stderr.write(f"[extract] {len(files)} capture files {a.since}->{a.until}, K={a.k}, horizon={a.label_horizon_s}s\n")
    sys.stderr.flush()
    out = open(a.out, "w")
    active = {}           # mint -> Mint (pre-decision OR within forward horizon)
    slot_exec = {}        # slot -> executed_transaction_count (rolling, bounded)
    slot_q = deque()      # FIFO of slots to bound slot_exec memory
    n_rows = 0; n_seen = 0; n_drop_short = 0; cur_t = 0.0

    def emit(m, censored):
        nonlocal n_rows
        if m.row_feats is None:
            return
        peak_ratio = 1.0 + m.run_max
        term_ratio = (m.term_mid / m.midK) if m.midK > 0 else 1.0
        row = dict(m.row_feats)
        row.update({
            "peak_ratio": peak_ratio, "term_ratio": term_ratio,
            "y_peak50": int(m.run_max >= 0.50), "y_peak100": int(m.run_max >= 1.00),
            "y_term0": int(term_ratio >= 1.0), "y_term25": int(term_ratio >= 1.25),
            "n_fwd": m.n_fwd, "censored": int(censored),
        })
        out.write(json.dumps(row, separators=(",", ":")) + "\n")
        n_rows += 1

    def evict(now_ts):
        # emit + drop mints whose forward horizon closed; also drop stale
        # undecided mints (instant-deaths that never reached K) to bound memory
        done = []; stale = []
        for mt, m in active.items():
            if m.decided:
                if (now_ts - m.trig_ts) > a.label_horizon_s:
                    done.append(mt)
            elif (now_ts - m.last_ts) > 900.0:
                stale.append(mt)
        for mt in done:
            emit(active.pop(mt), censored=False)
        for mt in stale:
            active.pop(mt, None)

    for fi, fn in enumerate(files):
        try:
            fh = open_any(fn)
        except Exception as e:
            sys.stderr.write(f"[extract] open fail {fn}: {e}\n"); continue
        for ln in fh:
            n_seen += 1
            if (n_seen % a.progress_every) == 0:
                sys.stderr.write(f"[extract] seen={n_seen} rows={n_rows} active={len(active)} file={fi+1}/{len(files)}\n")
                sys.stderr.flush()
            # fast-path: ~96% of lines are NoEvent/PumpSwap; skip parse unless relevant
            if '"TradeEvent"' not in ln and '"BlockMeta"' not in ln:
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            ev = e.get("event")
            t = e.get("t") or 0.0
            if t > cur_t:
                cur_t = t
            if ev == "BlockMeta":
                sl = e.get("slot"); ec = e.get("executed_transaction_count") or e.get("exec_tx_count")
                if sl and ec:
                    if sl not in slot_exec:
                        slot_q.append(sl)
                        if len(slot_q) > 200000:
                            slot_exec.pop(slot_q.popleft(), None)
                    slot_exec[sl] = int(ec)
                continue
            if ev != "TradeEvent":
                continue
            if e.get("failed"):
                continue
            vsol = e.get("vsol") or 0; vtok = e.get("vtok") or 0
            if vsol <= 0 or vtok <= 0:
                continue
            mint = e.get("mint")
            if not mint:
                continue
            mid = vsol / vtok
            m = active.get(mint)
            if m is None:
                # only anchor on a FRESH classic-curve launch we see near birth
                rsol = e.get("rsol") or 0
                is_classic = abs(vsol - PUMP_INIT_VSOL - rsol) < 50_000_000
                if not is_classic or rsol > a.fresh_rsol_max_lam:
                    continue
                active[mint] = Mint(e, mid)
                continue
            m.add_trade(e, mid)
            if (not m.decided) and m.n >= a.k:
                m.decided = True; m.midK = mid; m.trig_ts = t
                m.trig_slot = e.get("slot") or 0; m.n_at_trig = m.n
                base = m.baseline_feats(); comp = m.competition_feats(slot_exec)
                m.row_feats = {"mint": mint, "trig_ts": t, "trig_slot": m.trig_slot, "n_window": m.n, **base, **comp}
            # periodic eviction keyed off the stream clock
            if (n_seen % 50000) == 0:
                evict(cur_t)
        fh.close()
        evict(cur_t)

    # flush: decided mints -> emit (mark censored if horizon not fully elapsed);
    # never-decided mints (too few trades) -> dropped (instant deaths)
    for mt, m in list(active.items()):
        if m.decided:
            censored = (cur_t - m.trig_ts) <= a.label_horizon_s
            emit(m, censored=censored)
        else:
            n_drop_short += 1
    out.close()
    sys.stderr.write(f"[extract] DONE rows={n_rows} dropped_short={n_drop_short} seen={n_seen}\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
