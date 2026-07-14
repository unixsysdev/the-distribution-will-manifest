#!/usr/bin/env python3
"""2x-CONTINUATION augmented panel extractor — single causal pass over the RAW
gRPC firehose (the 'all data' source), driving the PROVEN RichTracker.

Decision = first 2x-from-launch cross (RichTracker, parity with live). Label =
TP(+0.5x) vs STOP(-0.3x) from the REALISTIC post-cross fill (next trade's mid,
NOT the cross mid -> no gap-0 lookahead). Every feature is frozen at-or-before
the cross trade; the label uses only trades strictly after the fill.

Features at the cross:
  RICH (22)            : the continuation_tracker_rich set (current model superset)
  firehose-only (NEW)  : per-mint LANDED-FAILED-BUY pressure (the 400% flood,
                         attributed via pump-ix Account[2]=mint) + per-slot BLOCK
                         CONGESTION (executed_transaction_count) at the cross slot.
All firehose-only aggregates accumulate as the stream flows and are snapshotted
at the cross => strictly causal.

Raw frame format (docs/schemas/grpc_firehose.md):
`<QQI recv_ns, slot, payload_len> + SubscribeUpdate`.
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, struct, sys, time
from collections import deque
from pathlib import Path

ROOT = os.getenv("PUMPFUN_ROOT", str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/grpc_stubs")
import geyser_pb2  # noqa
from .continuation_tracker_rich import RichTracker, RICH  # parity with live
from pumpfun_parse import parse_trade_event, TRADE_EVENT_DISC  # bonding-curve TradeEvent decoder
import base64, base58

PUMP_FUN_PROG = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
# pump buy discriminators (first 8 bytes of ix.data); mint = ix.accounts[2]
BUY_DISCS = {
    bytes.fromhex("66063d1201daebea"),  # buy
    bytes.fromhex("b817ee6167c5d33d"),  # buy_v2
    bytes.fromhex("c2ab1c46684d5b2f"),  # buy_exact_quote_in_v2
    bytes.fromhex("38fc74089edfcd5f"),  # buy_exact_sol_in
}
HDR = struct.calcsize("<QQI")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--firehose-dirs", nargs="+", default=[
        ROOT + "/buffer/grpc_firehose/ready",
        "/mnt/storagebox/backup/grpc_firehose",
    ])
    ap.add_argument("--since", default="20260613")
    ap.add_argument("--until", default="99999999")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--fresh-rsol-max-lam", type=int, default=8_000_000_000)
    ap.add_argument("--out", default=ROOT + "/bot_data/cont_2x_aug_panel.jsonl")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=2000000)
    return ap.parse_args()


def list_files(dirs, since, until):
    seen = {}
    for d in dirs:
        for fn in glob.glob(os.path.join(d, "grpc-firehose-*.bin*")):
            base = os.path.basename(fn)
            try:
                stamp = base.split("grpc-firehose-")[1][:8]
            except Exception:
                continue
            if stamp < since or stamp > until:
                continue
            key = base.replace(".gz", "")
            prev = seen.get(key)
            if prev is None or (prev.endswith(".gz") and not fn.endswith(".gz")):
                seen[key] = fn
    return [seen[k] for k in sorted(seen)]


def open_any(fn):
    return gzip.open(fn, "rb") if fn.endswith(".gz") else open(fn, "rb")


def find_failed_buy_mint(u):
    """For a FAILED pump tx, return (mint, signer) if it carries a pump BUY ix, else None.
    mint = pump-ix Account[2]; signer = Account[6] (intent_extractor layout)."""
    try:
        txi = u.transaction.transaction
        msg = txi.transaction.message
        keys = [base58.b58encode(bytes(k)).decode() for k in msg.account_keys]
        # resolved key space = static + loaded (V0 ALT) for account-index lookups
        meta = txi.meta
        full = keys + [base58.b58encode(bytes(a)).decode() for a in meta.loaded_writable_addresses] \
                    + [base58.b58encode(bytes(a)).decode() for a in meta.loaded_readonly_addresses]
        for ix in msg.instructions:
            if ix.program_id_index >= len(keys):
                continue
            if keys[ix.program_id_index] != PUMP_FUN_PROG:
                continue
            d = bytes(ix.data)[:8]
            if d not in BUY_DISCS:
                continue
            accs = list(ix.accounts)
            if len(accs) < 7:
                continue
            mi, si = accs[2], accs[6]
            mint = full[mi] if mi < len(full) else None
            signer = full[si] if si < len(full) else None
            return mint, signer
    except Exception:
        return None
    return None


class FailAcc:
    __slots__ = ("nfail", "signers", "prios", "nbuy_ok", "fail_ts", "last_t")
    def __init__(self):
        self.nfail = 0; self.signers = set(); self.prios = []
        self.nbuy_ok = 0; self.fail_ts = deque(maxlen=400); self.last_t = 0.0


def _prio_micro(u):
    """Extract requested priority-fee (micro-lamports/CU) from ComputeBudget ix 0x03."""
    try:
        msg = u.transaction.transaction.transaction.message
        keys = msg.account_keys
        CB = "ComputeBudget111111111111111111111111111111"
        for ix in msg.instructions:
            if ix.program_id_index < len(keys) and base58.b58encode(bytes(keys[ix.program_id_index])).decode() == CB:
                d = bytes(ix.data)
                if len(d) >= 9 and d[0] == 0x03:
                    return int.from_bytes(d[1:9], "little")
    except Exception:
        pass
    return 0


def main():
    a = parse_args()
    files = list_files(a.firehose_dirs, a.since, a.until)
    if a.max_files:
        files = files[:a.max_files]
    sys.stderr.write(f"[c2x] {len(files)} firehose files {a.since}->{a.until} k={a.k}\n"); sys.stderr.flush()
    out = open(a.out, "w")
    trk = RichTracker(k=a.k)
    fails = {}                 # mint -> FailAcc
    seen_mint = {}             # mint -> True(track)/False(ignore non-fresh)
    pending = {}               # mint -> partial row (cross snapshot awaiting outcome)
    slot_exec = {}; slot_entries = {}; slot_q = deque()
    recent_pump_ts = deque(); recent_fail_ts = deque()   # pump-family tx arrival times (causal congestion RATE)
    cur_slot_val = 0; cur_slot_pump = 0                   # intra-slot pump-tx count so far (causal)
    n_seen = n_cross = n_rows = n_failbuy = 0
    cur_t = 0.0; t0 = time.time()

    def snap_fail(mint, cross_t):
        fa = fails.get(mint)
        if fa is None:
            return {"ff_nfail": 0, "ff_nfail_signers": 0, "ff_fail_rate": 0.0,
                    "ff_fail_prio_mean": 0.0, "ff_fail_prio_max": 0.0, "ff_fail_recent5s": 0}
        nb = fa.nbuy_ok
        rate = fa.nfail / (fa.nfail + nb) if (fa.nfail + nb) > 0 else 0.0
        recent = sum(1 for ts in fa.fail_ts if cross_t - ts <= 5.0)
        pr = fa.prios
        return {"ff_nfail": fa.nfail, "ff_nfail_signers": len(fa.signers), "ff_fail_rate": rate,
                "ff_fail_prio_mean": (sum(pr) / len(pr)) if pr else 0.0,
                "ff_fail_prio_max": max(pr) if pr else 0.0, "ff_fail_recent5s": recent}

    def snap_cong(slot, cross_t):
        # STRICTLY-PRIOR slots only for block_meta: the cross slot's final tx-count
        # is unknown until it closes (~400ms after the decision) -> would peek ahead.
        if slot:
            ex = [slot_exec[s] for s in range(slot - 10, slot) if s in slot_exec]
            en = [slot_entries[s] for s in range(slot - 10, slot) if s in slot_entries]
            last = max((s for s in range(slot - 10, slot) if s in slot_exec), default=0)
            d = {"cong_exec_prior10": (sum(ex) / len(ex)) if ex else 0.0,
                 "cong_exec_last": slot_exec.get(last, 0),
                 "cong_entries_prior10": (sum(en) / len(en)) if en else 0.0,
                 "cong_slot_partial": cur_slot_pump}
        else:
            d = {"cong_exec_prior10": 0.0, "cong_exec_last": 0, "cong_entries_prior10": 0.0, "cong_slot_partial": 0}
        # REAL-TIME causal congestion RATE: pump-family tx (and failed) arrivals in the
        # trailing window up to the cross -> "how hard the flood is happening right now".
        d["cong_pumptx_1s"] = sum(1 for t in recent_pump_ts if cross_t - t <= 1.0)
        d["cong_pumptx_5s"] = sum(1 for t in recent_pump_ts if cross_t - t <= 5.0)
        d["cong_failtx_1s"] = sum(1 for t in recent_fail_ts if cross_t - t <= 1.0)
        d["cong_failtx_5s"] = sum(1 for t in recent_fail_ts if cross_t - t <= 5.0)
        return d

    for fi, fn in enumerate(files):
        try:
            f = open_any(fn)
        except Exception as e:
            sys.stderr.write(f"[c2x] open fail {fn}: {e}\n"); continue
        while True:
            h = f.read(HDR)
            if len(h) < HDR:
                break
            recv_ns, slot, pl = struct.unpack("<QQI", h)
            payload = f.read(pl)
            n_seen += 1
            if (n_seen % a.progress_every) == 0:
                sys.stderr.write(f"[c2x] seen={n_seen} cross={n_cross} rows={n_rows} failbuy={n_failbuy} "
                                 f"active={len(trk.state)} pend={len(pending)} f={fi+1}/{len(files)} "
                                 f"({n_seen/max(1,time.time()-t0):.0f}/s)\n"); sys.stderr.flush()
            u = geyser_pb2.SubscribeUpdate()
            try:
                u.ParseFromString(payload)
            except Exception:
                continue
            if u.HasField("block_meta"):
                bm = u.block_meta
                s = bm.slot
                if s not in slot_exec:
                    slot_q.append(s)
                    if len(slot_q) > 300000:
                        old = slot_q.popleft(); slot_exec.pop(old, None); slot_entries.pop(old, None)
                slot_exec[s] = int(bm.executed_transaction_count)
                slot_entries[s] = int(bm.entries_count)
                continue
            if not u.HasField("transaction"):
                continue
            ts = recv_ns / 1e9
            if ts > cur_t:
                cur_t = ts
            # causal real-time congestion: every firehose tx frame is pump-family (server-filtered)
            if slot != cur_slot_val:
                cur_slot_val = slot; cur_slot_pump = 0
            cur_slot_pump += 1
            recent_pump_ts.append(ts)
            lo = ts - 12.0
            while recent_pump_ts and recent_pump_ts[0] < lo:
                recent_pump_ts.popleft()
            meta = u.transaction.transaction.meta
            failed = bool(meta.err.err) if meta.HasField("err") else False
            if failed:
                recent_fail_ts.append(ts)
                while recent_fail_ts and recent_fail_ts[0] < lo:
                    recent_fail_ts.popleft()
                fb = find_failed_buy_mint(u)
                if fb and fb[0]:
                    mint, signer = fb
                    fa = fails.get(mint)
                    if fa is None:
                        fa = fails[mint] = FailAcc()
                    fa.nfail += 1; fa.last_t = ts
                    if signer:
                        fa.signers.add(signer)
                    fa.prios.append(_prio_micro(u)); fa.fail_ts.append(ts)
                    n_failbuy += 1
                continue
            # success: decode bonding-curve TradeEvent from Program data logs
            ev = None
            for ln in meta.log_messages:
                if "Program data:" in ln:
                    try:
                        data = base64.b64decode(ln.split("Program data:", 1)[1].strip())
                    except Exception:
                        continue
                    if len(data) >= 8 and bytes(data[:8]) == TRADE_EVENT_DISC:
                        ev = parse_trade_event(data);
                        if ev: break
            if ev is None:
                continue
            mint = ev.mint
            track = seen_mint.get(mint)
            if track is None:
                track = ev.is_classic_curve and ev.real_sol_reserves <= a.fresh_rsol_max_lam
                seen_mint[mint] = track
            if not track:
                continue
            # count successful buys for this mint (fail-rate denominator)
            if ev.is_buy:
                fa = fails.get(mint)
                if fa is None:
                    fa = fails[mint] = FailAcc()
                fa.nbuy_ok += 1
            evs = trk.update(mint, ev.virtual_sol_reserves, ev.virtual_token_reserves,
                             ev.real_sol_reserves, ev.sol_amount, ev.is_buy, ts, ev.user)
            for e in evs:
                k = e["kind"]
                if k == "cross":
                    n_cross += 1
                    row = {fld: e[fld] for fld in (RICH + ["cross_mid"])}
                    row["mint"] = mint; row["cross_t"] = e["t"]; row["cross_slot"] = slot
                    row.update(snap_fail(mint, e["t"]))
                    row.update(snap_cong(slot, e["t"]))
                    pending[mint] = row
                elif k == "fill":
                    if mint in pending:
                        fm = e["fill_mid"]; cm = pending[mint]["cross_mid"]
                        pending[mint]["fill_mid"] = fm
                        pending[mint]["entry_slip"] = (fm / cm - 1.0) if cm else 0.0
                elif k == "outcome":
                    row = pending.pop(mint, None)
                    if row is not None:
                        row["y"] = e["y"]; row["ret"] = e["ret"]
                        row["dur_s"] = e["t"] - row["cross_t"]
                        out.write(json.dumps(row, separators=(",", ":")) + "\n"); n_rows += 1
            # bounded memory: prune dead mints from tracker + fail/pending mirrors
            if (n_seen % 200000) == 0:
                dead = [m for m, s in trk.state.items() if (cur_t - s.last_t) > 1800 or s.phase == "done"]
                for m in dead:
                    trk.state.pop(m, None)
                    if m not in pending:
                        fails.pop(m, None)
                for m in [m for m, fa in fails.items() if (cur_t - fa.last_t) > 1800 and m not in trk.state and m not in pending]:
                    fails.pop(m, None)
        f.close()
    out.close()
    sys.stderr.write(f"[c2x] DONE rows={n_rows} cross={n_cross} failbuy={n_failbuy} seen={n_seen}\n"); sys.stderr.flush()


if __name__ == "__main__":
    main()
