"""continuation_executor.py — DRY-RUN, decision-realistic continuation executor
(2026-06-14).

PURE SIMULATION. No broker, no wallet, no network submission — it cannot place a
trade. It tails the live grpc capture, runs ContinuationTracker, SCORES each 2x
cross with the robust June model, and for the crosses the model SELECTS it computes
the correctly-sized would-be fill (continuation_sizing.plan_buy + simulate_fill:
hard cap spend<=bet, reverts fills that drifted beyond cap) and the HONEST return
(realized_return, on actual outlay incl. tip+fees).

This is the last bench step before arming a real (small) bet: it proves the full
plumbing — selection -> sizing -> fill/revert -> accounting — on live data, so the
ONLY thing left unmeasured is P(gap-0), which needs small-live.

It also logs, for every SELECTED-but-REVERTED entry, the counterfactual outcome the
tracker observed (trk_ret), so we can check whether the cap reverts losers (good) or
winners (bad) before risking anything.

Run live (tail):   python -m research.continuation.continuation_executor
Validate offline:  python -m research.continuation.continuation_executor --replay <capture.jsonl[.gz]> --max-outcomes 15
"""
import argparse, base64, glob, gzip, json, os, time, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pumpfun_parse import parse_trade_event
from .continuation_tracker import ContinuationTracker
from .continuation_sizing import plan_buy, simulate_fill, realized_return, LAMPORTS_PER_SOL

ROOT = "/root/the-distribution-will-manifest"
CAP_GLOB = f"{ROOT}/grpc_capture/capture_*.jsonl"
FE = ["dd", "bf", "ntr", "recent", "tps", "uniq"]   # training order (june_v2_panel)


def train_model(panel_path, tier):
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(panel_path)
    X = df[FE].values.astype(float); y = df["y"].values.astype(int)
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    cutoff = float(np.quantile(p, 1.0 - tier))
    insample_auc = roc_auc_score(y, p)
    print(f"[exec] model trained on {len(df)} June crosses  base_hit={y.mean():.1%}  "
          f"in-sample_AUC={insample_auc:.3f} (sanity only)  "
          f"tier=top{tier:.0%} -> p>={cutoff:.3f} selects", flush=True)
    return clf, cutoff


def feats_of(e):
    # tracker emits buy_frac; panel calls it bf — map by position into FE order
    return [e["dd"], e["buy_frac"], e["ntr"], e["recent"], e["tps"], e["uniq"]]


def newest_capture():
    fs = [f for f in glob.glob(CAP_GLOB) if not f.endswith(".gz")]
    return max(fs, key=os.path.getmtime) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bet-sol", type=float, default=0.05)
    ap.add_argument("--cap-bps", type=int, default=2500)
    ap.add_argument("--tip-sol", type=float, default=0.005)
    ap.add_argument("--tier", type=float, default=0.25, help="select top-`tier` by model score")
    ap.add_argument("--panel", default=f"{ROOT}/bot_data/june_v2_panel.parquet")
    ap.add_argument("--out", default=f"{ROOT}/bot_data/continuation_executor.jsonl")
    ap.add_argument("--replay", default=None, help="capture file to read from START (offline validate)")
    ap.add_argument("--max-outcomes", type=int, default=0, help="stop after N resolved selected entries (replay)")
    ap.add_argument("--max-lines", type=int, default=0, help="stop after N lines (replay safety bound)")
    ap.add_argument("--stdin", action="store_true", help="replay lines piped on stdin (zcat|grep|python multi-file)")
    args = ap.parse_args()

    bet_lam = int(args.bet_sol * LAMPORTS_PER_SOL)
    tip_lam = int(args.tip_sol * LAMPORTS_PER_SOL)
    clf, cutoff = train_model(args.panel, args.tier)
    trk = ContinuationTracker()
    sel = {}                         # mint -> {plan, cross_mid, p, ts, fill}
    out = open(args.out, "a")
    st = {"cross": 0, "selected": 0, "filled": 0, "reverted": 0,
          "net_lam": 0, "sum_outlay_ret": 0.0, "sum_curve_ret": 0.0, "sum_revert_trk": 0.0}

    def emit(rec):
        out.write(json.dumps(rec) + "\n"); out.flush()

    def on_event(e, vsol, vtok, ts):
        m = e["mint"]; k = e["kind"]
        if k == "cross":
            st["cross"] += 1
            p = float(clf.predict_proba(np.array([feats_of(e)], float))[0, 1])
            chosen = p >= cutoff
            rec = {"kind": "decision", "mint": m, "t": ts, "p": round(p, 4),
                   "selected": chosen, "dd": round(e["dd"], 4), "bf": round(e["buy_frac"], 3),
                   "ntr": e["ntr"], "recent": round(e["recent"], 4),
                   "tps": round(e["tps"], 3), "uniq": e["uniq"]}
            if chosen:
                st["selected"] += 1
            # Size + account EVERY cross and log its model `p`; we tier by the LIVE
            # score distribution OFFLINE (a frozen June cutoff over-selects the live
            # regime). `selected` is kept only as an informational flag.
            plan = plan_buy(vsol, vtok, bet_lam, args.cap_bps)
            sel[m] = {"plan": plan, "cross_mid": e["cross_mid"], "p": p, "ts": ts, "fill": None}
            rec["token_amount"] = plan.token_amount
            rec["max_sol_cost"] = plan.max_sol_cost_lam
            rec["ref_cost_lam"] = plan.ref_curve_cost_lam
            emit(rec)
        elif k == "fill" and m in sel:
            s = sel[m]
            fr = simulate_fill(vsol, vtok, s["plan"], tip_lam=tip_lam)
            s["fill"] = fr
            emit({"kind": "fill", "mint": m, "t": ts, "filled": fr.filled,
                  "curve_cost_lam": fr.curve_cost_lam, "outlay_lam": fr.total_outlay_lam,
                  "exec_slip": round(fr.exec_slip, 4) if fr.filled else None,
                  "revert_reason": fr.revert_reason})
        elif k == "outcome" and m in sel:
            s = sel.pop(m); fr = s["fill"]
            rec = {"kind": "outcome", "mint": m, "t": ts, "p": round(s["p"], 4),
                   "trk_y": e["y"], "trk_ret": round(e["ret"], 4)}
            if fr is not None and fr.filled:
                tr = realized_return(fr, s["plan"], vsol, vtok)
                rec.update({"filled": True, "ret_curve": round(tr.return_on_curve, 4),
                            "ret_outlay": round(tr.return_on_outlay, 4), "net_lam": tr.net_pnl_lam})
                st["filled"] += 1; st["net_lam"] += tr.net_pnl_lam
                st["sum_outlay_ret"] += tr.return_on_outlay; st["sum_curve_ret"] += tr.return_on_curve
            else:
                rec.update({"filled": False, "note": "reverted at fill (drift>cap) -> counterfactual only"})
                st["reverted"] += 1; st["sum_revert_trk"] += e["ret"]
            emit(rec)
            return True   # a selected entry resolved
        return False

    def print_stats(tag=""):
        f = st["filled"]; r = st["reverted"]
        fr_rate = f / max(1, f + r)
        line = (f"[exec]{tag} crosses={st['cross']} selected={st['selected']} "
                f"filled={f} reverted={r} fill_rate={fr_rate:.0%} "
                f"net={st['net_lam']/LAMPORTS_PER_SOL:+.4f}SOL")
        if f:
            line += (f" | filled mean ret_outlay={st['sum_outlay_ret']/f:+.3f} "
                     f"ret_curve={st['sum_curve_ret']/f:+.3f}")
        if r:
            line += f" | reverted mean trk_ret={st['sum_revert_trk']/r:+.3f} (cap's counterfactual)"
        line += "  [all crosses; tier top10/25% by p OFFLINE]"
        print(line, flush=True)

    def process_line(line):
        if '"TradeEvent"' not in line:
            return 0
        try:
            r = json.loads(line); ev = parse_trade_event(base64.b64decode(r["raw"]))
        except Exception:
            return 0
        if ev is None or not ev.is_classic_curve or ev.virtual_token_reserves <= 0:
            return 0
        ts = r.get("t") or time.time()
        resolved = 0
        for e in trk.update(ev.mint, ev.virtual_sol_reserves, ev.virtual_token_reserves,
                            ev.is_buy, ts, ev.user):
            if on_event(e, ev.virtual_sol_reserves, ev.virtual_token_reserves, ts):
                resolved += 1
        return resolved

    # ---------- REPLAY (offline validation) ----------
    if args.replay:
        print(f"[exec] REPLAY {args.replay}  bet={args.bet_sol} cap={args.cap_bps}bps "
              f"tip={args.tip_sol}  (DRY-RUN, no broker)", flush=True)
        opn = gzip.open if args.replay.endswith(".gz") else open
        resolved_total = nlines = 0
        with opn(args.replay, "rt") as f:
            for line in f:
                nlines += 1
                resolved_total += process_line(line)
                if args.max_lines and nlines >= args.max_lines:
                    print(f"[exec] hit max-lines {args.max_lines}", flush=True); break
                if args.max_outcomes and resolved_total >= args.max_outcomes:
                    print(f"[exec] hit max-outcomes {args.max_outcomes}", flush=True); break
                if nlines % 250_000 == 0:
                    print_stats(f" @{nlines//1000}k lines")
        print_stats(" FINAL")
        return

    # ---------- STDIN STREAM (fast multi-file replay: zcat|grep|python --stdin) ----------
    if args.stdin:
        import sys as _sys
        print("[exec] STDIN replay (DRY-RUN, no broker, no submit)", flush=True)
        nlines = 0
        for line in _sys.stdin:
            nlines += 1
            process_line(line)
            if nlines % 2_000_000 == 0:
                print_stats(f" @{nlines//1000}k lines")
        print_stats(" FINAL")
        return

    # ---------- LIVE TAIL ----------
    print(f"[exec] LIVE bet={args.bet_sol} cap={args.cap_bps}bps tip={args.tip_sol} "
          f"tier=top{args.tier:.0%}  DRY-RUN (no broker, no wallet, cannot submit)", flush=True)
    cur = newest_capture()
    while cur is None:
        time.sleep(2); cur = newest_capture()
    f = open(cur, "r"); f.seek(0, os.SEEK_END)
    print(f"[exec] tailing {cur} from EOF", flush=True)
    last_prune = last_stat = time.time()
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.25); now = time.time()
            nf = newest_capture()
            if nf and nf != cur:
                f.close(); cur = nf; f = open(cur, "r")
                print(f"[exec] rotated -> {cur}", flush=True)
            if now - last_prune > 120:
                trk.prune(now)
                for m in [m for m in sel if m not in trk.state]:
                    sel.pop(m, None)
                last_prune = now
            if now - last_stat > 120:
                print_stats(); last_stat = now
            continue
        process_line(line)


if __name__ == "__main__":
    main()
