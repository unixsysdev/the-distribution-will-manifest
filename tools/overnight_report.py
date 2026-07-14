"""Overnight session report.

Reads everything the bot + capture produced and prints a single comprehensive
markdown-formatted report:
  - Session summary (uptime, event counts)
  - Score distribution (live fires vs training reference, KS test)
  - Realized outcomes per fired mint (via grpc_capture peak_ret lookup)
  - Exit policy breakdown (de-risk / runner / death-cut / stale)
  - Paper book P&L distribution + comparison to analytical reference
  - Latency stats (bh_age, asm_ms, slot lag)
  - Reconciliation stats (landed / failed / pending)

Designed to be run any time as `python tools/overnight_report.py` and produce a
report writable straight to disk for tomorrow-morning review.

Paths default to the standard sol layout under /root/the-distribution-will-manifest.
Override via CLI flags if running locally on dumped logs.
"""
from __future__ import annotations
import argparse, gzip, glob, json, statistics as st, sys, time
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_ROOT = Path("/root/the-distribution-will-manifest")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=None, help="optional output md file path")
    ap.add_argument("--training-p50", type=float, default=0.189,
                    help="frozen training p50 score (fresh-rsol pop)")
    ap.add_argument("--training-p90", type=float, default=0.4453,
                    help="frozen training p90 score (= production threshold)")
    ap.add_argument("--analytical-plausible", type=float, default=0.401,
                    help="OOS PLAUSIBLE expected per-bet from K_combined backtest")
    return ap.parse_args()


def load_jsonl(path: Path, gz: bool = False):
    out = []
    if not path.exists(): return out
    opener = gzip.open if gz or path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for ln in f:
            try: out.append(json.loads(ln))
            except Exception: continue
    return out


def fmt_pct(p): return f"p25={p[0]:.4f} p50={p[1]:.4f} p75={p[2]:.4f} p90={p[3]:.4f} p95={p[4]:.4f} max={p[5]:.4f}"


def quantiles(xs):
    if not xs: return (0,)*6
    s = sorted(xs)
    def q(p): return s[min(int(len(s)*p), len(s)-1)]
    return (q(.25), q(.50), q(.75), q(.90), q(.95), s[-1])


def ks_2sample(a, b):
    """Simple KS statistic (no scipy dep). Returns D."""
    if not a or not b: return None
    a, b = sorted(a), sorted(b)
    i = j = 0; D = 0
    na, nb = len(a), len(b)
    while i < na and j < nb:
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1
        D = max(D, abs(i/na - j/nb))
    return D


def main():
    args = parse_args()
    root = Path(args.root)
    bot_data = root/"bot_data"
    logs = root/"logs"
    capture_dir = root/"grpc_capture"

    print(f"\n=== Overnight Report — generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===")
    print(f"root: {root}\n")

    # ---- 1) bot decisions + position history ----
    decisions = load_jsonl(bot_data/"shadow_run.jsonl")
    positions = load_jsonl(bot_data/"positions.jsonl")
    if not decisions:
        print("no shadow_run.jsonl found — bot hasn't written anything yet."); return

    # Group by event kind
    by_kind = Counter(r.get("kind","?") for r in decisions)
    entry_decisions = [r for r in decisions if r.get("kind") == "entry_decision"]
    fires = [r for r in entry_decisions if r.get("fire")]
    k_triggers = [r for r in decisions if r.get("kind") == "k_trigger"]
    v_triggers = [r for r in decisions if r.get("kind") == "v_trigger"]
    death_cuts = [r for r in decisions if r.get("kind") == "live_death_cut"]
    scale_slices = [r for r in decisions if r.get("kind") == "live_scale_slice"]
    runner_exits = [r for r in decisions if r.get("kind") == "live_runner_exit"]
    position_closes = [r for r in decisions if r.get("kind") == "position_close"]
    recon_rollbacks = [r for r in decisions if r.get("kind") == "recon_rollback"]
    recon_buy_failed = [r for r in decisions if r.get("kind") == "recon_buy_failed"]

    if decisions:
        t_start = min(r.get("t",0) for r in decisions if r.get("t",0) > 0)
        t_end = max(r.get("t",0) for r in decisions)
        uptime_s = t_end - t_start
    else:
        uptime_s = 0
    print(f"## Session")
    print(f"  uptime              {uptime_s/3600:.2f} h ({uptime_s/60:.1f} min)")
    print(f"  start (UTC)         {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t_start)) if uptime_s else '-'}")
    print(f"  end (UTC)           {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t_end)) if uptime_s else '-'}")
    print(f"  event kinds         {dict(by_kind)}")
    print()

    # ---- 2) trigger + fire activity ----
    print(f"## Activity")
    print(f"  k_triggers           {len(k_triggers)}")
    print(f"  v_triggers           {len(v_triggers)}")
    print(f"  ready (both fired)   {len(entry_decisions)}")
    print(f"  fires                {len(fires)} ({100*len(fires)/max(len(entry_decisions),1):.1f}% of ready)")
    if uptime_s > 0:
        print(f"  fire rate            {len(fires)*3600/uptime_s:.1f} fires/hour")
        print(f"  ready rate           {len(entry_decisions)*3600/uptime_s:.1f} ready/hour")
    print()

    # ---- 3) score distribution ----
    print(f"## Scores")
    all_scores = [r.get("score") for r in entry_decisions if r.get("score") is not None]
    fired_scores = [r.get("score") for r in fires if r.get("score") is not None]
    if all_scores:
        print(f"  all ready (n={len(all_scores)})  {fmt_pct(quantiles(all_scores))}")
    if fired_scores:
        print(f"  fired     (n={len(fired_scores)})  {fmt_pct(quantiles(fired_scores))}")
    print(f"  training reference  p50 {args.training_p50:.4f}  p90 {args.training_p90:.4f}")
    if all_scores:
        # KS-statistic approximation via uniform-mixed training distribution (we don't
        # have a fresh training sample on hand; just report quantile gap).
        live_p90 = quantiles(all_scores)[3]
        gap = live_p90 - args.training_p90
        print(f"  live p90 gap vs training p90:  {gap:+.4f}")
        if all_scores:
            cutoff_for_10pct = quantiles(all_scores)[3]
            print(f"  threshold for ~10% fire rate on observed live distribution: {cutoff_for_10pct:.4f}")
    print()

    # ---- 4) realized outcomes via grpc_capture lookup ----
    if not fires:
        print("## Realized outcomes\n  (no fires yet, skipping)\n")
    else:
        print(f"## Realized outcomes (peak_ret via grpc_capture lookup)")
        # Build index: mint -> list of (ev_ts, vsol, vtok) trades from capture
        fired_mints = set(r["mint"] for r in fires)
        midK_by_mint = {r["mint"]: r.get("midK") for r in fires}
        trigger_ts_by_mint = {r["mint"]: r.get("k_window_last_ts") or r.get("v_window_last_ts") for r in fires}
        cap_trades = defaultdict(list)
        files = sorted(glob.glob(str(capture_dir/"*.jsonl*")))
        for path in files:
            opener = gzip.open if path.endswith(".gz") else open
            try:
                with opener(path, "rt") as f:
                    for ln in f:
                        try: rec = json.loads(ln)
                        except: continue
                        m = rec.get("mint")
                        if m in fired_mints:
                            cap_trades[m].append((rec.get("ev_ts",0), rec.get("vsol",0), rec.get("vtok",0)))
            except Exception: continue

        # For each fired mint, compute peak_ret from forward trades
        outcomes = []
        for r in fires:
            m = r["mint"]; midK = midK_by_mint.get(m); trig_ts = trigger_ts_by_mint.get(m)
            if not midK or not trig_ts: continue
            forward = [(ts, vs, vt) for ts, vs, vt in cap_trades.get(m, []) if ts >= trig_ts]
            if len(forward) < 5: continue
            mids = [vs/vt for ts, vs, vt in forward if vt > 0]
            if not mids: continue
            peak_ret = max(mids)/midK - 1.0
            terminal_ret = mids[-1]/midK - 1.0
            outcomes.append({"mint": m[:14], "score": r["score"],
                             "peak_ret": peak_ret, "terminal_ret": terminal_ret,
                             "n_forward": len(forward)})

        if not outcomes:
            print(f"  ({len(fires)} fires but no sufficient capture data — capture may not have covered the time window)")
        else:
            n = len(outcomes)
            winners_2x = sum(1 for o in outcomes if o["peak_ret"] >= 1.0)
            winners_5x = sum(1 for o in outcomes if o["peak_ret"] >= 4.0)
            winners_10x = sum(1 for o in outcomes if o["peak_ret"] >= 9.0)
            print(f"  fires analyzed         {n} / {len(fires)}")
            print(f"  peak_ret >= 2x         {winners_2x} ({100*winners_2x/n:.1f}%)   training ref 28%")
            print(f"  peak_ret >= 5x         {winners_5x} ({100*winners_5x/n:.1f}%)")
            print(f"  peak_ret >= 10x        {winners_10x} ({100*winners_10x/n:.1f}%)")
            peaks = [o["peak_ret"] for o in outcomes]
            terms = [o["terminal_ret"] for o in outcomes]
            print(f"  realized peak_ret      mean {sum(peaks)/n:+.3f}  median {st.median(peaks):+.3f}  max {max(peaks):+.3f}")
            print(f"  realized terminal_ret  mean {sum(terms)/n:+.3f}  median {st.median(terms):+.3f}  max {max(terms):+.3f}")
            # score correlation
            win_scores = [o["score"] for o in outcomes if o["peak_ret"] >= 1.0]
            los_scores = [o["score"] for o in outcomes if o["peak_ret"] < 1.0]
            if win_scores and los_scores:
                print(f"  winners scores  mean {sum(win_scores)/len(win_scores):.4f}  median {st.median(win_scores):.4f}  max {max(win_scores):.4f}")
                print(f"  losers  scores  mean {sum(los_scores)/len(los_scores):.4f}  median {st.median(los_scores):.4f}  max {max(los_scores):.4f}")
        print()

    # ---- 5) exit policy breakdown ----
    print(f"## Exit policy mix (K_combined 4+4)")
    print(f"  total fires                {len(fires)}")
    print(f"  death-cuts                 {len(death_cuts)}")
    print(f"  scale_slice events         {len(scale_slices)}")
    if scale_slices:
        phase_count = Counter(s.get("phase") for s in scale_slices)
        for ph, n in phase_count.most_common():
            print(f"    phase={ph}             {n}")
    print(f"  runner exits (trailing)    {len(runner_exits)}")
    print(f"  positions closed (book)    {len(position_closes)}")
    if position_closes:
        close_reasons = Counter(c.get("reason") for c in position_closes)
        for rn, n in close_reasons.most_common():
            print(f"    reason={rn}             {n}")
    print()

    # ---- 6) paper book P&L (from positions.jsonl close events) ----
    print(f"## Paper book P&L")
    closes = [p for p in positions if p.get("kind") == "close"]
    rets = [p["net_return"] for p in closes if isinstance(p.get("net_return"), (int, float))]
    if rets:
        rets_sorted = sorted(rets)
        n = len(rets); mean = sum(rets)/n; med = st.median(rets)
        win = sum(1 for r in rets if r > 0)
        print(f"  closed positions          {n}")
        print(f"  mean per-bet              {mean:+.4f} SOL  (analytical ref: +{args.analytical_plausible:.4f})")
        print(f"  median per-bet            {med:+.4f} SOL")
        print(f"  win rate                  {100*win/n:.1f}%")
        print(f"  total session             {sum(rets):+.2f} SOL")
        print(f"  best / worst              {max(rets):+.3f} / {min(rets):+.3f} SOL")
        gap = mean - args.analytical_plausible
        print(f"  gap vs analytical ref     {gap:+.4f} SOL/bet")
    else:
        print("  (no closes yet)")
    print()

    # ---- 7) bundle assembly latency (DRY_RUN trace) ----
    bundles = load_jsonl(logs/"broker_jito.jsonl")
    if bundles:
        print(f"## Bundle assembly (DRY_RUN trace, n={len(bundles)})")
        bh_ages = [b.get("bh_age_ms",0) for b in bundles if b.get("bh_age_ms") is not None]
        asm = [b.get("asm_ms",0) for b in bundles if b.get("asm_ms") is not None]
        slots = [b.get("slot") for b in bundles if b.get("slot") is not None]
        ops = Counter(b.get("op") for b in bundles)
        print(f"  ops                       {dict(ops)}")
        if bh_ages:
            q = quantiles(bh_ages)
            print(f"  bh_age_ms (n={len(bh_ages)})        p50={q[1]:.0f} p90={q[3]:.0f} p95={q[4]:.0f} max={q[5]:.0f}")
        if asm:
            q = quantiles(asm)
            print(f"  asm_ms                    p50={q[1]:.0f} p90={q[3]:.0f} max={q[5]:.0f}")
        if slots:
            print(f"  ev_slot populated         {len(slots)}/{len(bundles)} (gRPC source on {100*len(slots)/len(bundles):.0f}% of bundles)")
        print()

    # ---- 8) reconciliation stats ----
    recon = load_jsonl(logs/"broker_recon.jsonl")
    print(f"## Reconciliation")
    print(f"  recon events              {len(recon)}")
    print(f"  recon_rollback (sells)    {len(recon_rollbacks)}")
    print(f"  recon_buy_failed          {len(recon_buy_failed)}")
    if recon:
        kinds = Counter(r.get("kind") for r in recon)
        print(f"  by kind                   {dict(kinds)}")
        landed = [r for r in recon if r.get("kind") == "landed"]
        failed = [r for r in recon if r.get("kind") == "failed"]
        if landed:
            lats = [r.get("latency_s",0) for r in landed]
            tips = [r.get("tip_lam",0) for r in landed]
            print(f"  landed n={len(landed)}: latency_s p50={st.median(lats):.2f} tip_lam p50={st.median(tips):.0f}")
        if failed:
            tips = [r.get("tip_lam",0) for r in failed]
            reasons = Counter(r.get("reason") for r in failed)
            print(f"  failed n={len(failed)}: tip_lam p50={st.median(tips):.0f}  reasons={dict(reasons)}")
            if landed and failed:
                print(f"  -> tip-vs-landing-rate: landed_tip_median={st.median([r.get('tip_lam',0) for r in landed]):.0f} "
                      f"vs failed_tip_median={st.median([r.get('tip_lam',0) for r in failed]):.0f}")
    else:
        print(f"  (no recon events — DRY_RUN, nothing submitted to chain)")
    print()

    # ---- 9) capture archive stats ----
    print(f"## Capture archive")
    files = sorted(glob.glob(str(capture_dir/"*.jsonl*")))
    total_sz = sum(Path(p).stat().st_size for p in files)
    print(f"  files                     {len(files)}")
    print(f"  total size                {total_sz/(1<<20):.1f} MB")
    print()

    if args.out:
        # also write to a file (re-run with output redirect to capture)
        print(f"\n(use shell redirect to save: python overnight_report.py > {args.out})")


if __name__ == "__main__":
    main()
