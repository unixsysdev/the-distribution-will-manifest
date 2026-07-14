"""Correlate policy swaps with realized outcomes.

For each swap event in logs/policy_decisions.jsonl, show:
  - swap time + from/to policies + uplift the selector expected
  - n_fires before/after the swap (capped at ~window before, all after)
  - mean/median realized peak_ret + paper P&L before vs after
  - actual uplift vs predicted uplift (did the selector pick well?)

Auditable cause -> effect chain so we can verify the bot's self-modifications
actually helped (or didn't) in retrospect.

Read-only; uses bot_data/shadow_run.jsonl (entry_decisions stamped with
exit_policy) + bot_data/positions.jsonl (close net_return) + grpc_capture for
peak_ret ground truth.
"""
from __future__ import annotations
import argparse, gzip, glob, json, sys, time
from pathlib import Path
from collections import defaultdict
import statistics as st

DEFAULT_ROOT = Path("/root/the-distribution-will-manifest")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--n-before", type=int, default=30, help="fires to summarize before each swap")
    ap.add_argument("--n-after",  type=int, default=30, help="fires to summarize after each swap")
    return ap.parse_args()


def load_jsonl(path: Path):
    if not path.exists(): return []
    out = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for ln in f:
            try: out.append(json.loads(ln))
            except: continue
    return out


def main():
    args = parse_args()
    root = Path(args.root)
    decisions = load_jsonl(root/"logs"/"policy_decisions.jsonl")
    swaps = [d for d in decisions if d.get("kind") == "swap"]
    if not swaps:
        print("no swap events in logs/policy_decisions.jsonl — auto-policy has not"
              " swapped yet (or it's running in static / dry-run only).")
        # Still helpful: show recent decisions summary
        recent_kinds = [d.get("kind") for d in decisions[-20:]]
        if decisions:
            from collections import Counter
            print(f"  recent decision kinds ({len(decisions)} total): {dict(Counter(recent_kinds))}")
        return 0
    # Load all fires with policy stamp
    sr = load_jsonl(root/"bot_data"/"shadow_run.jsonl")
    fires = [r for r in sr if r.get("kind") == "entry_decision" and r.get("fire")]
    # Closes from positions.jsonl
    pos = load_jsonl(root/"bot_data"/"positions.jsonl")
    closes = {p["mint"]: p for p in pos if p.get("kind") == "close"}
    # Peak_ret via capture lookup (lazy: only load capture if we need it)
    cap_trades = None
    def get_peak_ret(mint, midK, trig_ts):
        nonlocal cap_trades
        if cap_trades is None:
            cap_trades = defaultdict(list)
            for path in sorted(glob.glob(str(root/"grpc_capture"/"*.jsonl*"))):
                opener = gzip.open if path.endswith(".gz") else open
                try:
                    with opener(path, "rt") as f:
                        for ln in f:
                            try: rec = json.loads(ln)
                            except: continue
                            m = rec.get("mint")
                            if m: cap_trades[m].append((rec.get("ev_ts",0), rec.get("vsol",0), rec.get("vtok",0)))
                except: continue
        trades = [(ts, vs, vt) for ts, vs, vt in cap_trades.get(mint, []) if ts >= trig_ts and vs > 0 and vt > 0]
        if len(trades) < 5: return None
        peak = max(vs/vt for _, vs, vt in trades)
        return peak / midK - 1.0

    print(f"=== Policy-impact report: {len(swaps)} swap event(s) ===\n")
    for i, sw in enumerate(swaps):
        t = sw["t"]; t_iso = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(t))
        frm = sw.get("from_", "?"); to = sw.get("to", "?")
        expected_uplift = sw.get("uplift", 0)
        n_sample = sw.get("n_sample", 0)
        print(f"--- swap #{i+1} @ {t_iso} ---")
        print(f"  {frm}  ->  {to}    expected_uplift={expected_uplift:+.4f} SOL/bet  (n={n_sample})")
        before = [f for f in fires if f.get("t",0) < t][-args.n_before:]
        after  = [f for f in fires if f.get("t",0) >= t][:args.n_after]
        for label, group in [("before", before), ("after", after)]:
            if not group:
                print(f"  {label}: 0 fires"); continue
            # Pull paper net_return for each
            nets = []
            peaks = []
            for fire in group:
                c = closes.get(fire["mint"])
                if c and isinstance(c.get("net_return"), (int, float)):
                    nets.append(c["net_return"])
                midK = fire.get("midK"); trig = fire.get("k_window_last_ts") or fire.get("v_window_last_ts")
                if midK and trig:
                    pr = get_peak_ret(fire["mint"], midK, trig)
                    if pr is not None: peaks.append(pr)
            policy_set = set(f.get("exit_policy", "?") for f in group)
            net_str = f"net mean {sum(nets)/len(nets):+.4f}  median {st.median(nets):+.4f}" if nets else "no closes"
            peak_str = f"peak_ret mean {sum(peaks)/len(peaks):+.3f}  median {st.median(peaks):+.3f}  winners(>=2x) {sum(1 for p in peaks if p>=1.0)}/{len(peaks)}" if peaks else "no peaks"
            print(f"  {label} ({len(group)} fires, policies={policy_set})")
            print(f"    {net_str}")
            print(f"    {peak_str}")
        # Realized uplift = after_mean_net - before_mean_net
        before_nets = [closes[f["mint"]]["net_return"] for f in before
                        if f["mint"] in closes and isinstance(closes[f["mint"]].get("net_return"), (int,float))]
        after_nets = [closes[f["mint"]]["net_return"] for f in after
                       if f["mint"] in closes and isinstance(closes[f["mint"]].get("net_return"), (int,float))]
        if before_nets and after_nets:
            actual_uplift = sum(after_nets)/len(after_nets) - sum(before_nets)/len(before_nets)
            verdict = "GOOD CALL" if actual_uplift >= expected_uplift * 0.5 else "DISAPPOINTING" if actual_uplift < 0 else "MARGINAL"
            print(f"  realized uplift: {actual_uplift:+.4f} SOL/bet vs expected {expected_uplift:+.4f}  -> {verdict}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
