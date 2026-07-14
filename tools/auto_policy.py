"""Auto-select exit policy based on recent live fires.

Mirrors auto_retrain.py but for exit policy. Replays the last N actual fires
through every candidate exit policy (via strategy_ab_replay's policy functions),
picks the one with best mean P&L, gates on min_uplift + cooldown + sample-size,
optionally swaps cfg.exit.policy + restarts pumpfun-bot.service.

Hard read-only by default (dry-run prints decision). Pass --execute to actually
edit config.yaml + restart. Gated so it can't ping-pong.

Designed to run on a systemd timer (e.g. every 6 hours) once we have enough
fires. Or invoke via:
  scripts/pumpfun_ctl.sh policy-check   (dry-run; just prints what it would pick)
  scripts/pumpfun_ctl.sh policy-now     (--execute; swaps + restarts if gates pass)
"""
from __future__ import annotations
import argparse, gzip, glob, json, os, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))   # for strategy_ab_replay import
sys.path.insert(0, str(HERE.parent))   # for bot_config import

DEFAULT_ROOT = Path("/root/the-distribution-will-manifest")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--execute", action="store_true",
                    help="actually swap config.yaml + restart bot if gates pass. "
                         "default is dry-run.")
    ap.add_argument("--force", action="store_true",
                    help="bypass cooldown + sample-size gates (still requires uplift)")
    return ap.parse_args()


def load_cfg(root: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load((root / "config.yaml").read_text()) or {}
    except Exception:
        return {}


def write_cfg(root: Path, cfg: dict):
    import yaml
    (root / "config.yaml").write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))


def last_swap_age_h(root: Path) -> float:
    """Read logs/policy_decisions.jsonl for the last `swap` entry; return age in hours
    or +inf if no prior swap."""
    p = root / "logs" / "policy_decisions.jsonl"
    if not p.exists(): return float("inf")
    last_t = 0.0
    try:
        for ln in p.read_text().splitlines():
            try: r = json.loads(ln)
            except: continue
            if r.get("kind") == "swap" and r.get("t", 0) > last_t:
                last_t = r["t"]
    except Exception: pass
    if last_t == 0: return float("inf")
    return (time.time() - last_t) / 3600.0


def log_decision(root: Path, **kw):
    p = root / "logs" / "policy_decisions.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"t": time.time(), **kw}, default=str) + "\n")


def main():
    args = parse_args()
    root = Path(args.root)
    cfg = load_cfg(root)
    exit_cfg = cfg.get("exit", {})
    candidates = exit_cfg.get("dynamic_candidates",
                                ["k_combined", "h_time_spaced", "b_frontload",
                                 "c_hybrid_t30", "f_hybrid_t50"])
    current_policy   = exit_cfg.get("policy", "k_combined")
    window_fires     = int(exit_cfg.get("dynamic_window_fires", 30))
    min_uplift       = float(exit_cfg.get("dynamic_min_uplift_sol", 0.02))
    cooldown_h       = float(exit_cfg.get("dynamic_cooldown_h", 6))
    min_sample       = int(exit_cfg.get("dynamic_min_sample", 30))
    mode             = exit_cfg.get("mode", "static")
    bet_sol          = float(cfg.get("bot", {}).get("bet_sol", 1.0))

    now_iso = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"=== auto_policy @ {now_iso}  (mode={mode}, execute={args.execute}) ===")
    print(f"current policy: {current_policy}")
    print(f"candidates:     {candidates}")
    print(f"bet_sol:        {bet_sol}")

    if mode != "dynamic" and not args.execute and not args.force:
        # informational only
        print(f"\nmode=static — would not swap even with uplift. Set exit.mode=dynamic to enable.")

    # ---- Pull last N fires from bot_data/shadow_run.jsonl ----
    sr_path = root / "bot_data" / "shadow_run.jsonl"
    if not sr_path.exists():
        print(f"no shadow_run.jsonl — nothing to evaluate"); return 0
    with open(sr_path) as f:
        all_rows = [json.loads(ln) for ln in f if ln.strip().startswith("{")]
    fires = [r for r in all_rows if r.get("kind") == "entry_decision" and r.get("fire")]
    fires = fires[-window_fires:]   # last N
    print(f"\nfires in window:  {len(fires)} (window cap {window_fires})")
    if len(fires) < min_sample and not args.force:
        print(f"  below min_sample {min_sample} — skip evaluation")
        log_decision(root, kind="skip", reason="below_min_sample",
                     n_fires=len(fires), min_sample=min_sample)
        return 0

    # ---- Pull forward trades from grpc_capture for each fired mint ----
    mints = set(f["mint"] for f in fires)
    cap_dir = root / "grpc_capture"
    cap_trades = defaultdict(list)
    for path in sorted(glob.glob(str(cap_dir / "*.jsonl*"))):
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt") as f:
                for ln in f:
                    try: rec = json.loads(ln)
                    except: continue
                    m = rec.get("mint")
                    if m in mints:
                        vs = rec.get("vsol", 0); vt = rec.get("vtok", 0)
                        if vs > 0 and vt > 0:
                            cap_trades[m].append((rec.get("ev_ts", 0), vs, vt))
        except Exception: continue
    for m in cap_trades: cap_trades[m].sort(key=lambda x: x[0])

    # ---- Replay each fire through each policy via strategy_ab_replay ----
    # The legacy POLICY_FNS dict served the original 5 policies. New policies
    # are evaluated via the registry adapter (policy_via_registry) so adding
    # one to exit_policies/ automatically makes it auto_policy-replayable as
    # long as cfg.exit.dynamic_candidates lists it.
    from strategy_ab_replay import (build_snap_array, policy_k_combined,
                                     policy_time_spaced, policy_hybrid_trail,
                                     policy_frontload, policy_via_registry)
    # Build a cfg-like object the registry policies can read from
    class _Cfg:
        class exit:
            total_slices = int(exit_cfg.get("total_slices", 8))
            derisk_slices = int(exit_cfg.get("derisk_slices", 4))
            derisk_min_gap_s = float(exit_cfg.get("derisk_min_gap_s", 5.0))
            runner_min_gap_s = float(exit_cfg.get("runner_min_gap_s", 15.0))
            runner_retrace_frac = float(exit_cfg.get("runner_retrace_frac", 0.30))
            runner_min_arm_ret = float(exit_cfg.get("runner_min_arm_ret", 0.20))
            death_threshold = float(exit_cfg.get("death_threshold", 0.10))
            rl_artifact_dir = exit_cfg.get("rl_artifact_dir", "bot_artifacts_K7V_rl_layered")
            rl_q5_threshold = float(exit_cfg.get("rl_q5_threshold", 0.20))
    LEGACY = {
        "k_combined":    lambda *a: policy_k_combined(*a),
        "h_time_spaced": lambda *a: policy_time_spaced(*a, gap_s=15.0),
        "b_frontload":   lambda *a: policy_frontload(*a),
        "c_hybrid_t30":  lambda *a: policy_hybrid_trail(*a, runner_retrace=0.30),
        "f_hybrid_t50":  lambda *a: policy_hybrid_trail(*a, runner_retrace=0.50),
    }
    results = {p: [] for p in candidates}
    n_replayed = 0
    for fire in fires:
        m = fire["mint"]; midK = fire.get("midK")
        vsK = fire.get("vsK"); vtK = fire.get("vtK")
        trig = fire.get("k_window_last_ts") or fire.get("v_window_last_ts")
        if not all([midK, vsK, vtK, trig]): continue
        forward = [(ts, vs, vt) for ts, vs, vt in cap_trades.get(m, []) if ts >= trig]
        if len(forward) < 5: continue
        vsC, vtC = forward[-1][1], forward[-1][2]
        snaps, fwd, dts = build_snap_array(forward)
        if len(snaps) < 1: continue
        n_replayed += 1
        for p in candidates:
            try:
                if p in LEGACY:
                    pl = LEGACY[p](vsK, vtK, vsC, vtC, snaps, dts)
                else:
                    # Try via the new registry — supports rl_layered + any future plugin
                    pl = policy_via_registry(vsK, vtK, vsC, vtC, snaps, dts,
                                              policy_name=p, cfg=_Cfg, mint=m,
                                              entry_features=fire.get("features"),
                                              entry_score=float(fire.get("score") or 0.0))
                if pl is not None: results[p].append(pl)
            except Exception as e:
                pass
    print(f"replayed {n_replayed} / {len(fires)} fires (rest had insufficient capture coverage)")

    # ---- Score each policy ----
    print(f"\n{'policy':22s} {'n':>4s} {'mean':>9s} {'median':>9s} {'win%':>6s} {'total':>9s}")
    scored = []
    for p in candidates:
        rs = results[p]
        if not rs: continue
        mean = sum(rs)/len(rs); med = st.median(rs); win = sum(1 for r in rs if r > 0)
        scored.append((p, mean, med, 100*win/len(rs), sum(rs), len(rs)))
    scored.sort(key=lambda x: -x[1])   # by mean desc
    for p, mean, med, wp, total, n in scored:
        flag = " <-- current" if p == current_policy else ""
        print(f"{p:22s} {n:>4d} {mean:>+9.4f} {med:>+9.4f} {wp:>5.1f}% {total:>+9.3f}{flag}")
    if not scored:
        print("no policies produced usable results"); return 0

    best = scored[0]
    best_policy, best_mean = best[0], best[1]
    cur = next((s for s in scored if s[0] == current_policy), None)
    cur_mean = cur[1] if cur else None
    if cur_mean is None:
        print(f"\ncurrent policy {current_policy} not in candidate list — adding to result")
        cur_mean = -float("inf")
    uplift = best_mean - cur_mean
    age_h = last_swap_age_h(root)
    print(f"\nbest: {best_policy}  mean {best_mean:+.4f}")
    print(f"current: {current_policy}  mean {cur_mean:+.4f}")
    print(f"uplift = {uplift:+.4f}  (require >= {min_uplift})")
    print(f"hours since last swap = {age_h:.1f}  (require >= {cooldown_h})")

    # ---- Gate checks ----
    reasons = []
    if best_policy == current_policy: reasons.append(f"already_on_winner")
    if uplift < min_uplift:           reasons.append(f"uplift_below_threshold")
    if age_h < cooldown_h and not args.force: reasons.append(f"in_cooldown")
    if mode != "dynamic" and not args.force:  reasons.append(f"mode_static")
    if reasons:
        print(f"\nno swap: {' & '.join(reasons)}")
        log_decision(root, kind="no_swap", reasons=reasons,
                     best=best_policy, best_mean=best_mean,
                     current=current_policy, current_mean=cur_mean,
                     uplift=uplift, age_h=age_h)
        return 0

    # ---- All gates pass: would swap ----
    print(f"\nGATES PASS — would swap {current_policy} -> {best_policy}")
    if not args.execute:
        print(f"  (dry-run; pass --execute to actually swap config.yaml + restart bot)")
        log_decision(root, kind="dry_run_pass", best=best_policy, current=current_policy,
                     uplift=uplift, n=best[5])
        return 0

    # ---- Execute the swap ----
    cfg["exit"]["policy"] = best_policy
    write_cfg(root, cfg)
    log_decision(root, kind="swap", from_=current_policy, to=best_policy,
                 uplift=uplift, n_sample=best[5],
                 scored_by_mean={p:m for p,m,_,_,_,_ in scored})
    print(f"  wrote new exit.policy={best_policy} to config.yaml")
    try:
        r = subprocess.run(["systemctl", "restart", "pumpfun-bot.service"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            print(f"  restarted pumpfun-bot.service")
        else:
            print(f"  restart returned {r.returncode}: {r.stderr[-200:]}")
    except Exception as e:
        print(f"  restart error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
