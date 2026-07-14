"""Weekly auto-retrain candidate (idempotent, gated).

The pipeline (only runs if all preconditions met):
  1. Check capture archive has >= MIN_CAPTURE_DAYS of data
  2. Run extract_from_capture.py against grpc_capture (with same fresh_rsol filter)
     -> data/pumpfun_continuation_K7_capture/ + _V_capture/ parquets
  3. Run build_bot_artifacts_K7V.py with --suffix _capture
     -> bot_artifacts_K7V_capture/ pickles + spec
  4. Hold out 20% of capture-derived population as test set; compare candidate's
     train-set top-decile threshold + test-set AUC against the current production
     model on the SAME test split
  5. If candidate beats production by >= MIN_AUC_UPLIFT, swap symlink
       bot_artifacts_K7V -> bot_artifacts_K7V_capture_<timestamp>
  6. If swapped: trigger bot restart to pick up new artifacts
                 (systemctl restart pumpfun-bot.service)

The pipeline is IDEMPOTENT — running it multiple times in a row with the same
data does nothing on subsequent runs.

Designed to be invoked from systemd timer weekly OR manually via
  scripts/pumpfun_ctl.sh retrain-check  (dry-run, prints what would happen)
  scripts/pumpfun_ctl.sh retrain-now    (executes if gates pass)

Read-only by default unless --execute is passed; --execute is the only thing
that ACTUALLY swaps artifacts and restarts the bot.

Mostly skeleton at this stage — extract_from_capture.py is a real pipeline but
the model-comparison and symlink-swap pieces are the new work here.
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, shutil, subprocess, sys, time
from pathlib import Path

DEFAULT_ROOT = Path("/root/the-distribution-will-manifest")

MIN_CAPTURE_DAYS = 3
MIN_AUC_UPLIFT = 0.005          # holdout AUC uplift gate (in-sample-ish)
MIN_HOLDOUT_BETS = 200          # need this many in holdout for meaningful AUC
MIN_LIVE_SHADOW_FIRES = 30      # last N live fires for the out-of-sample gate
MIN_LIVE_AUC_UPLIFT = 0.02      # candidate must beat production on live fires by this
INPUT_SUFFIXES = ["_fresh", "_capture"]   # combine May + capture data
OUTPUT_SUFFIX = "_capture"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--execute", action="store_true",
                    help="actually swap artifacts + restart bot if criteria met. "
                         "Default is dry-run (prints plan only).")
    ap.add_argument("--force", action="store_true",
                    help="bypass MIN_CAPTURE_DAYS check (e.g., for testing)")
    return ap.parse_args()


def capture_data_days(capture_dir: Path) -> float:
    files = sorted(glob.glob(str(capture_dir / "*.jsonl*")))
    if len(files) < 2: return 0.0
    # Use the earliest and latest mtimes as a rough proxy
    times = [Path(p).stat().st_mtime for p in files]
    return (max(times) - min(times)) / 86400.0


def run_subprocess(cmd: list, cwd: Path, log_path: Path | None = None) -> tuple[int, str]:
    """Run a subprocess, return (returncode, last 500 chars of output)."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} === {' '.join(cmd)} ===\n")
            f.write(r.stdout + "\n" + r.stderr + "\n")
    out = (r.stdout + r.stderr).strip()
    return r.returncode, out[-500:] if len(out) > 500 else out


def compare_models(root: Path, log: Path) -> tuple[bool, dict]:
    """Compare candidate model in bot_artifacts_K7V_capture vs current production
    bot_artifacts_K7V on the candidate's holdout set.

    Returns (should_swap, comparison_dict)."""
    try:
        import pandas as pd
        import pickle
        import sklearn  # noqa
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        return False, {"error": f"missing deps: {e}"}

    K7_dir = root / "data" / "pumpfun_continuation_K7_capture"
    V_dir  = root / "data" / "pumpfun_continuation_V05_capture"
    if not (K7_dir / "token_level.parquet").exists():
        return False, {"error": "candidate parquets missing"}

    cand_dir = root / "bot_artifacts_K7V_capture"
    prod_dir = root / "bot_artifacts_K7V"
    if not (cand_dir / "entry_model.pkl").exists():
        return False, {"error": "candidate entry_model.pkl missing"}
    if not (prod_dir / "entry_model.pkl").exists():
        return False, {"error": "production entry_model.pkl missing"}

    ENTRY_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
               "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
    ENTRY_V = [f"{c}_v" for c in ENTRY_K]
    K_TO_V = {k: v for k, v in zip(ENTRY_K, ENTRY_V)}

    tk = pd.read_parquet(K7_dir / "token_level.parquet")
    tv = pd.read_parquet(V_dir / "token_level.parquet")
    joined = tk.merge(tv[["mint"] + ENTRY_K].rename(columns=K_TO_V), on="mint", how="inner")
    if len(joined) < MIN_HOLDOUT_BETS * 5:
        return False, {"error": f"joined dataset too small ({len(joined)} < {MIN_HOLDOUT_BETS*5})"}

    # 80/20 split (deterministic; use sorted first_slot if available)
    if "first_slot" in joined.columns:
        joined = joined.sort_values("first_slot").reset_index(drop=True)
    split = int(0.8 * len(joined))
    holdout = joined.iloc[split:].copy()
    n_hold = len(holdout)

    # Score holdout with both models
    cand = pickle.load(open(cand_dir / "entry_model.pkl", "rb"))
    prod = pickle.load(open(prod_dir / "entry_model.pkl", "rb"))
    X = holdout[ENTRY_K + ENTRY_V].values
    y = (holdout["peak_ret"] >= 1.0).astype(int).values
    cand_auc = roc_auc_score(y, cand.predict_proba(X)[:, 1])
    prod_auc = roc_auc_score(y, prod.predict_proba(X)[:, 1])
    uplift = cand_auc - prod_auc

    comp = {"n_holdout": n_hold, "cand_auc": float(cand_auc),
            "prod_auc": float(prod_auc), "uplift": float(uplift),
            "min_uplift_threshold": MIN_AUC_UPLIFT,
            "min_holdout_bets": MIN_HOLDOUT_BETS}
    if n_hold < MIN_HOLDOUT_BETS:
        comp["decision_reason"] = f"holdout too small ({n_hold} < {MIN_HOLDOUT_BETS})"
        return False, comp
    if uplift < MIN_AUC_UPLIFT:
        comp["decision_reason"] = f"uplift {uplift:+.4f} below threshold {MIN_AUC_UPLIFT}"
        return False, comp
    comp["decision_reason"] = f"uplift {uplift:+.4f} >= {MIN_AUC_UPLIFT}, SWAP"
    return True, comp


def compare_models_on_live_fires(root: Path) -> tuple[bool, dict]:
    """Out-of-sample gate: score the LAST N actual live fires (from
    shadow_run.jsonl, which now stamps each fire with its 22 features) with BOTH
    the candidate and production models. Compare AUCs of predicting peak_ret>=1.0
    using realized outcomes from grpc_capture.

    This is the meaningful gate: the candidate must beat production on data the
    bot actually saw recently, not just on the candidate's own training holdout."""
    try:
        import pickle
        import gzip, glob
        import numpy as np
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        return False, {"error": f"missing deps: {e}"}

    cand_dir = root / "bot_artifacts_K7V_capture"
    prod_dir = root / "bot_artifacts_K7V"
    if not (cand_dir / "entry_model.pkl").exists() or not (prod_dir / "entry_model.pkl").exists():
        return False, {"error": "missing model pickles"}
    import json as _json
    cand_spec = _json.load(open(cand_dir / "model_spec.json"))
    prod_spec = _json.load(open(prod_dir / "model_spec.json"))
    feats = cand_spec["entry"]["features"]
    if feats != prod_spec["entry"]["features"]:
        return False, {"error": "model feature lists differ; cannot compare on same vector"}
    cand_clf = pickle.load(open(cand_dir / "entry_model.pkl", "rb"))
    prod_clf = pickle.load(open(prod_dir / "entry_model.pkl", "rb"))

    # Read fires with full features from shadow_run.jsonl
    sr_path = root / "bot_data" / "shadow_run.jsonl"
    fires = []
    if sr_path.exists():
        for ln in sr_path.read_text().splitlines():
            try: r = _json.loads(ln)
            except: continue
            if r.get("kind") == "entry_decision" and r.get("features"):
                fires.append(r)
    fires = fires[-500:]   # last 500 ready events (not just fires)
    if len(fires) < MIN_LIVE_SHADOW_FIRES:
        return False, {"error": f"only {len(fires)} live fires with features; need {MIN_LIVE_SHADOW_FIRES}"}

    # Build feature matrix; look up realized peak_ret per mint from capture
    mints = set(f["mint"] for f in fires)
    cap_trades = {}
    for path in sorted(glob.glob(str(root / "grpc_capture" / "*.jsonl*"))):
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt") as f:
                for ln in f:
                    try: rec = _json.loads(ln)
                    except: continue
                    m = rec.get("mint")
                    if m in mints:
                        vs = rec.get("vsol", 0); vt = rec.get("vtok", 0)
                        if vs > 0 and vt > 0:
                            cap_trades.setdefault(m, []).append((rec.get("ev_ts", 0), vs, vt))
        except Exception: continue

    X = []; y = []
    for r in fires:
        midK = r.get("midK"); trig = r.get("k_window_last_ts") or r.get("v_window_last_ts")
        if not midK or not trig: continue
        trades = [(ts, vs, vt) for ts, vs, vt in cap_trades.get(r["mint"], []) if ts >= trig]
        if len(trades) < 5: continue
        peak = max(vs/vt for _, vs, vt in trades)
        peak_ret = peak / midK - 1.0
        feat_dict = r["features"]
        try:
            X.append([feat_dict[f] for f in feats])
            y.append(1 if peak_ret >= 1.0 else 0)
        except KeyError:
            continue
    if len(X) < MIN_LIVE_SHADOW_FIRES or sum(y) < 5 or sum(1 for v in y if v == 0) < 5:
        return False, {"error": f"insufficient live shadow data after capture lookup: n={len(X)}, winners={sum(y)}"}
    X = np.array(X); y = np.array(y)
    cand_scores = cand_clf.predict_proba(X)[:, 1]
    prod_scores = prod_clf.predict_proba(X)[:, 1]
    try:
        cand_auc = float(roc_auc_score(y, cand_scores))
        prod_auc = float(roc_auc_score(y, prod_scores))
    except ValueError as e:
        return False, {"error": f"AUC compute failed: {e}"}
    uplift = cand_auc - prod_auc
    pass_gate = uplift >= MIN_LIVE_AUC_UPLIFT
    return pass_gate, {
        "n_live_shadow": len(X), "n_winners": int(sum(y)),
        "cand_live_auc": cand_auc, "prod_live_auc": prod_auc,
        "live_uplift": uplift, "live_min_uplift": MIN_LIVE_AUC_UPLIFT,
        "decision": "live_shadow_PASS" if pass_gate else "live_shadow_FAIL",
    }


def swap_artifacts(root: Path, log: Path) -> bool:
    """Swap bot_artifacts_K7V symlink to point at the new _capture dir.
    First time: rename existing bot_artifacts_K7V/ to bot_artifacts_K7V_v_initial/
    and create the symlink. Subsequent: just update symlink target."""
    base = root / "bot_artifacts_K7V"
    cand = root / "bot_artifacts_K7V_capture"
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    versioned = root / f"bot_artifacts_K7V_capture_{ts}"
    try:
        shutil.move(str(cand), str(versioned))
        if base.is_symlink():
            os.remove(base)
        elif base.is_dir():
            shutil.move(str(base), str(root / f"bot_artifacts_K7V_v_initial_{ts}"))
        os.symlink(versioned.name, str(base))
        with open(log, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}: symlink swap "
                    f"bot_artifacts_K7V -> {versioned.name}\n")
        return True
    except Exception as e:
        with open(log, "a") as f:
            f.write(f"swap failed: {e}\n")
        return False


def main():
    args = parse_args()
    root = Path(args.root)
    log = root / "logs" / "auto_retrain.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    now_iso = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"=== auto_retrain @ {now_iso}  (execute={args.execute}) ===")

    # 1) capture days gate
    days = capture_data_days(root / "grpc_capture")
    print(f"  capture archive span: {days:.2f} days  (need >= {MIN_CAPTURE_DAYS})")
    if days < MIN_CAPTURE_DAYS and not args.force:
        print(f"  not enough capture data, skip"); return 0

    # 2) extract from capture
    print(f"  step 2/5: extract_from_capture ...")
    rc, out = run_subprocess(["./venv/bin/python", "tools/extract_from_capture.py",
                              "--capture-dir", "grpc_capture",
                              "--k7-out", "data/pumpfun_continuation_K7_capture",
                              "--v-out",  "data/pumpfun_continuation_V05_capture",
                              "--fresh-rsol-lam", "3000000000"], root, log)
    if rc != 0:
        print(f"    FAILED rc={rc}: {out[-300:]}"); return 1
    print(f"    OK")

    # 3) build candidate artifacts combining MAY (_fresh) + capture (_capture)
    print(f"  step 3/5: build_bot_artifacts_K7V combining {INPUT_SUFFIXES} -> {OUTPUT_SUFFIX} ...")
    if not (root / "tools" / "build_bot_artifacts_K7V.py").exists():
        print(f"    tools/build_bot_artifacts_K7V.py not found; cannot continue"); return 1
    rc, out = run_subprocess(["./venv/bin/python", "tools/build_bot_artifacts_K7V.py",
                              "--inputs", *INPUT_SUFFIXES,
                              "--out", OUTPUT_SUFFIX], root, log)
    if rc != 0:
        print(f"    FAILED rc={rc}: {out[-300:]}"); return 1
    print(f"    OK")

    # 4a) holdout comparison (in-sample gate)
    print(f"  step 4a/5: holdout comparison ...")
    holdout_pass, comp = compare_models(root, log)
    print(f"    {json.dumps(comp, indent=2)}")
    if not holdout_pass:
        print(f"    skip swap (holdout gate failed)"); return 0

    # 4b) LIVE-SHADOW comparison (out-of-sample gate on real recent fires)
    print(f"  step 4b/5: live-shadow comparison on last ~{MIN_LIVE_SHADOW_FIRES}+ fires ...")
    live_pass, live_comp = compare_models_on_live_fires(root)
    print(f"    {json.dumps(live_comp, indent=2)}")
    if not live_pass:
        print(f"    skip swap (live-shadow gate failed: candidate did NOT beat "
              f"production on actual recent live fires)"); return 0
    print(f"    BOTH gates passed: holdout AUC uplift +{comp.get('uplift',0):+.4f} "
          f"AND live AUC uplift +{live_comp.get('live_uplift',0):+.4f}")

    # 5) actually swap (only when --execute)
    if not args.execute:
        print(f"  step 5/5: swap would happen here (dry-run, skipping)"); return 0
    print(f"  step 5/5: swapping artifacts ...")
    if not swap_artifacts(root, log):
        print(f"    swap failed"); return 1
    print(f"    OK")

    # 6) restart bot
    print(f"  restarting pumpfun-bot.service ...")
    rc, out = run_subprocess(["systemctl", "restart", "pumpfun-bot.service"], root, log)
    if rc != 0:
        print(f"    restart returned {rc}: {out[-200:]}")
    else:
        print(f"    OK")

    # 7) threshold revisit reminder
    # User decision (2026-06-07): "let's let the threshold 0.30 so we fire often
    # but we should reconsider". A new model swap is a natural revisit point —
    # surface the comparison loudly in the retrain log so the next operator
    # check sees it. No auto-change to the runtime override.
    try:
        new_spec_path = root / "bot_artifacts_K7V" / "model_spec.json"
        with open(new_spec_path) as f:
            new_spec = json.load(f)
        native_thr = new_spec.get("entry", {}).get("entry_threshold_top_decile")
        runtime_override = _extract_runtime_threshold(root)
        msg = ("\n  THRESHOLD REVISIT REMINDER (user-requested 2026-06-07):\n"
               f"    new model's native top-decile threshold = {native_thr}\n"
               f"    current systemd --entry-threshold override = {runtime_override}\n"
               f"    decision standing: keep override at 0.30 (fire often, collect "
               f"data); revisit when ≥100 live fires accumulated post-swap OR\n"
               f"    drift_monitor says live p90 matches train p90 within ±10%.\n"
               f"    To lift override: edit /etc/systemd/system/pumpfun-bot.service\n"
               f"    --entry-threshold flag to {native_thr} (or remove it entirely\n"
               f"    to use the model's native value).")
        print(msg)
        with open(log, "a") as f: f.write(msg + "\n")
    except Exception as e:
        with open(log, "a") as f: f.write(f"threshold reminder error: {e}\n")

    return 0


def _extract_runtime_threshold(root: Path) -> str:
    """Read the active --entry-threshold from the systemd unit (best-effort)."""
    unit = Path("/etc/systemd/system/pumpfun-bot.service")
    if not unit.exists():
        return "?"
    try:
        for line in unit.read_text().splitlines():
            line = line.strip()
            if "--entry-threshold" in line:
                # tokens like  "--entry-threshold 0.30 \"
                parts = line.replace("\\", " ").split()
                for i, t in enumerate(parts):
                    if t == "--entry-threshold" and i + 1 < len(parts):
                        return parts[i + 1]
    except Exception:
        pass
    return "?"


if __name__ == "__main__":
    sys.exit(main())
