"""One-shot rebuild of bot_artifacts_K7V at snap_every=1 (matches live).

The previous artifacts were trained on offline parquets that sampled every
3rd forward event (SNAP_EVERY=3 in the K7/V extractor modules). The live
bot has snap_every=1 in config.yaml — every event scored. This script
forces the training extraction to use SNAP_EVERY=1 so the recovery model
sees the same per-snap distribution it'll see in production.

Steps:
  1. Monkey-patch pumpfun_continuation_value_K7.SNAP_EVERY and
     pumpfun_continuation_value_V.SNAP_EVERY to 1
  2. Run extract_from_capture against grpc_capture/ -> parquets at
     data/pumpfun_continuation_{K7,V05}_snap1/
  3. Run build_bot_artifacts_K7V with --inputs _snap1
     -> bot_artifacts_K7V_snap1/
  4. Print AUC comparison vs current production artifacts
  5. Print swap instructions (DO NOT auto-swap — user reviews first)

Idempotent: re-running just re-trains; the extracted parquets are reused
unless --force-reextract is passed.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path("/root/the-distribution-will-manifest")
SUFFIX = "_snap1"
K7_OUT = ROOT / f"data/pumpfun_continuation_K7{SUFFIX}"
V_OUT  = ROOT / f"data/pumpfun_continuation_V05{SUFFIX}"
ART_OUT = ROOT / f"bot_artifacts_K7V{SUFFIX}"
CURRENT_ARTIFACTS = ROOT / "bot_artifacts_K7V"


def patch_snap_every(value: int):
    """Edit SNAP_EVERY in both extractor modules. Returns the (old_K7, old_V)
    tuple so the caller can restore."""
    paths = {
        "K7": ROOT / "tools/pumpfun_continuation_value_K7.py",
        "V":  ROOT / "tools/pumpfun_continuation_value_V.py",
    }
    olds = {}
    for label, p in paths.items():
        src = p.read_text()
        # Find the existing line, capture its value, replace with new value
        import re
        m = re.search(r"^SNAP_EVERY\s*=\s*(\d+)", src, re.MULTILINE)
        if not m:
            raise RuntimeError(f"SNAP_EVERY not found in {p}")
        olds[label] = int(m.group(1))
        new = re.sub(r"^(SNAP_EVERY\s*=\s*)\d+", rf"\g<1>{value}", src,
                     count=1, flags=re.MULTILINE)
        p.write_text(new)
        print(f"  patched {label}: SNAP_EVERY {olds[label]} -> {value}")
    return olds


def restore_snap_every(olds: dict):
    paths = {
        "K7": ROOT / "tools/pumpfun_continuation_value_K7.py",
        "V":  ROOT / "tools/pumpfun_continuation_value_V.py",
    }
    import re
    for label, p in paths.items():
        src = p.read_text()
        src = re.sub(r"^(SNAP_EVERY\s*=\s*)\d+", rf"\g<1>{olds[label]}", src,
                     count=1, flags=re.MULTILINE)
        p.write_text(src)
        print(f"  restored {label}: SNAP_EVERY -> {olds[label]}")


def run(cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    full_env = os.environ.copy()
    if env: full_env.update(env)
    r = subprocess.run(cmd, cwd=str(cwd), env=full_env)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({dt:.1f}s): {cmd}")
    print(f"  ok ({dt:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true",
                    help="skip re-extraction; train on existing _snap1 parquets")
    ap.add_argument("--force-reextract", action="store_true",
                    help="re-run extraction even if parquets already exist")
    args = ap.parse_args()

    print(f"=== rebuild_snap1 @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")
    print(f"  K7_OUT:  {K7_OUT}")
    print(f"  V_OUT:   {V_OUT}")
    print(f"  ART_OUT: {ART_OUT}")

    py = str(ROOT / "venv/bin/python")

    # Step 1+2: extract at SNAP_EVERY=1
    parquets_exist = (K7_OUT / "token_level.parquet").exists() and \
                     (V_OUT / "token_level.parquet").exists()
    if args.skip_extract:
        print("  --skip-extract: assuming parquets exist")
    elif parquets_exist and not args.force_reextract:
        print(f"  parquets already at {K7_OUT}/ and {V_OUT}/ ; skipping extract "
              f"(pass --force-reextract to rebuild)")
    else:
        print("\n[1/3] PATCH + EXTRACT")
        olds = patch_snap_every(1)
        try:
            run([py, "tools/extract_from_capture.py",
                 "--capture-dir", "grpc_capture",
                 "--k7-out",      str(K7_OUT),
                 "--v-out",       str(V_OUT)])
        finally:
            restore_snap_every(olds)

    # Step 3: train
    print("\n[2/3] TRAIN")
    run([py, "tools/build_bot_artifacts_K7V.py",
         "--inputs", SUFFIX,
         "--out",    SUFFIX])

    # Step 4: compare AUCs
    print("\n[3/3] COMPARE")
    if not ART_OUT.exists():
        print(f"  expected {ART_OUT}/ from build step but didn't find it")
        sys.exit(2)
    new_spec = json.loads((ART_OUT / "model_spec.json").read_text())
    old_spec = json.loads((CURRENT_ARTIFACTS / "model_spec.json").read_text())
    new_e = new_spec["entry"]["train_auc_peak2x"]
    new_r = new_spec["recovery"]["train_auc"]
    old_e = old_spec["entry"]["train_auc_peak2x"]
    old_r = old_spec["recovery"]["train_auc"]
    new_thr = new_spec["entry"]["entry_threshold_top_decile"]
    old_thr = old_spec["entry"]["entry_threshold_top_decile"]
    print(f"  entry AUC    old={old_e:.4f}  new={new_e:.4f}  delta={new_e-old_e:+.4f}")
    print(f"  recovery AUC old={old_r:.4f}  new={new_r:.4f}  delta={new_r-old_r:+.4f}")
    print(f"  entry threshold (top-decile)  old={old_thr:.4f}  new={new_thr:.4f}")
    print(f"\n  n_train_tokens          old={old_spec.get('n_train_tokens','?')}  new={new_spec.get('n_train_tokens','?')}")
    print(f"  n_recovery_train_rows   old={old_spec.get('n_recovery_train_rows','?')}  new={new_spec.get('n_recovery_train_rows','?')}")

    print(f"\nTo SWAP this candidate live:")
    print("  ssh <research-host>")
    print(f"  cd /root/the-distribution-will-manifest")
    print(f"  # back up current symlink target, then atomically repoint")
    print(f"  mv bot_artifacts_K7V bot_artifacts_K7V_pre_snap1_swap_$(date +%s)")
    print(f"  ln -s bot_artifacts_K7V_snap1 bot_artifacts_K7V")
    print(f"  sudo systemctl restart pumpfun-bot")


if __name__ == "__main__":
    main()
