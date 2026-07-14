#!/usr/bin/env python3
"""collector_health.py — one-command health check for the data collectors.

Read-only by design: attaches nothing writable, restarts nothing, and never
touches the live SHM ring as a writer. Checks, per collector:
  - systemd active state + uptime
  - newest output file age (is data landing NOW)
  - output growth over a short sample window (is data FLOWING)
plus disk headroom, the storagebox mount, and the intent ring's write_seq
advancing. Exit code: 0 healthy, 1 warnings, 2 critical.

Run:  ./venv/bin/python tools/collector_health.py  [--sample-s 3]
"""
import argparse
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COLLECTORS = {
    "pumpfun-grpc-capture": ROOT / "grpc_capture",
    "pumpfun-grpc-firehose": ROOT / "buffer" / "grpc_firehose",
    "pumpfun-shred-firehose": ROOT / "buffer" / "raw_shred_entries",
    "pumpfun-shred-intents": ROOT / "shred_bot" / "intent_capture",
    "pumpfun-storagebox-shipper": None,  # service-state only; output is the mount
}
FRESH_WARN_S = 180.0
FRESH_CRIT_S = 600.0


def svc_state(name: str) -> tuple[str, str]:
    try:
        active = subprocess.run(["systemctl", "is-active", name], capture_output=True,
                                text=True).stdout.strip()
        since = subprocess.run(["systemctl", "show", name, "--property=ActiveEnterTimestamp",
                                "--value"], capture_output=True, text=True).stdout.strip()
        return active, since
    except Exception as e:
        return f"err:{e}", ""


def newest_file(d: Path) -> Path | None:
    best, best_m = None, -1.0
    if not d or not d.exists():
        return None
    for dirpath, _dirnames, filenames in os.walk(d):
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m > best_m:
                best, best_m = p, m
    return best


def ring_seq() -> int | None:
    """Read write_seq from the live ring header without registering with the
    resource tracker (so this process exiting can never unlink the segment)."""
    try:
        from multiprocessing import shared_memory, resource_tracker
        shm = shared_memory.SharedMemory(name="pumpfun_intents")
        try:
            resource_tracker.unregister("/pumpfun_intents", "shared_memory")
        except Exception:
            pass
        seq = struct.unpack_from("<Q", shm.buf, 24)[0]
        shm.close()
        return seq
    except FileNotFoundError:
        return None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-s", type=float, default=3.0)
    args = ap.parse_args()

    worst = 0

    def report(level: int, line: str):
        nonlocal worst
        worst = max(worst, level)
        tag = {0: "OK  ", 1: "WARN", 2: "CRIT"}[level]
        print(f"[{tag}] {line}")

    sizes0, files = {}, {}
    for svc, outdir in COLLECTORS.items():
        if outdir is not None:
            f = newest_file(outdir)
            files[svc] = f
            sizes0[svc] = f.stat().st_size if f else -1
    seq0 = ring_seq()
    t0 = time.time()
    time.sleep(args.sample_s)

    for svc, outdir in COLLECTORS.items():
        active, since = svc_state(svc)
        if active != "active":
            report(2, f"{svc}: systemd state '{active}'")
            continue
        line = f"{svc}: active since {since or '?'}"
        if outdir is None:
            report(0, line)
            continue
        f = files.get(svc)
        if f is None:
            report(2, f"{svc}: active but NO output files under {outdir}")
            continue
        age = time.time() - f.stat().st_mtime
        grew = f.stat().st_size - sizes0[svc]
        if age > FRESH_CRIT_S:
            report(2, f"{svc}: newest output {f.name} is {age:.0f}s old")
        elif age > FRESH_WARN_S:
            report(1, f"{svc}: newest output {f.name} is {age:.0f}s old")
        elif grew <= 0 and age > 30:
            report(1, f"{svc}: {f.name} did not grow in {args.sample_s:.0f}s (age {age:.0f}s)")
        else:
            report(0, f"{line}; {f.name} +{grew:,}B in {time.time()-t0:.1f}s")

    seq1 = ring_seq()
    if seq0 is None or seq1 is None:
        report(2, "intent ring: SHM segment 'pumpfun_intents' not readable")
    elif seq1 <= seq0:
        report(1, f"intent ring: write_seq stalled at {seq1:,} over {args.sample_s:.0f}s")
    else:
        report(0, f"intent ring: write_seq {seq0:,} -> {seq1:,} (+{seq1-seq0})")

    st = os.statvfs(ROOT)
    free_frac = st.f_bavail / st.f_blocks
    if free_frac < 0.10:
        report(2, f"disk: only {free_frac:.0%} free on {ROOT}")
    elif free_frac < 0.20:
        report(1, f"disk: {free_frac:.0%} free on {ROOT}")
    else:
        report(0, f"disk: {free_frac:.0%} free on {ROOT}")
    sb = Path("/mnt/storagebox")
    if not sb.exists() or not os.path.ismount(sb):
        report(1, "storagebox: /mnt/storagebox not mounted")
    else:
        report(0, "storagebox: mounted")

    print(f"\noverall: {'HEALTHY' if worst == 0 else 'WARNINGS' if worst == 1 else 'CRITICAL'}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
