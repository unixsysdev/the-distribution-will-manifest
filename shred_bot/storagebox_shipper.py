"""Storagebox shipper — compress + move local buffer files to the Hetzner mount.

Design (2026-06-09):
  - The firehose recorders (raw_shred_firehose + grpc_firehose) write files to
    a LOCAL buffer/<stream>/active/X.bin while recording. On rotation they
    atomically rename active/X.bin -> ready/X.bin and never block.
  - This shipper polls all configured ready/ dirs, gzips each file to a
    local tmp/, transfers the .gz to /mnt/storagebox/<stream>/.tmp/X.bin.gz,
    atomically renames it to X.bin.gz on the box, then deletes the local
    source. That order is intentional: the local file is only removed
    AFTER the storagebox file has its final visible name.
  - If the storagebox is unreachable, the shipper logs and retries on the
    next poll. ready/ accumulates locally. On a 460 GB SSD with 428 GB
    free, ~5 GB/h of raw shreds = 80-90 hours of buffer before pressure
    builds. Hours of mount downtime is recoverable without data loss.

What the shipper does NOT do:
  - It does not touch the active/ file the recorder is currently writing.
  - It does not gzip files in place on the local SSD as a fallback
    (that defeats the whole point of "save SSD space"). If storagebox is
    down, files stay raw in ready/ — they're still valid framed data
    that can be decoded directly.

Usage (via systemd unit pumpfun-storagebox-shipper):
    /root/the-distribution-will-manifest/venv/bin/python -u shred_bot/storagebox_shipper.py
"""
from __future__ import annotations
import gzip, os, shutil, signal, sys, time
from pathlib import Path

# Map: local buffer ready/ dir -> remote storagebox dir.
# Add entries here as new firehose streams are added (e.g. grpc_firehose).
# Note on the dst path: Hetzner exposes the user home as a CIFS share
# called "backup". When mounted at /mnt/storagebox, the user-home root is
# visible there, and the existing data sits in a /backup subdirectory
# (Hetzner default). So /mnt/storagebox/backup/<stream> is the same path
# the old sshfs mount (`:./backup`) was already writing to — no migration
# of existing files needed.
STREAMS = [
    {
        "name":  "raw_shred_entries",
        "ready": Path("/root/the-distribution-will-manifest/buffer/raw_shred_entries/ready"),
        "tmp":   Path("/root/the-distribution-will-manifest/buffer/raw_shred_entries/tmp"),
        "dst":   Path("/mnt/storagebox/backup/raw_shred_entries"),
    },
    {
        "name":  "grpc_firehose",
        "ready": Path("/root/the-distribution-will-manifest/buffer/grpc_firehose/ready"),
        "tmp":   Path("/root/the-distribution-will-manifest/buffer/grpc_firehose/tmp"),
        "dst":   Path("/mnt/storagebox/backup/grpc_firehose"),
    },
]

POLL_SECS         = float(os.getenv("POLL_SECS", "15.0"))
GZIP_LEVEL        = int(os.getenv("GZIP_LEVEL", "3"))     # speed/ratio sweet spot
CHUNK             = 1 << 20                                # 1 MB
STATS_INTERVAL_S  = float(os.getenv("STATS_INTERVAL_S", "60.0"))
# Network throttle for the CIFS upload step. The actual effective rate is
# ~70% of this value because CIFS with cache=strict does synchronous writes
# (each chunk blocks for the network round-trip BEFORE sleep kicks in, so
# the throttle compounds with the natural sync-write latency). 70 MB/sec
# nominal -> ~50 MB/sec effective, which keeps the storagebox upload below
# the sustained ~100 MB/sec we observed unthrottled, so SSH and gRPC inbound
# stay responsive during ship windows.
# Set to 0 to disable (uncapped, ~100 MB/sec on CIFS).
# Empirical measurement on this box (500MB synthetic):
#   nominal 50 -> actual 35 MB/sec  (30% under)
#   nominal 70 -> actual ~50 MB/sec (target)
#   unthrottled -> ~102 MB/sec
UPLOAD_MB_PER_SEC = float(os.getenv("UPLOAD_MB_PER_SEC", "70.0"))

_stop = False
def _sig(*_):
    global _stop
    _stop = True
    print("[shipper] shutdown signal received", flush=True)


def _mount_active(remote_root: Path) -> bool:
    """Best-effort check: does anything under /mnt/storagebox show up in
    /proc/mounts? Avoids slamming a dead mount with copy attempts."""
    try:
        with open("/proc/mounts") as fh:
            for line in fh:
                fields = line.split()
                if len(fields) >= 2 and fields[1].startswith("/mnt/storagebox"):
                    return True
    except Exception:
        pass
    return False


def ship_one(src_path: Path, tmp_dir: Path, dst_dir: Path, stats: dict) -> bool:
    """Compress src_path -> local tmp, then upload to dst_dir, then delete
    the local source. Returns True on success."""
    name = src_path.name              # e.g. raw-shreds-20260609T040020Z.bin
    gz_name = name + ".gz"            # raw-shreds-20260609T040020Z.bin.gz
    local_gz = tmp_dir / gz_name
    remote_tmp = dst_dir / (".tmp." + gz_name)
    remote_gz  = dst_dir / gz_name
    raw_size = src_path.stat().st_size

    # 1) gzip locally (CPU-bound, no network involvement yet).
    t0 = time.time()
    try:
        with open(src_path, "rb") as fsrc, \
             gzip.open(local_gz, "wb", compresslevel=GZIP_LEVEL) as fdst:
            while True:
                chunk = fsrc.read(CHUNK)
                if not chunk: break
                fdst.write(chunk)
    except Exception as e:
        print(f"[shipper] local gzip failed for {name}: {e}", flush=True)
        try: local_gz.unlink(missing_ok=True)
        except Exception: pass
        stats["fail_gzip"] += 1
        return False
    gzip_dt = time.time() - t0
    gz_size = local_gz.stat().st_size

    # 2) Make sure dst_dir exists on the mount (cheap stat).
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[shipper] mkdir({dst_dir}) failed: {e}; will retry next poll", flush=True)
        local_gz.unlink(missing_ok=True)
        stats["fail_mkdir"] += 1
        return False

    # 3) Copy local_gz -> remote_tmp (the network-bound step).
    # Throttled chunked write: caps bandwidth so concurrent SSH / gRPC
    # streams on the same link aren't degraded during ship windows.
    #
    # Uses 4MB chunks (matching the CIFS mount's rsize/wsize) so each
    # write is a single network round-trip and the per-chunk time at
    # the target rate is ~80ms — much larger than Linux sleep
    # granularity (~5-10ms), so the sleep error becomes negligible.
    # The 1MB chunk size used elsewhere undershoots target by ~25-30%
    # because the per-chunk sleep target (20ms at 50 MB/sec) is on
    # the same order as the sleep jitter.
    #
    # Cumulative budget timer: after each chunk, compute "where should
    # I be on the byte budget by now" and sleep the gap. Self-corrects
    # for any individual chunk being slow or fast.
    t1 = time.time()
    try:
        if UPLOAD_MB_PER_SEC > 0:
            target_bps = UPLOAD_MB_PER_SEC * 1e6
            chunk = 4 << 20   # 4 MB (matches CIFS rsize/wsize)
            bytes_done = 0
            t_start = time.perf_counter()
            with open(local_gz, "rb") as f_in, open(remote_tmp, "wb") as f_out:
                while True:
                    buf = f_in.read(chunk)
                    if not buf: break
                    f_out.write(buf)
                    bytes_done += len(buf)
                    budget_time = bytes_done / target_bps
                    elapsed     = time.perf_counter() - t_start
                    if elapsed < budget_time:
                        time.sleep(budget_time - elapsed)
        else:
            shutil.copyfile(local_gz, remote_tmp)
    except Exception as e:
        print(f"[shipper] upload failed for {name}: {e}; will retry next poll", flush=True)
        try: remote_tmp.unlink(missing_ok=True)
        except Exception: pass
        local_gz.unlink(missing_ok=True)
        stats["fail_upload"] += 1
        return False
    upload_dt = time.time() - t1

    # 4) Atomic rename on the storagebox: .tmp.X.gz -> X.gz. CIFS and sshfs
    # both support same-dir rename atomically.
    try:
        os.rename(remote_tmp, remote_gz)
    except Exception as e:
        print(f"[shipper] remote rename failed for {name}: {e}; will retry", flush=True)
        try: remote_tmp.unlink(missing_ok=True)
        except Exception: pass
        local_gz.unlink(missing_ok=True)
        stats["fail_rename"] += 1
        return False

    # 5) Success — delete local source + local tmp.
    try:
        src_path.unlink()
        local_gz.unlink()
    except Exception as e:
        # The data IS shipped (remote .gz exists with proper name); failure
        # to delete is non-fatal. Log loudly so we notice if it persists.
        print(f"[shipper] cleanup failed for {name}: {e}", flush=True)
        stats["fail_cleanup"] += 1

    ratio = gz_size / raw_size if raw_size else 0
    print(f"[shipper] ok  {name}  raw={raw_size/1e6:.0f}MB  "
          f"gz={gz_size/1e6:.0f}MB  ratio={ratio:.2f}  "
          f"gzip_t={gzip_dt:.1f}s  up_t={upload_dt:.1f}s", flush=True)
    stats["shipped"] += 1
    stats["raw_bytes"]  += raw_size
    stats["gz_bytes"]   += gz_size
    stats["gzip_secs"]  += gzip_dt
    stats["upload_secs"] += upload_dt
    return True


def main():
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _sig)

    for stream in STREAMS:
        stream["ready"].mkdir(parents=True, exist_ok=True)
        stream["tmp"].mkdir(parents=True, exist_ok=True)

    stats = {"shipped": 0, "raw_bytes": 0, "gz_bytes": 0,
             "gzip_secs": 0.0, "upload_secs": 0.0,
             "fail_gzip": 0, "fail_mkdir": 0, "fail_upload": 0,
             "fail_rename": 0, "fail_cleanup": 0}
    last_stats_t = time.time()
    print(f"[shipper] started; polling every {POLL_SECS:.0f}s; streams={[s['name'] for s in STREAMS]}",
          flush=True)

    while not _stop:
        # Mount sanity check. If not mounted, skip the iteration to avoid
        # filling the system log with errors. Local buffer keeps filling.
        any_mount = _mount_active(Path("/mnt/storagebox"))
        if not any_mount:
            print("[shipper] /mnt/storagebox not mounted; backing off", flush=True)
            time.sleep(POLL_SECS)
            continue

        for stream in STREAMS:
            ready_dir = stream["ready"]
            tmp_dir   = stream["tmp"]
            dst_dir   = stream["dst"]
            # Process oldest files first so ordering on the remote matches
            # the recording order.
            files = sorted(ready_dir.glob("*.bin"))
            for f in files:
                if _stop: break
                # Skip files that are still being written by an external
                # process — extremely defensive; recorder writes to active/
                # not ready/, but just in case.
                try:
                    if (time.time() - f.stat().st_mtime) < 1.0:
                        continue
                except Exception:
                    continue
                ship_one(f, tmp_dir, dst_dir, stats)

        # Periodic stats line.
        now = time.time()
        if now - last_stats_t >= STATS_INTERVAL_S:
            print(f"[shipper] stats: shipped={stats['shipped']}  "
                  f"raw_mb={stats['raw_bytes']/1e6:.0f}  "
                  f"gz_mb={stats['gz_bytes']/1e6:.0f}  "
                  f"avg_gzip_s={stats['gzip_secs']/max(stats['shipped'],1):.1f}  "
                  f"avg_up_s={stats['upload_secs']/max(stats['shipped'],1):.1f}  "
                  f"failures={stats['fail_gzip']+stats['fail_mkdir']+stats['fail_upload']+stats['fail_rename']+stats['fail_cleanup']}",
                  flush=True)
            last_stats_t = now

        if _stop: break
        time.sleep(POLL_SECS)

    print("[shipper] exit", flush=True)


if __name__ == "__main__":
    main()
