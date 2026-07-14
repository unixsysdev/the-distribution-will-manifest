"""Raw shred firehose — saves EVERY shred entry-message verbatim.

Unlike intent_recorder.py (which filters to pump.fun Buy/Sell only), this
recorder writes the RAW bincode-encoded entries bytes for every message
the ShredstreamProxy.SubscribeEntries stream delivers. Bulk archival.

Two-tier write architecture (2026-06-09):
  RECORDER (this file)
    - writes raw frames to a LOCAL active/ file on the main SSD
    - on rotation: atomic rename active/X.bin -> ready/X.bin
    - opens new active/ file immediately, NEVER blocks for gzip or network
    - that's the whole loss-prevention story: writes never wait for I/O
      slower than local SSD

  SHIPPER (storagebox_shipper.py — separate process / systemd unit)
    - watches buffer/*/ready/ for completed files
    - gzips to buffer/.../tmp/, transfers to /mnt/storagebox/.../X.bin.gz
    - on success deletes the local source (saves SSD space)
    - on storagebox unreachable: backs off, retries; local buffer fills up
      to ~hundreds of GB before pressure builds (428 GB free on the SSD)

File format (unchanged from earlier version, simple self-framed binary):
  Each frame:
      u64 LE   recv_ns          — our receive time (time.time_ns())
      u64 LE   slot             — the entry's slot (from msg.slot)
      u32 LE   payload_len      — bytes that follow
      payload_len bytes         — bincode-encoded Vec<solana_entry::Entry>

Files rotate hourly, name pattern raw-shreds-YYYYMMDDTHHMMSSZ.bin.
On rotation, the in-flight file is MOVED (rename, atomic on same fs)
to the ready/ subdir; the shipper picks it up from there. No gzip in
the recorder hot path anymore — that was the 30-60s stall every hour.

Usage (via systemd unit pumpfun-shred-firehose):
    /root/the-distribution-will-manifest/venv/bin/python -u shred_bot/raw_shred_firehose.py
"""
from __future__ import annotations
import asyncio, os, signal, struct, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "stubs"))

import grpc
import shredstream_pb2
import shredstream_pb2_grpc

ENDPOINT          = os.getenv("SHREDS_ENDPOINT", "shreds-fra6-1.erpc.global:80")
# Local buffer root. The recorder writes here on the main SSD.
# The shipper watches ready/ and ships gzipped files to the storagebox.
BUFFER_ROOT       = Path(os.getenv(
    "BUFFER_ROOT",
    "/root/the-distribution-will-manifest/buffer/raw_shred_entries"
))
ROTATE_SECS       = int(os.getenv("ROTATE_SECS", "3600"))
STATS_INTERVAL_S  = int(os.getenv("STATS_INTERVAL_S", "60"))
RECONNECT_DELAY_S = float(os.getenv("RECONNECT_DELAY_S", "1.0"))

FRAME_HEADER_FMT  = "<QQI"   # recv_ns, slot, payload_len
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FMT)


class FirehoseRecorder:
    def __init__(self, buffer_root: Path, rotate_secs: int):
        # Two sub-directories on the SAME filesystem so rename is atomic.
        self.active_dir = buffer_root / "active"
        self.ready_dir  = buffer_root / "ready"
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.ready_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_secs = rotate_secs
        self.fh = None
        self.fh_opened_at = 0
        self.fh_path = None
        self.bytes_written_total = 0
        self.bytes_written_window = 0
        self.frames_total = 0
        self.frames_window = 0
        self.last_stats_t = time.time()
        self.start_t = time.time()
        # First-run housekeeping: any *.bin lingering in active/ from a
        # previous crash should be moved to ready/ so the shipper picks
        # them up (they're already framed; we just didn't get to rotate).
        try:
            for stale in self.active_dir.glob("raw-shreds-*.bin"):
                target = self.ready_dir / stale.name
                stale.rename(target)
                print(f"[firehose] startup: recovered {stale.name} -> ready/", flush=True)
        except Exception as e:
            print(f"[firehose] startup recovery err: {e}", flush=True)

    def _open_new_file(self):
        """Open a fresh active/ file. Caller is responsible for handling the
        previous fh_path (rotation moves it to ready/ atomically before we
        get here)."""
        if self.fh is not None:
            try: self.fh.close()
            except Exception: pass
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.fh_path = self.active_dir / f"raw-shreds-{ts}.bin"
        self.fh = open(self.fh_path, "ab")  # binary append
        self.fh_opened_at = time.time()
        print(f"[firehose] opened {self.fh_path}", flush=True)

    def _maybe_rotate(self):
        if self.fh is None or (time.time() - self.fh_opened_at) >= self.rotate_secs:
            old_path = self.fh_path
            old_fh   = self.fh
            # Close the old file BEFORE renaming so kernel page cache flushes.
            if old_fh is not None:
                try: old_fh.close()
                except Exception: pass
            # Atomic rename within the same filesystem: active/ -> ready/.
            # This is ~µs even for multi-GB files because it's just a directory
            # entry update. Recorder is back to writing in <1 ms.
            if old_path is not None:
                try:
                    target = self.ready_dir / old_path.name
                    old_path.rename(target)
                    print(f"[firehose] rotated -> ready/{old_path.name}", flush=True)
                except Exception as e:
                    print(f"[firehose] rotation rename {old_path} failed: {e}", flush=True)
            # Open the new active/ file. Total stall: a few ms.
            self._open_new_file()

    def write_frame(self, recv_ns: int, slot: int, payload: bytes):
        hdr = struct.pack(FRAME_HEADER_FMT, recv_ns, slot, len(payload))
        self.fh.write(hdr)
        self.fh.write(payload)
        n = FRAME_HEADER_SIZE + len(payload)
        self.bytes_written_total += n
        self.bytes_written_window += n
        self.frames_total += 1
        self.frames_window += 1

    def maybe_print_stats(self):
        now = time.time()
        dt = now - self.last_stats_t
        if dt < STATS_INTERVAL_S: return
        up = now - self.start_t
        mb_w = self.bytes_written_window / 1e6
        mb_total = self.bytes_written_total / 1e6
        # Count files awaiting ship — gives an at-a-glance shipper-backlog
        # signal in the same log stream as the recorder.
        try:
            ready_n = sum(1 for _ in self.ready_dir.glob("raw-shreds-*.bin"))
        except Exception:
            ready_n = -1
        print(f"[firehose] uptime={up:.0f}s  "
              f"frames_window={self.frames_window} ({self.frames_window/dt:.0f}/s)  "
              f"mb_window={mb_w:.1f} ({mb_w/dt:.2f} MB/s)  "
              f"total_frames={self.frames_total}  total_mb={mb_total:.1f}  "
              f"ready_backlog={ready_n}",
              flush=True)
        self.frames_window = self.bytes_written_window = 0
        self.last_stats_t = now

    def close(self):
        """On shutdown: close the open file AND move it to ready/ so the
        shipper finishes the job (otherwise a clean shutdown would leave
        the last file stuck in active/ until the next startup)."""
        if self.fh is not None:
            try: self.fh.close()
            except Exception: pass
        if self.fh_path is not None and self.fh_path.exists():
            try:
                target = self.ready_dir / self.fh_path.name
                self.fh_path.rename(target)
                print(f"[firehose] shutdown: moved active/{self.fh_path.name} -> ready/",
                      flush=True)
            except Exception as e:
                print(f"[firehose] shutdown rename {self.fh_path} failed: {e}",
                      flush=True)


async def run(rec: FirehoseRecorder, stop_evt: asyncio.Event):
    while not stop_evt.is_set():
        try:
            print(f"[firehose] connecting {ENDPOINT}", flush=True)
            async with grpc.aio.insecure_channel(ENDPOINT) as ch:
                stub = shredstream_pb2_grpc.ShredstreamProxyStub(ch)
                req  = shredstream_pb2.SubscribeEntriesRequest()
                stream = stub.SubscribeEntries(req)
                print(f"[firehose] subscribed", flush=True)
                async for msg in stream:
                    if stop_evt.is_set(): break
                    rec._maybe_rotate()
                    rec.write_frame(time.time_ns(), int(msg.slot),
                                     bytes(msg.entries))
                    rec.maybe_print_stats()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[firehose] stream err: {e}; reconnect in {RECONNECT_DELAY_S}s",
                  flush=True)
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=RECONNECT_DELAY_S)
            except asyncio.TimeoutError:
                pass


def main():
    rec = FirehoseRecorder(BUFFER_ROOT, ROTATE_SECS)
    rec._open_new_file()
    stop_evt = asyncio.Event()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    def _shutdown(*_):
        print("[firehose] shutdown signal", flush=True)
        loop.call_soon_threadsafe(stop_evt.set)
    for s in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(s, _shutdown)
        except Exception: pass
    try:
        loop.run_until_complete(run(rec, stop_evt))
    finally:
        rec.close()
        print(f"[firehose] exit; total_frames={rec.frames_total}  "
              f"total_mb={rec.bytes_written_total/1e6:.1f}", flush=True)
        loop.close()


if __name__ == "__main__":
    main()
