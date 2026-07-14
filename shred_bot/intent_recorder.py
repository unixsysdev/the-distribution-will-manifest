"""Long-running shred-stream pump.fun intent recorder.

Subscribes to ERPC Direct Shreds (Jito ShredstreamProxy), parses every entry,
extracts every pump.fun Buy intent, and:
  - appends one JSON line per intent to a rotating jsonl file under
    shred_bot/intent_capture/intent-YYYYMMDDTHHMMSSZ.jsonl
  - prints rolling stats every STATS_INTERVAL_S
  - auto-reconnects on stream error
  - flushes + rotates cleanly on SIGTERM (so systemd-managed restarts dont
    drop the tail of an in-flight file)

Stays as a separate process (does not touch the live bot in any way). Output
is later consumable by the policy bot via either:
  - tailing the jsonl files for offline analysis, or
  - the shared-memory ring buffer (intent_ring.py) for live decisions.

Usage:
    ./venv/bin/python shred_bot/intent_recorder.py
"""
from __future__ import annotations
import asyncio, gzip, json, os, signal, sys, time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent / "stubs"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grpc
import shredstream_pb2
import shredstream_pb2_grpc

# Reuse the parser from intent_extractor (same module dir)
from intent_extractor import parse_entry_batch, extract_intents, PUMP_FUN_PROG
from intent_ring import IntentRingWriter

ENDPOINT          = os.getenv("SHREDS_ENDPOINT", "shreds-fra6-1.erpc.global:80")
OUT_DIR           = Path(os.getenv("OUT_DIR",
                                   "/root/the-distribution-will-manifest/shred_bot/intent_capture"))
ROTATE_SECS       = int(os.getenv("ROTATE_SECS", "3600"))   # 1 hour per file
STATS_INTERVAL_S  = int(os.getenv("STATS_INTERVAL_S", "60"))
RECONNECT_DELAY_S = float(os.getenv("RECONNECT_DELAY_S", "1.0"))
# Per-write timing instrumentation. OFF by default — turn on for an
# experimental run (export RECORDER_TIMING=1 then restart the service)
# to see jsonl-write vs ring-write vs total latency distributions in
# the journal. Adds ~3 perf_counter_ns calls per intent (a few ns each);
# safe to leave on but kept opt-in to be cautious in production.
TIMING_ENABLED    = os.getenv("RECORDER_TIMING", "0") == "1"


def _gzip_one(path: Path) -> bool:
    """Compress path to path.gz and unlink the original. Idempotent: skip if
    path.gz already exists. Returns True on success or already-done, False on
    error."""
    gz = Path(str(path) + ".gz")
    if gz.exists():
        if not path.exists():
            return True
        partial = gz.with_name(gz.name + f".partial_{int(time.time())}")
        try:
            gz.rename(partial)
            print(f"[recorder] quarantined pre-existing {gz.name} -> "
                  f"{partial.name}; recompressing {path.name}", flush=True)
        except Exception as e:
            print(f"[recorder] could not quarantine {gz}: {e}", flush=True)
            return False
    tmp = gz.with_name(gz.name + f".tmp.{os.getpid()}")
    try:
        try: tmp.unlink()
        except FileNotFoundError: pass
        with open(path, "rb") as src, gzip.open(tmp, "wb", compresslevel=3) as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk: break
                dst.write(chunk)
        tmp.replace(gz)
        path.unlink()
        return True
    except Exception as e:
        try: tmp.unlink()
        except Exception: pass
        print(f"[recorder] gzip {path} failed: {e}", flush=True)
        return False


class Recorder:
    def __init__(self, out_dir: Path, rotate_secs: int, ring: IntentRingWriter | None = None):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir     = out_dir
        self.rotate_secs = rotate_secs
        self.ring        = ring  # optional; if set, writes to SHM too
        self.fh          = None
        self.fh_opened_at = 0
        self.fh_path     = None
        self.ring_errors = 0
        # stats — reset at every stats print
        self.n_entries   = 0
        self.n_txs       = 0
        self.n_pump      = 0
        self.n_buys      = 0
        self.parse_err   = 0
        self.last_stats_t = time.time()
        self.start_t     = time.time()
        # totals — never reset, for the systemd journal "uptime" line
        self.total_buys  = 0
        # Per-write timing buffers (capped to avoid unbounded growth between
        # stats prints). Only used when TIMING_ENABLED.
        self._t_jsonl_ns = []
        self._t_ring_ns  = []
        self._t_total_ns = []
        # Startup orphan recovery: any .jsonl that the previous instance left
        # ungzipped (because it crashed / was restarted mid-hour) gets gzipped
        # now. Without this, every service restart strands an ungzipped file
        # taking ~5-10x the space of the gzipped version.
        try:
            orphans = list(self.out_dir.glob("intent-*.jsonl"))
            if orphans:
                print(f"[recorder] startup: found {len(orphans)} ungzipped orphan(s); "
                      f"compressing", flush=True)
                for o in orphans:
                    if _gzip_one(o):
                        print(f"[recorder] gzipped orphan {o.name}.gz", flush=True)
        except Exception as e:
            print(f"[recorder] startup orphan scan err: {e}", flush=True)

    def _open_new_file(self):
        if self.fh is not None:
            try: self.fh.close()
            except Exception: pass
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.fh_path = self.out_dir / f"intent-{ts}.jsonl"
        self.fh = open(self.fh_path, "a", buffering=1)
        self.fh_opened_at = time.time()
        print(f"[recorder] opened {self.fh_path}", flush=True)

    def _maybe_rotate(self):
        if self.fh is None or (time.time() - self.fh_opened_at) >= self.rotate_secs:
            old = self.fh_path
            self._open_new_file()
            if old is not None and _gzip_one(old):
                print(f"[recorder] gzipped {old.name}.gz", flush=True)

    def write_intent(self, intent: dict):
        # Optional per-step timing. Toggled by RECORDER_TIMING=1 env var on
        # service start. When off, the perf_counter calls are skipped
        # entirely — zero overhead in real runs. Stats roll up into
        # maybe_print_stats and print every 60s.
        if TIMING_ENABLED:
            _t0 = time.perf_counter_ns()
            self.fh.write(json.dumps(intent, separators=(",", ":")) + "\n")
            _t1 = time.perf_counter_ns()
            if self.ring is not None:
                try: self.ring.write(intent)
                except Exception: self.ring_errors += 1
            _t2 = time.perf_counter_ns()
            self._t_jsonl_ns.append(_t1 - _t0)
            self._t_ring_ns.append(_t2 - _t1)
            self._t_total_ns.append(_t2 - _t0)
        else:
            self.fh.write(json.dumps(intent, separators=(",", ":")) + "\n")
            if self.ring is not None:
                try: self.ring.write(intent)
                except Exception: self.ring_errors += 1
        self.total_buys += 1

    def maybe_print_stats(self):
        now = time.time()
        dt = now - self.last_stats_t
        if dt < STATS_INTERVAL_S: return
        up = now - self.start_t
        print(f"[recorder] uptime={up:.0f}s  entries={self.n_entries}  "
              f"tx={self.n_txs}  pump.fun_tx={self.n_pump}  "
              f"buys={self.n_buys} ({self.n_buys/dt:.2f}/s)  "
              f"parse_err={self.parse_err}  total_buys={self.total_buys}",
              flush=True)
        # Per-write latency summary (only if RECORDER_TIMING=1 was set)
        if TIMING_ENABLED and self._t_total_ns:
            import statistics as _st
            def _qs(name, arr):
                arr = sorted(arr)
                n = len(arr)
                p50 = arr[n//2]
                p90 = arr[int(n*0.9)]
                p99 = arr[int(n*0.99) if int(n*0.99) < n else n-1]
                mx  = arr[-1]
                return f"{name}: p50={p50/1000:.1f}us p90={p90/1000:.1f}us p99={p99/1000:.1f}us max={mx/1000:.0f}us"
            print(f"[recorder.timing] n={len(self._t_total_ns)} "
                  f"{_qs('jsonl', self._t_jsonl_ns)}  "
                  f"{_qs('ring',  self._t_ring_ns)}  "
                  f"{_qs('total', self._t_total_ns)}",
                  flush=True)
            self._t_jsonl_ns.clear()
            self._t_ring_ns.clear()
            self._t_total_ns.clear()
        self.n_entries = self.n_txs = self.n_pump = self.n_buys = 0
        self.parse_err = 0
        self.last_stats_t = now

    def close(self):
        if self.fh is not None:
            try: self.fh.close()
            except Exception: pass
        # Gzip the file we just closed so a clean shutdown doesn't leave it
        # ungzipped (startup orphan-scan would catch it anyway, but the right
        # time to compress is now).
        if self.fh_path is not None and self.fh_path.exists():
            if _gzip_one(self.fh_path):
                print(f"[recorder] shutdown: gzipped {self.fh_path.name}.gz",
                      flush=True)
        if self.ring is not None:
            try: self.ring.close()
            except Exception: pass


async def run(rec: Recorder, stop_evt: asyncio.Event):
    while not stop_evt.is_set():
        try:
            print(f"[recorder] connecting {ENDPOINT}", flush=True)
            async with grpc.aio.insecure_channel(ENDPOINT) as ch:
                stub = shredstream_pb2_grpc.ShredstreamProxyStub(ch)
                req  = shredstream_pb2.SubscribeEntriesRequest()
                stream = stub.SubscribeEntries(req)
                print(f"[recorder] subscribed", flush=True)
                async for msg in stream:
                    if stop_evt.is_set(): break
                    rec.n_entries += 1
                    try:
                        txs = parse_entry_batch(msg.entries)
                    except Exception:
                        rec.parse_err += 1
                        continue
                    rec.n_txs += len(txs)
                    pumps_in_msg = sum(1 for tx in txs if PUMP_FUN_PROG in tx["keys"])
                    rec.n_pump += pumps_in_msg
                    if pumps_in_msg == 0:
                        rec.maybe_print_stats(); continue
                    intents = extract_intents(txs, int(msg.slot))
                    rec.n_buys += len(intents)
                    rec._maybe_rotate()
                    now_ns = time.time_ns()
                    for it in intents:
                        it["recv_ns"] = now_ns
                        rec.write_intent(it)
                    rec.maybe_print_stats()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[recorder] stream error: {e}; reconnect in {RECONNECT_DELAY_S}s",
                  flush=True)
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=RECONNECT_DELAY_S)
            except asyncio.TimeoutError:
                pass


def main():
    # Open the shared-memory ring before subscribing (so a reader process can
    # attach immediately on startup). Continues even if SHM init fails — the
    # jsonl-on-disk path is the durable record; ring is for live consumers.
    ring = None
    try:
        ring = IntentRingWriter()
        print(f"[recorder] shm ring '{ring.name}': capacity={ring.capacity} "
              f"record_size={ring.record_size}  (writer)", flush=True)
    except Exception as e:
        print(f"[recorder] shm ring init failed: {e} -- continuing without ring",
              flush=True)
    rec = Recorder(OUT_DIR, ROTATE_SECS, ring=ring)
    rec._open_new_file()
    stop_evt = asyncio.Event()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    def _shutdown(*_):
        print("[recorder] shutdown signal", flush=True)
        loop.call_soon_threadsafe(stop_evt.set)
    for s in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(s, _shutdown)
        except Exception: pass
    try:
        loop.run_until_complete(run(rec, stop_evt))
    finally:
        rec.close()
        print(f"[recorder] exit; total_buys={rec.total_buys}", flush=True)
        loop.close()


if __name__ == "__main__":
    main()
