"""gRPC firehose — full-fidelity archive of every pump.fun-touching SubscribeUpdate.

Counterpart of raw_shred_firehose.py but for the Yellowstone gRPC executed-tx
stream. Saves the FULL proto-serialized SubscribeUpdate per message so any
future parser bug or new field we discover later can be replayed against the
historical archive without needing fresh capture.

Architecture (matches the shred firehose):
  - Writes raw frames to /root/the-distribution-will-manifest/buffer/grpc_firehose/active/
  - Hourly rotation: atomic rename active/X.bin -> ready/X.bin
  - Never blocks on gzip or network — storagebox_shipper handles compression
    + transfer to the Hetzner mount in a separate process

Subscription:
  Same scope as the augmented grpc_capture.py (so the firehose archive is a
  superset of what the structured capture saves):
    - account_include: bonding curve, PumpSwap AMM, pump_fees
    - failed: both success and failure
    - commitment: processed (low-latency, matches what the bot's gRPC source
      uses)
  Difference from grpc_capture.py: the firehose saves proto bytes verbatim
  rather than parsing into JSONL. The structured capture stays the queryable
  feed; the firehose is the regret-free byte archive.

Frame format (binary, same as raw_shred_firehose):
  u64 LE   recv_ns          — local time when we received the msg
  u64 LE   slot              — the tx's slot
  u32 LE   payload_len       — bytes that follow
  payload_len bytes          — SubscribeUpdate.SerializeToString()

To decode an archived frame:
    msg = geyser_pb2.SubscribeUpdate()
    msg.ParseFromString(payload)
    tx = msg.transaction.transaction       # SubscribeUpdateTransaction
    meta = tx.meta                          # full TransactionStatusMeta

Volume estimate: ~22 msgs/s × ~8 KB avg = ~633 MB/h uncompressed,
~250-350 MB/h gzipped. At 5 TB storage that's ~110-150 days runway when
combined with the shred firehose.

Usage (via systemd unit pumpfun-grpc-firehose):
    /root/the-distribution-will-manifest/venv/bin/python -u grpc_firehose.py
"""
from __future__ import annotations
import asyncio, os, signal, struct, sys, time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "grpc_stubs"))

import grpc
import geyser_pb2
import geyser_pb2_grpc
import config


ENDPOINT          = os.getenv("GRPC_FIREHOSE_ENDPOINT", "grpc-fra1-1.erpc.global:80")
INSECURE          = os.getenv("GRPC_FIREHOSE_INSECURE", "1") == "1"
COMMITMENT_NAME   = os.getenv("COMMITMENT", "processed")
BUFFER_ROOT       = Path(os.getenv(
    "BUFFER_ROOT",
    "/root/the-distribution-will-manifest/buffer/grpc_firehose"
))
ROTATE_SECS       = int(os.getenv("ROTATE_SECS", "3600"))
STATS_INTERVAL_S  = int(os.getenv("STATS_INTERVAL_S", "60"))
RECONNECT_DELAY_S = float(os.getenv("RECONNECT_DELAY_S", "5.0"))

PUMP_FUN_PROG  = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMPSWAP_PROG  = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
PUMP_FEES_PROG = "pfeeXjVdkLAjAsfFqdtshb3aJxJrAcj62YotL5XPFCq"

FRAME_HEADER_FMT  = "<QQI"   # recv_ns, slot, payload_len
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FMT)


def _commitment_to_enum(name: str) -> int:
    m = {"processed": geyser_pb2.CommitmentLevel.PROCESSED,
         "confirmed": geyser_pb2.CommitmentLevel.CONFIRMED,
         "finalized": geyser_pb2.CommitmentLevel.FINALIZED}
    return m.get(name, geyser_pb2.CommitmentLevel.PROCESSED)


class GrpcFirehoseRecorder:
    """Local-only writer with atomic-rename rotation. Same shape as the
    raw_shred_firehose recorder — see that file for the full design notes."""

    def __init__(self, buffer_root: Path, rotate_secs: int):
        self.active_dir = buffer_root / "active"
        self.ready_dir  = buffer_root / "ready"
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.ready_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_secs = rotate_secs
        self.fh = None
        self.fh_opened_at = 0
        self.fh_path = None
        self.bytes_total = 0
        self.bytes_window = 0
        self.frames_total = 0
        self.frames_window = 0
        self.last_stats_t = time.time()
        self.start_t = time.time()
        # Recover any orphan in active/ from previous crash.
        try:
            for stale in self.active_dir.glob("grpc-firehose-*.bin"):
                target = self.ready_dir / stale.name
                stale.rename(target)
                print(f"[grpc-firehose] startup: recovered {stale.name} -> ready/",
                      flush=True)
        except Exception as e:
            print(f"[grpc-firehose] startup recovery err: {e}", flush=True)

    def _open_new_file(self):
        if self.fh is not None:
            try: self.fh.close()
            except Exception: pass
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.fh_path = self.active_dir / f"grpc-firehose-{ts}.bin"
        self.fh = open(self.fh_path, "ab")
        self.fh_opened_at = time.time()
        print(f"[grpc-firehose] opened {self.fh_path}", flush=True)

    def _maybe_rotate(self):
        if self.fh is None or (time.time() - self.fh_opened_at) >= self.rotate_secs:
            old_path = self.fh_path
            old_fh   = self.fh
            if old_fh is not None:
                try: old_fh.close()
                except Exception: pass
            if old_path is not None:
                try:
                    target = self.ready_dir / old_path.name
                    old_path.rename(target)
                    print(f"[grpc-firehose] rotated -> ready/{old_path.name}",
                          flush=True)
                except Exception as e:
                    print(f"[grpc-firehose] rotation rename {old_path} failed: {e}",
                          flush=True)
            self._open_new_file()

    def write_frame(self, recv_ns: int, slot: int, payload: bytes):
        hdr = struct.pack(FRAME_HEADER_FMT, recv_ns, slot, len(payload))
        self.fh.write(hdr)
        self.fh.write(payload)
        n = FRAME_HEADER_SIZE + len(payload)
        self.bytes_total += n
        self.bytes_window += n
        self.frames_total += 1
        self.frames_window += 1

    def maybe_print_stats(self):
        now = time.time()
        dt = now - self.last_stats_t
        if dt < STATS_INTERVAL_S: return
        up = now - self.start_t
        mb_w = self.bytes_window / 1e6
        mb_total = self.bytes_total / 1e6
        try:
            ready_n = sum(1 for _ in self.ready_dir.glob("grpc-firehose-*.bin"))
        except Exception:
            ready_n = -1
        print(f"[grpc-firehose] uptime={up:.0f}s  "
              f"frames_window={self.frames_window} ({self.frames_window/dt:.0f}/s)  "
              f"mb_window={mb_w:.1f} ({mb_w/dt:.2f} MB/s)  "
              f"total_frames={self.frames_total}  total_mb={mb_total:.1f}  "
              f"ready_backlog={ready_n}",
              flush=True)
        self.frames_window = self.bytes_window = 0
        self.last_stats_t = now

    def close(self):
        if self.fh is not None:
            try: self.fh.close()
            except Exception: pass
        if self.fh_path is not None and self.fh_path.exists():
            try:
                target = self.ready_dir / self.fh_path.name
                self.fh_path.rename(target)
                print(f"[grpc-firehose] shutdown: moved active/{self.fh_path.name} -> ready/",
                      flush=True)
            except Exception as e:
                print(f"[grpc-firehose] shutdown rename failed: {e}", flush=True)


async def run(rec: GrpcFirehoseRecorder, stop_evt: asyncio.Event):
    token = config.GRPC_TOKEN
    metadata = (("x-token", token),) if token else None

    while not stop_evt.is_set():
        try:
            print(f"[grpc-firehose] connecting {ENDPOINT} (insecure={INSECURE})",
                  flush=True)
            ch_factory = (grpc.aio.insecure_channel if INSECURE
                          else lambda ep: grpc.aio.secure_channel(ep, grpc.ssl_channel_credentials()))
            async with ch_factory(ENDPOINT) as ch:
                stub = geyser_pb2_grpc.GeyserStub(ch)
                req = geyser_pb2.SubscribeRequest()
                req.transactions["pumpfun"].account_include.append(PUMP_FUN_PROG)
                req.transactions["pumpfun"].account_include.append(PUMPSWAP_PROG)
                req.transactions["pumpfun"].account_include.append(PUMP_FEES_PROG)
                # No failed filter — capture both successes and failures
                # Also subscribe to blocks_meta so the archive carries
                # block_time (the validator-witnessed UTC per slot). One
                # message per slot; trivial extra bandwidth. Combined with
                # the tx slot field, offline tools can join slot -> block_time.
                req.blocks_meta["pumpfun_bm"].SetInParent()
                req.commitment = _commitment_to_enum(COMMITMENT_NAME)

                async def req_iter():
                    yield req
                    while not stop_evt.is_set():
                        await asyncio.sleep(1)

                print(f"[grpc-firehose] subscribed "
                      f"(programs=[pumpfun,pumpswap,pump_fees], failed=both, "
                      f"commitment={COMMITMENT_NAME}, +blocks_meta)", flush=True)

                async for resp in stub.Subscribe(req_iter(), metadata=metadata):
                    if stop_evt.is_set(): break
                    # Keep BOTH transactions and block_meta updates. Skip
                    # everything else (ping/pong, etc.).
                    if resp.HasField("transaction"):
                        slot = int(resp.transaction.slot) if resp.transaction.slot else 0
                    elif resp.HasField("block_meta"):
                        # block_meta has the slot directly + block_time
                        slot = int(resp.block_meta.slot) if resp.block_meta.slot else 0
                    else:
                        continue
                    rec._maybe_rotate()
                    payload = resp.SerializeToString()
                    rec.write_frame(time.time_ns(), slot, payload)
                    rec.maybe_print_stats()
        except asyncio.CancelledError:
            break
        except grpc.aio.AioRpcError as e:
            print(f"[grpc-firehose] gRPC error {e.code()}: "
                  f"{str(e.details())[:200]}; reconnect in {RECONNECT_DELAY_S}s",
                  flush=True)
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=RECONNECT_DELAY_S)
            except asyncio.TimeoutError:
                pass
        except Exception as e:
            print(f"[grpc-firehose] error: {type(e).__name__}: {e}; "
                  f"reconnect in {RECONNECT_DELAY_S}s", flush=True)
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=RECONNECT_DELAY_S)
            except asyncio.TimeoutError:
                pass


def main():
    rec = GrpcFirehoseRecorder(BUFFER_ROOT, ROTATE_SECS)
    rec._open_new_file()
    stop_evt = asyncio.Event()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown(*_):
        print("[grpc-firehose] shutdown signal", flush=True)
        loop.call_soon_threadsafe(stop_evt.set)

    for s in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(s, _shutdown)
        except Exception: pass

    try:
        loop.run_until_complete(run(rec, stop_evt))
    finally:
        rec.close()
        print(f"[grpc-firehose] exit; total_frames={rec.frames_total}  "
              f"total_mb={rec.bytes_total/1e6:.1f}", flush=True)
        loop.close()


if __name__ == "__main__":
    main()
