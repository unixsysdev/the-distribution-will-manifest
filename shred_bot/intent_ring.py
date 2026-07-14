"""Single-producer / multi-consumer shared-memory ring buffer for pump.fun
buy intents extracted from the shred stream.

Design:
  - One writer process (intent_recorder.py) writes fixed-size records.
  - Any number of reader processes (the policy bot, a dashboard, a debugger)
    attach to the same SharedMemory segment and poll for new records.
  - Lock-free SPSC coordination via a monotonic u64 write_seq in the header.
    On x86_64 u64 writes at 8-byte-aligned addresses are atomic.
  - No allocations on the hot path (fixed-size records, in-place struct pack).

Layout (LE throughout):
  Header (64 bytes, padded):
      magic       u32   0x50465249 = "PFRI" sanity check
      version     u32   1
      capacity    u64   number of record slots
      record_size u64   bytes per record (must match struct)
      write_seq   u64   monotonic counter; reader sees writes once incremented
      (padding to 64)
  Body:
      capacity × record_size bytes

Record (128 bytes):
      recv_ns           u64    monotonic time we observed the intent (time.time_ns())
      slot              u64    Solana slot the entry came from
      mint              32B    raw pubkey
      user              32B    raw pubkey of the signer
      token_amount      u64    Buy.token_amount
      max_sol_cost      u64    Buy.max_sol_cost_lam (slippage cap)
      priority_fee_micro u64   ComputeBudget set_compute_unit_price arg
      jito_tip_lam      u64    lamports transferred to a Jito tip account
      cu_limit          u32    ComputeBudget set_compute_unit_limit arg
      reserved          u32    padding/future use

Default 65,536 capacity × 128 = 8 MB shared region. At ~5 intents/sec the
ring wraps every ~3.6 hours; readers that fall behind > capacity records
will start missing data (write_seq advances past them).
"""
from __future__ import annotations
import base58
import struct
import time
from multiprocessing import shared_memory

MAGIC          = 0x50465249  # "PFRI" little-endian
VERSION        = 3           # v3 adds probable_spoof in an existing padding byte
HEADER_FMT     = "<IIQQQ"
HEADER_SIZE    = 64          # padded
# Layout (120 bytes total; bumped from v2 by repurposing 1 padding byte for probable_spoof):
#   Q  recv_ns
#   Q  slot
#   32s mint
#   32s user
#   Q  token_amount
#   Q  sol_limit_lam (Buy: max_sol_cost; Sell: min_sol_output)
#   Q  priority_fee_micro
#   Q  jito_tip_lam        WARNING: trivially spoofable (revertable tip)
#   I  cu_limit
#   B  is_buy              1=Buy, 0=Sell
#   B  probable_spoof     1=likely spoof/revertable, 0=normal/unknown
#   2x padding
RECORD_FMT     = "<QQ32s32sQQQQIBB2x"
RECORD_SIZE    = struct.calcsize(RECORD_FMT)   # 120
DEFAULT_NAME   = "pumpfun_intents"
DEFAULT_CAPACITY = 65536


def _intent_to_bytes(it: dict, recv_ns: int | None = None) -> bytes:
    """Pack a dict intent into the fixed-size record bytes."""
    if recv_ns is None:
        recv_ns = int(it.get("recv_ns") or time.time_ns())
    mint_b = base58.b58decode(it["mint"])
    user_b = base58.b58decode(it["user"])
    if len(mint_b) != 32: raise ValueError(f"bad mint len {len(mint_b)}")
    if len(user_b) != 32: raise ValueError(f"bad user len {len(user_b)}")
    return struct.pack(RECORD_FMT,
                       recv_ns,
                       int(it["slot"]),
                       mint_b, user_b,
                       int(it["token_amount"]),
                       int(it.get("sol_limit_lam", it.get("max_sol_cost", 0))),
                       int(it.get("priority_fee_micro", 0)),
                       int(it.get("jito_tip_lam", 0)),
                       int(it.get("cu_limit", 0)),
                       1 if it.get("is_buy", True) else 0,
                       1 if it.get("probable_spoof", False) else 0)


def _bytes_to_intent(buf: bytes) -> dict:
    (recv_ns, slot, mint_b, user_b, tok, sol_lim, prio, tip, cu, is_buy_byte, spoof_byte) = \
        struct.unpack(RECORD_FMT, buf)
    is_buy = bool(is_buy_byte)
    return {
        "is_buy": is_buy,
        "recv_ns": recv_ns,
        "slot": slot,
        "mint": base58.b58encode(mint_b).decode(),
        "user": base58.b58encode(user_b).decode(),
        "token_amount": tok,
        "sol_limit_lam": sol_lim,
        "sol_limit_sol": sol_lim / 1e9,
        # Compat aliases keyed by direction
        ("max_sol_cost" if is_buy else "min_sol_output"): sol_lim,
        ("max_sol_cost_sol" if is_buy else "min_sol_output_sol"): sol_lim / 1e9,
        "priority_fee_micro": prio,
        "jito_tip_lam": tip,
        "cu_limit": cu,
        "probable_spoof": bool(spoof_byte),
    }


class IntentRingWriter:
    """Creates / re-creates the SHM segment, writes records in place,
    increments write_seq."""
    def __init__(self, name: str = DEFAULT_NAME,
                 capacity: int = DEFAULT_CAPACITY,
                 record_size: int = RECORD_SIZE):
        if record_size != RECORD_SIZE:
            raise ValueError(f"record_size must be {RECORD_SIZE}")
        total = HEADER_SIZE + capacity * record_size
        # Clean up any stale segment from a previous crash
        try:
            old = shared_memory.SharedMemory(name=name)
            old.close(); old.unlink()
        except FileNotFoundError:
            pass
        self.shm = shared_memory.SharedMemory(name=name, create=True, size=total)
        self.name = name
        self.capacity = capacity
        self.record_size = record_size
        self.buf = self.shm.buf
        # Initialize header
        struct.pack_into(HEADER_FMT, self.buf, 0,
                         MAGIC, VERSION, capacity, record_size, 0)

    def write(self, it: dict) -> None:
        """Append one intent. Single-producer; lock-free."""
        rec = _intent_to_bytes(it)
        write_seq = struct.unpack_from("<Q", self.buf, 24)[0]
        slot_idx  = write_seq % self.capacity
        offset    = HEADER_SIZE + slot_idx * self.record_size
        self.buf[offset:offset + self.record_size] = rec
        # Increment write_seq AFTER the record is written so readers never
        # see a partial record.
        struct.pack_into("<Q", self.buf, 24, write_seq + 1)

    def close(self) -> None:
        try: self.shm.close()
        except Exception: pass
        try: self.shm.unlink()
        except Exception: pass


def _detach_from_resource_tracker(name: str) -> None:
    """Python's multiprocessing.resource_tracker registers every attached
    shared_memory segment for cleanup on the attaching process's exit. For
    a single-producer / many-readers ring, that's catastrophic — the first
    reader to exit unlinks the segment the writer is still using. Standard
    workaround (cpython issue #82300): un-register the segment so only the
    owning writer is responsible for unlinking it on shutdown."""
    try:
        from multiprocessing import resource_tracker
        resource_tracker.unregister(f"/{name}", "shared_memory")
    except Exception:
        pass


class IntentRingReader:
    """Attaches to an existing SHM segment, polls for new records. Each
    reader maintains its own local read_seq cursor."""
    def __init__(self, name: str = DEFAULT_NAME):
        self.shm = shared_memory.SharedMemory(name=name)
        # Detach immediately so this readers exit doesn't unlink the segment.
        _detach_from_resource_tracker(name)
        self.name = name
        self.buf = self.shm.buf
        magic, version, capacity, record_size, _wseq = \
            struct.unpack_from(HEADER_FMT, self.buf, 0)
        if magic != MAGIC:
            raise ValueError(f"bad ring magic 0x{magic:08x} (expected 0x{MAGIC:08x})")
        if version != VERSION:
            raise ValueError(f"version mismatch {version} vs {VERSION}")
        self.capacity = capacity
        self.record_size = record_size
        # Start at current write_seq — only deliver records that arrive AFTER
        # this reader connected. Readers that want backfill should consume
        # the on-disk jsonl files first, then attach to the ring.
        self.read_seq = _wseq

    def write_seq(self) -> int:
        return struct.unpack_from("<Q", self.buf, 24)[0]

    def poll(self, max_records: int = 1000) -> list[dict]:
        """Return up to `max_records` new intents (oldest first). Caller
        should call frequently; if behind by more than `capacity` records,
        the oldest will be silently overwritten by the writer."""
        wseq = self.write_seq()
        if wseq <= self.read_seq: return []
        # If writer lapped us, snap forward (and we'll have missed some)
        if wseq - self.read_seq > self.capacity:
            missed = (wseq - self.read_seq) - self.capacity
            print(f"[ring-reader] LAPPED — missed {missed} records "
                  f"(read_seq={self.read_seq} -> {wseq - self.capacity})")
            self.read_seq = wseq - self.capacity
        n = min(max_records, wseq - self.read_seq)
        out = []
        for i in range(n):
            slot_idx = (self.read_seq + i) % self.capacity
            offset = HEADER_SIZE + slot_idx * self.record_size
            out.append(_bytes_to_intent(bytes(self.buf[offset:offset + self.record_size])))
        self.read_seq += n
        return out

    def close(self) -> None:
        try: self.shm.close()
        except Exception: pass


# ---------- self-test ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "reader":
        # test reader: poll forever, print whatever arrives
        r = IntentRingReader()
        print(f"attached to '{r.name}': capacity={r.capacity} record_size={r.record_size}")
        print(f"starting at write_seq={r.read_seq}")
        while True:
            batch = r.poll(max_records=100)
            for it in batch:
                print(f"  slot={it['slot']}  mint={it['mint'][:14]}  "
                      f"user={it['user'][:14]}  max_sol={it['max_sol_cost_sol']:.4f}  "
                      f"prio_μ={it['priority_fee_micro']}  tip={it['jito_tip_lam']}  "
                      f"recv_ns={it['recv_ns']}")
            time.sleep(0.05)
    else:
        # quick roundtrip test
        w = IntentRingWriter(capacity=16)
        try:
            sample = {
                "slot": 425150000,
                "mint": "GTgQ7kqosJx7etP6FmpPAdG1K2zKg4j6HgWrPcDapump",
                "user": "99QzsYDTWZYUkQ647pLCCoemYNVYGaA8vGEQJ4862fJA",
                "token_amount": 2727462359366,
                "max_sol_cost": 88593750,
                "priority_fee_micro": 1600165,
                "jito_tip_lam": 0,
                "cu_limit": 125000,
            }
            for i in range(3):
                sample["slot"] = 425150000 + i
                w.write(sample)
            r = IntentRingReader(name=w.name)
            r.read_seq = 0
            got = r.poll(100)
            print(f"wrote 3, read {len(got)}")
            for it in got:
                print(f"  slot={it['slot']}  mint={it['mint'][:14]}  cost={it['max_sol_cost_sol']:.4f}")
            assert len(got) == 3
            print("ROUNDTRIP_OK")
        finally:
            w.close()
