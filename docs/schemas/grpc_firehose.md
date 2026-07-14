# gRPC firehose data schema

**Last updated:** 2026-06-09
**Producer:** `grpc_firehose.py` (systemd: `pumpfun-grpc-firehose.service`)
**Source:** Yellowstone Geyser gRPC stream (`grpc-fra1-1.erpc.global:80`)
**Filtering:** server-side via SubscribeRequest — `account_include` on the
pump.fun program family (bonding curve + PumpSwap AMM + pump_fees), no
failed-tx filter (captures both success and failure).

## Frame format

Each frame is self-delimited:

```
offset  size  field         description
------  ----  -----         -----------
 0       8    recv_ns       u64 LE  — local time when we received the msg (time.time_ns())
 8       8    slot          u64 LE  — slot from the contained update (tx.slot or block_meta.slot)
16       4    payload_len   u32 LE  — bytes that follow
20       N    payload       serialised geyser_pb2.SubscribeUpdate (oneof)
```

## What's in the payload

The serialised `SubscribeUpdate` is a Protobuf oneof — each frame is exactly
ONE of these variants:

### Variant 1: `transaction` (most frames)
`SubscribeUpdate.transaction` set, containing:
- `SubscribeUpdateTransaction.slot`
- `SubscribeUpdateTransactionInfo.signature`
- `SubscribeUpdateTransactionInfo.transaction` — the Transaction (message + sigs)
- `SubscribeUpdateTransactionInfo.meta` — TransactionStatusMeta (logs,
  pre/post balances, inner_instructions, pre/post_token_balances,
  loaded_writable_addresses, loaded_readonly_addresses,
  compute_units_consumed, fee, err)

### Variant 2: `block_meta` (one per slot, ~400ms apart)
Added 2026-06-09 to carry **validator-witnessed `block_time`** (the gap
the audit flagged as H). `SubscribeUpdate.block_meta` set, containing
`SubscribeUpdateBlockMeta`:
- `slot`
- `blockhash`, `parent_blockhash`
- `parent_slot`
- **`block_time`** — UTC unix timestamp (the canonical chain time for this
  slot; replaces wall-clock `recv_ns` for any cross-slot or cross-source
  time joins)
- `block_height`
- `rewards`
- `executed_transaction_count`, `entries_count`

Roughly 1 block_meta frame per ~125 tx frames at typical pump.fun activity.

## To decode a frame offline

```python
import struct
from geyser_pb2 import SubscribeUpdate

HDR = struct.calcsize("<QQI")
with open("grpc-firehose-YYYYMMDDTHHMMSSZ.bin.gz", "rb") as f_gz:
    import gzip
    with gzip.GzipFile(fileobj=f_gz) as f:
        while True:
            h = f.read(HDR)
            if len(h) < HDR: break
            recv_ns, slot, pl = struct.unpack("<QQI", h)
            payload = f.read(pl)
            msg = SubscribeUpdate()
            msg.ParseFromString(payload)
            if msg.HasField("transaction"):
                ...  # process tx update
            elif msg.HasField("block_meta"):
                block_time = msg.block_meta.block_time.timestamp
                ...  # join to txs by slot to stamp block_time onto each tx
```

## Volume + storage

- Throughput: ~300-400 frames/sec (~3 MB/sec uncompressed)
- Daily volume: ~250 GB uncompressed → ~75 GB gzipped at compresslevel=3
- 5 TB storagebox → ~65-70 days of gzipped runway combined with shred firehose

## Changelog

- **2026-06-09** — added `blocks_meta` subscription. Frames now carry both
  `transaction` and `block_meta` variants. Older frames (pre this date) only
  contain `transaction` variants — block_time is unavailable for them and
  must be reconstructed from `recv_ns` if needed (less canonical but close).
- **2026-06-09 (creation)** — initial firehose with transactions subscription
  for the pump.fun program family.
