# Raw shred entries firehose data schema

**Last updated:** 2026-06-09  
**Producer:** `shred_bot/raw_shred_firehose.py` (systemd: `pumpfun-shred-firehose.service`)  
**Source:** Jito Shredstream Proxy gRPC stream (`shreds-fra6-1.erpc.global:80`)  
**Filtering:** **NONE** — every single shred entry message is saved verbatim.
This is the COLD PATH archival recorder; the filtered HOT PATH is the
intent capture (see sibling `intent_capture/SCHEMA.md`).

## Write target

- Primary: `/mnt/storagebox/raw_shred_entries/` (Hetzner BX21 5 TB sshfs mount)
- Fallback (if mount missing at startup):
  `/root/the-distribution-will-manifest/shred_bot/raw_shred_entries/`
  — much smaller disk, will fill fast

## File layout

- `raw-shreds-YYYYMMDDTHHMMSSZ.bin` — open file
- `raw-shreds-YYYYMMDDTHHMMSSZ.bin.gz` — closed + compressed
- Rotates every **3600 seconds** (1 hour, configurable via `ROTATE_SECS` env)
- On rotation, the previous file is gzipped inline (compresslevel=3), then
  the `.bin` is unlinked. The recorder briefly pauses during the gzip; for
  a ~5 GB file on sshfs this is on the order of 30-60 seconds.
- Throughput: ~60-100 frames/sec, ~1.4-1.7 MB/sec = **~5-6 GB/hour uncompressed**
- Gzipped at compresslevel=3: ~1-1.5 GB/hour
- 5 TB storage = ~90-100 days of gzipped runway

## On-disk frame format

Each frame is self-delimited:

```
offset  size  field         description
------  ----  -----         -----------
 0       8    recv_ns       u64 LE  — local time when we received the shred (time.time_ns())
 8       8    slot          u64 LE  — the Solana slot the entry is from (from msg.slot)
16       4    payload_len   u32 LE  — number of bytes that follow
20       N    payload       bincode-encoded `Vec<solana_entry::Entry>`
```

Total framing overhead: 20 bytes per shred message.

## Payload decoding

The `payload` is a **bincode-encoded `Vec<Entry>`** where each `Entry`
contains a vector of `VersionedTransaction` in **Solana wire format**
(NOT bincode for the txs themselves — it's a hybrid serialization).

Use the same decode chain as `shred_bot/intent_extractor.py`:

```python
# pseudo-code; see intent_extractor.parse_entries + parse_vt
def decode_payload(payload: bytes) -> list[Entry]:
    # bincode Vec<Entry> = u64 LE length, then that many Entry records.
    # Each Entry = num_hashes(u64) + hash(32) + Vec<VersionedTransaction>(wire)
    ...
```

A worked decoder lives in `shred_bot/intent_extractor.py:parse_entries()` —
that function is the canonical reference. It handles:
- The bincode Vec<Entry> outer wrapper
- The Entry tick/data discrimination (no separate marker — distinguished by
  whether `tx_count > 0`)
- The hybrid wire-format VersionedTransaction (Legacy + V0)
- ShortVec (Solana compact-u16) for array lengths inside the tx
- Message header → writability rules

## Why this exists separately from intent_capture

| | intent_capture (HOT) | raw firehose (COLD) |
|---|---|---|
| Filter | pump.fun program touches | none — everything |
| Format | JSONL (one record per pump.fun ix) | binary frames (one frame per shred msg) |
| Volume | ~5-20 MB/h (~3-15 records/s) | ~5-6 GB/h (~60-100 frames/s) |
| Write target | local disk (low latency critical) | Hetzner sshfs mount (latency tolerant) |
| Purpose | bot's front-run signal source | exhaustive replay archive for any future feature we don't anticipate today |

If we ever realize there's a signal we missed (e.g. ATA creates that
predict graduation, or non-pump.fun whale moves correlating with rugs),
the raw firehose lets us go back and extract it without having to wait
for fresh data.

## Replay tools (planned, not yet built)

- `shred_bot/replay_firehose.py` — stream raw frames back through the
  intent_extractor for offline JSONL regeneration with any extractor
  version (e.g. retroactive v2-fix replay over historical data)
- `shred_bot/firehose_to_parquet.py` — dump all pump.fun txs to parquet
  for offline ML feature extraction

## Changelog

- **2026-06-09** — initial deployment + schema doc
- **2026-06-09** — discovered that sshfs mount target is 5 TB not 1 TB —
  runway revised from ~20 days to ~90-100 days
