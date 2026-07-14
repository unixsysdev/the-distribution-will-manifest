#!/usr/bin/env bash
# pumpfun-archive-sync: copy the NON-regenerable / expensive-to-regenerate
# assets to the Hetzner storagebox. The shipper already moves the raw
# firehose buffers; this covers what it deliberately does not:
#   grpc_capture/*.gz        filtered TradeEvent capture (training source)
#   intent_capture/*.gz      decoded shred intents
#   data/*.parquet           training sets
#   bot_artifacts_*/         model artifacts + specs
#   bot_data/*.jsonl         live decision record (NOT regenerable)
#   repo.bundle              full git history
# Copies only (never deletes), bandwidth-capped and ionice'd so the
# shipper's own uploads and the collectors are never starved.
set -u
ROOT=/root/the-distribution-will-manifest
DST=/mnt/storagebox/backup/archive
BW=25000   # KB/s, below the shipper's 50MB/s cap

mountpoint -q /mnt/storagebox || { echo "[archive-sync] storagebox not mounted; abort"; exit 1; }
mkdir -p "$DST"/{grpc_capture,intent_capture,intent_capture/x2,data,artifacts,bot_data,repo}

RS="ionice -c3 nice -n 15 rsync -t --whole-file --bwlimit=$BW"
RS_MOVE="$RS --remove-source-files"   # MOVE not copy: bot uses the SHM ring, no local intent copy needed; storagebox is the durable store

# closed (gzipped) capture shards only; the active .jsonl is still growing
$RS "$ROOT"/grpc_capture/*.jsonl.gz       "$DST/grpc_capture/" 2>/dev/null
# keep most-recent N grpc_capture locally (training/replay); prune older -- already on the box (copy above + prior runs)
ls -1t "$ROOT"/grpc_capture/*.jsonl.gz 2>/dev/null | tail -n +$((${KEEP_GRPC:-96}+1)) | xargs -r rm -f
$RS_MOVE "$ROOT"/shred_bot/intent_capture/*.jsonl.gz "$DST/intent_capture/" 2>/dev/null
$RS_MOVE "$ROOT"/shred_bot/intent_capture/x2/*.jsonl.gz "$DST/intent_capture/x2/" 2>/dev/null
$RS "$ROOT"/shred_bot/intent_capture/SCHEMA.md  "$DST/intent_capture/" 2>/dev/null
$RS "$ROOT"/data/*.parquet                "$DST/data/" 2>/dev/null
$RS -r "$ROOT"/bot_artifacts_*/           "$DST/artifacts/" --exclude "*.bak*" 2>/dev/null
# append-only logs: point-in-time copy is fine and exactly what we want
$RS "$ROOT"/bot_data/shadow_run.jsonl "$ROOT"/bot_data/positions.jsonl \
    "$ROOT"/bot_data/status.json          "$DST/bot_data/" 2>/dev/null
# broker logs: submissions AND the recon/fill record (actual fills, slot_gap,
# fees, reverts/retries) -- the richest execution-analysis data; was missing.
$RS "$ROOT"/logs/broker_jito.jsonl "$ROOT"/logs/broker_recon.jsonl \
    "$ROOT"/logs/livewatch.jsonl          "$DST/bot_data/" 2>/dev/null
( cd "$ROOT" && git bundle create /tmp/repo.bundle --all -q 2>/dev/null \
    && $RS /tmp/repo.bundle "$DST/repo/" && rm -f /tmp/repo.bundle )

echo "[archive-sync] done $(date -u +%FT%TZ): $(du -sh "$DST" 2>/dev/null | cut -f1) total at destination"
