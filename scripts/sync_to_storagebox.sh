#!/bin/bash
# Periodically ship gzipped capture artifacts to the Hetzner storage box.
#
# Strategy:
#   - Only ship files that are at least MIN_AGE_MIN minutes old (so we don't
#     fight the active writer for the current open file).
#   - rsync with --remove-source-files: after a file is successfully copied,
#     it's deleted locally. Frees disk on sol; storage box becomes the
#     canonical archive.
#   - Per-directory: grpc_capture (.jsonl.gz), intent_capture (.jsonl.gz),
#     raw_shred_entries (.bin.gz). The currently-open .jsonl files are
#     ignored because they're younger than MIN_AGE_MIN (mtime is being
#     updated continuously by the recorder).
#
# Idempotent. Safe to run on a timer. Logs to journal via systemd unit.

set -u

USER=${STORAGEBOX_USER:?Set STORAGEBOX_USER}
HOST=${STORAGEBOX_HOST:?Set STORAGEBOX_HOST}
PORT=${STORAGEBOX_PORT:-23}
KEY=${STORAGEBOX_SSH_KEY:-}
REMOTE_BASE=backup
MIN_AGE_MIN=${MIN_AGE_MIN:-30}
BANDWIDTH_KBPS=${BANDWIDTH_KBPS:-0}    # 0 = unlimited; e.g. 10000 = 10 MB/s cap

LOCAL_BASE=/root/the-distribution-will-manifest

ssh_opts="-p $PORT -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
if [ -n "$KEY" ]; then ssh_opts="-i $KEY $ssh_opts"; fi

ship_dir() {
    local local_dir="$1" remote_dir="$2" pattern="$3"
    if [ ! -d "$local_dir" ]; then return 0; fi
    # Find files older than MIN_AGE_MIN minutes matching pattern
    local files
    files=$(find "$local_dir" -maxdepth 1 -type f -name "$pattern" -mmin "+$MIN_AGE_MIN" -print0 | xargs -0 -I{} echo {} 2>/dev/null)
    if [ -z "$files" ]; then
        echo "  [$remote_dir] nothing to ship"
        return 0
    fi
    local n
    n=$(echo "$files" | wc -l)
    echo "  [$remote_dir] $n files >= ${MIN_AGE_MIN}min old"
    local bw=""
    if [ "$BANDWIDTH_KBPS" -gt 0 ]; then bw="--bwlimit=$BANDWIDTH_KBPS"; fi
    # Use rsync's --files-from for explicit list; --remove-source-files
    # cleans up after successful transfer.
    echo "$files" | xargs -I{} basename {} | \
        rsync -av --remove-source-files $bw \
              --files-from=- \
              -e "ssh $ssh_opts" \
              "$local_dir/" \
              "$USER@$HOST:$REMOTE_BASE/$remote_dir/" 2>&1 | \
        tail -10
}

echo "=== sync_to_storagebox @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ship_dir "$LOCAL_BASE/grpc_capture"                grpc_capture       "*.jsonl.gz"
ship_dir "$LOCAL_BASE/shred_bot/intent_capture"    intent_capture     "*.jsonl.gz"
ship_dir "$LOCAL_BASE/shred_bot/raw_shred_entries" raw_shred_entries  "*.bin.gz"
echo "done."
