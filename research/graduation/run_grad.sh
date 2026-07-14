#!/bin/bash
# Detached graduation-panel extraction. Args: N_newest_local_gz_files OUT_panel [extra extractor args...]
cd /root/the-distribution-will-manifest || exit 1
N=${1:-24}; OUT=${2:-bot_data/grad_cont_panel.jsonl}; shift 2 2>/dev/null
files=$(ls -1 grpc_capture/*.jsonl.gz | sort | tail -"$N")
echo "[run_grad] $(date -u +%FT%TZ) N=$N OUT=$OUT files=$(echo "$files" | wc -l)"
(for f in $files; do zcat "$f"; done) \
  | grep -F -e PumpSwap.BuyEvent -e PumpSwap.SellEvent -e CreatePoolEvent -e TradeEvent \
  | nice -n 15 ionice -c3 ./venv/bin/python -u -m research.graduation.grad_cont_extract --stdin --out "$OUT" "$@"
echo "[run_grad] DONE $(date -u +%FT%TZ) panel_lines=$(wc -l < "$OUT")"
