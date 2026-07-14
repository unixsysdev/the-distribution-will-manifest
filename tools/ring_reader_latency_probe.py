"""Attach as a reader to the pumpfun_intents SHM ring and measure how long
it takes for newly-published intents to become visible to a consumer.

For each record consumed, latency = (time.time_ns() - record.recv_ns)
where recv_ns is the timestamp the SHRED ENTRY was received by
intent_recorder. So this measures the FULL pipeline:

    shred entry received by recorder
      -> parsed in intent_extractor
      -> written to JSONL on local SSD
      -> written to SHM ring
      -> visible to a reader (this probe)
      -> read + observed by this probe

Prints rolling p50/p90/p99 latency every 5s. Doesn't write anything.
Read-only attach. Safe to run alongside the live bot.

Usage:
    ./venv/bin/python tools/ring_reader_latency_probe.py
        [--duration-s 60]   # default: run forever
        [--print-every-s 5]
"""
from __future__ import annotations
import argparse, signal, sys, time
from pathlib import Path

ROOT = Path("/root/the-distribution-will-manifest")
sys.path.insert(0, str(ROOT / "shred_bot"))
from intent_ring import IntentRingReader


_stop = False
def _sig(*_):
    global _stop
    _stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pumpfun_intents")
    ap.add_argument("--duration-s", type=float, default=0,
                    help="how long to run (0 = forever)")
    ap.add_argument("--print-every-s", type=float, default=5.0)
    args = ap.parse_args()

    print(f"=== ring_reader_latency_probe @ "
          f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")
    print(f"  attaching to SHM ring '{args.name}'")

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    reader = IntentRingReader(name=args.name)

    latencies_ns = []   # within current window
    total_n = 0
    t_start = time.time()
    t_print = t_start

    try:
        while not _stop:
            batch = reader.poll(max_records=1000)
            if not batch:
                # No new records — short sleep, don't busy-spin
                time.sleep(0.001)
            else:
                now_ns = time.time_ns()
                for it in batch:
                    recv_ns = it.get("recv_ns", 0)
                    if recv_ns:
                        latencies_ns.append(now_ns - recv_ns)
                        total_n += 1

            now = time.time()
            if now - t_print >= args.print_every_s and latencies_ns:
                arr = sorted(latencies_ns)
                n = len(arr)
                p50 = arr[n//2] / 1e6        # ms
                p90 = arr[int(n*0.9)] / 1e6
                p99 = arr[int(n*0.99) if int(n*0.99) < n else n-1] / 1e6
                mx  = arr[-1] / 1e6
                mn  = arr[0] / 1e6
                rate = n / (now - t_print)
                print(f"  [{now-t_start:5.0f}s] window n={n:5d} ({rate:5.1f}/s)  "
                      f"latency ms: min={mn:.1f} p50={p50:.1f} p90={p90:.1f} "
                      f"p99={p99:.1f} max={mx:.1f}    total_seen={total_n}",
                      flush=True)
                latencies_ns.clear()
                t_print = now

            if args.duration_s > 0 and (now - t_start) >= args.duration_s:
                break
    finally:
        try: reader.close()
        except Exception: pass

    print(f"\n  exit; total intents observed: {total_n}")


if __name__ == "__main__":
    main()
