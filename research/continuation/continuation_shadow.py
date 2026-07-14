"""Continuation shadow (dry-run, 2026-06-13) — ISOLATED from the trading bot.

Tails the live grpc_capture JSONL (written by pumpfun-grpc-capture.service, no
model/wallet), decodes bonding-curve TradeEvents, runs ContinuationTracker, and logs
every 2x-launch cross + would-be gap-0 fill + realized +50%/-30% outcome to
bot_data/continuation_shadow.jsonl. ZERO submissions, zero coupling to pumpfun-bot.

Purpose: gather LIVE continuation data (features + fills + outcomes) so we can apply
the daily-refit model offline and measure realized top-decile EV across days — the
decisive evidence the capstone backtest could only estimate.
"""
import base64, glob, json, os, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pumpfun_parse import parse_trade_event
from .continuation_tracker import ContinuationTracker

ROOT = "/root/the-distribution-will-manifest"
CAP_GLOB = f"{ROOT}/grpc_capture/capture_*.jsonl"   # active (uncompressed) files only
OUT = f"{ROOT}/bot_data/continuation_shadow.jsonl"


def newest_capture():
    fs = [f for f in glob.glob(CAP_GLOB) if not f.endswith(".gz")]
    return max(fs, key=os.path.getmtime) if fs else None


def main():
    trk = ContinuationTracker()
    cross_mid = {}                      # mint -> cross_mid (to compute fill slip)
    out = open(OUT, "a")
    cur = newest_capture()
    while cur is None:
        time.sleep(2); cur = newest_capture()
    f = open(cur, "r"); f.seek(0, os.SEEK_END)   # start at END = live forward only
    print(f"[cont-shadow] tailing {cur} from EOF", flush=True)
    stats = {"cross": 0, "fill": 0, "win": 0, "loss": 0, "lines": 0}
    last_prune = last_stat = time.time()

    while True:
        line = f.readline()
        if not line:
            time.sleep(0.25)
            now = time.time()
            # rotation: a newer capture file appeared
            nf = newest_capture()
            if nf and nf != cur:
                f.close(); cur = nf; f = open(cur, "r")   # new file: read from start
                print(f"[cont-shadow] rotated -> {cur}", flush=True)
            if now - last_prune > 120:
                trk.prune(now); last_prune = now
            if now - last_stat > 120:
                wr = stats["win"] / max(1, stats["win"] + stats["loss"])
                print(f"[cont-shadow] lines={stats['lines']} crosses={stats['cross']} "
                      f"fills={stats['fill']} outcomes={stats['win']+stats['loss']} "
                      f"winrate={wr:.1%} tracked={len(trk.state)}", flush=True)
                last_stat = now
            continue
        stats["lines"] += 1
        if '"TradeEvent"' not in line:
            continue
        try:
            r = json.loads(line)
            ev = parse_trade_event(base64.b64decode(r["raw"]))
        except Exception:
            continue
        if ev is None or not ev.is_classic_curve or ev.virtual_token_reserves <= 0:
            continue
        ts = r.get("t") or time.time()
        for e in trk.update(ev.mint, ev.virtual_sol_reserves, ev.virtual_token_reserves,
                            ev.is_buy, ts, ev.user):
            e["slot"] = r.get("slot")
            if e["kind"] == "cross":
                cross_mid[e["mint"]] = e["cross_mid"]; stats["cross"] += 1
            elif e["kind"] == "fill":
                cm = cross_mid.get(e["mint"])
                e["slip_vs_cross"] = (e["fill_mid"] / cm - 1.0) if cm else None
                stats["fill"] += 1
            elif e["kind"] == "outcome":
                stats["win" if e["y"] == 1 else "loss"] += 1
                cross_mid.pop(e["mint"], None)
            out.write(json.dumps(e) + "\n"); out.flush()


if __name__ == "__main__":
    main()
