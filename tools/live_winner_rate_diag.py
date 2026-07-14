"""Live winner-rate diagnostic for ready mints.

For every entry_decision row in bot_data/shadow_run.jsonl, look up that mint's
forward trade history in the grpc_capture archive and compute peak_ret =
max(vsol/vtok forward) / midK - 1. A "winner" is peak_ret >= 1.0 (2x).

Compare live winner rate to training's 28% peak>=2x base rate. This is the
single most important calibration question: are we in a phase where the model
correctly refuses low-quality opportunities (live winners ~ 1-2%), or is the
model missing real winners (live winners ~ 25-30%)?
"""
from __future__ import annotations
import json
import gzip
import glob
from pathlib import Path
from collections import defaultdict

BOT_DATA = Path("/root/the-distribution-will-manifest/bot_data/shadow_run.jsonl")
CAPTURE_DIR = Path("/root/the-distribution-will-manifest/grpc_capture")


def main():
    # 1) collect ready mints + their midK from the bot's log
    ready_by_mint = {}   # mint -> (midK, ready_ev_ts)
    with open(BOT_DATA) as f:
        for ln in f:
            try: rec = json.loads(ln)
            except: continue
            if rec.get("kind") != "entry_decision": continue
            mint = rec["mint"]
            midK = rec.get("midK")
            ev_ts_at_ready = rec.get("k_window_last_ts") or rec.get("v_window_last_ts")
            if midK and ev_ts_at_ready and mint not in ready_by_mint:
                ready_by_mint[mint] = (float(midK), int(ev_ts_at_ready), float(rec.get("score", 0)))
    print(f"unique ready mints in bot log: {len(ready_by_mint)}")
    if not ready_by_mint:
        print("nothing to analyze"); return

    # 2) sweep gRPC capture files; collect post-trigger trades per mint
    fwd_trades = defaultdict(list)   # mint -> list of (vsol, vtok, ev_ts)
    files = sorted(glob.glob(str(CAPTURE_DIR / "*.jsonl*")))
    print(f"scanning {len(files)} capture file(s) ...")
    for path in files:
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt") as f:
                for ln in f:
                    try: rec = json.loads(ln)
                    except: continue
                    mint = rec.get("mint")
                    if mint not in ready_by_mint: continue
                    midK, ready_ts, _ = ready_by_mint[mint]
                    if rec.get("ev_ts", 0) < ready_ts: continue   # pre-trigger; skip
                    vsol = rec.get("vsol", 0); vtok = rec.get("vtok", 0)
                    if vsol <= 0 or vtok <= 0: continue
                    fwd_trades[mint].append((vsol, vtok, rec.get("ev_ts", 0)))
        except Exception as e:
            print(f"  err on {path}: {e}")
    n_with_fwd = sum(1 for m in ready_by_mint if fwd_trades.get(m))
    print(f"ready mints with >=1 forward trade in capture: {n_with_fwd} / {len(ready_by_mint)}")

    # 3) compute peak_ret per mint
    n_winner = 0; n_total = 0; peak_rets = []
    n_5x = 0; n_10x = 0
    score_winner = []; score_loser = []
    n_fwd_lt_5 = 0
    for mint, (midK, ready_ts, score) in ready_by_mint.items():
        trades = fwd_trades.get(mint, [])
        if len(trades) < 5:
            n_fwd_lt_5 += 1
            continue
        peak = max(vs/vt for vs, vt, _ in trades)
        peak_ret = peak / midK - 1.0
        peak_rets.append(peak_ret)
        n_total += 1
        if peak_ret >= 1.0:
            n_winner += 1; score_winner.append(score)
        else:
            score_loser.append(score)
        if peak_ret >= 4.0:  n_5x += 1
        if peak_ret >= 9.0:  n_10x += 1

    print(f"\nready mints with < 5 forward trades (skipped): {n_fwd_lt_5}")
    print(f"ready mints analyzed: {n_total}")
    if n_total == 0:
        print("not enough forward data yet (capture must run longer)")
        return
    print(f"\n  LIVE WINNER RATE (peak_ret >= 2x): {n_winner}/{n_total} = {100*n_winner/n_total:.1f}%")
    print(f"     5x rate: {n_5x}/{n_total} = {100*n_5x/n_total:.1f}%")
    print(f"    10x rate: {n_10x}/{n_total} = {100*n_10x/n_total:.1f}%")
    print(f"\n  TRAINING reference: 28% peak>=2x (fresh-rsol filtered)")
    print(f"  delta: live - training = {100*n_winner/n_total - 28:+.1f}pp")
    # score distribution by outcome
    import statistics as s
    if score_winner:
        print(f"\n  WINNER score distribution: n={len(score_winner)} "
              f"mean={s.mean(score_winner):.4f} median={s.median(score_winner):.4f} max={max(score_winner):.4f}")
    if score_loser:
        print(f"  LOSER score distribution:  n={len(score_loser)} "
              f"mean={s.mean(score_loser):.4f} median={s.median(score_loser):.4f} max={max(score_loser):.4f}")
    # If winners exist, what threshold would have caught them?
    if score_winner:
        wmin = min(score_winner)
        print(f"\n  min winner score = {wmin:.4f}  (firing at <= this would catch ALL "
              f"{len(score_winner)} winners but also fire on "
              f"{sum(1 for x in score_loser if x >= wmin)} losers)")


if __name__ == "__main__":
    main()
