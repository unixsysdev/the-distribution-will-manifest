#!/usr/bin/env python3
"""live_bucket_diag.py — the honest live test for the currently deployed entry model.

Reads shadow_run.jsonl for the CURRENT bot era (auto-detected from systemd start,
override with --since), joins entry decisions to position closes, and prints:
  - live score distribution + fire rate vs the deployed spec expectation
  - realized outcomes per fire (book net + configured-exit-policy net)
  - realized win-rate-by-SCORE-BUCKET (the only honest precision test)
Run it any time:  ./venv/bin/python tools/live_bucket_diag.py
Rules (SHADOW_HARNESS_LOG GOTCHAS): if the fire rate collapses vs expectation or
buckets are not monotonic at n>=50 fires, that is a population-mismatch signal.
Do NOT recalibrate the threshold; find the mismatch.
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def era_start() -> float:
    try:
        out = subprocess.check_output(
            ["systemctl", "show", "pumpfun-bot", "--property=ActiveEnterTimestamp"],
            text=True,
        )
        stamp = out.split("=", 1)[1].strip()
        from datetime import datetime

        return datetime.strptime(stamp, "%a %Y-%m-%d %H:%M:%S %Z").timestamp()
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=0.0, help="unix ts era cut (default: bot service start)")
    ap.add_argument("--data-dir", default=str(ROOT / "bot_data"))
    ap.add_argument("--artifact-dir", default=str(ROOT / "bot_artifacts_K7V"))
    args = ap.parse_args()

    cut = args.since or era_start()
    spec = {}
    spec_path = Path(args.artifact_dir) / "model_spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
    thr = float(spec.get("entry", {}).get("threshold", 0.0)) or None
    hold = spec.get("holdout_result", {})

    decs, closes = [], []
    with open(Path(args.data_dir) / "shadow_run.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if (r.get("t") or 0) < cut:
                continue
            k = r.get("kind")
            if k == "entry_decision":
                decs.append(r)
            elif k == "position_close":
                closes.append(r)

    if not decs:
        print(f"no entry_decision rows since {cut}; is the era cut right?")
        return
    scores = sorted(float(d["score"]) for d in decs)
    n = len(scores)

    def pct(q):
        return scores[min(n - 1, int(q * n))]

    print(f"era since t={cut:.0f}  decisions={n}")
    print(
        f"score: p50={pct(.5):.4f} p90={pct(.9):.4f} p95={pct(.95):.4f} "
        f"p99={pct(.99):.4f} max={scores[-1]:.4f}"
    )
    if thr:
        fr = sum(1 for s in scores if s >= thr) / n
        parts = []
        for w in ("val", "test"):
            v = hold.get(w + "_fire_rate")
            if v is not None:
                parts.append(f"{v:.2%}")
        exp = " / ".join(parts) or "n/a"
        print(f"fire rate: live {fr:.2%} @ thr {thr:.4f}   (spec val/test: {exp})")

    by_mint = {}
    for d in decs:
        if thr and float(d["score"]) >= thr:
            by_mint.setdefault(d["mint"], float(d["score"]))
    closed = [(by_mint.get(c["mint"]), c) for c in closes if c.get("mint") in by_mint]
    print(f"\nfires={len(by_mint)}  closed={len(closes)} (joined {len(closed)})")
    if closed:
        nets = [float(c.get("net", 0.0)) for _, c in closed]
        pol = [float(c.get("live_policy_net", c.get("net", 0.0))) for _, c in closed]
        wins = sum(1 for x in pol if x > 0)
        print(
            f"book net/fire: {sum(nets)/len(nets):+.4f}   "
            f"policy({spec.get('exit_policy','?')}) net/fire: {sum(pol)/len(pol):+.4f}   "
            f"win {wins}/{len(pol)}"
        )
        print("\nscore  policy_net  reason")
        for s, c in sorted(closed, key=lambda x: -(x[0] or 0)):
            print(f"{s:.3f}  {float(c.get('live_policy_net', 0)):+10.3f}  {c.get('reason','?')}")
        buckets = {}
        for s, c in closed:
            b = min(int(s * 10) / 10, 0.9)
            buckets.setdefault(b, []).append(float(c.get("live_policy_net", 0)))
        # Pattern dedup: launch farms replay byte-identical launch scripts,
        # which produce byte-identical feature vectors and therefore identical
        # scores. n fires of one pattern are ~1 independent observation, not n
        # (seen live 2026-06-10: four mints, 51 min, identical score/cum_buy/vsK,
        # one farm). Group by exact score as the pattern key.
        pats = {}
        for s_, c in closed:
            pats.setdefault(round(s_ or 0, 6), []).append(float(c.get("live_policy_net", 0)))
        n_pat = len(pats)
        print(f"\npattern dedup: {len(closed)} closed fires across {n_pat} distinct patterns")
        if n_pat < len(closed):
            for k in sorted(pats, reverse=True):
                v = pats[k]
                if len(v) > 1:
                    print(f"  REPEATED pattern score={k:.6f}: n={len(v)} mean={sum(v)/len(v):+.3f}"
                          f"  <- treat as ~1 independent obs (launch-farm replay)")
            dd = [sum(v)/len(v) for v in pats.values()]
            print(f"  deduped mean net/pattern: {sum(dd)/len(dd):+.4f}  (n_pat={n_pat})")

        # Shadow exit comparison (when policy_nets logged): the live deduped
        # tp_50 vs tp_100 vs tp_200 race that decides the exit-policy swap.
        pn_rows = [(s_, c.get("policy_nets")) for s_, c in closed
                   if isinstance(c.get("policy_nets"), dict)]
        if pn_rows:
            print(f"\nshadow exit policies (counterfactual, n={len(pn_rows)} closes):")
            for pname in ("level_tp_50", "level_tp_100", "level_tp_200", "level_tp_100_t120", "level_tp_50_stop30_cap120", "level_tp_100_stop30_cap120"):
                vals = [(s_, d_.get(pname)) for s_, d_ in pn_rows
                        if d_.get(pname) is not None]
                if not vals:
                    continue
                by = {}
                for s_, v in vals:
                    by.setdefault(round(s_ or 0, 6), []).append(float(v))
                pm = [sum(v) / len(v) for v in by.values()]
                raw = sum(v for _, v in vals) / len(vals)
                print(f"  {pname:14s} raw={raw:+.3f}  DEDUPED={sum(pm)/len(pm):+.3f} "
                      f"(n={len(vals)}, n_pat={len(by)})")

        print("\nrealized by score bucket (policy net):")
        for b in sorted(buckets):
            v = buckets[b]
            wr = sum(1 for x in v if x > 0) / len(v)
            print(f"  {b:.1f}-{b+0.1:.1f}  n={len(v):3d}  win={wr:5.1%}  mean={sum(v)/len(v):+.3f}")
        if len(closed) >= 50:
            ws = [
                sum(1 for x in buckets[b] if x > 0) / len(buckets[b])
                for b in sorted(buckets)
                if len(buckets[b]) >= 5
            ]
            if ws and any(ws[i] > ws[i + 1] + 0.15 for i in range(len(ws) - 1)):
                print(
                    "\n!! bucket monotonicity BROKEN at n>=50 — population-mismatch "
                    "signal, do not recalibrate"
                )
    else:
        print("(no closed fires joined yet)")


if __name__ == "__main__":
    main()
