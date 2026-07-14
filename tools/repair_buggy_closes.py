"""Retroactively compute the real P&L for closes corrupted by the
snap_every=1 modulo bug.

The bug (fixed in commit f49dfc9): for every entry fired while
`snap_every=1` was in config, `pos.vsC` / `pos.vtC` stayed frozen at
entry-time values. When the stale watchdog eventually closed the
position, the broker sold AT THE ENTRY PRICE — so the recorded
exit_ret was 0.0 and net was just the entry-fee cost (-0.0058 SOL).

What we have available offline that the live bot didn't:
  - The OPEN event (vsK, vtK at K=7 trigger) per mint
  - The CLOSE timestamp per mint (from positions.jsonl)
  - The REAL post-trigger trade events for that mint (from
    grpc_capture/*.jsonl.gz — what the bot SHOULD have been seeing
    but its snap-update path was disabled)

So we can compute for each buggy close:
  1) tokens_bought  = constant-product buy with bet_sol against (vsK, vtK)
  2) Find the last grpc_capture event for this mint at-or-before the
     close timestamp -> (vsol_close, vtok_close)
  3) sol_proceeds_at_close = constant-product sell with tokens_bought
                              against (vsol_close, vtok_close)
  4) net_real     = sol_proceeds_at_close - bet_sol - fees
  5) exit_ret_real = (vsol_close/vtok_close) / (vsK/vtK) - 1

Output:
  - logs/repaired_closes.jsonl  one repaired record per buggy close,
                                 with both original and recomputed fields
  - summary printed to stdout: mean/p25/p50/p75/win-rate, before vs after

DOES NOT MODIFY positions.jsonl.  Read-only on capture data.
"""
from __future__ import annotations
import argparse, gzip, json, time
from pathlib import Path
from collections import defaultdict
import statistics

ROOT = Path("/root/the-distribution-will-manifest")


def _cp_buy(vs: float, vt: float, sol_in: float) -> float:
    """Constant-product: tokens out when you put `sol_in` SOL into the pool."""
    k = vs * vt
    new_vs = vs + sol_in
    new_vt = k / new_vs
    return vt - new_vt


def _cp_sell(vs: float, vt: float, tok_in: float) -> float:
    """Constant-product: SOL out when you dump `tok_in` tokens into the pool."""
    k = vs * vt
    new_vt = vt + tok_in
    new_vs = k / new_vt
    return vs - new_vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="bot_data/positions.jsonl")
    ap.add_argument("--shadow",    default="bot_data/shadow_run.jsonl")
    ap.add_argument("--capture-dir", default="grpc_capture")
    ap.add_argument("--out", default="logs/repaired_closes.jsonl")
    ap.add_argument("--bet-sol", type=float, default=0.1,
                    help="bet size used per entry (config.yaml bot.bet_sol)")
    ap.add_argument("--fees-sol", type=float, default=0.0058,
                    help="round-trip fees + tip in SOL (entry fee + sell fee + jito tip + slippage)")
    args = ap.parse_args()

    pos_path = ROOT / args.positions
    sr_path  = ROOT / args.shadow
    cap_dir  = ROOT / args.capture_dir
    out_path = ROOT / args.out

    print(f"=== repair_buggy_closes @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")

    # 1. Load positions.jsonl - opens + closes
    opens = {}    # mint -> open record
    closes = []   # list of close records (in order)
    with open(pos_path) as f:
        for ln in f:
            try: r = json.loads(ln)
            except Exception: continue
            k = r.get("kind"); m = r.get("mint")
            if not m: continue
            if k == "open":
                opens[m] = r
            elif k == "close":
                closes.append(r)
    print(f"  loaded {len(opens):,} opens / {len(closes):,} closes from positions.jsonl")

    # 2. Filter to "buggy" closes: reason=stale, exit_ret==0, kind=hold
    buggy = []
    for c in closes:
        if c.get("reason") != "stale": continue
        if c.get("exit_kind") != "hold": continue
        er = c.get("exit_ret")
        if er is None: continue
        if abs(er) > 1e-9: continue
        if c.get("mint") not in opens: continue
        buggy.append(c)
    print(f"  buggy candidates (reason=stale, exit_ret=0): {len(buggy)}")

    # 3. Some opens may be missing the K-anchored reserves (vsK, vtK).
    # Look those up from shadow_run.jsonl entry_decision events by mint+t window.
    need_vsk = {}  # mint -> open record needing fill
    for c in buggy:
        o = opens[c["mint"]]
        if "vsK" in o and "vtK" in o:
            continue
        need_vsk[c["mint"]] = o
    if need_vsk:
        print(f"  resolving vsK/vtK for {len(need_vsk)} opens from shadow_run.jsonl ...")
        # Build entry_decision lookup keyed by mint
        latest_entry = {}
        with open(sr_path) as f:
            for ln in f:
                if '"kind": "entry_decision"' not in ln: continue
                if '"fire": true' not in ln: continue
                try: r = json.loads(ln)
                except Exception: continue
                m = r.get("mint")
                if m in need_vsk:
                    latest_entry[m] = r   # keep last
        for m, o in need_vsk.items():
            e = latest_entry.get(m)
            if e:
                o["vsK"] = e.get("vsK"); o["vtK"] = e.get("vtK")
        print(f"    filled vsK/vtK on {sum(1 for o in need_vsk.values() if 'vsK' in o)} opens")

    # 4. Build a per-mint timeline of grpc_capture (vsol, vtok, t) events.
    # We only need it for the buggy mints — saves a lot of memory.
    buggy_mints = {c["mint"] for c in buggy}
    print(f"  scanning {sum(1 for _ in cap_dir.glob('*.jsonl*'))} capture files for "
          f"{len(buggy_mints)} mints ...")
    per_mint = defaultdict(list)
    n_evt = 0
    for path in sorted(cap_dir.glob("*.jsonl*")):
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            with opener(path, "rt") as f:
                for ln in f:
                    n_evt += 1
                    if n_evt % 1_000_000 == 0:
                        print(f"    .. {n_evt:,} events scanned")
                    # Cheap pre-filter: must contain one of the buggy mints
                    if '"mint"' not in ln: continue
                    try: r = json.loads(ln)
                    except Exception: continue
                    m = r.get("mint")
                    if m in buggy_mints:
                        vs = r.get("vsol") if r.get("vsol") is not None else r.get("vs")
                        vt = r.get("vtok") if r.get("vtok") is not None else r.get("vt")
                        t  = r.get("t", 0)
                        if vs and vt and t:
                            per_mint[m].append((t, float(vs), float(vt)))
        except Exception as e:
            print(f"  warn: {path}: {e}")
    print(f"  captured events for {len(per_mint)} of {len(buggy_mints)} buggy mints")
    # Sort each mint's timeline by t
    for m in per_mint: per_mint[m].sort()

    # 5. For each buggy close, compute the corrected exit_ret + net.
    def _last_at_or_before(timeline, t_close):
        """Binary-search-ish: return the (t, vs, vt) at-or-before t_close,
        or None if there's no such event."""
        prev = None
        for ev in timeline:
            if ev[0] > t_close:
                break
            prev = ev
        return prev

    repaired = []
    no_data = 0
    no_vsk = 0
    bet_lam = args.bet_sol * 1e9
    for c in buggy:
        o = opens[c["mint"]]
        vsK = o.get("vsK"); vtK = o.get("vtK")
        if vsK is None or vtK is None:
            no_vsk += 1
            continue
        tl = per_mint.get(c["mint"], [])
        last = _last_at_or_before(tl, c.get("t", time.time()))
        if last is None:
            no_data += 1
            continue
        _, vs_close, vt_close = last
        # Compute realized P&L from constant-product math
        try:
            tokens_bought = _cp_buy(float(vsK), float(vtK), bet_lam)
            sol_out_lam   = _cp_sell(vs_close, vt_close, tokens_bought)
        except Exception:
            no_data += 1
            continue
        gross_sol = (sol_out_lam - bet_lam) / 1e9
        net_real  = gross_sol - args.fees_sol
        exit_ret_real = (vs_close / vt_close) / (vsK / vtK) - 1.0
        rec = {
            "mint": c["mint"],
            "t_open":  o.get("t"),
            "t_close": c.get("t"),
            "duration_s": (c.get("t", 0) - o.get("t", 0)),
            "vsK": vsK, "vtK": vtK,
            "vs_close": vs_close, "vt_close": vt_close,
            "original": {
                "exit_ret": c.get("exit_ret"),
                "net":      c.get("net_return"),
                "kind":     c.get("exit_kind"),
                "reason":   c.get("reason"),
            },
            "repaired": {
                "exit_ret": exit_ret_real,
                "gross_sol": gross_sol,
                "net_sol":   net_real,
                "tokens_bought": tokens_bought,
                "fees_assumed_sol": args.fees_sol,
                "bet_sol": args.bet_sol,
            },
        }
        repaired.append(rec)

    # 6. Write + summarise
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in repaired:
            f.write(json.dumps(r) + "\n")
    print(f"\n  wrote {out_path} ({len(repaired)} repaired records)")
    print(f"  no_vsk={no_vsk}  no_capture_data={no_data}")

    if not repaired:
        return

    # Summary stats: ORIGINAL (broken) vs REPAIRED
    orig_net = [r["original"]["net"] for r in repaired if r["original"]["net"] is not None]
    rep_net  = [r["repaired"]["net_sol"] for r in repaired]
    rep_ret  = [r["repaired"]["exit_ret"] for r in repaired]
    rep_dur  = [r["duration_s"] for r in repaired]

    def _qstats(label, arr):
        if not arr:
            print(f"  {label}: n=0"); return
        q = sorted(arr)
        n = len(q)
        p25 = q[int(n*0.25)]
        p50 = q[n//2]
        p75 = q[int(n*0.75)]
        mean = sum(q)/n
        wins = sum(1 for x in q if x > 0)
        print(f"  {label}: n={n}  mean={mean:+.4f}  p25={p25:+.4f}  p50={p50:+.4f}  p75={p75:+.4f}  win%={wins*100/n:.0f}")

    print(f"\n=== ORIGINAL (broken-sale-at-entry-price) ===")
    _qstats("net_sol", orig_net)
    print(f"\n=== REPAIRED (sale at actual market price at close time) ===")
    _qstats("net_sol", rep_net)
    _qstats("exit_ret", rep_ret)
    _qstats("duration_s", rep_dur)

    # Big winners / big losers
    repaired.sort(key=lambda r: -r["repaired"]["net_sol"])
    print(f"\n=== top 5 winners (repaired net_sol) ===")
    for r in repaired[:5]:
        print(f"  {r['mint'][:16]}..  exit_ret={r['repaired']['exit_ret']:+.3f}  net={r['repaired']['net_sol']:+.4f}  dur={r['duration_s']:.0f}s")
    print(f"\n=== top 5 losers (repaired net_sol) ===")
    for r in repaired[-5:]:
        print(f"  {r['mint'][:16]}..  exit_ret={r['repaired']['exit_ret']:+.3f}  net={r['repaired']['net_sol']:+.4f}  dur={r['duration_s']:.0f}s")

    print(f"\n=== AGGREGATE delta vs ORIGINAL ===")
    if orig_net:
        delta = sum(rep_net) - sum(orig_net)
        print(f"  original total: {sum(orig_net):+.4f} SOL  (n={len(orig_net)})")
        print(f"  repaired total: {sum(rep_net):+.4f} SOL  (n={len(rep_net)})")
        print(f"  delta:           {delta:+.4f} SOL")


if __name__ == "__main__":
    main()
