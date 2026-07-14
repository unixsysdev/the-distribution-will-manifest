"""Partial-history diagnostic.

For each `ready` mint in bot_data/shadow_run.jsonl, find ALL its trades in the
grpc_capture archive (= ground truth) and replay them through a fresh TokenState.
Compare:
  - bot-logged midK/vsK/vtK vs capture-replayed midK/vsK/vtK
  - bot-logged midV/vsV/vtV vs capture-replayed midV/vsV/vtV
  - bot-logged n_at_ready vs capture-replayed n_at_ready
  - count of trades the capture saw BEFORE the bot's first observation

If the bot's first observation was trade #4 (capture saw 3 earlier trades), then
the bot's K=7 fired on the actual 10th on-chain trade, not the 7th. Features
would diverge. Repeat the test for many mints; if the gap is systemic, tighten
the fresh_rsol filter.
"""
from __future__ import annotations
import json, gzip, glob, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_accum import TokenState

BOT_LOG = Path("/root/the-distribution-will-manifest/bot_data/shadow_run.jsonl")
CAPTURE = Path("/root/the-distribution-will-manifest/grpc_capture")


def main():
    # collect bot's ready events
    bot_ready = {}
    bot_observed_first_seen_rsol = {}
    with open(BOT_LOG) as f:
        for ln in f:
            try: rec = json.loads(ln)
            except: continue
            if rec.get("kind") == "entry_decision":
                m = rec["mint"]
                if m not in bot_ready:
                    bot_ready[m] = rec
    print(f"ready mints in bot log: {len(bot_ready)}")
    if not bot_ready: return

    # collect all trades per mint from capture
    trades_by_mint = defaultdict(list)
    files = sorted(glob.glob(str(CAPTURE / "*.jsonl*")))
    print(f"scanning {len(files)} capture file(s) ...")
    for path in files:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            for ln in f:
                try: rec = json.loads(ln)
                except: continue
                m = rec.get("mint")
                if m in bot_ready:
                    trades_by_mint[m].append(rec)

    # for each, sort by (ev_ts, slot), replay through TokenState until ready
    # capture replay = ground truth. Compare to bot-logged.
    results = []
    for mint, br in bot_ready.items():
        trades = trades_by_mint.get(mint, [])
        if len(trades) < 7: continue   # not enough capture data for this mint
        trades.sort(key=lambda r: (r.get("ev_ts", 0), r.get("slot", 0)))
        # replay
        st = None; n_pre_obs = None; cap_midK = cap_midV = None
        cap_vsK = cap_vtK = cap_vsV = cap_vtV = 0; cap_n_at_ready = None
        for i, r in enumerate(trades):
            vsol = r["vsol"]; vtok = r["vtok"]
            if vsol <= 0 or vtok <= 0: continue
            sol = r["sol"] / 1e9
            is_buy = bool(r["is_buy"]); user = r["user"]; ts = r["ev_ts"]
            if st is None:
                st = TokenState(vsol, vtok, sol, is_buy, user, ts)
                continue
            result = st.update(vsol, vtok, sol, is_buy, user, ts)
            if result == "ready":
                cap_midK = st.midK; cap_vsK = st.vsK; cap_vtK = st.vtK
                cap_midV = st.midV; cap_vsV = st.vsV; cap_vtV = st.vtV
                cap_n_at_ready = st.n
                break
        if cap_n_at_ready is None: continue
        # compare
        bot_midK = br.get("midK"); bot_vsK = br.get("vsK"); bot_vtK = br.get("vtK")
        bot_n = br.get("n_at_ready"); bot_first_rsol = br.get("first_seen_rsol", None)
        # estimate n_pre_obs: bot saw n_at_ready trades total; capture saw N=len(trades)
        # before bot's "first observation", capture had K trades where K = n_pre_obs
        # bot's first observed trade = capture's trade #(n_pre_obs); after that bot saw
        # bot_n trades to reach ready (where bot's "ready" = cap_n_at_ready - n_pre_obs)
        # so n_pre_obs ~ cap_n_at_ready - bot_n
        n_pre = cap_n_at_ready - bot_n if (cap_n_at_ready and bot_n) else None
        results.append({
            "mint": mint[:14],
            "bot_n": bot_n,
            "cap_n": cap_n_at_ready,
            "n_pre_obs": n_pre,
            "midK_match": abs((bot_midK or 0) - cap_midK) < 1e-12 if bot_midK else False,
            "vsK_match": int(bot_vsK or 0) == int(cap_vsK),
            "vtK_match": int(bot_vtK or 0) == int(cap_vtK),
            "bot_midK": bot_midK, "cap_midK": cap_midK,
            "first_rsol": bot_first_rsol,
        })

    if not results:
        print("not enough overlap between bot log and capture archive (capture started "
              "after most ready events). Re-run after more capture accumulates.")
        return
    print(f"\nanalyzed {len(results)} mints with sufficient capture data\n")
    print(f"{'mint':16s} {'bot_n':>5s} {'cap_n':>5s} {'n_pre':>5s} {'midK_ok':>7s} {'vsK_ok':>6s} {'vtK_ok':>6s} {'first_rsol':>12s}")
    for r in results[:25]:
        print(f"{r['mint']:16s} {r['bot_n']:>5} {r['cap_n']:>5} {str(r['n_pre_obs']):>5} "
              f"{str(r['midK_match']):>7s} {str(r['vsK_match']):>6s} {str(r['vtK_match']):>6s} "
              f"{str(r['first_rsol']):>12s}")

    n_total = len(results)
    n_pre_obs_zero = sum(1 for r in results if r['n_pre_obs'] == 0)
    n_pre_obs_positive = sum(1 for r in results if r['n_pre_obs'] and r['n_pre_obs'] > 0)
    n_pre_obs_negative = sum(1 for r in results if r['n_pre_obs'] and r['n_pre_obs'] < 0)
    n_midK_ok = sum(1 for r in results if r['midK_match'])
    n_vsK_ok = sum(1 for r in results if r['vsK_match'])
    n_vtK_ok = sum(1 for r in results if r['vtK_match'])
    pre_obs_dist = sorted(r['n_pre_obs'] for r in results if r['n_pre_obs'] is not None)
    print(f"\n=== SUMMARY ({n_total} mints) ===")
    print(f"  n_pre_obs == 0 (bot saw all trades):  {n_pre_obs_zero} ({100*n_pre_obs_zero/n_total:.0f}%)")
    print(f"  n_pre_obs > 0  (bot missed trades):   {n_pre_obs_positive} ({100*n_pre_obs_positive/n_total:.0f}%)")
    print(f"  n_pre_obs < 0  (bot saw EXTRA somehow): {n_pre_obs_negative}")
    if pre_obs_dist:
        def pct(p): return pre_obs_dist[int(len(pre_obs_dist)*p/100)]
        print(f"  n_pre_obs distribution: p25={pct(25)} p50={pct(50)} p75={pct(75)} p90={pct(90)} max={pre_obs_dist[-1]}")
    print(f"\n  midK exact match: {n_midK_ok}/{n_total} ({100*n_midK_ok/n_total:.0f}%)")
    print(f"  vsK exact match:  {n_vsK_ok}/{n_total} ({100*n_vsK_ok/n_total:.0f}%)")
    print(f"  vtK exact match:  {n_vtK_ok}/{n_total} ({100*n_vtK_ok/n_total:.0f}%)")
    if n_midK_ok == n_total:
        print("\nVERDICT: NO partial-history issue. bot and capture replay agree on reserves.")
        print("   Calibration drift must come from elsewhere (regime / model overfit / WS drops).")
    elif n_pre_obs_positive > n_total * 0.5:
        print("\nVERDICT: CONFIRMED partial-history issue. Bot missed pre-observation trades on")
        print(f"   {n_pre_obs_positive}/{n_total} mints. Tightening fresh_rsol filter is justified.")


if __name__ == "__main__":
    main()
