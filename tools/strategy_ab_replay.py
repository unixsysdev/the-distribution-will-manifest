"""Multi-policy A/B replay: re-run each fire through several exit policies.

Reads bot_data/shadow_run.jsonl for actual entry_decision fires + grpc_capture
for the forward trade history of each fired mint. Replays each fire through:
  A. K_combined 4+4 g15s         (current live policy)
  H. H_time_spaced 15s           (highest-mean from OOS sweep)
  F. F_hybrid 4+4 t50            (looser trailing stop)
  C_orig. C_hybrid 4+4 t30       (previous live policy)
  B_frontload                    (original wiring)
For each policy, computes per-fire net P&L using the SAME PaperBook AMM math.
Prints side-by-side comparison. Empirical evidence beats backtest projections.

READ-ONLY. Does not touch the running bot.

Death-cut layer is INTENTIONALLY excluded from this replay because it depends on
the recovery model + path features which add complexity without changing the
relative ranking between exit policies. (Death-cut affects all policies equally
since they all share the same death-cut threshold.) The absolute numbers will
be slightly more optimistic than reality; the rankings are comparable to the
OOS sweep.
"""
from __future__ import annotations
import argparse, gzip, glob, json, sys
from collections import defaultdict
from pathlib import Path
import statistics as st

DEFAULT_ROOT = Path("/root/the-distribution-will-manifest")
COST_BPS = 250.0
FEE_PER_TX_SOL = 0.0015        # PLAUSIBLE scenario
ENTRY_LAT_SNAPS = 1            # PLAUSIBLE: enter at the next snap after trigger
TOTAL_SLICES = 8
TIP_LAM = 100_000
SNAP_EVERY = 3                  # bot's snap cadence in forward trades


def buy_tokens(vs, vt, d): return vt - (vs * vt) / (vs + d)
def sell_sol(vs, vt, d):  return vs - (vs * vt) / (vt + d)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--min-fires", type=int, default=5,
                    help="don't report if fewer than this many fires available")
    return ap.parse_args()


def load_capture_for_mints(capture_dir: Path, mints: set[str]) -> dict[str, list]:
    """Returns mint -> sorted list of (ev_ts, vsol, vtok) trades from capture."""
    cap = defaultdict(list)
    for path in sorted(glob.glob(str(capture_dir / "*.jsonl*"))):
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt") as f:
                for ln in f:
                    try: rec = json.loads(ln)
                    except: continue
                    m = rec.get("mint")
                    if m in mints:
                        vs = rec.get("vsol", 0); vt = rec.get("vtok", 0)
                        if vs > 0 and vt > 0:
                            cap[m].append((rec.get("ev_ts", 0), vs, vt))
        except Exception:
            continue
    # sort by ev_ts within each mint
    for m in cap:
        cap[m].sort(key=lambda x: x[0])
    return dict(cap)


def build_snap_array(forward_trades: list[tuple], snap_every: int = SNAP_EVERY):
    """Subsample forward trades to bot's snap cadence: every Nth trade is a snap.
    Returns: (vs/vt array shape (n,2), fwd indices array, dts array)."""
    snaps = []; fwd = []; dts = []
    if not forward_trades: return [], [], []
    t0 = forward_trades[0][0]
    for i, (ts, vs, vt) in enumerate(forward_trades):
        if (i + 1) % snap_every == 1 and i > 0:   # match offline extractor: snap_every==3 -> fwd 1, 4, 7, ...
            snaps.append((vs, vt))
            fwd.append(i)
            dts.append(ts - t0)
    return snaps, fwd, dts


# ---------- Policy implementations ----------

class ReplayContext:
    """Tracks state across a single mint's replay for a single policy."""
    def __init__(self, vsK, vtK, vsC, vtC, snaps, dts, entry_lat=None):
        # Look up ENTRY_LAT_SNAPS at call-time so module-level overrides (e.g.
        # rl_backtest --entry-lat 2) actually take effect. Default-arg capture
        # at def-time would freeze the original value.
        if entry_lat is None:
            entry_lat = ENTRY_LAT_SNAPS
        self.qlam = 1e9
        self.cost = COST_BPS / 1e4
        self.fee_lam = FEE_PER_TX_SOL * 1e9
        # entry
        if entry_lat == 0:
            self.vse, self.vte = vsK, vtK
            self.start = 0
        else:
            ei = min(entry_lat - 1, len(snaps) - 1)
            if ei < 0:
                self.vse, self.vte = vsK, vtK
                self.start = 0
            else:
                self.vse, self.vte = snaps[ei]
                self.start = entry_lat
        if self.vse <= 0 or self.vte <= 0:
            self.valid = False
            return
        self.valid = True
        self.pos = buy_tokens(self.vse, self.vte, self.qlam)
        self.emk = self.vse / self.vte
        self.pool = snaps[self.start:]
        self.dts = dts[self.start:]
        self.vsC = vsC; self.vtC = vtC
        self.cum_received = 0.0
        self.cum_sold = 0.0
        self.n_tx = 1  # buy
        self.run_max = 0.0

    def my_ret_at(self, i):
        vs, vt = self.pool[i]
        return vs / vt / self.emk - 1.0

    def sell_at(self, i, token_amount):
        vs, vt = self.pool[i]
        vs_eff = max(vs - self.cum_received, 1.0)
        vt_eff = vt + self.cum_sold
        got = sell_sol(vs_eff, vt_eff, token_amount)
        self.cum_received += got
        self.cum_sold += token_amount
        self.n_tx += 1
        return got

    def finalize(self):
        # close remainder at terminal
        remaining = self.pos - self.cum_sold
        if remaining > 0:
            vs_eff = max(self.vsC - self.cum_received, 1.0)
            vt_eff = self.vtC + self.cum_sold
            got = sell_sol(vs_eff, vt_eff, remaining)
            self.cum_received += got
            self.n_tx += 1
        return self.cum_received / self.qlam - 1 - self.cost - (self.fee_lam * self.n_tx) / self.qlam


def policy_k_combined(vsK, vtK, vsC, vtC, snaps, dts,
                       derisk=4, derisk_gap_s=5.0, runner_gap_s=15.0):
    """4 paced de-risk slices (require ret>0, 5s gap) + 4 force-spaced (15s gap)."""
    ctx = ReplayContext(vsK, vtK, vsC, vtC, snaps, dts)
    if not ctx.valid: return None
    if not ctx.pool: return ctx.finalize()
    n_sold = 0; last_slice_t = -1e9
    for i in range(len(ctx.pool)):
        ret_i = ctx.my_ret_at(i)
        if ret_i > ctx.run_max: ctx.run_max = ret_i
        ts_i = ctx.dts[i] if i < len(ctx.dts) else 0.0
        if n_sold < derisk:
            if ret_i > 0 and (ts_i - last_slice_t) >= derisk_gap_s:
                slice_tok = (ctx.pos - ctx.cum_sold) / (TOTAL_SLICES - n_sold)
                if slice_tok > 0:
                    ctx.sell_at(i, slice_tok)
                    n_sold += 1; last_slice_t = ts_i
        else:
            if (ts_i - last_slice_t) >= runner_gap_s:
                slice_tok = (ctx.pos - ctx.cum_sold) / (TOTAL_SLICES - n_sold)
                if slice_tok > 0:
                    ctx.sell_at(i, slice_tok)
                    n_sold += 1; last_slice_t = ts_i
                    if n_sold >= TOTAL_SLICES: break
    return ctx.finalize()


def policy_time_spaced(vsK, vtK, vsC, vtC, snaps, dts, gap_s=15.0):
    """8 slices at fixed gap_s intervals; no ret check."""
    ctx = ReplayContext(vsK, vtK, vsC, vtC, snaps, dts)
    if not ctx.valid: return None
    if not ctx.pool: return ctx.finalize()
    target_times = [k * gap_s for k in range(1, TOTAL_SLICES + 1)]
    n_sold = 0
    t0 = ctx.dts[0] if ctx.dts else 0.0
    for i in range(len(ctx.pool)):
        ts = (ctx.dts[i] if i < len(ctx.dts) else 0.0) - t0
        while n_sold < TOTAL_SLICES and ts >= target_times[n_sold]:
            slice_tok = (ctx.pos - ctx.cum_sold) / (TOTAL_SLICES - n_sold)
            if slice_tok > 0:
                ctx.sell_at(i, slice_tok)
            n_sold += 1
        if n_sold >= TOTAL_SLICES: break
    return ctx.finalize()


def policy_hybrid_trail(vsK, vtK, vsC, vtC, snaps, dts,
                         derisk=4, derisk_gap_s=5.0,
                         runner_retrace=0.30, runner_min_arm=0.20):
    """4 paced de-risk + 4 runner held until trailing stop at retrace_frac."""
    ctx = ReplayContext(vsK, vtK, vsC, vtC, snaps, dts)
    if not ctx.valid: return None
    if not ctx.pool: return ctx.finalize()
    n_sold = 0; last_slice_t = -1e9
    for i in range(len(ctx.pool)):
        ret_i = ctx.my_ret_at(i)
        if ret_i > ctx.run_max: ctx.run_max = ret_i
        ts_i = ctx.dts[i] if i < len(ctx.dts) else 0.0
        if n_sold < derisk:
            if ret_i > 0 and (ts_i - last_slice_t) >= derisk_gap_s:
                slice_tok = (ctx.pos - ctx.cum_sold) / (TOTAL_SLICES - n_sold)
                if slice_tok > 0:
                    ctx.sell_at(i, slice_tok)
                    n_sold += 1; last_slice_t = ts_i
        else:
            if ctx.run_max >= runner_min_arm and (1 + ctx.run_max) > 0:
                if (ctx.run_max - ret_i) / (1 + ctx.run_max) >= runner_retrace:
                    remaining = ctx.pos - ctx.cum_sold
                    if remaining > 0: ctx.sell_at(i, remaining)
                    break
    return ctx.finalize()


def policy_frontload(vsK, vtK, vsC, vtC, snaps, dts):
    """Sell on every profitable snap, 8 slices total."""
    ctx = ReplayContext(vsK, vtK, vsC, vtC, snaps, dts)
    if not ctx.valid: return None
    if not ctx.pool: return ctx.finalize()
    n_sold = 0
    for i in range(len(ctx.pool)):
        ret_i = ctx.my_ret_at(i)
        if ret_i > 0 and n_sold < TOTAL_SLICES:
            slice_tok = (ctx.pos - ctx.cum_sold) / (TOTAL_SLICES - n_sold)
            if slice_tok > 0:
                ctx.sell_at(i, slice_tok); n_sold += 1
                if n_sold >= TOTAL_SLICES: break
    return ctx.finalize()


# ---------- Registry adapter: replay ANY policy in exit_policies/ ------------
# This adapter wraps the new pluggable ExitPolicy interface so the same policy
# code that ships in the live bot is also what auto_policy A/B-tests offline.
# No double-bookkeeping: the auto_policy registry list is authoritative.
def policy_via_registry(vsK, vtK, vsC, vtC, snaps, dts, *,
                         policy_name: str, cfg, entry_features: dict | None = None,
                         entry_score: float = 0.0, mint: str = "REPLAY",
                         snap_extras: list[dict] | None = None) -> float | None:
    """Run the registered ExitPolicy `policy_name` against this fire's snaps.

    Uses the same per-position lifecycle the live harness uses:
      on_entry(mint, fake_ev, entry_features, entry_score)
      decide(...) per snap
      executes slice / sell_all via ReplayContext (own-impact correct)

    snap_extras (optional): list of dicts aligned with `snaps`, providing
    per-snap features beyond ret/run_max/dts (e.g. fill_k, buy_frac_w,
    nsell_w, solo_sell_w, vel_w). Merged into pf for policies that need them
    (lsm_continuation). Live harness already populates these from FeatureAccum.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from exit_policies import get_policy
    from exit_policies.base import HarnessConsts

    ctx = ReplayContext(vsK, vtK, vsC, vtC, snaps, dts)
    if not ctx.valid: return None
    if not ctx.pool: return ctx.finalize()
    policy = get_policy(policy_name, cfg)
    # entry features only matter for policies that route on entry (rl_layered)
    if entry_features is None: entry_features = {}
    class _EV: pass
    fake_ev = _EV()
    try:
        policy.on_entry(mint, fake_ev, entry_features, float(entry_score))
    except Exception:
        pass
    # Read consts from cfg (mirrors what the live harness builds in __init__)
    try:
        consts = HarnessConsts(
            max_slices=int(cfg.exit.total_slices),
            derisk_slices=int(cfg.exit.derisk_slices),
            derisk_min_gap_s=float(cfg.exit.derisk_min_gap_s),
            runner_min_gap_s=float(cfg.exit.runner_min_gap_s),
            runner_retrace_frac=float(cfg.exit.runner_retrace_frac),
            runner_min_arm_ret=float(cfg.exit.runner_min_arm_ret),
            death_threshold=float(cfg.exit.death_threshold),
        )
    except Exception:
        consts = HarnessConsts(8, 4, 5.0, 15.0, 0.30, 0.20, 0.10)
    n_sold = 0; last_slice_t = -1e9
    t0 = ctx.dts[0] if ctx.dts else 0.0
    for i in range(len(ctx.pool)):
        ret_i = ctx.my_ret_at(i)
        if ret_i > ctx.run_max: ctx.run_max = ret_i
        ts_i  = ctx.dts[i] if i < len(ctx.dts) else 0.0
        # p_rec proxy (replay-side; matches what rl_backtest's evaluation uses
        # when no real model is loaded). The live harness passes the real
        # recovery model's p_rec; here we approximate.
        prec_i = max(0.0, min(1.0,
                              1.0 - max(0.0, -ret_i) / (1.0 + max(ctx.run_max, 0.0))))
        # The "now" the policy sees in offline replay = the snap's timestamp.
        # Wall-clock-since-entry approximation: ts_i - t0.
        now = ts_i
        pf = {"ret": ret_i, "run_max_ret": ctx.run_max, "dts": ts_i - t0}
        # Pass through any extra per-snap features (path-local) that the live
        # FeatureAccum would also have. Idx into snap_extras follows ctx.start
        # because the pool was sliced.
        if snap_extras is not None:
            extra_idx = ctx.start + i
            if 0 <= extra_idx < len(snap_extras):
                for k, v in snap_extras[extra_idx].items():
                    pf.setdefault(k, v)
        try:
            dec = policy.decide(mint, n_sold, last_slice_t, now, pf, ctx.run_max,
                                 prec_i, fwd_n=i+1, consts=consts)
        except Exception:
            continue
        if dec.action == "hold":
            continue
        if dec.action == "sell_all":
            remaining = ctx.pos - ctx.cum_sold
            if remaining > 0: ctx.sell_at(i, remaining)
            break
        if dec.action == "slice" and dec.frac > 0:
            slice_tok = (ctx.pos - ctx.cum_sold) * dec.frac
            if slice_tok > 0:
                ctx.sell_at(i, slice_tok)
                n_sold += 1; last_slice_t = ts_i
                if n_sold >= TOTAL_SLICES or dec.frac >= 0.999: break
    return ctx.finalize()


POLICIES = [
    ("K_combined 4+4 g15s (LIVE)", lambda *a: policy_k_combined(*a)),
    ("H_time_spaced 15s",         lambda *a: policy_time_spaced(*a, gap_s=15.0)),
    ("F_hybrid 4+4 t50",          lambda *a: policy_hybrid_trail(*a, runner_retrace=0.50)),
    ("C_hybrid 4+4 t30",          lambda *a: policy_hybrid_trail(*a, runner_retrace=0.30)),
    ("B_frontload",               lambda *a: policy_frontload(*a)),
]


def main():
    args = parse_args()
    root = Path(args.root)
    # Load fires
    decisions = []
    sr = root / "bot_data" / "shadow_run.jsonl"
    if sr.exists():
        with open(sr) as f:
            for ln in f:
                try: decisions.append(json.loads(ln))
                except: continue
    fires = [r for r in decisions if r.get("kind") == "entry_decision" and r.get("fire")]
    print(f"=== A/B exit-policy replay on actual fires ===")
    print(f"fires found: {len(fires)}")
    if len(fires) < args.min_fires:
        print(f"need >= {args.min_fires} fires; current bot's fire rate ~few per hour")
        print("(re-run after the bot has run longer)")
        return

    # Load capture forward trades for each fired mint
    mints = set(r["mint"] for r in fires)
    cap_dir = root / "grpc_capture"
    print(f"loading capture for {len(mints)} mints ...")
    cap_trades = load_capture_for_mints(cap_dir, mints)
    print(f"capture coverage: {len(cap_trades)}/{len(mints)} mints have at least 1 forward trade")

    # Per-fire replay through each policy
    results = {label: [] for label, _ in POLICIES}
    n_analyzed = 0
    for fire in fires:
        m = fire["mint"]
        midK = fire.get("midK")
        vsK = fire.get("vsK"); vtK = fire.get("vtK")
        trigger_ts = fire.get("k_window_last_ts") or fire.get("v_window_last_ts")
        if not all([midK, vsK, vtK, trigger_ts]): continue
        forward = [(ts, vs, vt) for ts, vs, vt in cap_trades.get(m, []) if ts >= trigger_ts]
        if len(forward) < 5: continue
        # vsC, vtC = last seen reserves for hold-to-end fallback
        vsC, vtC = forward[-1][1], forward[-1][2]
        snaps, fwd, dts = build_snap_array(forward)
        if len(snaps) < 1: continue
        n_analyzed += 1
        for label, fn in POLICIES:
            try:
                pl = fn(vsK, vtK, vsC, vtC, snaps, dts)
                if pl is not None:
                    results[label].append({"mint": m[:14], "score": fire["score"], "pl": pl})
            except Exception as e:
                pass

    print(f"\nfires fully replayed: {n_analyzed} / {len(fires)}")
    if n_analyzed < args.min_fires:
        print(f"too few replays for meaningful stats"); return

    print(f"\n{'policy':30s} {'n':>4s} {'mean':>9s} {'median':>9s} {'win%':>6s} {'total':>8s} {'best':>8s} {'worst':>8s}")
    summary = {}
    for label, _ in POLICIES:
        rs = [x["pl"] for x in results[label]]
        if not rs: continue
        n = len(rs); mean = sum(rs)/n; med = st.median(rs)
        win = sum(1 for r in rs if r > 0)
        summary[label] = {"n": n, "mean": mean, "median": med, "win_pct": 100*win/n,
                          "total": sum(rs), "best": max(rs), "worst": min(rs)}
        print(f"{label:30s} {n:>4d} {mean:>+9.4f} {med:>+9.4f} {100*win/n:>5.1f}% "
              f"{sum(rs):>+8.3f} {max(rs):>+8.3f} {min(rs):>+8.3f}")

    # Per-fire breakdown (which policy won each)
    print(f"\nPer-fire winning policy distribution:")
    win_counter = defaultdict(int)
    for i in range(n_analyzed):
        best_pl = -1e9; best_label = None
        for label, _ in POLICIES:
            if i < len(results[label]):
                if results[label][i]["pl"] > best_pl:
                    best_pl = results[label][i]["pl"]
                    best_label = label
        if best_label: win_counter[best_label] += 1
    for label, count in sorted(win_counter.items(), key=lambda x: -x[1]):
        print(f"  {label:30s} winner on {count}/{n_analyzed} fires ({100*count/n_analyzed:.0f}%)")

    # Sample fires where policies differ most
    print(f"\nFires where policies disagree most (delta = best - worst):")
    rows = []
    for i in range(n_analyzed):
        pls = []
        for label, _ in POLICIES:
            if i < len(results[label]):
                pls.append((label, results[label][i]["pl"]))
        if pls:
            mn, mx = min(p[1] for p in pls), max(p[1] for p in pls)
            mint = results[POLICIES[0][0]][i]["mint"]
            score = results[POLICIES[0][0]][i]["score"]
            rows.append((mx - mn, mint, score, pls))
    rows.sort(reverse=True)
    for delta, mint, score, pls in rows[:5]:
        print(f"  mint={mint:14} score={score:.4f} delta={delta:+.3f}")
        for label, pl in pls:
            print(f"    {label:30s} {pl:+.4f}")

    out_path = root / "logs" / "ab_replay_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n_analyzed": n_analyzed, "summary": summary,
                   "winner_distribution": dict(win_counter)}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
