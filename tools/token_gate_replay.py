"""token_gate_replay — what did the two model heads say about ONE token?

Pulls a single mint's lifecycle from bot_data/shadow_run.jsonl and prints a
two-block timeline:

  [ENTRY GATE] — V+K7 stacked head
      score, threshold, fire?, 22 features summary, K/V trigger reserves

  [LIFECYCLE GATE] — recovery head + slice policy
      per-snap p_rec, slice fires (paced/runner/trail), death-cuts, runner-exit

  [OUTCOME]
      position_close (book net), broker_jito.jsonl realized sum

Usage:
  pumpfun_ctl.sh gate-replay <mint>            # specific mint
  pumpfun_ctl.sh gate-replay --last 1          # most recent fire
  pumpfun_ctl.sh gate-replay --last 5          # last 5 fires (one per token)
  pumpfun_ctl.sh gate-replay --dumped          # show only fires that ended
                                                 in death-cut or strong loss
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bot_data"
SHADOW_LOG = DATA / "shadow_run.jsonl"
POS_LOG    = DATA / "positions.jsonl"
BROKER_LOG = DATA / "broker_jito.jsonl"

# Top features to surface in the entry block (highest population-level
# importance from the V+K7 training run; we don't have shap per-fire so
# we just show these consistently for human eyeballing).
ENTRY_FEATS_SHOW = [
    "win_ret", "dir_eff", "buy_frac", "uniq", "net_sol", "single_actor_share",
    "trades_per_sec", "win_drawup", "win_drawdown",
    "win_ret_v", "dir_eff_v", "buy_frac_v", "uniq_v", "net_sol_v",
]


def _fmt_ts(t: float | None) -> str:
    if t is None: return "?"
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%H:%M:%S")


def _fmt_age(t0: float | None, t: float | None) -> str:
    if t0 is None or t is None: return "?"
    return f"{t - t0:+.0f}s"


def _load_bet_sol() -> float:
    # tools/ is one directory below the project root; add the root so we can
    # import bot_config without having to cd. Falls back to parsing config.yaml
    # if bot_config isn't importable (older snapshots without the loader).
    try:
        sys.path.insert(0, str(ROOT))
        from bot_config import cfg
        return float(cfg.bot.bet_sol)
    except Exception:
        pass
    try:
        import yaml
        with (ROOT / "config.yaml").open() as f:
            y = yaml.safe_load(f)
        return float(y.get("bot", {}).get("bet_sol", 1.0))
    except Exception:
        return 1.0


def _stream_events_for_mint(mint: str):
    """Yield events from shadow_run.jsonl that mention this mint."""
    if not SHADOW_LOG.exists(): return
    with SHADOW_LOG.open() as f:
        for line in f:
            try: e = json.loads(line)
            except Exception: continue
            if e.get("mint") == mint:
                yield e


def _broker_events_for_mint(mint: str):
    """Yield broker_jito.jsonl events for this mint (if file exists)."""
    if not BROKER_LOG.exists(): return
    with BROKER_LOG.open() as f:
        for line in f:
            try: e = json.loads(line)
            except Exception: continue
            if e.get("mint") == mint:
                yield e


def _position_close_records(mint: str):
    """Yield close records for this mint from positions.jsonl."""
    if not POS_LOG.exists(): return
    with POS_LOG.open() as f:
        for line in f:
            try: e = json.loads(line)
            except Exception: continue
            if e.get("kind") == "close" and e.get("mint") == mint:
                yield e


def _list_fires(limit: int | None = None,
                only_dumped: bool = False) -> list[tuple[float, str, float]]:
    """Return [(t, mint, score), ...] of fires (entry_decision with fire=true).
    Most recent first. If only_dumped, restrict to mints whose lifecycle had
    a death_cut or whose final pos.net_return < -0.10 (fractional)."""
    fires = []
    if not SHADOW_LOG.exists(): return fires
    death_cut_mints = set()
    if only_dumped:
        with SHADOW_LOG.open() as f:
            for line in f:
                try: e = json.loads(line)
                except Exception: continue
                if e.get("kind") == "live_death_cut":
                    death_cut_mints.add(e.get("mint"))
        bad_closes = set()
        if POS_LOG.exists():
            with POS_LOG.open() as f:
                for line in f:
                    try: e = json.loads(line)
                    except Exception: continue
                    if e.get("kind") == "close" and e.get("net_return", 0) < -0.10:
                        bad_closes.add(e.get("mint"))
        keep = death_cut_mints | bad_closes
    with SHADOW_LOG.open() as f:
        for line in f:
            try: e = json.loads(line)
            except Exception: continue
            if e.get("kind") == "entry_decision" and e.get("fire"):
                m = e.get("mint")
                if only_dumped and m not in keep:
                    continue
                fires.append((e.get("t"), m, e.get("score")))
    fires.sort(key=lambda r: r[0] or 0, reverse=True)
    if limit: fires = fires[:limit]
    return fires


def _print_header(mint: str, fire_t: float | None, score: float | None,
                  threshold: float | None, bet_sol: float):
    print()
    print("=" * 90)
    print(f"  token: {mint}")
    if fire_t:
        print(f"  fired at {datetime.fromtimestamp(fire_t, tz=timezone.utc).isoformat()}  "
              f"(bet={bet_sol:g} SOL)")
    if score is not None and threshold is not None:
        mar = score - threshold
        print(f"  entry score = {score:.4f}   threshold = {threshold:.4f}   "
              f"margin = {mar:+.4f}  ({'PASS' if mar >= 0 else 'BLOCK'})")
    print("=" * 90)


def replay_one(mint: str, bet_sol: float | None = None) -> None:
    bet_sol = bet_sol if bet_sol is not None else _load_bet_sol()
    events = list(_stream_events_for_mint(mint))
    if not events:
        print(f"[gate-replay] no events found for mint {mint}", file=sys.stderr)
        return

    # ---- Locate the entry_decision that fired ----
    fire_event = None
    last_decision = None
    for e in events:
        if e.get("kind") == "entry_decision":
            last_decision = e
            if e.get("fire"):
                fire_event = e
    de = fire_event or last_decision

    fire_t = de.get("t") if de else None
    score = de.get("score") if de else None
    thr   = de.get("threshold") if de else None

    _print_header(mint, fire_t, score, thr, bet_sol)

    # ---- ENTRY GATE block ----
    print()
    print("ENTRY GATE  (V+K7 stacked head — fires if score >= threshold)")
    if de is None:
        print("  (no entry_decision found — token never reached K=7+V=0.5 ready)")
    else:
        midK = de.get("midK"); vsK = de.get("vsK"); vtK = de.get("vtK")
        midV = de.get("midV"); vsV = de.get("vsV"); vtV = de.get("vtV")
        print(f"  K=7 trigger : ts_window_last={de.get('k_window_last_ts')}  "
              f"midK={midK}  vsK_lam={vsK}  vtK={vtK}")
        print(f"  V=0.5 trigger: ts_window_last={de.get('v_window_last_ts')}  "
              f"midV={midV}  vsV_lam={vsV}  vtV={vtV}")
        print(f"  cum_buy_sol_at_v_trigger = {de.get('cum_buy_sol'):.4f}  "
              f"first_seen_rsol_lam = {de.get('first_seen_rsol')}  "
              f"slot = {de.get('ev_slot')}")
        feats = de.get("features") or {}
        if feats:
            print()
            print(f"  features (showing {len(ENTRY_FEATS_SHOW)} of {len(feats)}):")
            for fname in ENTRY_FEATS_SHOW:
                if fname in feats:
                    v = feats[fname]
                    try: vs = f"{float(v):+.4f}"
                    except Exception: vs = str(v)
                    print(f"    {fname:24s} = {vs}")
        # were there earlier non-firing decisions for this mint?
        n_decisions = sum(1 for e in events if e.get("kind") == "entry_decision")
        if n_decisions > 1:
            n_fires = sum(1 for e in events if e.get("kind")=="entry_decision" and e.get("fire"))
            print(f"  ({n_decisions} entry decisions on this mint; {n_fires} fired)")

    # ---- LIFECYCLE block (recovery head + slice policy) ----
    snaps  = [e for e in events if e.get("kind") == "path_snap"]
    slices = [e for e in events if e.get("kind") == "live_scale_slice"]
    deaths = [e for e in events if e.get("kind") == "live_death_cut"]
    rexits = [e for e in events if e.get("kind") == "live_runner_exit"]
    closes = [e for e in events if e.get("kind") == "position_close"]

    print()
    print(f"LIFECYCLE GATE  (recovery head P(recover) per snap + slice/death decisions)")
    print(f"  total: {len(snaps)} snaps, {len(slices)} slices, "
          f"{len(deaths)} death-cuts, {len(rexits)} runner-exits, {len(closes)} closes")
    print()
    if not snaps and not slices:
        print("  (no path snaps after fire — buy never settled into a routed snap)")
    else:
        # Build a merged timeline of (t, kind, summary) for snaps + slices + deaths + rexits
        rows = []
        for s in snaps:
            pf = s.get("path_feats") or {}
            ret  = pf.get("ret")
            rmax = pf.get("run_max_ret")
            p    = s.get("p_rec")
            rows.append((s.get("t"), "SNAP",
                         f"fwd={s.get('fwd'):>3}  ret={ret:+.3f}  run_max={rmax:+.3f}  "
                         f"p_rec={p:.3f}  slot={s.get('ev_slot')}"))
        for sl in slices:
            rows.append((sl.get("t"), "SLICE",
                         f"fwd={sl.get('fwd'):>3}  slice {sl.get('slice_n')}/8  "
                         f"phase={sl.get('phase'):<6}  policy={sl.get('policy')}  "
                         f"frac={sl.get('frac'):.2f}  ret={sl.get('ret'):+.3f}  "
                         f"run_max={sl.get('run_max'):+.3f}  slot={sl.get('slot')}"))
        for d in deaths:
            rows.append((d.get("t"), "DEATH",
                         f"fwd={d.get('fwd'):>3}  n_sold={d.get('n_sold')}/8  "
                         f"phase={d.get('phase')}  ret={d.get('ret'):+.3f}  "
                         f"p_rec={d.get('p_rec'):.3f}  run_max={d.get('run_max'):+.3f}  "
                         f"slot={d.get('slot')}"))
        for r in rexits:
            rows.append((r.get("t"), "RUNNX",
                         f"fwd={r.get('fwd'):>3}  reason={r.get('reason')}  "
                         f"ret={r.get('ret'):+.3f}  run_max={r.get('run_max'):+.3f}  "
                         f"slot={r.get('slot')}"))
        for c in closes:
            rows.append((c.get("t"), "CLOSE",
                         f"exit_kind={c.get('exit_kind')}  net={c.get('net'):+.4f}  "
                         f"reason={c.get('reason')}"))
        rows.sort(key=lambda r: r[0] or 0)
        # Print timeline with age relative to fire
        for t, kind, summary in rows:
            age = _fmt_age(fire_t, t)
            ts  = _fmt_ts(t)
            print(f"  {ts}  {age:>6}  {kind:<5}  {summary}")

    # ---- OUTCOME block ----
    print()
    print("OUTCOME")
    if closes:
        c = closes[-1]
        frac = c.get("net", 0.0)
        abs_sol = frac * bet_sol
        print(f"  PaperBook close: exit_kind={c.get('exit_kind')}  reason={c.get('reason')}")
        print(f"    fractional net = {frac:+.4f}  (absolute {abs_sol:+.4f} SOL @ bet={bet_sol:g})")
    else:
        print(f"  (no PaperBook close yet — position still open or never closed cleanly)")

    # broker_jito.jsonl summary (real on-chain realized)
    brk = list(_broker_events_for_mint(mint))
    if brk:
        kinds = defaultdict(int)
        for b in brk: kinds[b.get("kind", "?")] += 1
        print(f"  broker_jito events: {dict(kinds)}")
        landed = [b for b in brk if b.get("status") == "landed"]
        not_yet = [b for b in brk if b.get("status") == "not_yet_wired"]
        if landed: print(f"    landed bundles: {len(landed)}")
        if not_yet: print(f"    not_yet_wired (DRY_RUN): {len(not_yet)}")
    else:
        print("  (no broker_jito events for this mint)")

    print()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("mint", nargs="?", help="specific mint address")
    g.add_argument("--last", type=int, default=None,
                   help="replay the last N fired tokens (most recent first)")
    g.add_argument("--dumped", action="store_true",
                   help="replay tokens that hit death-cut or closed below -10%")
    ap.add_argument("--bet-sol", type=float, default=None,
                    help="override config bet_sol when converting fractional → SOL")
    args = ap.parse_args()
    bet_sol = args.bet_sol if args.bet_sol is not None else _load_bet_sol()

    if args.mint:
        replay_one(args.mint, bet_sol=bet_sol)
        return

    fires = _list_fires(limit=args.last, only_dumped=args.dumped)
    if not fires:
        msg = "dumped" if args.dumped else "fired"
        print(f"[gate-replay] no {msg} tokens found in {SHADOW_LOG}", file=sys.stderr)
        return
    for t, mint, score in fires:
        replay_one(mint, bet_sol=bet_sol)


if __name__ == "__main__":
    main()
