"""rl_backtest — offline Fitted-Q-Iteration on the May parquets.

Frames the exit-timing problem as a discrete MDP:

  state  s = (ret_bin, runmax_bin, prec_bin, dts_bin, n_sold_bin)
  action a ∈ {0, 0.125, 0.25, 0.50, 1.0}    # fraction of remaining to sell
  reward r = SOL realized this snap × bet_sol fraction
  transition s -> s' empirical from May per-snap data

Solves V*(s) = max_a [r(s,a) + E[V*(s') | s,a]] via tabular value iteration.

CRITICAL ASSUMPTION: own-impact ignored at transition level. We use the
path-snapshot AMM state as-is, ignoring that our selling shifts (vs, vt).
For 0.1-SOL bets into 30-100 SOL pools this is ~3-bp distortion — small
relative to the policy gap we are measuring. The ReplayContext below
DOES compound own-impact correctly when we evaluate the policy. So the
TRAINING ignores own-impact (one-step Bellman uses snap state) but the
EVALUATION measures real P&L under own-impact.

Pipeline:
  1. Load token_level + path_snapshots parquets from data/.
  2. Reconstruct each position's lifetime as a list of (s, r, s') tuples.
  3. Bin state, bin action, build empirical r(s,a) and P(s'|s,a) tables.
  4. Run value iteration to convergence (~50 sweeps usually).
  5. Extract greedy policy π*(s).
  6. Evaluate π* via ReplayContext (own-impact correct) and report
     mean/median/p25/win% vs K_combined.

Usage:
  pumpfun_ctl.sh rl-backtest                          # train on K7_fresh, eval same
  pumpfun_ctl.sh rl-backtest --eval-on OOS            # train on K7_fresh, eval on OOS
  pumpfun_ctl.sh rl-backtest --bins 5,5,5,4,4         # fewer bins (faster, more biased)
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from strategy_ab_replay import (
    ReplayContext, TOTAL_SLICES, SNAP_EVERY,
    COST_BPS, FEE_PER_TX_SOL,
    policy_k_combined, policy_time_spaced, policy_hybrid_trail, policy_frontload,
)

# ----------------------- Recovery model integration -------------------------
#
# The 20-feature recovery head expects PATH + ENTRY_K columns in this order.
# Must mirror build_bot_artifacts_K7V.py exactly so the feature vector is
# identical to what model_serve sends at runtime.
PATH_COLS  = ["ret", "run_max_ret", "dd", "fill_k", "buy_frac_w",
              "nsell_w", "solo_sell_w", "vel_w", "dts"]
ENTRY_COLS = ["win_ret", "dir_eff", "buy_frac", "uniq", "net_sol", "tot_sol",
              "single_actor_share", "trades_per_sec", "entry_sol",
              "win_drawup", "win_drawdown"]


def compute_real_prec(snap_df: pd.DataFrame, token_df: pd.DataFrame,
                       model_path: Path) -> pd.DataFrame:
    """For every (mint, fwd) row in snap_df, compute P(recover) from the
    trained 20-feature recovery model. Returns a DataFrame with columns
    [mint, fwd, p_rec] suitable for merging into snap_df.

    The model expects the same feature ORDER as build_bot_artifacts_K7V.py
    used during training: PATH + ENTRY_K = 20 features.

    For training, the recovery model only sees drawdown snaps (ret < 0). At
    serve time / RL backtest, we still score every snap — but we clamp
    P(recover) = 1.0 wherever ret >= 0 to match the harness's runtime
    convention. That keeps the bin distribution identical to live serving.
    """
    import pickle
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    # Join entry features into every snap row by mint
    tk = token_df.reset_index() if "mint" not in token_df.columns else token_df.copy()
    snaps = snap_df.merge(tk[["mint"] + ENTRY_COLS], on="mint", how="left")
    # Some old rows might have NaN in entry cols (mint not in token_level); drop
    feat = snaps[PATH_COLS + ENTRY_COLS].values
    valid_mask = ~np.isnan(feat).any(axis=1)
    p_rec = np.ones(len(snaps), dtype=np.float32)
    if valid_mask.sum() > 0:
        p_rec_pred = clf.predict_proba(feat[valid_mask])[:, 1].astype(np.float32)
        p_rec[valid_mask] = p_rec_pred
    # clamp to 1.0 where ret >= 0 (matches harness behavior)
    p_rec[snaps["ret"].values >= 0] = 1.0
    out = snaps[["mint", "fwd"]].copy()
    out["p_rec"] = p_rec
    return out


# ----------------------- State / action discretization -----------------------

# These bins encode where the heavy-tailed pump.fun action lives.
# We make the cut-points NON-UNIFORM so heavy bins (around 0 ret) get more
# resolution than tails. This is informative-prior, not optimization.
RET_BINS    = np.array([-1.0, -0.5, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 100.0])
RUNMAX_BINS = np.array([ 0.0,  0.10, 0.25,  0.50, 1.0, 2.0, 5.0, 100.0])
PREC_BINS   = np.array([0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.01])
DTS_BINS    = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 1e9])
NSOLD_BINS  = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9])  # n_sold ∈ [0..8]

# Action space — fraction of remaining to sell. Including 0 (hold) is critical.
ACTIONS = np.array([0.0, 0.125, 0.25, 0.50, 1.0])
N_ACTIONS = len(ACTIONS)


def state_key(ret, runmax, prec, dts, n_sold):
    i_ret    = int(np.searchsorted(RET_BINS,    ret,    side="right") - 1)
    i_runmax = int(np.searchsorted(RUNMAX_BINS, runmax, side="right") - 1)
    i_prec   = int(np.searchsorted(PREC_BINS,   prec,   side="right") - 1)
    i_dts    = int(np.searchsorted(DTS_BINS,    dts,    side="right") - 1)
    i_nsold  = int(np.searchsorted(NSOLD_BINS,  n_sold, side="right") - 1)
    # clip to legal ranges
    i_ret    = max(0, min(len(RET_BINS)-2,    i_ret))
    i_runmax = max(0, min(len(RUNMAX_BINS)-2, i_runmax))
    i_prec   = max(0, min(len(PREC_BINS)-2,   i_prec))
    i_dts    = max(0, min(len(DTS_BINS)-2,    i_dts))
    i_nsold  = max(0, min(len(NSOLD_BINS)-2,  i_nsold))
    return (i_ret, i_runmax, i_prec, i_dts, i_nsold)


# ----------------------- Build transition / reward tables --------------------

def build_mdp_tables(snap_df: pd.DataFrame, token_df: pd.DataFrame,
                      p_rec_df: pd.DataFrame | None = None):
    """Returns:
        Rsa[s][a]   -> sum of immediate-reward, count
        Psa[s][a][s'] -> count of transitions

    Approximations:
      - reward at (s,a) = expected SOL realized from selling action a's
        fraction of remaining holdings at the snap's current AMM state,
        using the cleanest "small-bet" linear approximation (ret × frac × bet).
        For bigger bets (>5% of pool) we would need to do the impact-correct
        AMM sell here; for 0.1 SOL bets this is fine.
      - we ignore that selling fraction `a` *removes inventory* from future
        snaps. Instead we treat n_sold as a counter that increments by 1
        on any non-zero action (matches K_combined's slice-counting).
    """
    snap_df = snap_df.sort_values(["mint", "fwd"]).reset_index(drop=True)
    # Optionally attach p_rec from a passed prediction. If not provided, we
    # use the recovery proxy: 1 - max(0, -ret/(1+runmax)) — a cheap monotonic
    # stand-in so we can run FQI without the recovery model right here.
    if p_rec_df is not None and "p_rec" in p_rec_df.columns:
        snap_df = snap_df.merge(p_rec_df[["mint", "fwd", "p_rec"]],
                                on=["mint", "fwd"], how="left")
        snap_df["p_rec"] = snap_df["p_rec"].fillna(1.0)
    else:
        # cheap monotone proxy bounded in [0,1]
        ret = snap_df["ret"].values
        rmax = snap_df["run_max_ret"].values
        snap_df["p_rec"] = np.clip(1.0 - np.maximum(0.0, -ret) / (1.0 + np.maximum(rmax, 0.0)), 0.0, 1.0)

    R_sum   = defaultdict(lambda: np.zeros(N_ACTIONS))
    R_count = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.int64))
    T_count = defaultdict(lambda: defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.int64)))

    # Iterate per-mint trajectories
    for mint, g in snap_df.groupby("mint", sort=False):
        n_sold = 0
        held_frac = 1.0
        rows = g.itertuples(index=False)
        prev_state = None; prev_action = None
        for r in rows:
            ret    = float(r.ret)
            rmax   = float(r.run_max_ret)
            prec   = float(r.p_rec)
            dts    = float(r.dts)
            s = state_key(ret, rmax, prec, dts, n_sold)
            # Record transition for previous step (s_prev, a_prev) -> s
            if prev_state is not None:
                T_count[prev_state][s][prev_action] += 1
            # For each action a, record the immediate reward expected if taken now
            for ai, frac in enumerate(ACTIONS):
                if frac == 0.0:
                    rwd = 0.0
                else:
                    # SOL realized if we sell `frac * held_frac` at current ret.
                    # The "1 + ret" is the per-token value relative to entry.
                    # We deduct cost & per-tx fee proportionally on non-zero a.
                    qty_frac = held_frac * frac
                    gross    = qty_frac * (1.0 + ret)
                    cost_per_slice = (COST_BPS / 1e4) * qty_frac
                    fee_per_slice = (FEE_PER_TX_SOL / 1.0) * 1.0  # 0.0015 SOL per tx
                    rwd = gross - cost_per_slice - fee_per_slice
                R_sum[s][ai]   += rwd
                R_count[s][ai] += 1
            # Pick an action stochastically only for transition logging — we
            # cycle through actions weighted by how often each one is taken
            # under K_combined-ish behavior. For simplicity we record the
            # transition under the "no-sell" action so we have the dense path
            # of (s -> s') under no-op, which is the dominant transition
            # distribution we care about. (Q-iteration uses one-step ahead
            # under SOME policy; "no-sell" gives unbiased next-state.)
            prev_state  = s
            prev_action = 0   # no-sell transition kernel
        # end mint loop

    return R_sum, R_count, T_count


def value_iteration(R_sum, R_count, T_count,
                     n_iter: int = 200, gamma: float = 0.999,
                     tol: float = 1e-5) -> tuple[dict, dict]:
    """Tabular VI on the empirical tables. Returns (V*, π*).

    Each iteration:
      Q(s, a) = mean_r(s, a)  +  gamma * sum_s' P(s'|s, a) V(s')
    For terminal/never-visited states we use V=0 (absorbing).

    For actions that ARE sells (a > 0), we approximate the next-state
    transition as identical to the no-sell transition (since the AMM path
    doesn't change with our small bet) but the SLICE COUNTER s'[nsold]
    increments by 1. We re-key s' accordingly.
    """
    # Build mean-reward table
    R_mean = {}
    for s, sums in R_sum.items():
        counts = R_count[s]
        R_mean[s] = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    # Pre-compute next-state distributions
    # T_count[s] is dict s' -> [counts per action]; we used action 0 only.
    # For action a>0 next state is "same path next snap but n_sold += 1".
    P_next = {}  # s -> list of (s', prob) under action 0
    for s, succ_dict in T_count.items():
        total = sum(arr[0] for arr in succ_dict.values())
        if total <= 0: continue
        P_next[s] = [(sp, arr[0] / total) for sp, arr in succ_dict.items()]

    V = defaultdict(float)
    for it in range(n_iter):
        delta = 0.0
        new_V = {}
        for s in R_mean.keys():
            qvals = np.zeros(N_ACTIONS)
            for ai, frac in enumerate(ACTIONS):
                # Reward part
                qvals[ai] = R_mean[s][ai]
                # Continuation part: weight V of next state.
                # For a > 0, the next state has n_sold incremented; if
                # n_sold already at 8 (saturated) we treat the position
                # as closed (continuation 0).
                next_n_sold_inc = 1 if frac > 0 else 0
                successors = P_next.get(s)
                if successors is None:
                    continue
                # For action 0 we use the no-sell transitions as-is.
                # For action a>0 we shift s'.nsold by 1 (cap at last bin).
                cont = 0.0
                for sp, pr in successors:
                    if next_n_sold_inc == 0:
                        cont += pr * V.get(sp, 0.0)
                    else:
                        sp_inc = (sp[0], sp[1], sp[2], sp[3],
                                  min(sp[4] + 1, len(NSOLD_BINS) - 2))
                        cont += pr * V.get(sp_inc, 0.0)
                # If selling 100% (frac=1.0), we exit — terminal continuation 0
                if frac >= 0.999:
                    cont = 0.0
                qvals[ai] += gamma * cont
            new_v = qvals.max()
            old_v = V.get(s, 0.0)
            delta = max(delta, abs(new_v - old_v))
            new_V[s] = new_v
        # Apply
        for s, v in new_V.items(): V[s] = v
        if delta < tol:
            print(f"  VI converged at iter {it+1} (delta={delta:.2e})")
            break
    # Policy extraction
    pi = {}
    for s in R_mean.keys():
        qvals = np.zeros(N_ACTIONS)
        for ai, frac in enumerate(ACTIONS):
            qvals[ai] = R_mean[s][ai]
            next_n_sold_inc = 1 if frac > 0 else 0
            successors = P_next.get(s)
            if successors is not None and frac < 0.999:
                for sp, pr in successors:
                    if next_n_sold_inc == 0:
                        qvals[ai] += gamma * pr * V.get(sp, 0.0)
                    else:
                        sp_inc = (sp[0], sp[1], sp[2], sp[3],
                                  min(sp[4] + 1, len(NSOLD_BINS) - 2))
                        qvals[ai] += gamma * pr * V.get(sp_inc, 0.0)
        pi[s] = int(np.argmax(qvals))
    return dict(V), pi


# ------------------------- Policy evaluation under own-impact -----------------

def policy_rl_lookup(pi: dict, p_rec_per_snap: dict | None = None):
    """Return a policy function compatible with ReplayContext-based replay.
    Uses the learned π* lookup at each snap; falls back to "no sell" if state
    unseen. n_sold is tracked locally.

    `p_rec_per_snap`: optional {(mint, fwd_index_within_pool): p_rec}. If
    provided, we use the real recovery-model p_rec at evaluation. Otherwise
    we fall back to the same monotone proxy used at training time.
    """
    def fn(vsK, vtK, vsC, vtC, snaps, dts, mint=None):
        ctx = ReplayContext(vsK, vtK, vsC, vtC, snaps, dts)
        if not ctx.valid: return None
        if not ctx.pool: return ctx.finalize()
        n_sold = 0
        for i in range(len(ctx.pool)):
            ret_i  = ctx.my_ret_at(i)
            if ret_i > ctx.run_max: ctx.run_max = ret_i
            ts_i  = ctx.dts[i] if i < len(ctx.dts) else 0.0
            # p_rec: real model prediction if mint+i available, else proxy
            prec_i = None
            if p_rec_per_snap is not None and mint is not None:
                prec_i = p_rec_per_snap.get((mint, i))
            if prec_i is None:
                prec_i = max(0.0, min(1.0,
                                      1.0 - max(0.0, -ret_i) / (1.0 + max(ctx.run_max, 0.0))))
            s = state_key(ret_i, ctx.run_max, prec_i, ts_i, n_sold)
            ai = pi.get(s, 0)
            frac = float(ACTIONS[ai])
            if frac > 0 and (ctx.pos - ctx.cum_sold) > 0:
                slice_tok = (ctx.pos - ctx.cum_sold) * frac
                if slice_tok > 0:
                    ctx.sell_at(i, slice_tok)
                    n_sold += 1
                    if n_sold >= TOTAL_SLICES or frac >= 0.999: break
        return ctx.finalize()
    return fn


def policy_calibrated_soft_cut(vsK, vtK, vsC, vtC, snaps, dts):
    """Replace the binary death-cut with a sell-fraction-by-P(recover):
        sell_fraction_now = max(0, 1 - 2 * p_rec_proxy)
    So p_rec=0.5 → sell 0; p_rec=0.3 → sell 40%; p_rec=0.1 → sell 80%; p_rec=0 → sell 100%.
    Plus paced de-risk on the way up (matching K_combined).
    """
    ctx = ReplayContext(vsK, vtK, vsC, vtC, snaps, dts)
    if not ctx.valid: return None
    if not ctx.pool: return ctx.finalize()
    n_sold = 0; last_slice_t = -1e9
    DERISK = 4; DERISK_GAP = 5.0; RUNNER_GAP = 15.0
    for i in range(len(ctx.pool)):
        ret_i = ctx.my_ret_at(i)
        if ret_i > ctx.run_max: ctx.run_max = ret_i
        ts_i  = ctx.dts[i] if i < len(ctx.dts) else 0.0
        # CALIBRATED CUT — fires anywhere if recovery is fading
        prec_proxy = max(0.0, min(1.0, 1.0 - max(0.0, -ret_i) / (1.0 + max(ctx.run_max, 0.0))))
        if ret_i < 0 and prec_proxy < 0.5:
            # sell-fraction = (1 - 2*p_rec) clamped to [0, 1]
            cut_frac = max(0.0, min(1.0, 1.0 - 2.0 * prec_proxy))
            if cut_frac > 0:
                remaining = ctx.pos - ctx.cum_sold
                if remaining > 0:
                    slice_tok = remaining * cut_frac
                    if slice_tok > 0: ctx.sell_at(i, slice_tok)
                    if cut_frac >= 0.999: break
                    last_slice_t = ts_i
                    n_sold += 1
                    continue  # don't also do paced de-risk this step
        # Paced de-risk + runner like K_combined
        if n_sold < DERISK:
            if ret_i > 0 and (ts_i - last_slice_t) >= DERISK_GAP:
                slice_tok = (ctx.pos - ctx.cum_sold) / (TOTAL_SLICES - n_sold)
                if slice_tok > 0:
                    ctx.sell_at(i, slice_tok); n_sold += 1; last_slice_t = ts_i
        else:
            if (ts_i - last_slice_t) >= RUNNER_GAP:
                slice_tok = (ctx.pos - ctx.cum_sold) / (TOTAL_SLICES - n_sold)
                if slice_tok > 0:
                    ctx.sell_at(i, slice_tok); n_sold += 1; last_slice_t = ts_i
                    if n_sold >= TOTAL_SLICES: break
    return ctx.finalize()


# -------------------------------- Main ---------------------------------------

def _stats(name, rs):
    if not rs: return {"name": name, "n": 0}
    a = np.array(rs)
    return {"name": name, "n": int(len(a)),
            "mean":   float(a.mean()),
            "median": float(np.median(a)),
            "p25":    float(np.percentile(a, 25)),
            "win_pct": float(100 * (a > 0).mean()),
            "best":   float(a.max()),
            "worst":  float(a.min())}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="data/pumpfun_continuation_K7_fresh")
    ap.add_argument("--eval-dir",  default=None,
                    help="default: same as train-dir; pass _oos_ path for true OOS")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--n-iter", type=int, default=200)
    ap.add_argument("--save-policy", default=None,
                    help="optional pickle path to save π* + V*")
    ap.add_argument("--recovery-model", default=None,
                    help="path to recovery_model.pkl (default: "
                         "bot_artifacts_K7V/recovery_model.pkl); pass empty string "
                         "to force proxy mode")
    ap.add_argument("--use-recovery-model", action="store_true", default=False,
                    help="compute real p_rec from the trained recovery model "
                         "(instead of the cheap proxy) for both train+eval")
    ap.add_argument("--entry-lat", type=int, default=1,
                    help="entry latency in snaps (default 1 = fill at next "
                         "snap; 2 = fill at snap+1 to model slot-delay slippage)")
    ap.add_argument("--peak-max", type=float, default=None,
                    help="exclude tokens with peak_ret >= this from TRAINING. "
                         "Eg. --peak-max 1.46 trains on Q1-Q4 (no Q5 moonshots) so "
                         "the policy cannot lean on diamond-hands-for-moonshot logic")
    ap.add_argument("--peak-min", type=float, default=None,
                    help="exclude tokens with peak_ret < this from TRAINING. "
                         "Eg. --peak-min 1.46 trains on Q5 only (the moonshot-holder)")
    return ap.parse_args()


def main():
    args = parse_args()
    train_dir = ROOT / args.train_dir
    eval_dir  = ROOT / (args.eval_dir or args.train_dir)
    print(f"=== rl_backtest @ {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ===")
    print(f"train: {train_dir}")
    print(f"eval : {eval_dir}")
    # Override entry latency globally so all policies see the same fill model.
    # entry_lat=2 simulates one additional slot of delay between gRPC observation
    # and bundle landing — a more realistic stress test of bang-bang policies
    # that rely on hitting a specific (ret, p_rec) configuration.
    import strategy_ab_replay as _sar
    _sar.ENTRY_LAT_SNAPS = args.entry_lat
    print(f"entry_lat: {_sar.ENTRY_LAT_SNAPS} snap(s) (1=instant, 2=one-slot delay)")

    # Load train
    sk_train = pd.read_parquet(train_dir / "path_snapshots.parquet")
    tk_train = pd.read_parquet(train_dir / "token_level.parquet").set_index("mint")
    # Optional peak_ret filter (mixture-of-experts split). Apply BEFORE subset
    # so subset reflects the filtered population.
    n_before = len(tk_train)
    if args.peak_max is not None and "peak_ret" in tk_train.columns:
        tk_train = tk_train[tk_train["peak_ret"] < args.peak_max]
        print(f"  filter: peak_ret < {args.peak_max} → kept {len(tk_train)}/{n_before} mints")
    if args.peak_min is not None and "peak_ret" in tk_train.columns:
        tk_train = tk_train[tk_train["peak_ret"] >= args.peak_min]
        print(f"  filter: peak_ret ≥ {args.peak_min} → kept {len(tk_train)}/{n_before} mints")
    if args.peak_max is not None or args.peak_min is not None:
        sk_train = sk_train[sk_train.mint.isin(tk_train.index)]
    if args.subset:
        mints = tk_train.index.tolist()[:args.subset]
        sk_train = sk_train[sk_train.mint.isin(mints)]
        tk_train = tk_train.loc[mints]
    print(f"train mints: {len(tk_train)}, snaps: {len(sk_train)}")

    # Optionally compute REAL p_rec from the trained recovery head.
    p_rec_train = None
    if args.use_recovery_model:
        model_path = (ROOT / (args.recovery_model
                              or "bot_artifacts_K7V/recovery_model.pkl"))
        print(f"computing real p_rec from {model_path} ...")
        t0 = time.time()
        p_rec_train = compute_real_prec(sk_train, tk_train, model_path)
        nz = (p_rec_train["p_rec"] < 1.0).sum()
        print(f"  p_rec computed for {len(p_rec_train)} snaps ({time.time()-t0:.1f}s); "
              f"{nz} ({100*nz/len(p_rec_train):.1f}%) in drawdown region")

    print("building MDP tables ...")
    t0 = time.time()
    R_sum, R_count, T_count = build_mdp_tables(sk_train, tk_train, p_rec_df=p_rec_train)
    n_states = len(R_sum)
    n_transitions = sum(sum(arr[0] for arr in d.values()) for d in T_count.values())
    print(f"  {n_states} unique states, {n_transitions} transitions  ({time.time()-t0:.1f}s)")

    print(f"value iteration (gamma={args.gamma}, n_iter={args.n_iter}) ...")
    t0 = time.time()
    V, pi = value_iteration(R_sum, R_count, T_count,
                             n_iter=args.n_iter, gamma=args.gamma)
    print(f"  done in {time.time()-t0:.1f}s — policy table has {len(pi)} entries")

    # Action distribution in π*
    act_counts = defaultdict(int)
    for s, a in pi.items(): act_counts[a] += 1
    print(f"\nlearned policy action distribution over visited states:")
    for ai, frac in enumerate(ACTIONS):
        print(f"  sell {frac:>5.3f}: {act_counts[ai]:>6d} states ({100*act_counts[ai]/len(pi):.1f}%)")

    if args.save_policy:
        with open(ROOT / args.save_policy, "wb") as f:
            pickle.dump({"V": dict(V), "pi": dict(pi),
                          "RET_BINS": RET_BINS, "RUNMAX_BINS": RUNMAX_BINS,
                          "PREC_BINS": PREC_BINS, "DTS_BINS": DTS_BINS,
                          "NSOLD_BINS": NSOLD_BINS, "ACTIONS": ACTIONS}, f)
        print(f"saved policy to {ROOT / args.save_policy}")

    # ---- Evaluation ----
    sk_eval = pd.read_parquet(eval_dir / "path_snapshots.parquet")
    tk_eval = pd.read_parquet(eval_dir / "token_level.parquet").set_index("mint")
    sk_eval = sk_eval.sort_values(["mint", "fwd"])
    sk_eval_groups = {m: g for m, g in sk_eval.groupby("mint", sort=False)}

    # Real p_rec for eval (separate computation — eval set ≠ train set on OOS)
    p_rec_eval_map = None
    if args.use_recovery_model:
        model_path = (ROOT / (args.recovery_model
                              or "bot_artifacts_K7V/recovery_model.pkl"))
        print(f"computing real p_rec for eval ({eval_dir.name}) ...")
        t0 = time.time()
        p_rec_eval = compute_real_prec(sk_eval, tk_eval, model_path)
        # Build a (mint, snap_index_within_pool) -> p_rec map. The snap index is
        # the row offset within each mint's sorted-by-fwd snap list, matching
        # what ReplayContext sees inside policy_rl_lookup.
        p_rec_eval_map = {}
        for m, g in p_rec_eval.groupby("mint", sort=False):
            ps = g["p_rec"].tolist()
            for i, p in enumerate(ps):
                p_rec_eval_map[(m, i)] = float(p)
        print(f"  built ({len(p_rec_eval_map)} lookups, {time.time()-t0:.1f}s)")

    rl_policy_fn = policy_rl_lookup(pi, p_rec_per_snap=p_rec_eval_map)
    policies = [
        ("K_combined (LIVE)",            lambda *a, mint=None: policy_k_combined(*a)),
        ("H_time_spaced 15s",            lambda *a, mint=None: policy_time_spaced(*a, gap_s=15.0)),
        ("Calibrated soft-cut",          lambda *a, mint=None: policy_calibrated_soft_cut(*a)),
        ("RL π* (FQI)",                  rl_policy_fn),
    ]
    results = {n: [] for n, _ in policies}
    n_evaluated = 0
    for m in tk_eval.index:
        if m not in sk_eval_groups: continue
        g = sk_eval_groups[m]
        if len(g) < 2: continue
        vsK = float(tk_eval.at[m, "vsK"]) if "vsK" in tk_eval.columns else None
        vtK = float(tk_eval.at[m, "vtK"]) if "vtK" in tk_eval.columns else None
        if vsK is None or vtK is None or vsK <= 0 or vtK <= 0: continue
        snaps = list(zip(g["vs"].tolist(), g["vt"].tolist()))
        dts = g["dts"].tolist()
        vsC, vtC = snaps[-1]
        n_evaluated += 1
        for name, fn in policies:
            try:
                pl = fn(vsK, vtK, vsC, vtC, snaps, dts, mint=m)
                if pl is not None: results[name].append(pl)
            except Exception: pass

    print(f"\neval: {n_evaluated} mints replayed under each policy")
    print(f"\n{'policy':32s} {'n':>5s} {'mean':>9s} {'median':>9s} {'p25':>9s} {'win%':>6s} {'best':>8s} {'worst':>8s}   uplift_vs_K")
    print("-" * 110)
    ref_mean = next((np.mean(rs) for n, _ in policies if n.startswith("K_combined") for rs in [results[n]] if rs), 0.0)
    for name, _ in policies:
        rs = results[name]
        if not rs: continue
        s = _stats(name, rs)
        upl = s["mean"] - ref_mean
        marker = "  <-- BEATS" if upl > 0.005 else ("  (worse)" if upl < -0.005 else "")
        print(f"{name:32s} {s['n']:>5d} {s['mean']:>+9.4f} {s['median']:>+9.4f} "
              f"{s['p25']:>+9.4f} {s['win_pct']:>5.1f}% {s['best']:>+8.3f} {s['worst']:>+8.3f}   "
              f"{upl:+.4f}{marker}")


if __name__ == "__main__":
    main()
