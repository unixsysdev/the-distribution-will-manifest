"""lsm_continuation: optimal-stopping policy via learned continuation value.

At each snap, predicts the conditional expected terminal return given the
full state (ret, run_max, velocity, acceleration, flow, single-actor share,
fill_k, entry_score). Sells iff current ret already exceeds that expectation
plus a small margin epsilon.

Rule:
    sell_all  iff  ret_now  >  f(s_t) + epsilon

This is the Longstaff-Schwartz approximation of the Bellman value function
applied to the realized path empirical distribution. It does NOT assume a
model of token dynamics — it learns the conditional expectation from
realized paths.

Subsumes:
  - level_tp_X    — when E[future | high ret] is consistently low (spike-collapse)
  - trailing stop — when E[future | high run_max, falling ret] is low
  - death cut     — when E[future | low ret, low p_rec, decelerating flow] is very negative

Maintains per-mint snap history so velocity / acceleration can be computed
on-line during live operation.

Loads bot_artifacts_K7V/lsm_continuation.pkl trained by tools/train_lsm_continuation.py.
"""
from __future__ import annotations
import pickle
import time
from pathlib import Path
import numpy as np

from .base import ExitPolicy, ExitDecision, HarnessConsts, register


@register("lsm_continuation")
class LSMContinuationPolicy(ExitPolicy):

    def __init__(self, cfg, **kw):
        super().__init__(cfg, **kw)
        artifact = Path(getattr(cfg.exit, "lsm_artifact",
                                "bot_artifacts_K7V/lsm_continuation.pkl"))
        if not artifact.is_absolute():
            artifact = Path.cwd() / artifact
        with open(artifact, "rb") as f:
            payload = pickle.load(f)
        self.model    = payload["model"]
        self.features = payload["features"]  # ordered list of 15 feature names
        # Decision margin (epsilon). Positive => sell-more-eagerly; negative =>
        # hold longer. Default 0 = exact Bellman boundary.
        self.epsilon = float(getattr(cfg.exit, "lsm_epsilon", 0.0))
        print(f"[lsm_continuation] loaded {artifact}")
        print(f"[lsm_continuation]   n_features={len(self.features)}  "
              f"train_r2={payload.get('train_r2','?')}  "
              f"eval_r2={payload.get('eval_r2','?')}  "
              f"epsilon={self.epsilon}")

    # ---------- lifecycle ----------

    def on_entry(self, mint, ev, entry_features, score):
        self.per_mint[mint] = {
            "entry_score": float(score),
            "ret_hist":  [],   # [(dts, ret)]
            "vel_prev":  None,  # previous velocity_ret_1 (for accel)
            "last_runmax_fwd": None,
            "entry_t": time.time(),
        }

    def _build_state(self, mint_st, pf, run_max, fwd_n) -> np.ndarray | None:
        """Assemble the 15-feature state vector matching training."""
        ret = float(pf.get("ret", 0.0))
        dts = float(pf.get("dts", 0.0))  # seconds since entry; harness should pass this
        fill_k = float(pf.get("fill_k", 0.0))
        buy_frac_w = float(pf.get("buy_frac_w", 0.0))
        nsell_w = float(pf.get("nsell_w", 0.0))
        solo_sell_w = float(pf.get("solo_sell_w", 0.0))
        vel_w = float(pf.get("vel_w", 0.0))

        hist = mint_st["ret_hist"]
        hist.append((dts, ret))
        # cap history to last 10 snaps
        if len(hist) > 10: hist[:] = hist[-10:]

        # velocity_ret_1 = (ret_t - ret_{t-1}) / (dts_t - dts_{t-1})
        vel1 = vel3 = accel = np.nan
        if len(hist) >= 2:
            d_dt = max(hist[-1][0] - hist[-2][0], 1e-6)
            vel1 = (hist[-1][1] - hist[-2][1]) / d_dt
            if mint_st["vel_prev"] is not None:
                accel = vel1 - mint_st["vel_prev"]
            mint_st["vel_prev"] = vel1
        if len(hist) >= 4:
            d_dt3 = max(hist[-1][0] - hist[-4][0], 1e-6)
            vel3 = (hist[-1][1] - hist[-4][1]) / d_dt3

        rm = float(run_max)
        retracement_norm = (rm - ret) / max(1.0 + rm, 1e-6)

        # time_since_run_max in snaps
        if abs(ret - rm) < 1e-9:
            mint_st["last_runmax_fwd"] = fwd_n
        if mint_st["last_runmax_fwd"] is None:
            tsrm = 0
        else:
            tsrm = max(0, fwd_n - mint_st["last_runmax_fwd"])

        dd = ret - rm  # negative

        feat = {
            "ret": ret, "run_max_ret": rm, "dd": dd,
            "retracement_norm": retracement_norm,
            "velocity_ret_1": vel1, "velocity_ret_3": vel3, "accel_ret": accel,
            "time_since_run_max": float(tsrm),
            "dts": dts, "fill_k": fill_k,
            "buy_frac_w": buy_frac_w, "nsell_w": nsell_w,
            "solo_sell_w": solo_sell_w, "vel_w": vel_w,
            "entry_score": mint_st["entry_score"],
        }
        return np.array([feat[k] for k in self.features], dtype=np.float64).reshape(1, -1)

    def decide(self, mint, n_sold, last_slice_t, now, pf, run_max, p_rec, fwd_n,
               consts: HarnessConsts) -> ExitDecision:
        st = self.per_mint.get(mint)
        if st is None:
            return ExitDecision("hold", reason="no_entry_state")
        x = self._build_state(st, pf, run_max, fwd_n)
        if x is None:
            return ExitDecision("hold")
        try:
            yhat = float(self.model.predict(x)[0])
        except Exception:
            return ExitDecision("hold")
        ret_now = float(pf.get("ret", 0.0))
        if ret_now > yhat + self.epsilon:
            return ExitDecision(action="sell_all",
                                phase="lsm",
                                reason=f"ret={ret_now:.3f}>E[future]={yhat:.3f}+eps={self.epsilon}",
                                extra={"yhat": yhat, "ret": ret_now})
        return ExitDecision("hold",
                            extra={"yhat": yhat, "ret": ret_now,
                                   "gap": ret_now - yhat - self.epsilon})
