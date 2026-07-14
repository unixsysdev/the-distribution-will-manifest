"""rl_layered: mixture-of-experts FQI exit policy.

At entry: predict P(Q5 moonshot) from the 22 V+K7 entry features via a trained
HGB classifier. If P(Q5) >= threshold (default 0.20), use the HOLDER policy
(trained on full data, optimized for catching 10x+ runners). Otherwise use
the DISCRIMINATOR policy (trained on Q1-Q4 only, optimized for aggressive
scale-out on boring tokens).

Per-snap: bin state (ret, run_max, p_rec, dts_since_entry, n_sold), look up
greedy action in the selected π* table. Action ∈ {0, 0.125, 0.25, 0.50, 1.0}
of REMAINING. Action 0.0 = hold; 1.0 = sell_all.

Validated OOS (96K mints):
  K_combined mean   = -0.062
  RL π* layered     = -0.015   (uplift +0.046 SOL/bet)
  oracle ceiling    = +0.126   (peak-routed)

Artifacts loaded from cfg.exit.rl_artifact_dir (default
bot_artifacts_K7V_rl_layered/):
    pi_disc.pkl         — discriminator π* + bin edges + action set
    pi_hold.pkl         — holder π* (must use same bin edges)
    q5_classifier.pkl   — HGB clf + feature names + routing threshold
"""
from __future__ import annotations
import pickle
import time
from pathlib import Path

import numpy as np

from .base import ExitPolicy, ExitDecision, HarnessConsts, register


@register("rl_layered")
class RLLayeredPolicy(ExitPolicy):

    def __init__(self, cfg, **kw):
        super().__init__(cfg, **kw)
        artifact_dir = Path(getattr(cfg.exit, "rl_artifact_dir",
                                    "bot_artifacts_K7V_rl_layered"))
        # If not absolute, resolve relative to project root (where bot runs)
        if not artifact_dir.is_absolute():
            artifact_dir = Path.cwd() / artifact_dir
        with open(artifact_dir / "pi_disc.pkl", "rb") as f:
            self._disc = pickle.load(f)
        with open(artifact_dir / "pi_hold.pkl", "rb") as f:
            self._hold = pickle.load(f)
        with open(artifact_dir / "q5_classifier.pkl", "rb") as f:
            self._q5 = pickle.load(f)
        # Both policy tables must share the same bin edges + actions. We assume
        # that and warn loudly if not.
        for key in ("RET_BINS", "RUNMAX_BINS", "PREC_BINS", "DTS_BINS",
                    "NSOLD_BINS", "ACTIONS"):
            d_arr = self._disc[key]; h_arr = self._hold[key]
            if not np.array_equal(d_arr, h_arr):
                raise ValueError(f"pi_disc and pi_hold disagree on {key} — re-train "
                                 f"both with the same rl_backtest invocation")
        self.RET_BINS    = self._disc["RET_BINS"]
        self.RUNMAX_BINS = self._disc["RUNMAX_BINS"]
        self.PREC_BINS   = self._disc["PREC_BINS"]
        self.DTS_BINS    = self._disc["DTS_BINS"]
        self.NSOLD_BINS  = self._disc["NSOLD_BINS"]
        self.ACTIONS     = self._disc["ACTIONS"]
        self.pi_disc     = self._disc["pi"]
        self.pi_hold     = self._hold["pi"]
        self.clf         = self._q5["clf"]
        self.q5_features = self._q5["features"]   # ordered list of 22 names
        self.q5_threshold = float(getattr(cfg.exit, "rl_q5_threshold",
                                          self._q5.get("threshold", 0.20)))
        print(f"[rl_layered] loaded from {artifact_dir}/")
        print(f"[rl_layered]   pi_disc states={len(self.pi_disc)}  "
              f"pi_hold states={len(self.pi_hold)}  "
              f"q5_features={len(self.q5_features)}  threshold={self.q5_threshold:.3f}")

    # ---------- state binning ----------
    def _bin(self, ret: float, runmax: float, prec: float,
              dts_s: float, n_sold: int) -> tuple:
        def _s(arr, x, hi_default):
            i = int(np.searchsorted(arr, x, side="right") - 1)
            return max(0, min(len(arr) - 2, i)) if hi_default is None else max(0, min(hi_default, i))
        i_ret    = _s(self.RET_BINS,    ret,    None)
        i_runmax = _s(self.RUNMAX_BINS, runmax, None)
        i_prec   = _s(self.PREC_BINS,   prec,   None)
        i_dts    = _s(self.DTS_BINS,    dts_s,  None)
        i_nsold  = _s(self.NSOLD_BINS,  n_sold, None)
        return (i_ret, i_runmax, i_prec, i_dts, i_nsold)

    # ---------- lifecycle ----------
    def on_entry(self, mint, ev, entry_features, score):
        # Build feature vector in the same column order the classifier expects
        x = np.array([float(entry_features.get(f, 0.0)) for f in self.q5_features],
                     dtype=np.float32).reshape(1, -1)
        try:
            p_q5 = float(self.clf.predict_proba(x)[0, 1])
        except Exception as e:
            p_q5 = 0.0
        route = "hold" if p_q5 >= self.q5_threshold else "disc"
        self.per_mint[mint] = {
            "route": route,
            "p_q5":  p_q5,
            "entry_t": time.time(),
        }

    def decide(self, mint, n_sold, last_slice_t, now, pf, run_max, p_rec, fwd_n,
               consts: HarnessConsts) -> ExitDecision:
        st = self.per_mint.get(mint)
        if st is None:
            # We somehow didn't see on_entry — fall back to hold (safe)
            return ExitDecision("hold", reason="no_entry_state")
        # dts measured as wall-clock since entry. The May parquets used
        # "dts since K7 trigger" which is effectively the same: entry happens
        # 1 snap after K7 trigger, and entry_t ~= snap-0 wall clock.
        dts_s = now - st["entry_t"]
        s = self._bin(pf["ret"], run_max, p_rec, dts_s, n_sold)
        pi_table = self.pi_hold if st["route"] == "hold" else self.pi_disc
        action_idx = pi_table.get(s, 0)    # default: hold on unseen state
        frac = float(self.ACTIONS[action_idx])
        if frac <= 0.0:
            return ExitDecision("hold")
        if frac >= 0.999:
            return ExitDecision(action="sell_all",
                                phase=f"rl_{st['route']}_kill",
                                reason="pi*=1.0",
                                extra={"p_q5": st["p_q5"], "route": st["route"],
                                        "state_bins": list(s)})
        return ExitDecision(action="slice",
                            phase=f"rl_{st['route']}",
                            frac=frac,
                            extra={"p_q5": st["p_q5"], "route": st["route"],
                                    "state_bins": list(s)})
