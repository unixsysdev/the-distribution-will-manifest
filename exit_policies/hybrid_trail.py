"""hybrid_trail: paced de-risk (4 slices, ret>0, 5s gap) + trailing stop on
the remainder. When run_max armed (>= runner_min_arm_ret) and a retrace of
retrace_frac happens, sell_all the rest.

Two registered variants:
  c_hybrid_t30   retrace_frac = 0.30
  f_hybrid_t50   retrace_frac = 0.50
"""
from __future__ import annotations
from .base import ExitPolicy, ExitDecision, HarnessConsts, register


class _HybridTrail(ExitPolicy):
    """Shared base. Subclass sets `retrace_frac_override` to fix the trail."""
    retrace_frac_override: float | None = None  # None = use consts.runner_retrace_frac

    def decide(self, mint, n_sold, last_slice_t, now, pf, run_max, p_rec, fwd_n,
               consts: HarnessConsts) -> ExitDecision:
        retrace = (self.retrace_frac_override
                   if self.retrace_frac_override is not None
                   else consts.runner_retrace_frac)
        if n_sold < consts.derisk_slices:
            # paced de-risk (same shape as K_combined)
            if pf["ret"] > 0 and (now - last_slice_t) >= consts.derisk_min_gap_s:
                return ExitDecision(action="slice", phase="derisk",
                                    frac=1.0 / (consts.max_slices - n_sold))
            return ExitDecision("hold")
        # trailing stop on remainder
        if run_max >= consts.runner_min_arm_ret and (1 + run_max) > 0:
            if (run_max - pf["ret"]) / (1 + run_max) >= retrace:
                return ExitDecision(action="sell_all", phase="runner_trail",
                                    reason=f"retrace>={retrace:.2f} after run_max={run_max:.3f}",
                                    extra={"retrace_frac": retrace, "run_max": run_max})
        return ExitDecision("hold")


@register("c_hybrid_t30")
class CHybridT30(_HybridTrail):
    retrace_frac_override = 0.30


@register("f_hybrid_t50")
class FHybridT50(_HybridTrail):
    retrace_frac_override = 0.50


# Back-compat alias for older configs
@register("hybrid_trail")
class HybridTrailDefault(_HybridTrail):
    retrace_frac_override = None  # use consts.runner_retrace_frac (cfg-driven)
