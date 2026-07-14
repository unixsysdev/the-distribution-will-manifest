"""h_time_spaced: 8 slices at fixed runner_min_gap_s, no ret check.

Marginally beats K_combined on mean per bet in OOS A/B replay but with
worse median — sells regardless of profitability so it exits red faster
than K_combined would.
"""
from __future__ import annotations
from .base import ExitPolicy, ExitDecision, HarnessConsts, register


@register("h_time_spaced")
class HTimeSpacedPolicy(ExitPolicy):
    def decide(self, mint, n_sold, last_slice_t, now, pf, run_max, p_rec, fwd_n,
               consts: HarnessConsts) -> ExitDecision:
        if n_sold < consts.max_slices and (now - last_slice_t) >= consts.runner_min_gap_s:
            return ExitDecision(action="slice", phase="time_spaced",
                                frac=1.0 / (consts.max_slices - n_sold))
        return ExitDecision("hold")
