"""b_frontload: sell on every profitable snap, no time gap.

Aggressive scale-out — captures lots of small wins but tail-truncated.
Beats K_combined on win-rate (37%) but loses on best-trade ceiling.
"""
from __future__ import annotations
from .base import ExitPolicy, ExitDecision, HarnessConsts, register


@register("b_frontload")
class BFrontloadPolicy(ExitPolicy):
    def decide(self, mint, n_sold, last_slice_t, now, pf, run_max, p_rec, fwd_n,
               consts: HarnessConsts) -> ExitDecision:
        if pf["ret"] > 0 and n_sold < consts.max_slices:
            return ExitDecision(action="slice", phase="frontload",
                                frac=1.0 / (consts.max_slices - n_sold))
        return ExitDecision("hold")
