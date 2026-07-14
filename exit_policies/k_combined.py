"""k_combined: 4 paced de-risk (ret>0, 5s gap) + 4 force-spaced runner (15s gap).

This is the LIVE default. See the main A/B replay results in
docs/notes/SHADOW_HARNESS_LOG.md — beats H_time_spaced on win-rate (26 vs 21%) and
matches it on mean per bet within sampling noise.
"""
from __future__ import annotations
from .base import ExitPolicy, ExitDecision, HarnessConsts, register


@register("k_combined")
class KCombinedPolicy(ExitPolicy):
    def decide(self, mint, n_sold, last_slice_t, now, pf, run_max, p_rec, fwd_n,
               consts: HarnessConsts) -> ExitDecision:
        if n_sold < consts.derisk_slices:
            if pf["ret"] > 0 and (now - last_slice_t) >= consts.derisk_min_gap_s:
                return ExitDecision(action="slice", phase="derisk",
                                    frac=1.0 / (consts.max_slices - n_sold))
        else:
            if (now - last_slice_t) >= consts.runner_min_gap_s:
                return ExitDecision(action="slice", phase="runner",
                                    frac=1.0 / (consts.max_slices - n_sold))
        return ExitDecision("hold")
