"""Exit policy registry + level_tp semantics (the live exit family)."""
import pytest

import exit_policies  # noqa: F401  (imports register all policies)
from exit_policies.base import ExitDecision, HarnessConsts, get_policy

CONSTS = HarnessConsts(max_slices=8, derisk_slices=4, derisk_min_gap_s=5.0,
                       runner_min_gap_s=15.0, runner_retrace_frac=0.3,
                       runner_min_arm_ret=0.2, death_threshold=0.1)


def _decide(pol, ret: float) -> ExitDecision:
    return pol.decide("m", 0, 0.0, 0.0, {"ret": ret}, run_max=ret, p_rec=1.0,
                      fwd_n=1, consts=CONSTS)


@pytest.mark.parametrize("name,tp", [
    ("level_tp_50", 0.50), ("level_tp_100", 1.00), ("level_tp_200", 2.00),
])
def test_level_tp_sells_at_level(name, tp):
    pol = get_policy(name, cfg=None, fresh=True)
    assert _decide(pol, tp - 0.01).action == "hold"
    dec = _decide(pol, tp)
    assert dec.action == "sell_all"
    assert dec.phase == "tp_level"


def test_registry_has_all_production_policies():
    for name in ("level_tp_50", "level_tp_100", "level_tp_200",
                 "level_tp_50_stop30_cap120", "level_tp_100_stop30_cap120",
                 "c_hybrid_t30", "f_hybrid_t50", "k_combined",
                 "h_time_spaced", "b_frontload"):
        assert get_policy(name, cfg=None, fresh=True) is not None


def test_unknown_policy_raises_with_listing():
    with pytest.raises(ValueError, match="registered"):
        get_policy("definitely_not_a_policy", cfg=None, fresh=True)
