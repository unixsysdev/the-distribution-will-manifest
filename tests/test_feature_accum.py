"""TokenState trigger + feature contract at the live K=3/V=0.3 operating point.

train==live rests on extraction tools running THIS code over capture data, so
the trigger semantics and the 22-feature vector shape are load-bearing.
"""
import math

from feature_accum import ENTRY_FEATURE_NAMES, K_TRIGGER, V_TRIGGER, TokenState

BASE_VS = 30_000_000_000.0
BASE_VT = 1_000_000_000_000_000.0


def _grow(st_or_none, n_trades: int, sol_each: float, t0: float = 1_000_000.0,
          users=None):
    """Build/extend a TokenState with n buy trades of sol_each SOL."""
    vs, vt = BASE_VS, BASE_VT
    st = st_or_none
    for i in range(n_trades):
        lam = sol_each * 1e9
        vs += lam
        vt -= lam / (vs / vt)  # rough constant-product-ish move; shape only
        user = (users[i] if users else f"user{i}")
        if st is None:
            st = TokenState(vs, vt, sol_each, True, user, t0 + i)
        else:
            st.update(vs, vt, sol_each, True, user, t0 + i)
    return st


def test_env_trigger_values():
    assert K_TRIGGER == 3 and V_TRIGGER == 0.3  # conftest pins the live env


def test_joint_trigger_fires_at_k3_with_volume():
    st = _grow(None, 3, sol_each=0.15)
    assert st.k_fired and st.v_fired  # 3 trades, 0.45 cum buy SOL >= 0.3


def test_insufficient_volume_does_not_fire_v():
    st = _grow(None, 3, sol_each=0.05)  # cum 0.15 < 0.3
    assert st.k_fired and not st.v_fired


def test_feature_vector_contract():
    st = _grow(None, 3, sol_each=0.15)
    feats = st.combined_entry_features()
    assert len(feats) == len(ENTRY_FEATURE_NAMES) == 22
    finite = [f for f in feats if isinstance(f, float) and not math.isnan(f)]
    assert len(finite) >= 20  # vector is essentially fully populated at ready
    # uniq is capped by the K window
    uniq_idx = ENTRY_FEATURE_NAMES.index("uniq")
    assert feats[uniq_idx] <= 3


def test_deterministic():
    a = _grow(None, 3, sol_each=0.15).combined_entry_features()
    b = _grow(None, 3, sol_each=0.15).combined_entry_features()
    assert a == b


def test_forward_tracking_after_ready():
    st = _grow(None, 3, sol_each=0.15)
    st = _grow(st, 2, sol_each=0.2, t0=1_000_010.0)  # two forward trades
    assert st.fwd == 2
    assert st.run_max_ret >= 0.0
