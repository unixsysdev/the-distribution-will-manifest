"""ShredWindow signal/feature semantics: the 2026-06-10 additions (visible-tip
p90 for the adaptive Jito bump, presence flags for intent features so missing
coverage is distinguishable from quiet) plus drain_now safety without a ring."""
import time

from shred_window import ShredWindow

MINT = "So11111111111111111111111111111111111111112"


def _rec(now_ns: int, age_ms: float, tip: int = 0, is_buy: bool = True,
         user: str = "u1", prio: int = 0) -> dict:
    return {
        "recv_ns": int(now_ns - age_ms * 1e6),
        "is_buy": is_buy,
        "user": user,
        "sol_limit_lam": 100_000_000,
        "jito_tip_lam": tip,
        "priority_fee_micro": prio,
        "probable_spoof": False,
    }


def test_signal_tip_p90_and_counts():
    sw = ShredWindow(ring_name="unused_test")
    now_ns = time.time_ns()
    dq = sw._by_mint[MINT]
    dq.append(_rec(now_ns, 1800, tip=100_000, user="a"))
    dq.append(_rec(now_ns, 900, tip=1_000_000, user="b"))
    dq.append(_rec(now_ns, 100, tip=0, user="c"))
    sig = sw.signal(MINT, now_ns=now_ns)
    assert sig["shred_buy_2000ms"] == 3
    assert sig["shred_buy_500ms"] == 1
    assert sig["shred_unique_signers_2000ms"] == 3
    # 2 of 3 tipped -> rate 2/3; p90 of nonzero tips ~ the 1M outlier
    assert abs(sig["shred_jito_tip_rate_2000ms"] - 2 / 3) < 1e-9
    assert sig["shred_jito_tip_p90_2000ms"] == 1_000_000


def test_signal_empty_mint_is_zeroed():
    sw = ShredWindow(ring_name="unused_test")
    sig = sw.signal("missing", now_ns=time.time_ns())
    assert sig["shred_buy_2000ms"] == 0
    assert sig["shred_jito_tip_p90_2000ms"] == 0


def test_intent_features_presence_flags():
    sw = ShredWindow(ring_name="unused_test")
    now_ns = time.time_ns()
    feats = sw.intent_features("missing", now_ns=now_ns)
    for w in ("intent_0p5s", "intent_2p0s", "intent_5p0s"):
        assert feats[f"{w}_present"] == 0.0
        assert feats[f"{w}_n"] == 0.0
    sw._by_mint[MINT].append(_rec(now_ns, 100, tip=5))
    feats = sw.intent_features(MINT, now_ns=now_ns)
    assert feats["intent_0p5s_present"] == 1.0
    assert feats["intent_0p5s_n"] == 1.0


def test_drain_now_without_reader_is_safe():
    sw = ShredWindow(ring_name="unused_test")
    assert sw.drain_now() == 0
