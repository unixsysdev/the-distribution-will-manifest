"""SPSC shared-memory ring roundtrip. Uses a throwaway segment name so the
live 'pumpfun_intents' ring is never touched."""
import os

import base58
import pytest

from intent_ring import IntentRingReader, IntentRingWriter, RECORD_SIZE


def _fake_intent(slot: int, tip: int = 0, is_buy: bool = True, spoof: bool = False) -> dict:
    key = base58.b58encode(bytes(32)).decode()
    return {
        "mint": key,
        "user": key,
        "slot": slot,
        "token_amount": 123,
        "sol_limit_lam": 456_000_000,
        "priority_fee_micro": 789,
        "jito_tip_lam": tip,
        "cu_limit": 100_000,
        "is_buy": is_buy,
        "probable_spoof": spoof,
    }


@pytest.fixture
def ring():
    name = f"pumpbot_test_ring_{os.getpid()}"
    w = IntentRingWriter(name=name, capacity=64)
    r = IntentRingReader(name=name)
    yield w, r
    r.close()
    w.close()


def test_roundtrip_fields(ring):
    w, r = ring
    w.write(_fake_intent(slot=11, tip=1_000_000, is_buy=True, spoof=True))
    out = r.poll(max_records=10)
    assert len(out) == 1
    it = out[0]
    assert it["slot"] == 11
    assert it["jito_tip_lam"] == 1_000_000
    assert it["is_buy"] is True
    assert it["probable_spoof"] is True
    assert it["sol_limit_lam"] == 456_000_000
    assert it["max_sol_cost"] == 456_000_000  # buy-direction alias


def test_reader_cursor_advances(ring):
    w, r = ring
    for i in range(5):
        w.write(_fake_intent(slot=i))
    assert len(r.poll(max_records=100)) == 5
    assert r.poll(max_records=100) == []  # nothing new
    w.write(_fake_intent(slot=99))
    out = r.poll(max_records=100)
    assert len(out) == 1 and out[0]["slot"] == 99


def test_record_size_contract():
    assert RECORD_SIZE == 120  # v3 layout; bump VERSION if this ever changes
