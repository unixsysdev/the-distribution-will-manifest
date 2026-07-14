"""Shred-intent pipeline surface (canonical: shred_bot/*).

NOTE: intent_ring + intent_recorder are in the COLLECTOR FROZEN SET
(tests/test_collector_freeze.py). Read-side use only from here: never
instantiate IntentRingWriter against the live 'pumpfun_intents' segment;
the single producer is pumpfun-shred-intents.service.
"""
from ._lazy import make_lazy

make_lazy(__name__, {
    "ShredWindow": ("shred_window", "ShredWindow"),
    "IntentRingReader": ("intent_ring", "IntentRingReader"),
    "IntentRingWriter": ("intent_ring", "IntentRingWriter"),
})
