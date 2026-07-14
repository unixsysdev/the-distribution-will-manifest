"""Regression guard for JitoBroker.reconcile_grpc_tx — the gRPC-native
confirmation+fill path (primary since 2026-06-12, replacing the degraded eRPC
getTransaction/getSignatureStatuses). Feeds a fake geyser tx meta and asserts:
a landed buy/sell emits a 'fill' recon record (source=grpc) with correct actual
token/SOL deltas + slot_gap; a reverted tx routes to failure; a foreign sig is a
no-op. Mirrors test_actual_fill but for the feed path."""
import sys
from collections import deque
from types import MethodType
sys.path.insert(0, "/root/the-distribution-will-manifest")
from jito_broker import JitoBroker, PendingBundle

OWNER = "So11111111111111111111111111111111111111112"
MINT = "9AYX2NmPw1VLqKpump1111111111111111111111111"


class _A:
    def __init__(self, **kw): self.__dict__.update(kw)


class _ErrNone:
    err = None            # geyser meta.err.err is None when the tx succeeded


class _ErrSet:
    err = b"InstructionError"


def _meta(pre_tok, post_tok, pre_sol, post_sol, fee=5000, failed=False):
    def tb(amt):
        return _A(owner=OWNER, mint=MINT, ui_token_amount=_A(amount=str(amt)))
    return _A(err=(_ErrSet() if failed else _ErrNone()),
              fee=fee,
              pre_token_balances=[tb(pre_tok)] if pre_tok is not None else [],
              post_token_balances=[tb(post_tok)] if post_tok is not None else [],
              pre_balances=[pre_sol, 0], post_balances=[post_sol, 0])


class _Stub:
    user_pk = OWNER
    dry_run = True
    def __init__(self):
        self.pending_bundles = {}
        self.recent_outcomes = deque(maxlen=500)
        self.realized_pnl_lam = 0
        self._buy_cost_lam = {}
        self.holdings = {}
        self.bg_tasks = set()
        self.failure_callback = None
        self.fill_callback = None
        self.rec = []
        self._emit_buy_fill = MethodType(JitoBroker._emit_buy_fill, self)
        # use the REAL landed/fail handlers (they only touch attrs above)
        self._handle_bundle_landed = MethodType(JitoBroker._handle_bundle_landed, self)
        self._handle_bundle_failed = MethodType(JitoBroker._handle_bundle_failed, self)
        self._fail_or_retry = MethodType(JitoBroker._fail_or_retry, self)
    def _recon_log(self, kind, **kw): self.rec.append((kind, kw))


def _fills(s):
    return [kw for (k, kw) in s.rec if k == "fill"]


def test_grpc_buy_fill():
    s = _Stub()
    s.pending_bundles["sigBUY"] = PendingBundle(
        sig="sigBUY", mint=MINT, op="buy", tok_delta=1585561555093,
        tip_lamports=30000, submitted_t=0.0, expected_sol_cost=100_000_000,
        target_slot=425, bh_slot=424)
    # bought 1.585e12 tokens; wallet SOL 0.2 -> 0.084 (spent 0.116 incl fee+tip)
    meta = _meta(pre_tok=0, post_tok=1585561555093, pre_sol=200_000_000, post_sol=84_000_000)
    out = JitoBroker.reconcile_grpc_tx(s, "sigBUY", 425, meta)
    assert out is True
    assert "sigBUY" not in s.pending_bundles
    f = _fills(s); assert len(f) == 1, s.rec
    assert f[0]["source"] == "grpc"
    assert f[0]["actual_tok_delta"] == 1585561555093
    assert f[0]["actual_sol_delta_lam"] == -116_000_000
    assert f[0]["slot_gap"] == 0                       # landed in decision slot
    assert s._buy_cost_lam[MINT] == -116_000_000       # buy cost recorded for P&L pairing


def test_grpc_buy_fill_emits_anchor():
    # a landed buy must hand the harness our realized fill mid for fill-anchored exits
    s = _Stub()
    got = []
    s.fill_callback = lambda m, fm: got.append((m, fm))
    s.pending_bundles["sigBUY"] = PendingBundle(
        sig="sigBUY", mint=MINT, op="buy", tok_delta=638487865452,
        tip_lamports=30000, submitted_t=0.0, target_slot=425, bh_slot=424)
    meta = _meta(pre_tok=0, post_tok=638487865452, pre_sol=200_000_000, post_sol=84_000_000)
    JitoBroker.reconcile_grpc_tx(s, "sigBUY", 425, meta)
    assert len(got) == 1 and got[0][0] == MINT
    # curve_in = |−116M| − tip 30k − fee 5k = 115,965,000 ; fill_mid = curve_in/tokens
    assert abs(got[0][1] - 115_965_000 / 638487865452) < 1e-12


def test_grpc_sell_fill_realizes_pnl():
    s = _Stub()
    s._buy_cost_lam[MINT] = -116_000_000               # prior buy cost
    s.pending_bundles["sigSELL"] = PendingBundle(
        sig="sigSELL", mint=MINT, op="sell_all", tok_delta=-1585561555093,
        tip_lamports=30000, submitted_t=0.0, expected_sol_out=173_000_000,
        target_slot=426, bh_slot=425)
    # sold all; wallet SOL 0.084 -> 0.257 (received 0.173)
    meta = _meta(pre_tok=1585561555093, post_tok=0, pre_sol=84_000_000, post_sol=257_000_000)
    out = JitoBroker.reconcile_grpc_tx(s, "sigSELL", 428, meta)
    assert out is True
    f = _fills(s); assert len(f) == 1
    assert f[0]["actual_sol_delta_lam"] == 173_000_000
    assert f[0]["slot_gap"] == 2                        # 428 - 426
    # realized P&L = sell proceeds + buy cost = +173M + (-116M) = +57M lamports
    assert s.realized_pnl_lam == 57_000_000
    assert MINT not in s._buy_cost_lam                  # popped after pairing


def test_grpc_reverted_tx_no_fill():
    s = _Stub()
    s.holdings[MINT] = 1585561555093
    s.pending_bundles["sigREV"] = PendingBundle(
        sig="sigREV", mint=MINT, op="buy", tok_delta=1585561555093,
        tip_lamports=30000, submitted_t=0.0)
    meta = _meta(pre_tok=0, post_tok=0, pre_sol=200_000_000, post_sol=199_995_000, failed=True)
    out = JitoBroker.reconcile_grpc_tx(s, "sigREV", 430, meta)
    assert out is True
    assert _fills(s) == []                              # reverted -> no fill
    assert any(k == "failed" for (k, _) in s.rec), s.rec
    assert s.holdings.get(MINT, 0) == 0                 # buy reservation rolled back


def test_grpc_foreign_sig_noop():
    s = _Stub()
    meta = _meta(pre_tok=0, post_tok=1, pre_sol=1, post_sol=1)
    out = JitoBroker.reconcile_grpc_tx(s, "not_ours", 1, meta)
    assert out is False
    assert s.rec == []
