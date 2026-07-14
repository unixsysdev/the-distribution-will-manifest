"""Regression guard for JitoBroker._log_actual_fill: invokes the real method
with a stub self + fake RPC client over a mock landed tx, and asserts a buy and
a sell each emit a 'fill' recon record with correct actual token / lamport deltas
(so real slippage = actual vs expected is captured for the live plumbing test)."""
import asyncio, sys
sys.path.insert(0, "/root/the-distribution-will-manifest")
from solders.signature import Signature
from jito_broker import JitoBroker

OWNER = "So11111111111111111111111111111111111111112"
MINT = "9AYX2NmPw1VLqKpump1111111111111111111111111"


class _A:
    def __init__(self, **kw): self.__dict__.update(kw)


class _Stub:
    user_pk = OWNER
    dry_run = True   # ATA-close hook is live-only; off for the parsing tests
    def __init__(self): self.rec = []; self.realized_pnl_lam = 0; self._buy_cost_lam = {}
    def _recon_log(self, kind, **kw): self.rec.append((kind, kw))
    def _emit_buy_fill(self, *a, **k): pass


def _cli(meta):
    class FakeCli:
        async def get_transaction(self, *a, **k):
            return _A(value=_A(transaction=_A(meta=meta)))
    return FakeCli()


def _run(meta, b):
    s = _Stub()
    asyncio.run(JitoBroker._log_actual_fill(s, _cli(meta), str(Signature.default()), b))
    assert len(s.rec) == 1 and s.rec[0][0] == "fill", s.rec
    return s.rec[0][1]


def test_buy_fill_parsing():
    meta = _A(
        pre_token_balances=[_A(owner=OWNER, mint=MINT, ui_token_amount=_A(amount="0"))],
        post_token_balances=[_A(owner=OWNER, mint=MINT, ui_token_amount=_A(amount="1585561555093"))],
        pre_balances=[200_000_000, 0], post_balances=[84_000_000, 0], fee=5000)
    b = _A(mint=MINT, op="buy", tok_delta=1585561555093, tip_lamports=5_000_000,
           expected_sol_out=0, expected_sol_cost=100_000_000,
           landed_slot=425, target_slot=425, bh_slot=424)
    kw = _run(meta, b)
    assert kw["actual_tok_delta"] == 1585561555093
    assert kw["actual_sol_delta_lam"] == -116_000_000   # spent 0.116 incl fee+tip
    assert kw["fee_lam"] == 5000
    assert kw["slot_gap"] == 0      # landed in the decision slot (same-slot fill)


def test_sell_fill_parsing():
    meta = _A(
        pre_token_balances=[_A(owner=OWNER, mint=MINT, ui_token_amount=_A(amount="1585561555093"))],
        post_token_balances=[_A(owner=OWNER, mint=MINT, ui_token_amount=_A(amount="0"))],
        pre_balances=[84_000_000, 0], post_balances=[257_000_000, 0], fee=5000)
    b = _A(mint=MINT, op="sell_all", tok_delta=-1585561555093, tip_lamports=5_000_000,
           expected_sol_out=173_798_790, expected_sol_cost=0,
           landed_slot=427, target_slot=426, bh_slot=425)
    kw = _run(meta, b)
    assert kw["actual_tok_delta"] == -1585561555093
    assert kw["actual_sol_delta_lam"] == 173_000_000     # received ~0.173
    assert kw["expected_sol_out"] == 173_798_790          # vs expected -> sell slippage
    assert kw["slot_gap"] == 1      # landed one slot after the decision (next-slot fill)


def test_no_meta_is_graceful():
    s = _Stub()
    class NilCli:
        async def get_transaction(self, *a, **k): return _A(value=None)
    b = _A(mint=MINT, op="buy", tok_delta=1, tip_lamports=0, landed_slot=3)
    asyncio.run(JitoBroker._log_actual_fill(s, NilCli(), str(Signature.default()), b))
    assert s.rec and s.rec[0][0] == "fill_no_meta"


class _StubLive:
    """Live-path stub (dry_run False) with a recording _close_ata, to assert the
    rent-reclaim hook fires ONLY when a landed sell fully exits the position."""
    user_pk = OWNER
    dry_run = False
    def __init__(self):
        self.rec = []; self.realized_pnl_lam = 0; self._buy_cost_lam = {}
        self.bg_tasks = set(); self.closed = []
    def _recon_log(self, kind, **kw): self.rec.append((kind, kw))
    async def _close_ata(self, cli, mint): self.closed.append(mint)


def _drain(meta, b):
    s = _StubLive()
    async def go():
        await JitoBroker._log_actual_fill(s, _cli(meta), str(Signature.default()), b)
        for _ in range(3):
            await asyncio.sleep(0)   # let the fire-and-forget close task run
    asyncio.run(go())
    return s


def _bal(pre, post):
    return dict(pre_token_balances=[_A(owner=OWNER, mint=MINT, ui_token_amount=_A(amount=pre))],
                post_token_balances=[_A(owner=OWNER, mint=MINT, ui_token_amount=_A(amount=post))],
                pre_balances=[84_000_000, 0], post_balances=[257_000_000, 0], fee=5000)


def test_close_fires_only_on_full_exit():
    # full exit (held -> 0) on a landed sell  ->  close fires
    s = _drain(_A(**_bal("1585561555093", "0")),
               _A(mint=MINT, op="sell_all", tok_delta=-1585561555093, tip_lamports=0,
                  expected_sol_out=1, expected_sol_cost=0, landed_slot=2))
    assert s.closed == [MINT], s.closed
    # a buy  ->  never closes
    s = _drain(_A(**_bal("0", "1585561555093")),
               _A(mint=MINT, op="buy", tok_delta=1585561555093, tip_lamports=0,
                  expected_sol_out=0, expected_sol_cost=1, landed_slot=1))
    assert s.closed == [], s.closed
    # a partial sell (still holding)  ->  never closes
    s = _drain(_A(**_bal("1585561555093", "500000000000")),
               _A(mint=MINT, op="sell_slice", tok_delta=-1085561555093, tip_lamports=0,
                  expected_sol_out=1, expected_sol_cost=0, landed_slot=3))
    assert s.closed == [], s.closed
