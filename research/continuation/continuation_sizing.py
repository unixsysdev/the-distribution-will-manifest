"""continuation_sizing.py — SOL-in-disciplined buy sizing + honest cost-basis
accounting for the continuation executor (2026-06-14).

WHY THIS EXISTS
The live launch-sniper sized buys exact-OUT: it committed to a fixed token_amount
(computed off a STALE decision-time reserve snapshot) and set
    max_sol_cost = bet * (1 + slippage_bps_buy/10000),   slippage_bps_buy = 20000 (200%)
so a 0.05 SOL bet was legally allowed to spend up to 0.15, and DID balloon to ~0.10
on fast runners: the price moved up between snapshot and landing, and buying the
stale token count then cost ~2x. Worse, paper_book._close_one reported return as
    net = proceeds / q_lam - 1
i.e. divided by the NOTIONAL bet (0.05) at the SNAPSHOT price, while the wallet
actually spent ~0.10  ->  reported return overstated, real capital-at-risk uncounted.
That is the "mirage" (optimistic entry anchor) realised as actual overspend.

WHAT THIS MODULE DOES
1. Makes `bet` mean MAX SOL AT RISK with a hard invariant: curve spend <= bet, ALWAYS.
   We set max_sol_cost = bet and size the token request for the CAPPED price, so a
   fill that drifts up to +cap% from our quote still fills (for fewer tokens), and a
   fill beyond +cap% REVERTS rather than overspends.
2. Realises slip as FEWER TOKENS at a higher entry price — the exact-in model the
   +0.14 edge was measured under — never as more SOL out the door.
3. Computes return on ACTUAL total outlay (curve cost + pump 1% + tip + base fee),
   never on the notional or the entry snapshot. Also reports the curve-only price
   return for parity with continuation_shadow's `ret`.

TWO SLIPS, DO NOT CONFLATE THEM
  - token momentum slip  (fill_mid/cross_mid - 1): a property of the TOKEN, measured
    passively in the shadow. High = strong continuation = the SIGNAL. Used to SELECT
    which token to buy (model features). NEVER gate selection on it (proven backfire).
  - our execution slip   (fill_price/our_quote - 1): a property of OUR latency. High =
    we landed late (lost the gap-0 race). The max_sol_cost cap bounds THIS — it both
    caps overspend and doubles as a latency filter (reverts the races we lost badly).
Selection and execution are separate stages; this module governs execution only.

Pure integer-lamport / raw-token math; matches pump_fun_ix on-chain rounding.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

LAMPORTS_PER_SOL = 1_000_000_000
PUMP_FEE_BPS = 100          # pump.fun ~1% trade fee, charged each side (configurable)
BASE_TX_FEE_LAM = 5_000     # ~base signature fee per tx
DEFAULT_CAP_BPS = 2500      # 25% execution-slip / overspend cap (tune from live gap data)


# ---------- constant-product AMM primitives (match pump_fun_ix integer rounding) ----------
def tokens_out_for_sol(vsol: int, vtok: int, sol_in: int) -> int:
    """Tokens received for `sol_in` lamports (buy). vtok*ds/(vsol+ds), floored."""
    if vsol <= 0 or vtok <= 0 or sol_in <= 0:
        return 0
    return int(vtok - (vsol * vtok) // (vsol + sol_in))


def sol_out_for_tokens(vsol: int, vtok: int, tok_in: int) -> int:
    """Lamports received for selling `tok_in` tokens. vsol*tk/(vtok+tk), floored."""
    if vsol <= 0 or vtok <= 0 or tok_in <= 0:
        return 0
    return int(vsol - (vsol * vtok) // (vtok + tok_in))


def sol_cost_for_tokens(vsol: int, vtok: int, tok_out: int):
    """Minimal lamports to BUY `tok_out` tokens (inverse of tokens_out_for_sol).
    Rounds up (curve-favourable), matching how the chain charges. None if infeasible."""
    if vsol <= 0 or vtok <= 0 or tok_out <= 0 or tok_out >= vtok:
        return None
    new_vsol = -(-(vsol * vtok) // (vtok - tok_out))   # ceil(vsol*vtok / (vtok - tok_out))
    return int(new_vsol - vsol)


# ---------- buy planning ----------
@dataclass
class BuyPlan:
    token_amount: int        # exact-out token request (raw units)
    max_sol_cost_lam: int    # HARD cap on curve spend (= bet); chain reverts above this
    bet_lam: int             # the intended bet (max SOL at risk in the token)
    cap_bps: int             # execution-slip tolerance we sized for
    ref_curve_cost_lam: int  # curve cost at the fresh quote (spend if zero drift)


def _reserves_after_drift(vsol: int, vtok: int, cap_bps: int):
    """Reserves once the price (vsol/vtok) has risen by cap_bps, k held constant.
    Closed form: vsol*sqrt(1+cap), vtok/sqrt(1+cap)."""
    r = math.sqrt(1.0 + cap_bps / 10_000)
    return int(vsol * r), int(vtok / r)


def plan_buy(vsol: int, vtok: int, bet_lam: int, cap_bps: int = DEFAULT_CAP_BPS) -> BuyPlan:
    """Size an exact-out buy so curve spend can NEVER exceed `bet_lam`, while filling iff
    the price we land at has drifted no more than +cap_bps from our fresh quote.

    We request the token count that costs EXACTLY `bet` at the capped price (reserves
    after a +cap_bps move), and set max_sol_cost = bet. Then: below the cap that token
    count costs < bet (fills, under-deploying slightly); at the cap it costs bet (fills);
    above the cap it costs > bet and the chain reverts. So `cap_bps` is the precise fill
    ceiling AND curve spend is hard-bounded by bet — both, exactly. Sizing against the
    capped reserves (not the fresh ones) accounts for the curve's convexity so the knob
    means what it says.
    """
    assert bet_lam > 0 and cap_bps >= 0
    vs_cap, vt_cap = _reserves_after_drift(vsol, vtok, cap_bps)
    tok = tokens_out_for_sol(vs_cap, vt_cap, bet_lam)            # costs `bet` at the cap price
    ref_cost = sol_cost_for_tokens(vsol, vtok, tok) or bet_lam   # what it costs at zero drift
    return BuyPlan(token_amount=tok, max_sol_cost_lam=bet_lam, bet_lam=bet_lam,
                   cap_bps=cap_bps, ref_curve_cost_lam=ref_cost)


def legacy_plan_buy(vsol: int, vtok: int, bet_lam: int, slippage_bps_buy: int = 20_000) -> BuyPlan:
    """The OLD behaviour, for contrast in the self-test: token count sized at the
    snapshot for the FULL bet, max_sol_cost = bet*(1+slip). With slip=20000 (200%)
    the chain happily lets a runner cost up to 3x the bet."""
    tok = tokens_out_for_sol(vsol, vtok, bet_lam)
    max_cost = bet_lam * (10_000 + slippage_bps_buy) // 10_000
    ref_cost = sol_cost_for_tokens(vsol, vtok, tok) or bet_lam
    return BuyPlan(token_amount=tok, max_sol_cost_lam=max_cost, bet_lam=bet_lam,
                   cap_bps=slippage_bps_buy, ref_curve_cost_lam=ref_cost)


# ---------- fill realisation ----------
@dataclass
class FillResult:
    filled: bool
    curve_cost_lam: int      # SOL into the token (recoverable by selling)
    pump_fee_lam: int        # ~1% pump fee on entry
    tip_lam: int             # Jito tip (small; on pump.fun the race is won mostly via priority fee)
    priority_fee_lam: int    # ComputeBudget priority fee on the buy (the real race lever)
    base_fee_lam: int
    total_outlay_lam: int    # everything that left the wallet on entry
    fill_price: float        # curve_cost / token_amount (lamports per raw token)
    exec_slip: float         # curve_cost/ref_curve_cost - 1  (OUR execution slip)
    revert_reason: str | None


def simulate_fill(vsol_fill: int, vtok_fill: int, plan: BuyPlan, tip_lam: int = 0,
                  priority_fee_lam: int = 0, base_fee_lam: int = BASE_TX_FEE_LAM) -> FillResult:
    """Model the on-chain buy at the reserves we actually LAND in. Reverts if buying
    plan.token_amount would exceed max_sol_cost (we landed too late / too much drift).
    Entry cost = curve + pump(1%) + priority_fee + tip + base. On pump.fun the race is won
    mostly by the PRIORITY FEE (and speed), so the tip is small and the priority fee is the lever."""
    cost = sol_cost_for_tokens(vsol_fill, vtok_fill, plan.token_amount)
    if cost is None or cost > plan.max_sol_cost_lam:
        return FillResult(False, 0, 0, 0, 0, 0, 0, float("nan"), float("nan"),
                          revert_reason=f"curve_cost>{plan.max_sol_cost_lam} (drift beyond cap)")
    pump_fee = cost * PUMP_FEE_BPS // 10_000
    total = cost + pump_fee + tip_lam + priority_fee_lam + base_fee_lam
    return FillResult(True, cost, pump_fee, tip_lam, priority_fee_lam, base_fee_lam, total,
                      cost / plan.token_amount, cost / plan.ref_curve_cost_lam - 1.0, None)


# ---------- return accounting (always on ACTUAL outlay) ----------
@dataclass
class TradeResult:
    net_pnl_lam: int
    return_on_outlay: float   # net_pnl / entry total outlay  <- THE honest number
    return_on_curve: float    # exit_curve / entry_curve - 1  <- parity with shadow `ret`
    exit_proceeds_lam: int


def realized_return(fill: FillResult, plan: BuyPlan, vsol_exit: int, vtok_exit: int, exit_tip_lam: int = 0,
                    priority_fee_lam: int = 0, base_fee_lam: int = BASE_TX_FEE_LAM) -> TradeResult:
    """Sell plan.token_amount into the exit reserves; net against the ACTUAL entry outlay.
    Exit cost = pump(1%) + priority_fee + (small/zero) exit_tip + base. return_on_outlay is the
    real SOL return on what we actually risked."""
    gross_out = sol_out_for_tokens(vsol_exit, vtok_exit, plan.token_amount)
    pump_fee_exit = gross_out * PUMP_FEE_BPS // 10_000
    proceeds = gross_out - pump_fee_exit - exit_tip_lam - priority_fee_lam - base_fee_lam
    net = proceeds - fill.total_outlay_lam
    return TradeResult(net_pnl_lam=net,
                       return_on_outlay=net / fill.total_outlay_lam,
                       return_on_curve=gross_out / fill.curve_cost_lam - 1.0,
                       exit_proceeds_lam=proceeds)


# ---------- self-test ----------
if __name__ == "__main__":
    import math

    def reserves_at_price_mult(vsol: int, vtok: int, mult: float):
        """Reserves after the price (vsol/vtok) has moved by `mult`, k constant."""
        k = vsol * vtok
        price = (vsol / vtok) * mult
        vs = int(round(math.sqrt(k * price)))
        vt = int(round(math.sqrt(k / price)))
        return vs, vt

    SOL = LAMPORTS_PER_SOL
    # realistic pump.fun reserves at ~2x from launch (the milestone cross)
    VSOL0, VTOK0 = 42_400_000_000, 758_800_000_000_000     # ~42.4 SOL, ~7.59e14 raw
    BET = int(0.05 * SOL)
    TIP = int(0.005 * SOL)

    def s(lam): return f"{lam / SOL:.4f}"

    print(f"setup: cross reserves vsol={s(VSOL0)} SOL  bet={s(BET)}  cap={DEFAULT_CAP_BPS}bps  tip={s(TIP)}\n")

    # --- LEGACY vs NEW on a +100% runner (price doubled between our quote and landing) ---
    vsR, vtR = reserves_at_price_mult(VSOL0, VTOK0, 2.0)
    leg = legacy_plan_buy(VSOL0, VTOK0, BET)
    legf = simulate_fill(vsR, vtR, leg, tip_lam=TIP)
    new = plan_buy(VSOL0, VTOK0, BET)
    newf = simulate_fill(vsR, vtR, new, tip_lam=TIP)
    print("+100% runner (we landed very late):")
    print(f"  LEGACY: filled={legf.filled}  curve_spend={s(legf.curve_cost_lam)}  "
          f"(cap was {s(leg.max_sol_cost_lam)}) -> overspends to ~2x the bet")
    print(f"  NEW:    filled={newf.filled}  reason={newf.revert_reason}  "
          f"-> correctly DECLINES the lost race (no overspend)\n")
    assert legf.filled and legf.curve_cost_lam > int(0.09 * SOL), "legacy should balloon to ~0.10"
    assert not newf.filled, "new must revert a +100% late fill, not overspend"

    # --- NEW: hard-cap invariant across drift scenarios ---
    print("NEW sizing across execution-slip scenarios (invariant: curve_spend <= bet):")
    print(f"  {'drift':>6} {'filled':>7} {'curve_spend':>12} {'tokens':>16} {'exec_slip':>10}")
    for mult in (1.00, 1.10, 1.20, 1.30, 1.45):
        vs, vt = reserves_at_price_mult(VSOL0, VTOK0, mult)
        f = simulate_fill(vs, vt, new, tip_lam=TIP)
        if f.filled:
            assert f.curve_cost_lam <= new.bet_lam, "INVARIANT BREACH: curve spend exceeded bet"
            print(f"  {mult-1:>+5.0%} {'yes':>7} {s(f.curve_cost_lam):>12} "
                  f"{new.token_amount:>16d} {f.exec_slip:>+9.1%}")
        else:
            print(f"  {mult-1:>+5.0%} {'REVERT':>7} {'-':>12} {'-':>16} {'(beyond cap)':>10}")
    print()

    # --- return accounting on ACTUAL outlay: a +50% winner and a -30% loser ---
    vsE, vtE = reserves_at_price_mult(VSOL0, VTOK0, 1.10)        # we filled at +10% drift
    fill = simulate_fill(vsE, vtE, new, tip_lam=TIP)
    print(f"filled at +10% drift: curve={s(fill.curve_cost_lam)} pump_fee={s(fill.pump_fee_lam)} "
          f"tip={s(fill.tip_lam)} -> total outlay={s(fill.total_outlay_lam)}")
    for name, exit_mult in (("winner +50% from fill", 1.50), ("loser -30% from fill", 0.70)):
        # exit price relative to the FILL price (mult 1.10 was our fill)
        vsx, vtx = reserves_at_price_mult(VSOL0, VTOK0, 1.10 * exit_mult)
        tr = realized_return(fill, new, vsx, vtx, exit_tip_lam=0)
        print(f"  {name:>22}: return_on_curve={tr.return_on_curve:+.3f}  "
              f"return_on_outlay={tr.return_on_outlay:+.3f}  net={s(tr.net_pnl_lam)} SOL")
    # the curve return must track the price move; the outlay return is a touch lower (tip+fees)
    vsW, vtW = reserves_at_price_mult(VSOL0, VTOK0, 1.10 * 1.50)
    win = realized_return(fill, new, vsW, vtW)
    assert 0.45 < win.return_on_curve < 0.55, "winner curve return should be ~+0.50"
    assert win.return_on_outlay < win.return_on_curve, "outlay return must carry the tip+fee drag"

    print("\nOK: continuation_sizing self-test passed "
          "(spend hard-capped at bet, late races reverted, return on actual outlay).")
