"""SUCCESS-path canary: ONE tiny real buy + immediate sell on a live pump.fun
curve, with a real tip so the bundle LANDS, to validate the full fire path
end-to-end (blockhash -> bundle -> Jito -> land -> recon -> actual-fill logging)
BEFORE turning the bot loose on real signals.

Complements tools/tiny_tip_test.py (which only proves the FAILURE path: tip=1,
never lands). This proves a SUCCESSFUL fire and that real slippage is captured.

SAFETY:
  - tiny --sol (default 0.002) and immediate round-trip => worst case ~0.002 SOL.
  - asserts the curve is live (complete=False, sane reserves) before buying.
  - requires explicit --yes to submit; --dry runs the whole flow with no submission.
  - needs a funded wallet (.env WALLET_PRIVATE_KEY) + this process sets JITO_DRY_RUN
    per --yes/--dry; it does NOT touch the running bot or the collectors.

USAGE (after the key is placed on the host):
  ./venv/bin/python tools/tiny_live_roundtrip.py --mint <LIVE_PUMP_MINT> --dry      # rehearse
  ./venv/bin/python tools/tiny_live_roundtrip.py --mint <LIVE_PUMP_MINT> --sol 0.002 --tip-lam 300000 --yes
"""
from __future__ import annotations
import argparse, asyncio, json, os, struct, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def parse_curve(data: bytes):
    vtok, vsol, rtok, rsol, supply = struct.unpack_from("<QQQQQ", data, 8)
    complete = bool(data[48]) if len(data) > 48 else None
    return vsol, vtok, complete


async def fetch_reserves(cli, mint_str):
    from solders.pubkey import Pubkey
    from solana.rpc.commitment import Processed
    from pump_fun_ix import derive_bonding_curve
    bc = derive_bonding_curve(Pubkey.from_string(mint_str))
    # PROCESSED + timeout to match the broker: a FRESH curve is invisible at
    # the default (finalized) commitment for ~13s, and eRPC stalls on that
    # lookup rather than returning None (the same class of bug we fixed in
    # JitoBroker._get_mint_meta).
    resp = await asyncio.wait_for(
        cli.get_account_info(bc, encoding="base64", commitment=Processed), timeout=5.0)
    if resp.value is None:
        raise RuntimeError("bonding curve account not found (migrated/closed?)")
    return parse_curve(bytes(resp.value.data))


async def ata_balance(cli, user_pk, mint_str):
    from solders.pubkey import Pubkey
    from solana.rpc.commitment import Processed
    from pump_fun_ix import derive_ata, TOKEN_2022_PROGRAM
    ata = derive_ata(user_pk, Pubkey.from_string(mint_str), TOKEN_2022_PROGRAM)
    try:
        resp = await asyncio.wait_for(
            cli.get_token_account_balance(ata, commitment=Processed), timeout=5.0)
        return int(resp.value.amount) if resp.value else 0
    except Exception:
        return 0


async def wait_for_outcome(broker, prev_n, op, max_wait):
    """Block until a NEW recent_outcome for `op` appears (landed or failed)."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        await asyncio.sleep(1.0)
        outs = [o for o in list(broker.recent_outcomes)[prev_n:] if o["op"] == op]
        if outs:
            return outs[-1]
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mint", required=True)
    ap.add_argument("--sol", type=float, default=0.002)
    ap.add_argument("--tip-lam", type=int, default=300_000)
    ap.add_argument("--hold-s", type=float, default=1.0)
    ap.add_argument("--max-wait", type=float, default=45.0)
    ap.add_argument("--yes", action="store_true", help="actually submit (real SOL)")
    ap.add_argument("--dry", action="store_true", help="rehearse with no submission")
    a = ap.parse_args()
    if not (a.yes or a.dry):
        print("refusing: pass --dry to rehearse or --yes to submit real SOL"); return 2

    os.environ["PUMPFUN_LIVE_OK"] = "1"
    os.environ["JITO_DRY_RUN"] = "1" if a.dry else "0"
    os.environ["PUMPFUN_BROKER_TIP_LAMPORTS"] = str(a.tip_lam)

    from solana.rpc.async_api import AsyncClient
    import config
    from jito_broker import JitoBroker

    async with AsyncClient(config.rpc_http_url()) as cli:
        vsol, vtok, complete = await fetch_reserves(cli, a.mint)
        print(f"[canary] {a.mint}")
        print(f"[canary] curve: vsol={vsol/1e9:.3f} SOL  vtok={vtok}  complete={complete}")
        if complete or not (28e9 <= vsol <= 130e9):
            print("[canary] ABORT: curve not live/sane (migrated or off-range)"); return 3
        print(f"[canary] mode={'DRY (no submission)' if a.dry else 'LIVE (REAL SOL)'}  "
              f"sol={a.sol}  tip_lam={a.tip_lam}")
        if a.yes:
            print("[canary] submitting REAL bundles in 5s ... Ctrl-C to abort"); await asyncio.sleep(5)

        broker = await JitoBroker.create(bet_sol=a.sol, dry_run=a.dry)
        print(f"[canary] broker: wallet={broker.user_pk} tip={broker.tip_lamports} dry={broker.dry_run}")

        # --- BUY ---
        n0 = len(broker.recent_outcomes)
        t0 = time.time()
        await broker.buy(mint=a.mint, sol=a.sol, vsol_lam=vsol, vtok=vtok, slot=None)
        print("[canary] buy() submitted; waiting for land ...")
        if a.dry:
            print("[canary] DRY: no pending bundle (DRY skips submission); flow OK up to buy.")
        else:
            res = await wait_for_outcome(broker, n0, "buy", a.max_wait)
            if res is None:
                print("[canary] BUY did not classify within max_wait -> check broker_recon.jsonl"); return 4
            if not res.get("landed"):
                print(f"[canary] BUY FAILED: {res.get('reason')} (tip may be too low) -> no sell"); return 5
            print(f"[canary] BUY LANDED slot={res.get('slot')} latency={res.get('latency_s'):.2f}s")
            await asyncio.sleep(a.hold_s)
            bal = await ata_balance(cli, broker.user_pk, a.mint)
            print(f"[canary] on-chain token balance after buy: {bal}")
            if bal <= 0:
                print("[canary] WARN: zero balance post-buy (fill mismatch) -> skip sell"); return 6
            broker.holdings[a.mint] = bal   # sell exactly what we actually hold
            vsol2, vtok2, _ = await fetch_reserves(cli, a.mint)
            n1 = len(broker.recent_outcomes)
            await broker.sell_all(mint=a.mint, vsol_lam=vsol2, vtok=vtok2, slot=None)
            print("[canary] sell_all() submitted; waiting for land ...")
            sres = await wait_for_outcome(broker, n1, "sell_all", a.max_wait)
            if sres is None or not sres.get("landed"):
                print(f"[canary] SELL not landed: {sres} -> position may still be open!"); return 7
            print(f"[canary] SELL LANDED slot={sres.get('slot')} latency={sres.get('latency_s'):.2f}s")

        # --- report fills from recon log ---
        await asyncio.sleep(1.0)
        print("\n[canary] === fill records for this mint (actual vs expected) ===")
        rp = HERE.parent / "logs" / "broker_recon.jsonl"
        for ln in rp.read_text().splitlines()[-40:]:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("mint") == a.mint and r.get("kind") in ("fill", "landed", "failed", "fill_err", "fill_no_meta"):
                print("  ", json.dumps({k: r.get(k) for k in
                      ("kind", "op", "landed_slot", "actual_tok_delta", "expected_tok_delta",
                       "actual_sol_delta_lam", "expected_sol_out", "fee_lam", "tip_lam", "reason")}))
        print("\n[canary] recon_summary:", json.dumps(broker.recon_summary()))
        print("[canary] DONE.", "Rehearsal OK." if a.dry else "Live round-trip complete.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
