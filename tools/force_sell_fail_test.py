"""CONTROLLED sell-failure test: buy tokens, then force the sell to revert on EVERY
escalation step (monkeypatch the retry slippage schedule to impossible values), and
verify the broker's panic-retry chain escalates to the terminal MARKET sell and
recovers the position (ends at 0 tokens) rather than holding the bag."""
import asyncio, os, sys
sys.path.insert(0, "/root/the-distribution-will-manifest")
import jito_broker
# force attempts 1-3 to revert (min_sol > achievable) -> must reach attempt 4 = market sell
jito_broker.SELL_RETRY_SLIPPAGE_BPS = [-10000, -10000, -10000]
os.environ["PUMPFUN_LIVE_OK"] = "1"; os.environ["JITO_DRY_RUN"] = "0"; os.environ["PUMPFUN_BROKER_TIP_LAMPORTS"] = "100000"
from jito_broker import JitoBroker

MINT = "35FwyPFD3QdkzzR9j5daycxXQSspiePr5Gu34jekpump"

async def main():
    broker = await JitoBroker.create(bet_sol=0.002, dry_run=False)
    res = await broker._get_curve_reserves(MINT)
    if res is None or res[2]:
        print("curve unavailable/migrated; abort"); return
    vsol, vtok, _ = res
    await broker.buy(MINT, sol=0.002, vsol_lam=vsol, vtok=vtok)
    bal = 0
    for _ in range(20):
        await asyncio.sleep(2)
        bal = await broker._token_balance(MINT)
        if bal > 0: break
    print(f"post-buy token balance: {bal}")
    if bal <= 0:
        print("buy did not land; abort"); return
    # FORCE the first sell to fail (impossible min_sol), then let the reconciler retry-chain run
    broker.holdings[MINT] = 0
    vsol2, vtok2, _ = await broker._get_curve_reserves(MINT)
    await broker._do_sell(MINT, bal, vsol2, vtok2, None, op_label="sell_all", slippage_bps_override=-10000)
    print("forced-fail sell submitted (attempt 0 impossible); watching panic-retry chain ...")
    recovered_at = None
    for i in range(60):
        await asyncio.sleep(3)
        b = await broker._token_balance(MINT)
        if b == 0:
            recovered_at = (i + 1) * 3; break
    final = await broker._token_balance(MINT)
    print(f"RESULT: final token balance={final}  recovered={'YES at ~%ds' % recovered_at if recovered_at else 'NO'}")
asyncio.run(main())
