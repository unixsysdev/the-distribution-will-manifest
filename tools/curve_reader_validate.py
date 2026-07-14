"""READ-ONLY: validate an on-chain pump.fun BondingCurve reserve reader against
recent live mints, so the success-path canary can fetch correct reserves. No key,
no submission: just getAccountInfo + struct parse + sanity checks."""
import json, struct, asyncio, sys
from pathlib import Path
sys.path.insert(0, "/root/the-distribution-will-manifest")
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
import config
from pump_fun_ix import derive_bonding_curve, tokens_out_for_sol

ROOT = Path("/root/the-distribution-will-manifest")

def parse_curve(data: bytes):
    # pump.fun BondingCurve: 8 disc + vtok u64 + vsol u64 + real_tok u64 + real_sol u64 + supply u64 + complete bool
    vtok, vsol, rtok, rsol, supply = struct.unpack_from("<QQQQQ", data, 8)
    complete = bool(data[48]) if len(data) > 48 else None
    return vsol, vtok, rsol, rtok, supply, complete

# recent distinct mints from broker_jito.jsonl (most recent first)
mints = []
for ln in open(ROOT / "logs/broker_jito.jsonl"):
    try:
        m = json.loads(ln).get("mint")
        if m and m not in mints: mints.append(m)
    except Exception: pass
mints = list(reversed(mints))[:10]

async def main():
    ok = 0
    async with AsyncClient(config.rpc_http_url()) as cli:
        for m in mints:
            bc = derive_bonding_curve(Pubkey.from_string(m))
            try:
                resp = await cli.get_account_info(bc, encoding="base64")
            except Exception as e:
                print(f"  {m[:12]} RPC err: {e}"); continue
            v = resp.value
            if v is None:
                print(f"  {m[:12]} curve=NONE (migrated/closed)"); continue
            data = bytes(v.data)
            vsol, vtok, rsol, rtok, supply, complete = parse_curve(data)
            tok002 = tokens_out_for_sol(vsol, vtok, 2_000_000)
            sane = (28e9 <= vsol <= 130e9) and (1e13 <= vtok <= 1.2e15) and (tok002 > 0)
            ok += sane
            print(f"  {m[:12]} vsol={vsol/1e9:6.2f}SOL vtok={vtok/1e6:12.0f} complete={complete} "
                  f"len={len(data)} 0.002SOL->{tok002/1e6:.0f}tok  {'SANE' if sane else '?? CHECK'}")
    print(f"\nlayout valid on {ok}/{len(mints)} readable curves "
          f"(virtual_token_reserves@8, virtual_sol_reserves@16, complete@48)")

asyncio.run(main())
