import asyncio, sys, json
sys.path.insert(0, "/root/the-distribution-will-manifest")
from pathlib import Path
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
import config
TOKEN = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN22 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
ROOT = Path("/root/the-distribution-will-manifest")
mints = []
for ln in open(ROOT / "logs/broker_jito.jsonl"):
    try:
        m = json.loads(ln).get("mint")
        if m and m not in mints: mints.append(m)
    except Exception: pass
mints = mints[-12:]
async def main():
    counts = {"LEGACY": 0, "TOKEN-2022": 0, "OTHER": 0}
    async with AsyncClient(config.rpc_http_url()) as c:
        for m in mints:
            try:
                r = await c.get_account_info(Pubkey.from_string(m))
                o = str(r.value.owner)
                tag = "TOKEN-2022" if o == TOKEN22 else "LEGACY" if o == TOKEN else "OTHER"
            except Exception:
                tag = "OTHER"
            counts[tag] += 1
            print(f"  {m[:12]} {tag}")
    print("counts:", counts)
asyncio.run(main())
