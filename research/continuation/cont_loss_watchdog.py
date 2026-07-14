"""cont_loss_watchdog.py — external, decisive loss kill-switch for the continuation bot (2026-06-16).

Runs as a SEPARATE root systemd service so it can stop the bot regardless of the bot's own state
(a bug, a runaway, a stuck position). Two independent trip conditions, whichever fires first:
  1) REALIZED P&L  <= -CONT_LOSS_FLOOR     (default -1.0 SOL) — exact, from status.realized_net_sol
  2) RAW DRAWDOWN  >= CONT_DRAWDOWN_FLOOR  (default  1.5 SOL) — start_balance - current on-chain balance
     (backstop: catches stuck/unrealized losses or any accounting gap the realized number misses)
On a trip: `systemctl stop pumpfun-continuation-bot`, log CRITICAL, exit. Open positions are left
JOURNALED (cont_open_positions.json) so a later restart recovers+manages them — investigate first.

Every tick is logged to bot_data/cont_watchdog.jsonl for post-mortem. The start balance is captured
once (before any live buy) and persisted, so a watchdog restart keeps the same baseline."""
import asyncio, json, os, subprocess, sys, time
sys.path.insert(0, "/root/the-distribution-will-manifest")
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
import config

ROOT = "/root/the-distribution-will-manifest"
WALLET = os.getenv("CONT_WALLET", "")
SERVICE = os.getenv("CONT_SERVICE", "pumpfun-continuation-bot")
STATUS = f"{ROOT}/bot_data/continuation_status.json"
WLOG = f"{ROOT}/bot_data/cont_watchdog.jsonl"
START_FILE = f"{ROOT}/bot_data/cont_watchdog_start.json"

REALIZED_FLOOR = float(os.getenv("CONT_LOSS_FLOOR", "-1.0"))      # realized net SOL floor
DRAWDOWN_FLOOR = float(os.getenv("CONT_DRAWDOWN_FLOOR", "1.5"))   # raw balance drop from start
POLL_S = float(os.getenv("CONT_WATCHDOG_POLL_S", "12"))


def log(d):
    d["t"] = time.time()
    try:
        with open(WLOG, "a") as f:
            f.write(json.dumps(d) + "\n")
    except Exception:
        pass
    print("[watchdog]", json.dumps(d), flush=True)


def read_status():
    try:
        return json.load(open(STATUS))
    except Exception:
        return {}


def stop_service(reason, **ctx):
    log({"kind": "TRIP", "reason": reason, **ctx})
    try:
        subprocess.run(["systemctl", "stop", SERVICE], timeout=45, check=False)
        log({"kind": "STOPPED", "reason": reason, "service": SERVICE, **ctx})
    except Exception as e:
        log({"kind": "STOP_ERR", "reason": reason, "err": str(e)[:160], **ctx})


async def get_balance(cli, pk):
    return (await cli.get_balance(pk)).value / 1e9


async def main():
    if not WALLET:
        raise RuntimeError("CONT_WALLET must be set to the monitored public address")
    pk = Pubkey.from_string(WALLET)
    cli = AsyncClient(config.rpc_http_url())
    if os.path.exists(START_FILE):
        start = float(json.load(open(START_FILE))["start_balance"])
        log({"kind": "watchdog_resume", "start_balance": start})
    else:
        start = await get_balance(cli, pk)
        json.dump({"start_balance": start, "t": time.time()}, open(START_FILE, "w"))
        log({"kind": "watchdog_start", "start_balance": start})
    log({"kind": "config", "realized_floor": REALIZED_FLOOR, "drawdown_floor": DRAWDOWN_FLOOR,
         "poll_s": POLL_S, "service": SERVICE, "wallet": WALLET})
    while True:
        try:
            bal = await get_balance(cli, pk)
            st = read_status()
            realized = float(st.get("realized_net_sol", 0.0))
            drawdown = start - bal
            log({"kind": "tick", "balance": round(bal, 6), "realized_net_sol": realized,
                 "drawdown": round(drawdown, 6), "mode": st.get("mode"), "open": st.get("open"),
                 "halted": st.get("halted"), "bot_status_age_s": round(time.time() - st.get("ts", 0), 1) if st.get("ts") else None})
            if realized <= REALIZED_FLOOR:
                stop_service("realized_loss", realized=realized, balance=round(bal, 6), floor=REALIZED_FLOOR)
                break
            if drawdown >= DRAWDOWN_FLOOR:
                stop_service("balance_drawdown", drawdown=round(drawdown, 6), balance=round(bal, 6),
                             realized=realized, floor=DRAWDOWN_FLOOR)
                break
        except Exception as e:
            log({"kind": "watchdog_err", "err": str(e)[:200]})
        await asyncio.sleep(POLL_S)


if __name__ == "__main__":
    asyncio.run(main())
