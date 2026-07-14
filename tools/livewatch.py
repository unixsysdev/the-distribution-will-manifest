"""READ-ONLY live-trade heartbeat monitor (2026-06-12, for the armed 0.05 night).

Runs as its own systemd service (pumpfun-livewatch.service, Restart=always) so it
watches the trader independently of any interactive session. Every INTERVAL it
appends one JSON heartbeat to logs/livewatch.jsonl (and journal) with: bot
active-state, dry_run, fire count, open positions, circuit-breaker hits, recon
landed/failed/fill counts in the last window + the latest slot_gap, the funded
wallet balance and its delta from the armed baseline.

STRICTLY READ-ONLY: reads status.json + the recon log + systemctl/journalctl
state + a public getBalance. It NEVER submits, never touches the bot, the
wallet, or the collectors. A crash of this monitor cannot affect trading (it is
a separate process; the in-process risk circuits remain the autonomous safety)."""
from __future__ import annotations
import json, os, subprocess, time, urllib.request

BASE = "/root/the-distribution-will-manifest"
WALLET = os.getenv("LIVEWATCH_WALLET", "")
PUBLIC_RPC = "https://api.mainnet-beta.solana.com"
OUT = BASE + "/logs/livewatch.jsonl"
INTERVAL = 120.0
_baseline = os.getenv("LIVEWATCH_BASELINE_SOL", "").strip()
BASELINE_SOL = float(_baseline) if _baseline else None


def balance():
    if not WALLET:
        return None
    try:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "getBalance",
                           "params": [WALLET]}).encode()
        req = urllib.request.Request(PUBLIC_RPC, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read())["result"]["value"] / 1e9
    except Exception:
        return None


def recon_window(window_s=130.0):
    landed = failed = fills = retries = 0
    last_gap = None
    try:
        lines = open(BASE + "/logs/broker_recon.jsonl").readlines()[-600:]
        cut = time.time() - window_s
        for ln in lines:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("t", 0) < cut:
                continue
            k = r.get("kind")
            if k == "landed":
                landed += 1
            elif k == "failed":
                failed += 1
            elif k == "fill":
                fills += 1
                if r.get("slot_gap") is not None:
                    last_gap = r.get("slot_gap")
            elif k in ("sell_retry", "sell_will_retry", "sell_retry_no_curve_market"):
                retries += 1
    except Exception:
        pass
    return {"landed": landed, "failed": failed, "fills": fills,
            "retries": retries, "last_slot_gap": last_gap}


def sysval(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def main():
    while True:
        st = {}
        try:
            st = json.load(open(BASE + "/bot_data/status.json"))
        except Exception:
            pass
        bal = balance()
        active = sysval("systemctl", "is-active", "pumpfun-bot").strip() or "?"
        jtail = sysval("journalctl", "-u", "pumpfun-bot", "--since",
                       "130 seconds ago", "--no-pager")
        n_err = sum(1 for l in jtail.splitlines()
                    if "Traceback" in l or "error:" in l.lower())
        stats = st.get("stats", {})
        hb = {"t": time.time(), "bot": active, "dry_run": st.get("dry_run"),
              "fires": stats.get("entry_fire"), "open": st.get("n_open_paper"),
              "broker_calls": stats.get("broker_calls"),
              "grpc_wallet_recon": stats.get("grpc_wallet_recon"),
              "cb_active": stats.get("circuit_breaker_active"),
              "risk_refusal_daily_loss": stats.get("risk_refusal_daily_loss"),
              "recon_2min": recon_window(),
              "wallet_sol": bal,
              "wallet_delta_sol": (round(bal - BASELINE_SOL, 6)
                                   if bal is not None and BASELINE_SOL is not None else None),
              "errs_2min": n_err}
        try:
            with open(OUT, "a") as f:
                f.write(json.dumps(hb) + "\n")
        except Exception:
            pass
        print("livewatch " + json.dumps(hb), flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
