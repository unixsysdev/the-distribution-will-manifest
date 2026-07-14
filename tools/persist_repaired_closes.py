"""Write the repaired close values from logs/repaired_closes.jsonl back into
bot_data/positions.jsonl.

For each repaired record (matched by mint + close timestamp), the
corresponding close record in positions.jsonl gets:
  - exit_ret     <- repaired exit_ret
  - net_return   <- repaired net_sol
  - exit_kind    <- "hold" -> classified based on repaired exit_ret:
                     >= +1.0 -> "tp_level_hit_repaired"
                     >  0    -> "winner_repaired"
                     <= 0    -> "rugged_repaired"
  - reason       stays "stale" so we know this was a stale-watchdog close
  - repaired_at  <- ISO timestamp
  - orig_exit_ret, orig_net_return, orig_exit_kind preserved

A backup of the pre-rewrite positions.jsonl is saved alongside.
"""
from __future__ import annotations
import json, shutil, time
from pathlib import Path

ROOT = Path("/root/the-distribution-will-manifest")
POS  = ROOT / "bot_data/positions.jsonl"
REP  = ROOT / "logs/repaired_closes.jsonl"


def main():
    if not REP.exists():
        raise SystemExit(f"repaired_closes.jsonl not found at {REP}")
    if not POS.exists():
        raise SystemExit(f"positions.jsonl not found at {POS}")

    # Load repaired records keyed by (mint, t_close) for safe matching
    repaired = {}
    with open(REP) as f:
        for ln in f:
            try: r = json.loads(ln)
            except Exception: continue
            key = (r["mint"], round(r.get("t_close", 0), 3))
            repaired[key] = r
    print(f"loaded {len(repaired)} repaired records")
    # repaired_closes.jsonl stores `net_sol` as ABSOLUTE SOL.
    # positions.jsonl stores `net_return` as a RATIO of bet_sol.
    # To match the existing convention we divide net_sol by bet_sol.
    bet_sol = None
    for r in repaired.values():
        bet_sol = r["repaired"].get("bet_sol")
        if bet_sol: break
    if not bet_sol:
        raise SystemExit("bet_sol missing in repaired records — can't convert SOL -> ratio")
    print(f"using bet_sol={bet_sol} for SOL->ratio conversion")

    # Backup positions.jsonl
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bak = POS.with_suffix(f".jsonl.bak.{ts}")
    shutil.copy2(POS, bak)
    print(f"backed up -> {bak.name}")

    # Pass through positions.jsonl, rewriting matched close records
    in_lines = POS.read_text().splitlines()
    out_lines = []
    n_rewritten = 0
    for ln in in_lines:
        try: r = json.loads(ln)
        except Exception:
            out_lines.append(ln); continue
        if r.get("kind") != "close":
            out_lines.append(ln); continue
        key = (r.get("mint",""), round(r.get("t", 0), 3))
        rep = repaired.get(key)
        if rep is None:
            out_lines.append(ln); continue
        # rewrite — preserve originals for audit
        new = dict(r)
        new["orig_exit_ret"]  = r.get("exit_ret")
        new["orig_net_return"]= r.get("net_return")
        new["orig_exit_kind"] = r.get("exit_kind")
        new["exit_ret"]   = rep["repaired"]["exit_ret"]
        # SOL -> ratio: net_return is stored as a fraction of bet_sol
        # (e.g. +1.45 = +145% return on a 0.1 SOL bet = +0.145 SOL absolute)
        new["net_return"] = rep["repaired"]["net_sol"] / bet_sol
        new["net_sol_absolute"] = rep["repaired"]["net_sol"]  # also keep absolute for clarity
        # Reclassify based on repaired return
        er = rep["repaired"]["exit_ret"]
        if   er >= 1.0: new["exit_kind"] = "tp_level_hit_repaired"
        elif er >  0:   new["exit_kind"] = "winner_repaired"
        else:           new["exit_kind"] = "rugged_repaired"
        new["repaired_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out_lines.append(json.dumps(new))
        n_rewritten += 1

    POS.write_text("\n".join(out_lines) + "\n")
    print(f"rewrote {n_rewritten} close records in {POS}")

    # Sanity check the result
    counts = {}
    with open(POS) as f:
        for ln in f:
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("kind") == "close" and r.get("repaired_at"):
                k = r.get("exit_kind","?")
                counts[k] = counts.get(k, 0) + 1
    print(f"\nrepaired closes by exit_kind:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s} {v}")

    print(f"\nrollback if needed:")
    print(f"  cp {bak} {POS}")


if __name__ == "__main__":
    main()
