"""Per-mint sliding window over the shred intent stream.

Consumes from the pumpfun_intents SHM ring (written by intent_recorder)
and maintains a small bounded deque per mint. Designed for query-time
inspection at the bot's entry trigger:

    sw = ShredWindow()
    sw.start()                          # background asyncio task drains the ring

    # at entry trigger for `mint`:
    sig = sw.signal(mint, now_ns=time.time_ns())
    # sig = {"shred_buy_500ms": 3, "shred_buy_2000ms": 7,
    #        "shred_jito_tip_rate_500ms": 0.67, ...}

Memory bounds:
  - Per-mint deque: maxlen=200 (caps memory per mint at ~24 KB)
  - Mint dict: pruned every PRUNE_INTERVAL_S of mints whose newest record
    is older than PRUNE_AGE_S (default 60s)
  - Bounded; will not grow unboundedly.

End-to-end shred-receive -> ring-visible has been measured at ~1 ms
on this system; signal queries are O(window contents) which is small.
"""
from __future__ import annotations
import asyncio, os, time
from collections import deque, defaultdict
from pathlib import Path
import sys

# import the ring reader from the sibling package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from intent_ring import IntentRingReader

# Background GC/warm cadence. Decision-time freshness is from drain_now() at
# the fire instant, not this loop, so 50ms is fine; env override for experiments.
POLL_INTERVAL_S    = float(os.getenv("SHRED_POLL_INTERVAL_S", "0.05"))
# parity_intent (2026-06-10) showed 200 never truncates the 0.5/2/5s windows.
PER_MINT_MAXLEN    = int(os.getenv("SHRED_PER_MINT_MAXLEN", "200"))
PRUNE_INTERVAL_S   = 30.0    # how often to GC stale mints
PRUNE_AGE_S        = 60.0    # drop a mint's deque if no record in this long


def _percentile(vals, q: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


class ShredWindow:
    def __init__(self, ring_name: str = "pumpfun_intents"):
        self._ring_name = ring_name
        self._reader: IntentRingReader | None = None
        self._by_mint: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=PER_MINT_MAXLEN))
        self._task: asyncio.Task | None = None
        self._stop = False
        # stats
        self.total_intents = 0
        self.total_polls   = 0
        self.last_prune_t  = time.time()

    def start(self) -> None:
        """Attach to the ring and start the background drain task."""
        if self._reader is not None: return
        self._reader = IntentRingReader(name=self._ring_name)
        self._task = asyncio.create_task(self._drain_loop())
        print(f"[shred-window] attached to '{self._ring_name}', "
              f"draining every {POLL_INTERVAL_S*1000:.0f}ms",
              flush=True)

    def drain_now(self, max_records: int = 2000) -> int:
        """Synchronously drain pending ring records ahead of a decision.
        Median shred lead over the gRPC feed is ~12ms (measured 2026-06-10),
        so with only the 50ms background tick the trigger's own intents are
        often still sitting in the ring at decision time. Poll cost is
        microseconds."""
        if self._reader is None:
            return 0
        try:
            batch = self._reader.poll(max_records=max_records)
        except Exception:
            return 0
        for it in batch:
            m = it.get("mint")
            if not m:
                continue
            self._by_mint[m].append(it)
            self.total_intents += 1
        return len(batch)

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None:
            try: await self._task
            except Exception: pass
        if self._reader is not None:
            try: self._reader.close()
            except Exception: pass

    async def _drain_loop(self) -> None:
        while not self._stop:
            try:
                batch = self._reader.poll(max_records=1000)
                self.total_polls += 1
                for it in batch:
                    m = it.get("mint")
                    if not m: continue
                    self._by_mint[m].append(it)
                    self.total_intents += 1
                # Periodic GC of stale mints (cap dict size)
                now = time.time()
                if now - self.last_prune_t > PRUNE_INTERVAL_S:
                    self._gc_stale_mints(now_ns=time.time_ns())
                    self.last_prune_t = now
            except Exception as e:
                # Don't kill the loop on transient errors
                print(f"[shred-window] drain err: {e}", flush=True)
            await asyncio.sleep(POLL_INTERVAL_S)

    def _gc_stale_mints(self, now_ns: int) -> None:
        """Drop mints whose newest record is older than PRUNE_AGE_S."""
        cutoff_ns = now_ns - int(PRUNE_AGE_S * 1e9)
        stale = [m for m, dq in self._by_mint.items()
                 if not dq or dq[-1].get("recv_ns", 0) < cutoff_ns]
        for m in stale:
            del self._by_mint[m]

    def signal(self, mint: str, now_ns: int | None = None) -> dict:
        """Compute pending-intent stats for `mint` over two windows
        (500ms and 2000ms). Returns a dict with both window counts +
        per-window sophistication aggregates. All fields are bounded
        ints/floats so it's cheap to log."""
        if now_ns is None:
            now_ns = time.time_ns()
        dq = self._by_mint.get(mint)
        out = {
            "shred_buy_500ms":     0,
            "shred_buy_2000ms":    0,
            "shred_sell_500ms":    0,
            "shred_sell_2000ms":   0,
            "shred_unique_signers_2000ms": 0,
            "shred_buy_sol_2000ms":    0.0,
            "shred_jito_tip_rate_2000ms": 0.0,
            "shred_priority_fee_p90_2000ms": 0,
            "shred_jito_tip_p90_2000ms": 0,
            "shred_probable_spoof_rate_2000ms": 0.0,
            "shred_n_records_total": 0,
        }
        if not dq: return out

        cutoff_500  = now_ns - 500_000_000
        cutoff_2000 = now_ns - 2_000_000_000

        signers = set()
        sol_sum = 0
        jito_tipped = 0
        tips_nz = []
        n_2k = 0
        spoofed = 0
        fees = []

        # Walk newest-to-oldest; break when below 2000ms cutoff
        for it in reversed(dq):
            rn = it.get("recv_ns", 0)
            if rn < cutoff_2000: break
            out["shred_n_records_total"] += 1
            is_buy = bool(it.get("is_buy", False))
            if is_buy:
                out["shred_buy_2000ms"] += 1
                if rn >= cutoff_500: out["shred_buy_500ms"] += 1
                sol_sum += it.get("sol_limit_lam", it.get("max_sol_cost", 0)) or 0
            else:
                out["shred_sell_2000ms"] += 1
                if rn >= cutoff_500: out["shred_sell_500ms"] += 1
            signers.add(it.get("user", ""))
            tl = it.get("jito_tip_lam", 0) or 0
            if tl:
                jito_tipped += 1
                tips_nz.append(tl)
            if it.get("probable_spoof"): spoofed += 1
            pf = it.get("priority_fee_micro", 0)
            if pf: fees.append(pf)
            n_2k += 1

        if n_2k > 0:
            out["shred_unique_signers_2000ms"]      = len(signers - {""})
            out["shred_buy_sol_2000ms"]             = sol_sum / 1e9
            out["shred_jito_tip_rate_2000ms"]       = jito_tipped / n_2k
            out["shred_probable_spoof_rate_2000ms"] = spoofed / n_2k
            if fees:
                fees.sort()
                out["shred_priority_fee_p90_2000ms"] = int(fees[int(len(fees)*0.9)])
            if tips_nz:
                tips_nz.sort()
                out["shred_jito_tip_p90_2000ms"] = int(tips_nz[int(len(tips_nz)*0.9)]
                                                       if len(tips_nz) > 1 else tips_nz[0])
        return out

    def intent_features(self, mint: str, now_ns: int | None = None) -> dict:
        """Feature columns matching tools/train_june_causal_sweep.py.

        Uses the same 0.5s / 2s / 5s lookback windows and the same column names:
        intent_<window>_{n,buy,sell,buy_frac,net_limit_sol,uniq_signers,
        tip_rate,tip_max_lam,priority_p90,spoof_rate}.
        """
        if now_ns is None:
            now_ns = time.time_ns()
        dq = self._by_mint.get(mint)
        out = {}
        for w_ns, label in (
            (500_000_000, "intent_0p5s"),
            (2_000_000_000, "intent_2p0s"),
            (5_000_000_000, "intent_5p0s"),
        ):
            cutoff = now_ns - w_ns
            rows = []
            if dq:
                for it in reversed(dq):
                    rn = it.get("recv_ns", 0)
                    if rn < cutoff:
                        break
                    rows.append(it)
            n = len(rows)
            if n == 0:
                out.update({
                    f"{label}_present": 0.0,
                    f"{label}_n": 0.0,
                    f"{label}_buy": 0.0,
                    f"{label}_sell": 0.0,
                    f"{label}_buy_frac": 0.0,
                    f"{label}_net_limit_sol": 0.0,
                    f"{label}_uniq_signers": 0.0,
                    f"{label}_tip_rate": 0.0,
                    f"{label}_tip_max_lam": 0.0,
                    f"{label}_priority_p90": 0.0,
                    f"{label}_spoof_rate": 0.0,
                })
                continue
            buys = [it for it in rows if it.get("is_buy")]
            sells = [it for it in rows if not it.get("is_buy")]
            buy_lim = sum((it.get("sol_limit_lam") or it.get("max_sol_cost") or 0) for it in buys)
            sell_lim = sum((it.get("sol_limit_lam") or it.get("min_sol_output") or 0) for it in sells)
            tips = [it.get("jito_tip_lam", 0) or 0 for it in rows]
            pri = [it.get("priority_fee_micro", 0) or 0 for it in rows]
            out.update({
                f"{label}_present": 1.0,
                f"{label}_n": float(n),
                f"{label}_buy": float(len(buys)),
                f"{label}_sell": float(len(sells)),
                f"{label}_buy_frac": float(len(buys) / n),
                f"{label}_net_limit_sol": float((buy_lim - sell_lim) / 1e9),
                f"{label}_uniq_signers": float(len({it.get("user", "") for it in rows} - {""})),
                f"{label}_tip_rate": float(sum(1 for x in tips if x > 0) / n),
                f"{label}_tip_max_lam": float(max(tips) if tips else 0),
                f"{label}_priority_p90": _percentile(pri, 90),
                f"{label}_spoof_rate": float(sum(1 for it in rows if it.get("probable_spoof")) / n),
            })
        return out


# ---- simple smoke ----
if __name__ == "__main__":
    async def main():
        sw = ShredWindow()
        sw.start()
        try:
            for _ in range(30):
                await asyncio.sleep(1)
                # Pick a couple of arbitrary mints currently in the window
                mints = list(sw._by_mint.keys())[:3]
                for m in mints:
                    sig = sw.signal(m)
                    print(f"  {m[:16]}  buy500={sig['shred_buy_500ms']}  "
                          f"buy2k={sig['shred_buy_2000ms']}  "
                          f"sell2k={sig['shred_sell_2000ms']}  "
                          f"unique={sig['shred_unique_signers_2000ms']}  "
                          f"tip%={sig['shred_jito_tip_rate_2000ms']*100:.0f}")
                print(f"  -- dict size={len(sw._by_mint)}, total_intents={sw.total_intents}")
        finally:
            await sw.stop()

    asyncio.run(main())
