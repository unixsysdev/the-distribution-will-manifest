"""continuation_bot.py — the continuation-strategy bot (2026-06-14).

Strategy: enter when a coin first crosses 2x from launch AND the model scores it top-tier;
bet it reaches +0.5x before -0.3x from the fill.

DUAL-TIER + BACKFILL: fills the wider tier (--tier, default top-10%); top-5% is a strict
subset by score, tracked side-by-side. At startup it BACKFILLS both tiers from the 4-5 day
replay (cont_multi_panel.jsonl) as a "backtest baseline", so the dashboard is populated
immediately and you can compare LIVE forward vs backtest.

SAFETY: DRY-RUN by default (pure sim, no broker/wallet/RPC, CANNOT submit). LIVE needs
--live AND PUMPFUN_LIVE_OK=1 AND JITO_DRY_RUN=0 (arm-time validation still pending).

Writes bot_data/continuation_status.json + continuation_bot.jsonl.
"""
import argparse, asyncio, base64, collections, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pumpfun_parse import parse_trade_event
from .continuation_tracker import ContinuationTracker
from .continuation_sizing import plan_buy, simulate_fill, realized_return, LAMPORTS_PER_SOL

ROOT = "/root/the-distribution-will-manifest"
CAP_GLOB = f"{ROOT}/grpc_capture/capture_*.jsonl"
PANEL_BF = f"{ROOT}/bot_data/cont_multi_panel.jsonl"   # backfill source (4-5 day replay)
FE = ["dd", "bf", "ntr", "recent", "tps", "uniq"]
STATUS = f"{ROOT}/bot_data/continuation_status.json"
LOG = f"{ROOT}/bot_data/continuation_bot.jsonl"
NARROW = 0.05
PUMP_RT = 0.02
REVERT_COST = 0.0006   # a cap-reverted buy still pays ~buy-side priority+base (no fill, no pump fee, no sell)


def train_model(panel):
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    df = pd.read_parquet(panel)
    X = df[FE].values.astype(float); y = df["y"].values.astype(int)
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05).fit(X, y)
    return clf, clf.predict_proba(X)[:, 1]


def newest_capture():
    fs = [f for f in glob.glob(CAP_GLOB) if not f.endswith(".gz")]
    return max(fs, key=os.path.getmtime) if fs else None


def _empty():
    return {"5": collections.Counter(), "10": collections.Counter()}


class ContinuationBot:
    def __init__(self, args):
        self.args = args
        self.WIDE = args.tier
        self.bet = args.bet_sol; self.tip = args.tip_sol
        self.bet_lam = int(args.bet_sol * LAMPORTS_PER_SOL)
        self.tip_lam = int(args.tip_sol * LAMPORTS_PER_SOL)
        self.prio_lam = int(args.prio_fee_micro * args.cu_limit / 1e6)               # priority fee per tx (lamports)
        self.fixed_rt = (self.tip_lam + 2 * self.prio_lam + 2 * 5000) / LAMPORTS_PER_SOL  # round-trip fixed cost (SOL): tip(entry) + 2*prio + 2*base
        self.clf, preds = train_model(args.panel)
        self.warm = {"5": float(np.quantile(preds, 1 - NARROW)), "10": float(np.quantile(preds, 1 - self.WIDE))}
        self.trk = ContinuationTracker()
        self.live = bool(args.live) and os.getenv("PUMPFUN_LIVE_OK") == "1" and os.getenv("JITO_DRY_RUN", "1") == "0"
        self.broker = None
        self.crosses = 0
        self.t = _empty()            # LIVE forward stats
        self.bf = _empty()           # BACKTEST baseline (backfilled at startup, static)
        self.bf_n = 0
        self.bf_test_frac = 1.0      # fraction of replay coins held out for the OOS baseline
        self.sel = {}; self.pos = {}
        self.recent_p = collections.deque(maxlen=500)
        self.recent_closes = collections.deque(maxlen=40)
        self.t0 = time.time()
        self.out = open(LOG, "a")
        self.backfill()

    @staticmethod
    def _bump(store, is5, key, val=1):
        store["10"][key] += val
        if is5:
            store["5"][key] += val

    def _score(self, c):
        return float(self.clf.predict_proba(
            np.array([[c["dd"], c["buy_frac"], c["ntr"], c["recent"], c.get("tps", 0), c.get("uniq", 0)]], float))[0, 1])

    def backfill(self):
        """Seed recent_p (live-cutoff calibration via the deployed model) + the bf BASELINE.
        The baseline is BY-COIN OOS: a fresh model trains on train-coins and scores held-out
        test-coins (no coin in both), so the dashboard shows the real OOS edge, not an
        in-sample-optimistic number. net_lam includes the small cost of cap-reverted attempts,
        so bf net_sol is net-per-SELECTED-cross. Counts are the held-out test slice (bf_test_frac)."""
        if not os.path.exists(PANEL_BF):
            return
        cr = {}; fl = {}; oc = {}
        for l in open(PANEL_BF):
            try: e = json.loads(l)
            except Exception: continue
            if e.get("mult") != 2.0: continue
            m = e.get("mint"); k = e.get("kind")
            if k == "cross": cr[m] = e
            elif k == "fill": fl[m] = e
            elif k == "outcome": oc[m] = e
        keys = [m for m in oc if m in cr]
        if not keys:
            return
        self.bf_n = len(keys)
        X = np.array([[cr[m]["dd"], cr[m]["buy_frac"], cr[m]["ntr"], cr[m]["recent"],
                       cr[m].get("tps", 0), cr[m].get("uniq", 0)] for m in keys], float)
        for p in self.clf.predict_proba(X)[:, 1][-500:].tolist():
            self.recent_p.append(p)                       # live-cutoff calibration (deployed model)
        import hashlib
        from sklearn.ensemble import HistGradientBoostingClassifier
        bk = np.array([int(hashlib.md5(m.encode()).hexdigest(), 16) % 100 for m in keys])
        y = np.array([1 if oc[m]["y"] == 1 else 0 for m in keys])
        trm = bk < 70; idx_te = np.where(bk >= 70)[0]
        self.bf_test_frac = float(len(idx_te) / len(keys))
        if int(trm.sum()) < 50 or len(idx_te) < 20:
            return
        clf_bf = HistGradientBoostingClassifier(max_depth=4, max_iter=400, learning_rate=0.05,
                                                l2_regularization=1.0).fit(X[trm], y[trm])
        p_te = clf_bf.predict_proba(X[idx_te])[:, 1]
        cut5 = float(np.quantile(p_te, 1 - NARROW)); cut10 = float(np.quantile(p_te, 1 - self.WIDE))
        cap_frac = self.args.cap_bps / 10000.0
        revert_lam = int(REVERT_COST * LAMPORTS_PER_SOL)
        closes = []
        for jj, gi in enumerate(idx_te):
            p = float(p_te[jj])
            if p < cut10:
                continue
            m = keys[gi]; is5 = bool(p >= cut5)
            self._bump(self.bf, is5, "selected")
            c = cr[m]; f = fl.get(m); o = oc[m]
            slip = (f["fill_mid"] / c["cross_mid"] - 1.0) if (f and c.get("cross_mid")) else 0.0
            if slip > cap_frac:
                self._bump(self.bf, is5, "reverted"); self._bump(self.bf, is5, "net_lam", -revert_lam)
                continue
            self._bump(self.bf, is5, "filled")
            ret = o["ret"]; ro = ret - PUMP_RT - self.fixed_rt / self.bet
            net = self.bet * ret - PUMP_RT * self.bet - self.fixed_rt
            self._bump(self.bf, is5, "closed"); self._bump(self.bf, is5, "win" if o["y"] == 1 else "loss")
            self._bump(self.bf, is5, "net_lam", int(net * LAMPORTS_PER_SOL))
            self._bump(self.bf, is5, "sum_curve", ret); self._bump(self.bf, is5, "sum_outlay", ro)
            closes.append((p, is5, o["y"], ret, ro, net))
        for (p, is5, y0, ret, ro, net) in closes[-12:]:
            self.recent_closes.append({"mint": "(backfill)", "p": round(p, 3), "is5": is5, "y": y0,
                                       "ret_curve": round(ret, 3), "ret_outlay": round(ro, 3),
                                       "net_sol": round(net, 5), "dur_s": 0, "bf": True})
        print(f"[cont-bot] backfilled OOS {len(idx_te)}/{len(keys)} test crosses (frac {self.bf_test_frac:.2f}): "
              f"top5 {self.bf['5']['closed']} / top10 {self.bf['10']['closed']} closes", flush=True)

    def emit(self, rec):
        self.out.write(json.dumps(rec) + "\n"); self.out.flush()

    def _cut(self, key, frac):
        return float(np.quantile(self.recent_p, 1 - frac)) if len(self.recent_p) >= 100 else self.warm[key]

    async def setup(self):
        if self.live:
            from jito_broker import JitoBroker
            self.broker = await JitoBroker.create(bet_sol=self.bet)
            print("[cont-bot] LIVE broker up (double-gated)", flush=True)
        else:
            print("[cont-bot] DRY-RUN (pure sim, no broker, cannot submit)", flush=True)

    async def on_events(self, events, vsol, vtok, ts, slot):
        for e in events:
            k = e["kind"]; m = e["mint"]
            if k == "cross":
                self.crosses += 1
                p = self._score({"dd": e["dd"], "buy_frac": e["buy_frac"], "ntr": e["ntr"],
                                 "recent": e["recent"], "tps": e["tps"], "uniq": e["uniq"]})
                self.recent_p.append(p)
                if p < self._cut("10", self.WIDE):
                    continue
                is5 = bool(p >= self._cut("5", NARROW))
                self._bump(self.t, is5, "selected")
                plan = plan_buy(vsol, vtok, self.bet_lam, self.args.cap_bps)
                self.sel[m] = {"plan": plan, "p": p, "is5": is5, "t": ts}
                self.emit({"kind": "decision", "mint": m, "t": ts, "p": round(p, 4), "is5": is5,
                           "token_amount": plan.token_amount, "max_sol_cost": plan.max_sol_cost_lam})
                if self.live:
                    try: await self.broker.buy(m, self.bet, vsol, vtok, slot=slot, tip_lamports_override=self.tip_lam)
                    except Exception as ex: self.emit({"kind": "submit_err", "mint": m, "err": str(ex)[:120]})
            elif k == "fill" and m in self.sel:
                s = self.sel[m]; is5 = s["is5"]
                fr = simulate_fill(vsol, vtok, s["plan"], tip_lam=self.tip_lam, priority_fee_lam=self.prio_lam)
                if fr.filled:
                    self._bump(self.t, is5, "filled")
                    self.pos[m] = {"p": s["p"], "is5": is5, "fill": fr, "plan": s["plan"], "fill_t": ts, "exec_slip": fr.exec_slip}
                else:
                    self._bump(self.t, is5, "reverted")
                    self._bump(self.t, is5, "net_lam", -int(REVERT_COST * LAMPORTS_PER_SOL))
                    self.sel.pop(m, None)
                self.emit({"kind": "fill", "mint": m, "t": ts, "is5": is5, "filled": fr.filled,
                           "exec_slip": round(fr.exec_slip, 4) if fr.filled else None, "revert": fr.revert_reason})
            elif k == "outcome" and m in self.pos:
                pos = self.pos.pop(m); self.sel.pop(m, None); is5 = pos["is5"]
                tr = realized_return(pos["fill"], pos["plan"], vsol, vtok, exit_tip_lam=0, priority_fee_lam=self.prio_lam)
                self._bump(self.t, is5, "closed"); self._bump(self.t, is5, "win" if e["y"] == 1 else "loss")
                self._bump(self.t, is5, "net_lam", tr.net_pnl_lam)
                self._bump(self.t, is5, "sum_curve", tr.return_on_curve); self._bump(self.t, is5, "sum_outlay", tr.return_on_outlay)
                cl = {"mint": m, "p": round(pos["p"], 4), "is5": is5, "y": e["y"], "bf": False,
                      "ret_curve": round(tr.return_on_curve, 4), "ret_outlay": round(tr.return_on_outlay, 4),
                      "net_sol": round(tr.net_pnl_lam / LAMPORTS_PER_SOL, 5), "dur_s": round(ts - pos["fill_t"], 1)}
                self.recent_closes.appendleft(cl)
                self.emit({"kind": "outcome", "t": ts, **cl})
                if self.live:
                    try: await self.broker.sell_all(m, vsol, vtok, slot=slot)
                    except Exception: pass

    def _block(self, store, key):
        c = store[key]; f = c["win"] + c["loss"]
        return {"selected": c["selected"], "filled": c["filled"], "reverted": c["reverted"], "closed": c["closed"],
                "wins": c["win"], "fill_rate": c["filled"] / max(1, c["filled"] + c["reverted"]),
                "win_rate": (c["win"] / f) if f else 0.0, "net_sol": round(c["net_lam"] / LAMPORTS_PER_SOL, 5),
                "mean_curve": (c["sum_curve"] / f) if f else 0.0, "mean_outlay": (c["sum_outlay"] / f) if f else 0.0}

    def write_status(self):
        now = time.time()
        opens = [{"mint": m, "p": round(v["p"], 3), "is5": v["is5"], "age_s": round(now - v["fill_t"], 0),
                  "exec_slip": round(v["exec_slip"], 3)} for m, v in list(self.pos.items())[:30]]
        status = {"ts": now, "mode": "LIVE" if self.live else "DRY-RUN", "uptime_s": round(now - self.t0),
                  "bet_sol": self.bet, "tip_sol": self.tip, "prio_sol": round(self.prio_lam / LAMPORTS_PER_SOL, 5),
                  "fixed_rt_sol": round(self.fixed_rt, 5), "cap_bps": self.args.cap_bps, "bf_n": self.bf_n,
                  "bf_test_frac": round(self.bf_test_frac, 3),
                  "crosses": self.crosses, "open": len(self.pos),
                  "cut5": round(self._cut("5", NARROW), 4), "cut10": round(self._cut("10", self.WIDE), 4),
                  "bf5": self._block(self.bf, "5"), "bf10": self._block(self.bf, "10"),
                  "live5": self._block(self.t, "5"), "live10": self._block(self.t, "10"),
                  "open_positions": opens, "recent_closes": list(self.recent_closes)}
        tmp = STATUS + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(status, fh)
        os.replace(tmp, STATUS)

    async def run(self):
        await self.setup()
        cur = newest_capture()
        while cur is None:
            await asyncio.sleep(2); cur = newest_capture()
        f = open(cur, "r"); f.seek(0, os.SEEK_END)
        print(f"[cont-bot] tailing {cur} (fill top{self.WIDE:.0%}, compare top{NARROW:.0%}, bet={self.bet} cap={self.args.cap_bps}bps)", flush=True)
        last_prune = last_status = time.time(); nlines = 0
        self.write_status()
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.2)
                nf = newest_capture()
                if nf and nf != cur:
                    f.close(); cur = nf; f = open(cur, "r"); print(f"[cont-bot] rotated -> {cur}", flush=True)
            elif '"TradeEvent"' in line:
                try:
                    r = json.loads(line); ev = parse_trade_event(base64.b64decode(r["raw"]))
                except Exception:
                    ev = None
                if ev is not None and ev.is_classic_curve and ev.virtual_token_reserves > 0:
                    ts = r.get("t") or time.time()
                    evs = self.trk.update(ev.mint, ev.virtual_sol_reserves, ev.virtual_token_reserves, ev.is_buy, ts, ev.user)
                    if evs:
                        await self.on_events(evs, ev.virtual_sol_reserves, ev.virtual_token_reserves, ts, r.get("slot"))
                nlines += 1
                if nlines % 300 == 0:
                    await asyncio.sleep(0)
            now = time.time()
            if now - last_status > 5:
                self.write_status(); last_status = now
            if now - last_prune > 120:
                self.trk.prune(now)
                for m in [m for m in self.sel if m not in self.trk.state]: self.sel.pop(m, None)
                for m in [m for m in self.pos if m not in self.trk.state]: self.pos.pop(m, None)
                last_prune = now


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bet-sol", type=float, default=0.1)
    ap.add_argument("--tier", type=float, default=0.10)
    ap.add_argument("--cap-bps", type=int, default=2500)
    ap.add_argument("--tip-sol", type=float, default=0.0005, help="small Jito tip (race is won mostly via priority fee + speed)")
    ap.add_argument("--prio-fee-micro", type=int, default=5_000_000, help="priority fee uLamports/CU (the real race lever)")
    ap.add_argument("--cu-limit", type=int, default=120_000)
    ap.add_argument("--panel", default=f"{ROOT}/bot_data/june_v2_panel.parquet")
    ap.add_argument("--live", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(ContinuationBot(parse_args()).run())
