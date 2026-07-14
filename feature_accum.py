"""Per-mint feature accumulator for the V+K7 stacked entry head (Finding 35).

UPDATED for bot_artifacts_K7V: tracks BOTH triggers in parallel from the same
running accumulator and snapshots two independent 11-feature vectors:

  - K=7 trade-count window  -> snapshots 11 K-features + (vsK, vtK) reserves at trade #7
  - V=0.5 cumulative-buy SOL window
                            -> snapshots 11 V-features when cum_buy_sol >= 0.5 AND n >= 3

Decision moment = whichever trigger fires LATER. Entry reserves used by the paper
book are K=7-anchored (vsK, vtK) to match the training in pumpfun_K7_V_book.py.
Path/recovery features are also K-anchored (ret = mid/midK - 1, run_max_ret tracked
from the K=7 trigger trade onward — even while waiting on V to fire).

Causal: each snapshot freezes window_last_ts at its trigger so trades_per_sec equals
the offline parquet exactly. Verified byte-identical against the offline K7 and V
extractors when run on the same trade stream.

Ingests generic trades (vsol, vtok, sol, is_buy, user, ts) so this module backs:
  - offline replay (parity test against the K7 / V token_level.parquets), and
  - live parsed TradeEvents.
Gating to classic curve / fresh-launch detection is the HARNESS's job.
"""
from __future__ import annotations
from collections import deque

import os as _os
K_TRIGGER = int(_os.getenv("K_TRIGGER", "5"))   # default 5 (was 7, swapped 2026-06-09 for tp50_k5 model)
V_TRIGGER = float(_os.getenv("V_TRIGGER", "0.5"))    # cumulative buy SOL trigger
MIN_WINDOW_N = 3   # V trigger additionally requires at least 3 trades (skip degenerate)
W = 15             # rolling window for path/recovery features


class TokenState:
    __slots__ = ("n", "mid0", "first_ts", "last_ts",
                 "mids", "users", "user_sol", "n_buy", "cum_buy_sol",
                 "net_sol", "tot_sol", "entry_sol", "win_dup", "win_ddown",
                 # K=7 snapshot (frozen at trade #7)
                 "k_fired", "midK", "vsK", "vtK", "k_feats", "k_window_last_ts",
                 # V=0.5 snapshot (frozen at first trade with cum_buy_sol>=0.5 AND n>=3)
                 "v_fired", "midV", "vsV", "vtV", "v_feats", "v_window_last_ts",
                 # forward state for path/recovery (K-anchored)
                 "fwd", "run_max_ret",
                 "win")

    def __init__(self, vsol, vtok, sol, is_buy, user, ts):
        mid = vsol / vtok
        self.n = 1; self.mid0 = mid
        self.first_ts = ts; self.last_ts = ts
        self.mids = [mid]
        self.users = {user}; self.user_sol = {user: sol}
        self.n_buy = 1 if is_buy else 0
        self.cum_buy_sol = sol if is_buy else 0.0
        self.net_sol = sol if is_buy else -sol
        self.tot_sol = sol; self.entry_sol = sol
        self.win_dup = 0.0; self.win_ddown = 0.0
        # K snapshot defaults
        self.k_fired = False; self.midK = 0.0; self.vsK = 0.0; self.vtK = 0.0
        self.k_feats = None; self.k_window_last_ts = ts
        # V snapshot defaults
        self.v_fired = False; self.midV = 0.0; self.vsV = 0.0; self.vtV = 0.0
        self.v_feats = None; self.v_window_last_ts = ts
        # forward
        self.fwd = 0; self.run_max_ret = 0.0
        self.win = deque(maxlen=W); self.win.append((user, sol, is_buy, ts))

    def _snapshot_entry_feats(self) -> tuple:
        """Compute the 11 entry features from the current accumulator state.
        Called at trigger time so last_ts is the trigger trade's timestamp."""
        mids = self.mids
        d = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
        sa = sum(abs(x) for x in d)
        dir_eff = abs(sum(d)) / sa if sa > 0 else 0.0
        win_ret = mids[-1] / self.mid0 - 1.0 if self.mid0 > 0 else 0.0
        span = max(1e-6, self.last_ts - self.first_ts)
        sas = max(self.user_sol.values()) / self.tot_sol if self.tot_sol > 0 else 0.0
        return (win_ret, dir_eff, self.n_buy / self.n, len(self.users), self.net_sol,
                self.tot_sol, sas, self.n / span, self.entry_sol, self.win_dup, self.win_ddown)

    def update(self, vsol, vtok, sol, is_buy, user, ts) -> str:
        """Returns one of:
          'skip'    invalid trade (vsol or vtok <= 0)
          'window'  still accumulating; no new trigger fired this trade
          'k_only'  K=7 fired this trade but V=0.5 not yet
          'v_only'  V=0.5 fired this trade but K=7 not yet
          'ready'   BOTH triggers now fired (either both this trade or one already)
          'fwd'     already in forward phase (both fired in a prior trade)
        """
        if vsol <= 0 or vtok <= 0:
            return "skip"
        mid = vsol / vtok
        self.win.append((user, sol, is_buy, ts))
        already_ready = self.k_fired and self.v_fired
        if not already_ready:
            # still accumulating windows; update the shared running state
            self.n += 1; self.mids.append(mid); self.users.add(user)
            self.user_sol[user] = self.user_sol.get(user, 0.0) + sol
            if is_buy:
                self.n_buy += 1; self.net_sol += sol; self.cum_buy_sol += sol
            else:
                self.net_sol -= sol
            self.tot_sol += sol; self.last_ts = ts
            rr = mid / self.mid0 - 1.0 if self.mid0 > 0 else 0.0
            self.win_dup = max(self.win_dup, rr); self.win_ddown = min(self.win_ddown, rr)
            fired_now = []
            # K=7 trigger (snapshot reserves+features at the K-th trade)
            if not self.k_fired and self.n >= K_TRIGGER:
                self.k_fired = True
                self.midK = mid; self.vsK = vsol; self.vtK = vtok
                self.k_window_last_ts = ts
                self.k_feats = self._snapshot_entry_feats()
                fired_now.append("k")
            # V=0.5 trigger
            if not self.v_fired and self.cum_buy_sol >= V_TRIGGER and self.n >= MIN_WINDOW_N:
                self.v_fired = True
                self.midV = mid; self.vsV = vsol; self.vtV = vtok
                self.v_window_last_ts = ts
                self.v_feats = self._snapshot_entry_feats()
                fired_now.append("v")
            # Start tracking K-anchored run_max_ret as soon as K has fired,
            # even before V fires (so by the time we enter the position the
            # path snapshots are K-anchored from the K-trigger moment, matching
            # the offline K7 path_snapshots.parquet).
            if self.k_fired and self.midK > 0:
                ret = mid / self.midK - 1.0
                if ret > self.run_max_ret:
                    self.run_max_ret = ret
            if self.k_fired and self.v_fired:
                return "ready"
            if "k" in fired_now:
                return "k_only"
            if "v" in fired_now:
                return "v_only"
            return "window"
        # both triggers already fired; we're tracking forward path
        self.fwd += 1; self.last_ts = ts
        if self.midK > 0:
            ret = mid / self.midK - 1.0
            if ret > self.run_max_ret:
                self.run_max_ret = ret
        return "fwd"

    def combined_entry_features(self) -> list[float]:
        """The 22-feature V+K7 vector in the model's expected order:
        K-features first (11), then V-features (11). Both triggers must have fired."""
        if self.k_feats is None or self.v_feats is None:
            raise RuntimeError("combined_entry_features called before both triggers fired")
        return list(self.k_feats) + list(self.v_feats)

    def k_entry_features(self) -> list[float]:
        """K-only entry features (used by the recovery model). Requires k_fired."""
        if self.k_feats is None:
            raise RuntimeError("k_entry_features called before K=7 trigger fired")
        return list(self.k_feats)

    # --- persistence (atomic checkpoint / restore across restarts) -----------
    # We persist the full accumulator state so that restarts don't lose
    # in-flight mid-window mints. Without this the bot misses ~60% of mints
    # (verified via coverage_diag.py 2026-06-08): fast-moving high-scoring
    # mints reach >3 SOL fresh-rsol threshold within seconds of birth, so any
    # restart in their early window causes the bot to first-observe them
    # past the fresh filter and drop them. With persistence, restarts cost
    # only the trades that arrived DURING the restart (typically a few seconds)
    # rather than the entire accumulator state.
    def to_dict(self) -> dict:
        """JSON-safe snapshot of every __slots__ field. Loaded via from_dict."""
        return {
            "n": self.n, "mid0": self.mid0,
            "first_ts": self.first_ts, "last_ts": self.last_ts,
            "mids": list(self.mids),
            "users": list(self.users),
            "user_sol": dict(self.user_sol),
            "n_buy": self.n_buy, "cum_buy_sol": self.cum_buy_sol,
            "net_sol": self.net_sol, "tot_sol": self.tot_sol,
            "entry_sol": self.entry_sol,
            "win_dup": self.win_dup, "win_ddown": self.win_ddown,
            "k_fired": self.k_fired, "midK": self.midK, "vsK": self.vsK, "vtK": self.vtK,
            "k_feats": list(self.k_feats) if self.k_feats is not None else None,
            "k_window_last_ts": self.k_window_last_ts,
            "v_fired": self.v_fired, "midV": self.midV, "vsV": self.vsV, "vtV": self.vtV,
            "v_feats": list(self.v_feats) if self.v_feats is not None else None,
            "v_window_last_ts": self.v_window_last_ts,
            "fwd": self.fwd, "run_max_ret": self.run_max_ret,
            "win": [list(t) for t in self.win],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenState":
        """Inverse of to_dict. Reconstructs a TokenState without re-running update()."""
        # Bypass __init__ since we don't have the (vsol, vtok, ...) of trade #1
        self = cls.__new__(cls)
        self.n        = int(d["n"])
        self.mid0     = float(d["mid0"])
        self.first_ts = float(d["first_ts"])
        self.last_ts  = float(d["last_ts"])
        self.mids     = list(d["mids"])
        self.users    = set(d["users"])
        self.user_sol = dict(d["user_sol"])
        self.n_buy        = int(d["n_buy"])
        self.cum_buy_sol  = float(d["cum_buy_sol"])
        self.net_sol      = float(d["net_sol"])
        self.tot_sol      = float(d["tot_sol"])
        self.entry_sol    = float(d["entry_sol"])
        self.win_dup      = float(d["win_dup"])
        self.win_ddown    = float(d["win_ddown"])
        self.k_fired     = bool(d["k_fired"])
        self.midK        = float(d["midK"])
        self.vsK         = float(d["vsK"])
        self.vtK         = float(d["vtK"])
        self.k_feats     = tuple(d["k_feats"]) if d["k_feats"] is not None else None
        self.k_window_last_ts = float(d["k_window_last_ts"])
        self.v_fired     = bool(d["v_fired"])
        self.midV        = float(d["midV"])
        self.vsV         = float(d["vsV"])
        self.vtV         = float(d["vtV"])
        self.v_feats     = tuple(d["v_feats"]) if d["v_feats"] is not None else None
        self.v_window_last_ts = float(d["v_window_last_ts"])
        self.fwd         = int(d["fwd"])
        self.run_max_ret = float(d["run_max_ret"])
        self.win = deque((tuple(t) for t in d["win"]), maxlen=W)
        return self

    def path_features(self, vsol, vtok) -> dict:
        """K-anchored recovery / scale-out state at the current forward trade."""
        mid = vsol / vtok if vtok else 0.0
        ret = mid / self.midK - 1.0 if self.midK > 0 else 0.0
        dd = (mid / (self.midK * (1 + self.run_max_ret)) - 1.0) if self.run_max_ret > -1 else 0.0
        wl = list(self.win); nb = sum(1 for _, _, b, _ in wl if b); nw = len(wl)
        buy_frac = nb / nw if nw else 0.0
        sellers = {}; sell_tot = 0.0
        for u, s, b, _ in wl:
            if not b:
                sellers[u] = sellers.get(u, 0.0) + s; sell_tot += s
        solo = (max(sellers.values()) / sell_tot) if sell_tot > 0 else 0.0
        dt = max(1e-6, wl[-1][3] - wl[0][3])
        net = sum((s if b else -s) for _, s, b, _ in wl)
        fill_k = max(0.0, min(1.0, (vsol / 1e9 - 30.0) / 85.0))
        return {"ret": ret, "run_max_ret": self.run_max_ret, "dd": dd, "fill_k": fill_k,
                "buy_frac_w": buy_frac, "nsell_w": len(sellers), "solo_sell_w": solo,
                "vel_w": net / dt, "dts": self.last_ts - self.first_ts, "mid": mid}


# Feature-name conventions matching bot_artifacts_K7V/model_spec.json
ENTRY_FEATURE_NAMES_K = ["win_ret","dir_eff","buy_frac","uniq","net_sol","tot_sol",
                         "single_actor_share","trades_per_sec","entry_sol","win_drawup","win_drawdown"]
ENTRY_FEATURE_NAMES_V = [f"{c}_v" for c in ENTRY_FEATURE_NAMES_K]
ENTRY_FEATURE_NAMES   = ENTRY_FEATURE_NAMES_K + ENTRY_FEATURE_NAMES_V  # 22-feature V+K7
