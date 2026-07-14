"""level_tp: fixed take-profit. sell_all when ret >= tp_level. Nothing else.

Motivated by the project's FolioBot-port research (Finding 7): pump.fun
winners spike-and-collapse rather than trend, so on the frozen snapshot a
fixed TP at +50% / +100% produced positive median net (+49% / +92%) and
60-72% win rates, while trailing stops at 20/30/50% retrace had NEGATIVE
median (-14% to -25%). The hypothesis to test under the bot's actual
selection (entry_score >= τ) is whether that shape transfers.

This policy is intentionally one rule, no scaling-out, no time gates, no
recovery model. If Finding 7 transfers, this naked level-TP should beat
the incumbent scale-out family on bot-selected mints. If not, the path
forward is the LSM continuation-value policy.

Variants registered:
  level_tp_50    sell_all at ret >= 0.50  (+50%, Finding 7 winner)
  level_tp_100   sell_all at ret >= 1.00  (+100%, Finding 7 higher-cap variant)
  level_tp_200   sell_all at ret >= 2.00  (+200%, the "2x continuation" filter)
"""
from __future__ import annotations
from .base import ExitPolicy, ExitDecision, HarnessConsts, register


class _LevelTP(ExitPolicy):
    tp_level: float = 0.50

    def decide(self, mint, n_sold, last_slice_t, now, pf, run_max, p_rec, fwd_n,
               consts: HarnessConsts) -> ExitDecision:
        if pf["ret"] >= self.tp_level:
            return ExitDecision(action="sell_all", phase="tp_level",
                                reason=f"ret>={self.tp_level:.2f}",
                                extra={"tp_level": self.tp_level})
        return ExitDecision("hold")


@register("level_tp_50")
class LevelTP50(_LevelTP):
    tp_level = 0.50


@register("level_tp_100")
class LevelTP100(_LevelTP):
    tp_level = 1.00


@register("level_tp_200")
class LevelTP200(_LevelTP):
    tp_level = 2.00


# --- time-capped take-profit (optimal_exit P2, 2026-06-11) ---
# Sell at +100% if reached, ELSE liquidate at a time cap. exit_lab + optimal_exit
# showed tp_100 with a ~120s cap beat uncapped tp_100 on the OOS fold (median time
# to +100% was ~99s; holding dead past the cap only adds collapse risk). The cap
# uses pf["dts"] (seconds since entry/first snap), which the harness + replay both
# populate.
class _LevelTPCap(_LevelTP):
    tp_level: float = 1.00
    time_cap_s: float = 120.0

    def decide(self, mint, n_sold, last_slice_t, now, pf, run_max, p_rec, fwd_n,
               consts: HarnessConsts) -> ExitDecision:
        if pf["ret"] >= self.tp_level:
            return ExitDecision(action="sell_all", phase="tp_level",
                                reason=f"ret>={self.tp_level:.2f}",
                                extra={"tp_level": self.tp_level})
        if pf.get("dts", 0.0) >= self.time_cap_s:
            return ExitDecision(action="sell_all", phase="time_cap",
                                reason=f"dts>={self.time_cap_s:.0f}s",
                                extra={"time_cap_s": self.time_cap_s})
        return ExitDecision("hold")


@register("level_tp_100_t120")
class LevelTP100T120(_LevelTPCap):
    tp_level = 1.00
    time_cap_s = 120.0


# --- take-profit + HARD STOP + time-cap (loss_control 2026-06-11) ---
# The live 300s stale watchdog lets bleeders decay (non-winner mean -0.72). A
# -30% hard stop cuts that to -0.49 and lifts elite deduped net -0.008 -> +0.065
# (a timeout alone barely helps: bleeders keep trading through it). This is the
# proposed armed-phase downside rule. Uses pf["ret"] and pf["dts"].
class _LevelTPStopCap(_LevelTP):
    tp_level: float = 0.50
    stop_level: float = -0.30
    time_cap_s: float = 120.0

    def decide(self, mint, n_sold, last_slice_t, now, pf, run_max, p_rec, fwd_n,
               consts: HarnessConsts) -> ExitDecision:
        r = pf["ret"]
        if r >= self.tp_level:
            return ExitDecision("sell_all", phase="tp_level",
                                reason=f"ret>={self.tp_level:.2f}", extra={"tp_level": self.tp_level})
        if r <= self.stop_level:
            return ExitDecision("sell_all", phase="stop_loss",
                                reason=f"ret<={self.stop_level:.2f}", extra={"stop_level": self.stop_level})
        if pf.get("dts", 0.0) >= self.time_cap_s:
            return ExitDecision("sell_all", phase="time_cap",
                                reason=f"dts>={self.time_cap_s:.0f}s", extra={"time_cap_s": self.time_cap_s})
        return ExitDecision("hold")


@register("level_tp_50_stop30_cap120")
class LevelTP50Stop30Cap120(_LevelTPStopCap):
    tp_level = 0.50
    stop_level = -0.30
    time_cap_s = 120.0


@register("level_tp_100_stop30_cap120")
class LevelTP100Stop30Cap120(_LevelTPStopCap):
    tp_level = 1.00
    stop_level = -0.30
    time_cap_s = 120.0
