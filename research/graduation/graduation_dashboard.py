"""Continuation bot live dashboard (Textual TUI). Shows top-5% and top-10% SIDE BY SIDE,
each with a BACKTEST baseline (backfilled from the 4-5d replay) and the LIVE forward run.
Reads bot_data/graduation_status.json. Refresh 1s, read-only.
Run from the repository root: `python -m research.graduation.graduation_dashboard`.
"""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, DataTable, Header, Footer
from rich.text import Text

DEFAULT_ROOT = Path("/root/the-distribution-will-manifest")


def load(path):
    try: return json.loads(Path(path).read_text())
    except Exception: return None


def fmt_age(s):
    if not s or s <= 0: return "-"
    if s < 60: return f"{int(s)}s"
    if s < 3600: return f"{int(s/60)}m"
    return f"{int(s/3600)}h{int((s%3600)/60):02d}m"


def cv(x, signed=True, pct=False):
    if x is None: return Text("-", style="dim")
    s = (f"{x:+.0%}" if pct else f"{x:+.3f}") if signed else (f"{x:.0%}" if pct else f"{x:.3f}")
    return Text(s, style="bold green" if x > 0 else ("bold red" if x < 0 else "dim white"))


def fmt_sol(x, signed=True):
    if x is None: return Text("-", style="dim")
    s = f"{x:+.4f}" if signed else f"{x:.4f}"
    return Text(s, style="bold green" if x > 0 else ("bold red" if x < 0 else "white"))


def _pad(t, w):
    """Right-align a styled Text in a width-w field (f-string width ignores Text)."""
    pad = w - len(t.plain)
    return Text(" " * pad) + t if pad > 0 else t


def crep(x):
    """Buyer-cluster reputation 0-1 (neutral 0.5): green = good bots piling in, red = bad."""
    if x is None: return Text("-", style="dim")
    return Text(f"{x:.2f}", style="bold green" if x > 0.55 else ("bold red" if x < 0.45 else "white"))


class ContinuationDashboard(App):
    CSS = """
    Screen { background: black; }
    #hdr { height: 5; border: solid magenta; padding: 0 1; }
    #ops { height: 4; border: solid rgb(80,80,100); padding: 0 1; }
    #tiers { height: 9; }
    #t5 { border: solid #00ff66; width: 50%; padding: 0 1; }
    #t10 { border: solid cyan; width: 50%; padding: 0 1; }
    #econ { height: 9; }
    #ev { border: solid yellow; width: 60%; padding: 0 1; }
    #timing { border: solid rgb(80,80,100); width: 40%; padding: 0 1; }
    #lower { height: 1fr; }
    DataTable { background: black; }
    DataTable > .datatable--header { background: rgb(30,30,50); color: white; }
    """
    BINDINGS = [Binding("q", "quit", "quit"), Binding("r", "refresh", "refresh")]

    def __init__(self, root, refresh_s=1.0):
        super().__init__(); self.root = Path(root); self.refresh_s = refresh_s

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="hdr")
        yield Static(id="ops")
        yield Horizontal(Static(id="t5"), Static(id="t10"), id="tiers")
        yield Horizontal(Static(id="ev"), Static(id="timing"), id="econ")
        yield Vertical(DataTable(id="open"), DataTable(id="closes"), id="lower")
        yield Footer()

    def on_mount(self):
        o = self.query_one("#open", DataTable)
        o.add_columns("OPEN mint", "tier", "p", "rep", "age", "slip"); o.zebra_stripes = True; o.cursor_type = "row"
        c = self.query_one("#closes", DataTable)
        c.add_columns("CLOSED mint", "src", "tier", "rep", "outcome", "ret_curve", "ret_outlay", "net SOL", "dur")
        c.zebra_stripes = True; c.cursor_type = "row"
        self.set_interval(self.refresh_s, self.refresh_all); self.refresh_all()

    def action_refresh(self): self.refresh_all()

    def refresh_all(self):
        try: self._render()
        except Exception as e:
            self.query_one("#hdr", Static).update(Text(f"refresh err: {e}", style="red"))

    def _tier_panel(self, title, bf, live, rz=None):
        t = Text()
        t.append(f"{title}\n", style="bold underline")
        t.append("  backtest  ", style="dim")
        t.append(f"n={bf['closed']:<4} win ")
        t.append(f"{bf['win_rate']:.0%}", style="bold green" if bf['win_rate'] >= 0.5 else "bold red")
        t.append("  curve "); t.append_text(cv(bf["mean_curve"]))
        t.append("  outlay "); t.append_text(cv(bf["mean_outlay"]))
        t.append("\n")
        t.append("  LIVE      ", style="bold")
        t.append(f"n={live['closed']:<4} win ")
        wr = live["win_rate"]
        t.append(f"{wr:.0%}" if live["closed"] else "—", style="bold green" if (live["closed"] and wr >= 0.5) else ("bold red" if live["closed"] else "dim"))
        t.append("  curve "); t.append_text(cv(live["mean_curve"]) if live["closed"] else Text("—", style="dim"))
        t.append("  outlay "); t.append_text(cv(live["mean_outlay"]) if live["closed"] else Text("—", style="dim"))
        t.append("\n  LIVE funnel  ", style="dim")
        t.append(f"sel {live['selected']} -> fill {live['filled']} / rev {live['reverted']} "
                 f"({live['fill_rate']:.0%}) -> closed {live['closed']}\n")
        t.append("  LIVE net  "); t.append_text(cv(live["net_sol"])); t.append(" SOL", style="dim")
        if rz is not None:
            sb_n = live["closed"] - rz["closed"]; sb_net = live["net_sol"] - rz["net_sol"]
            rwr = rz["win_rate"]
            t.append("\n  REALIZABLE ", style="bold yellow"); t.append("(excl same-block)  ", style="dim")
            t.append(f"n={rz['closed']:<4} win ")
            t.append(f"{rwr:.0%}" if rz["closed"] else "—",
                     style="bold green" if (rz["closed"] and rwr >= 0.5) else ("bold red" if rz["closed"] else "dim"))
            t.append("  net "); t.append_text(cv(rz["net_sol"]) if rz["closed"] else Text("—", style="dim")); t.append(" SOL", style="dim")
            t.append(f"   [excl {sb_n} same-block ", style="dim"); t.append_text(cv(sb_net)); t.append(" mirage]", style="dim")
        return t

    def _ops_panel(self, s):
        """BLOCK 2 - OPS / HEALTH. Mode, status age, uptime, crosses/open, event
        plumbing, cost split, and wallet/realized P&L (N/A under dry-run)."""
        age = time.time() - s.get("ts", 0)
        o = Text()
        o.append("OPS / HEALTH\n", style="bold #00ff66")
        o.append("mode ", style="white")
        mode = s.get("mode", "?")
        o.append(f"{mode}", style="bold green" if mode == "DRY-RUN" else "bold red")
        o.append("  status ", style="white")
        o.append(f"{age:.0f}s", style="cyan" if age < 20 else "bold yellow")
        o.append(f"  uptime {fmt_age(s.get('uptime_s', 0))}", style="white")
        o.append(f"  crosses {s.get('crosses', 0)}  open {s.get('open', 0)}", style="white")
        o.append("  events ", style="white")
        o.append(f"{s.get('events', '-')}", style="dim")
        o.append("  intent ", style="white")
        ir = "ON" if s.get("intent_ring") else ("off" if "intent_ring" in s else "-")
        o.append(ir, style="bold cyan" if ir == "ON" else "dim")
        o.append("\n")
        o.append("cost  ", style="white")
        o.append(f"bet {s.get('bet_sol')}  cap {s.get('cap_bps')}bps  prio {s.get('prio_sol', 0):.4f} "
                 f"tip {s.get('tip_sol')}  rt {s.get('fixed_rt_sol', 0):.4f}\n", style="dim")
        o.append("wallet ", style="white")
        w = s.get("wallet_sol")
        o.append_text(fmt_sol(w, signed=False) if isinstance(w, (int, float)) else Text(s.get("wallet_sol", "N/A (dry-run)"), style="dim"))
        o.append("  realized P&L ", style="white")
        r = s.get("realized_pnl_sol")
        o.append_text(fmt_sol(r) if isinstance(r, (int, float)) else Text(s.get("realized_pnl_sol", "N/A (dry-run)"), style="dim"))
        return o

    def _ev_panel(self, s):
        """BLOCK 1 - EV / PAYOFF PROJECTION. Honest per-SELECTED-cross economics (by-coin OOS;
        net_sol includes the small cost of cap-reverted attempts):
          net_per_sel  = block.net_sol / selected
          sel_per_day  = selected / days   (bt: 4.8d x bf_test_frac held-out slice; live: uptime)
          SOL_per_day@P = sel_per_day * net_per_sel * P
        '-' when a block has selected==0."""
        tf = max(s.get("bf_test_frac", 1.0), 1e-9)
        days_bt = 4.8 * tf
        days_live = max(s.get("uptime_s", 0) / 86400.0, 1e-9)
        Ps = (1.0, 0.50, 0.35, 0.20)
        rows = [
            ("TOP-5% bt", s.get("bf5", {}), days_bt, False),
            ("TOP-5% live", s.get("live5", {}), days_live, True),
            ("TOP-10% bt", s.get("bf10", {}), days_bt, False),
            ("TOP-10% live", s.get("live10", {}), days_live, True),
        ]
        e = Text()
        e.append("EV / PAYOFF PROJECTION  ", style="bold yellow")
        e.append("(net/SELECTED cross, by-coin OOS incl cap-reverts; SOL/day = sel/day x net x P)\n", style="dim")
        e.append(f"{'tier':<13}{'net/sel':>10}{'sel/day':>9}"
                 f"{'@P1.0':>9}{'@.50':>8}{'@.35':>8}{'@.20':>8}\n", style="bold white")
        for label, blk, days, is_live in rows:
            sel = blk.get("selected", 0)
            e.append(f"{label:<13}", style="white")
            if sel == 0:
                e.append(f"{'-':>10}{'-':>9}{'-':>9}{'-':>8}{'-':>8}{'-':>8}\n", style="dim")
                continue
            npt = blk.get("net_sol", 0.0) / max(sel, 1)
            tpd = sel / days
            e.append_text(_pad(cv(npt), 10))
            e.append(f"{tpd:>9.0f}", style="dim")
            for P, w in zip(Ps, (9, 8, 8, 8)):
                e.append_text(_pad(cv(tpd * npt * P), w))
            e.append("\n")
        e.append("P(gap-0) = live race-win rate (unmeasured); .20 ~ breakeven. net incl cap-reverts.", style="dim")
        return e

    def _timing_panel(self, s):
        """BLOCK 3 - EXECUTION TIMING. Backtest constants (measured n=9945) plus
        live duration spread from recent live closes (bf falsy => live)."""
        t = Text()
        t.append("EXECUTION TIMING\n", style="bold #00ff66")
        t.append("backtest ", style="white")
        t.append("(n=9945)\n", style="dim")
        t.append("  win med 12.6s  loss med 15.1s\n", style="dim")
        t.append("  9% wins <0.4s (1 slot)  13% <1s\n", style="dim")
        t.append("  of <1s wins: 41% cap-reverted\n", style="dim")
        live_durs = [c.get("dur_s") for c in s.get("recent_closes", [])
                     if not c.get("bf") and isinstance(c.get("dur_s"), (int, float))]
        t.append("LIVE ", style="bold")
        if live_durs:
            t.append(f"(n={len(live_durs)})\n", style="dim")
            t.append("  min ", style="white"); t.append(f"{min(live_durs):.1f}s", style="cyan")
            t.append("  med ", style="white"); t.append(f"{statistics.median(live_durs):.1f}s", style="bold cyan")
            t.append("  max ", style="white"); t.append(f"{max(live_durs):.1f}s", style="cyan")
        else:
            t.append("-", style="dim")
        return t

    def _render(self):
        s = load(self.root / "bot_data" / "graduation_status.json")
        if not s:
            self.query_one("#hdr", Static).update(Text("waiting for graduation bot status…", style="yellow")); return
        age = time.time() - s.get("ts", 0); mode = s.get("mode", "?")
        h = Text()
        h.append("GRADUATION BOT (DRY-RUN) ", style="bold white")
        h.append(f"[{s.get('model','base')}]", style="bold magenta")
        h.append(f" {mode} ", style="bold black on green" if mode == "DRY-RUN" else "bold white on red")
        h.append(f"  status {age:.0f}s   uptime {fmt_age(s.get('uptime_s',0))}", style="cyan" if age < 20 else "bold yellow")
        h.append(f"   rep map {s.get('rep_signers','?')} bots  events {s.get('events','-')}\n", style="dim")
        h.append(f"  2x -> top10% LEAN+buyer-REP filter (p>={s.get('cut10',0):.3f}, top5% {s.get('cut5',0):.3f}) -> +0.5x/-0.3x", style="dim")
        h.append(f"   bet {s.get('bet_sol')} cap {s.get('cap_bps')}bps  cost: pump 2% + prio {s.get('prio_sol',0):.4f} + tip {s.get('tip_sol')} (rt fixed {s.get('fixed_rt_sol',0):.4f})\n", style="white")
        h.append(f"  live crosses {s.get('crosses',0)}  open {s.get('open',0)}   ", style="white")
        h.append(f"backtest baseline backfilled from {s.get('bf_n',0)} replay crosses (4-5 days)", style="dim")
        self.query_one("#hdr", Static).update(h)
        self.query_one("#ops", Static).update(self._ops_panel(s))
        self.query_one("#t5", Static).update(self._tier_panel("TOP-5%", s.get("bf5", {}), s.get("live5", {}), s.get("live5_rz")))
        self.query_one("#t10", Static).update(self._tier_panel("TOP-10%", s.get("bf10", {}), s.get("live10", {}), s.get("live10_rz")))
        self.query_one("#ev", Static).update(self._ev_panel(s))
        self.query_one("#timing", Static).update(self._timing_panel(s))

        ot = self.query_one("#open", DataTable); ot.clear()
        for p in s.get("open_positions", []):
            ot.add_row(p["mint"][:16], "top5" if p.get("is5") else "5-10", f"{p.get('p',0):.3f}",
                       crep(p.get("buy_rep")), fmt_age(p.get("age_s", 0)), cv(p.get("exec_slip", 0), pct=True))
        if not s.get("open_positions"): ot.add_row("none", "", "", "", "", "")
        ct = self.query_one("#closes", DataTable); ct.clear()
        for c in s.get("recent_closes", []):
            sb = c.get("same_block")
            base = "WIN" if c.get("y") == 1 else "loss"
            out = Text(base + ("·SB" if sb else ""),
                       style="yellow" if sb else ("bold green" if c.get("y") == 1 else "bold red"))
            ct.add_row(c["mint"][:16], "bt" if c.get("bf") else "live", "top5" if c.get("is5") else "5-10",
                       crep(c.get("buy_rep")), out,
                       cv(c.get("ret_curve")), cv(c.get("ret_outlay")), cv(c.get("net_sol")), f"{c.get('dur_s',0):.1f}s")
        if not s.get("recent_closes"): ct.add_row("none yet", "", "", "", "", "", "", "", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT)); ap.add_argument("--refresh-s", type=float, default=5.0)
    a = ap.parse_args()
    ContinuationDashboard(Path(a.root), a.refresh_s).run()


if __name__ == "__main__":
    main()
