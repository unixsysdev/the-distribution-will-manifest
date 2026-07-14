"""pump.fun bot live dashboard (Textual TUI).

Reads:
  bot_data/status.json          (canonical bot state, refreshed every 30s by bot)
  bot_data/shadow_run.jsonl     (entry_decision, scale_slice, death_cut, etc.)
  bot_data/positions.jsonl      (open / snap / close events for paper book)
  logs/broker_jito.jsonl        (bundle assembly / submission events)
  logs/broker_recon.jsonl       (recon: landed / failed / holdings_reconcile)
  logs/drift_log.jsonl          (drift_check events from drift monitor)
  bot_artifacts_K7V/model_spec.json (threshold + features for display)

Refresh: 1s. No writes. SSH into the research host, then:
  scripts/pumpfun_ctl.sh dashboard
"""
from __future__ import annotations
import argparse, gzip, json, os, time
from collections import deque
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, DataTable, Header, Footer
from rich.text import Text
from rich.console import Group

DEFAULT_ROOT = Path("/root/the-distribution-will-manifest")


def load_json_safe(path: Path):
    try: return json.loads(path.read_text())
    except Exception: return None


def tail_jsonl(path: Path, n: int = 200, tail_bytes: int = 4 << 20):
    """Return last n parsed JSON lines from a jsonl file. Returns oldest-first.
    tail_bytes (default 4MB): how much to read from the end when the file is bigger
    than that. shadow_run.jsonl can grow fast with path_snap rows; need a generous
    tail or older entry_decision rows fall out of view."""
    if not path.exists(): return []
    try:
        sz = path.stat().st_size
        if sz < tail_bytes:
            lines = path.read_text().splitlines()
        else:
            with open(path, "rb") as f:
                f.seek(max(0, sz - tail_bytes))
                lines = f.read().decode(errors="ignore").splitlines()[1:]
        out = []
        for ln in lines[-n:]:
            try: out.append(json.loads(ln))
            except Exception: continue
        return out
    except Exception:
        return []


def fmt_age(seconds: float) -> str:
    if seconds is None or seconds <= 0: return "-"
    if seconds < 60: return f"{int(seconds)}s"
    if seconds < 3600: return f"{int(seconds/60)}m{int(seconds%60):02d}s"
    return f"{int(seconds/3600)}h{int((seconds%3600)/60):02d}m"


# Plain mint cell — no underline, no link, no pump.fun URL.
# Just the truncated address as text. (Removed the OSC 8 hyperlink path
# because Textual DataTable strips the escape sequences anyway and the
# clutter wasn't worth keeping.)
def _mint_link(full_mint: str, *, max_chars: int = 14) -> Text:
    if not full_mint:
        return Text("-", style="dim")
    return Text(full_mint[:max_chars])


def fmt_sol(x: float | None, signed: bool = True) -> Text:
    if x is None: return Text("-", style="dim")
    s = f"{x:+.4f}" if signed else f"{x:.4f}"
    if x > 0: return Text(s, style="bold green")
    if x < 0: return Text(s, style="bold red")
    return Text(s, style="white")


class PumpfunDashboard(App):
    CSS = """
    Screen { background: black; }
    #top { height: 7; }
    #mid { height: 6; }
    #drift-panel { height: 3; }
    #lower { height: 1fr; }
    .panel {
        background: rgb(15,15,25);
        border: solid rgb(80,80,100);
        padding: 0 1;
    }
    #status-panel    { border: solid magenta; width: 65%; }
    #risk-panel      { border: solid yellow;  width: 35%; }
    #latency-panel   { border: solid #00ff66; width: 60%; }
    #recon-panel     { border: solid cyan;    width: 40%; }
    #drift-panel     { border: solid #ff9900; }
    DataTable        { background: black; }
    DataTable > .datatable--header { background: rgb(30,30,50); color: white; }
    """
    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("r", "refresh", "refresh"),
    ]

    def __init__(self, root: Path, refresh_s: float = 1.0):
        super().__init__()
        self.root = Path(root)
        self.refresh_s = refresh_s
        # Row-index -> full mint, so 'o' / 'c' bindings can resolve the URL
        # for whichever row is currently highlighted. Maintained by the
        # render functions for each DataTable.
        self._row_mints_open: list[str] = []
        self._row_mints_fires: list[str] = []
        # last status note shown in the footer/title for o/c actions
        self._last_action_msg: str = ""

    def _focused_mint(self) -> str | None:
        """Return the full mint of the row currently highlighted in whichever
        DataTable has focus. Falls back to the recent-fires table's first row
        if nothing is explicitly focused — common case for users who just
        launched the dashboard and haven't clicked into a table yet."""
        try:
            # Try the actually-focused widget first
            focused = self.focused
            if focused is not None:
                fid = getattr(focused, "id", None)
                if fid == "open-positions":
                    r = focused.cursor_row
                    if r is not None and 0 <= r < len(self._row_mints_open):
                        return self._row_mints_open[r]
                elif fid == "recent-fires":
                    r = focused.cursor_row
                    if r is not None and 0 <= r < len(self._row_mints_fires):
                        return self._row_mints_fires[r]
            # Fallback: read the recent-fires table's current cursor even
            # without focus (so 'o' works on first key press too).
            try:
                t = self.query_one("#recent-fires", DataTable)
                r = t.cursor_row
                if r is not None and 0 <= r < len(self._row_mints_fires):
                    return self._row_mints_fires[r]
            except Exception:
                pass
            try:
                t = self.query_one("#open-positions", DataTable)
                r = t.cursor_row
                if r is not None and 0 <= r < len(self._row_mints_open):
                    return self._row_mints_open[r]
            except Exception:
                pass
            return None
        except Exception:
            return None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            Static("status",  id="status-panel",  classes="panel"),
            Static("risk",    id="risk-panel",    classes="panel"),
            id="top",
        )
        yield Horizontal(
            Static("latency", id="latency-panel", classes="panel"),
            Static("recon",   id="recon-panel",   classes="panel"),
            id="mid",
        )
        yield Static("drift", id="drift-panel", classes="panel")
        yield Vertical(
            DataTable(id="open-positions"),
            DataTable(id="recent-fires"),
            id="lower",
        )
        yield Footer()

    def on_mount(self):
        t1 = self.query_one("#open-positions", DataTable)
        t1.add_columns("mint", "score", "age", "slice", "ret_last", "run_max", "phase")
        t1.zebra_stripes = True
        t1.cursor_type = "row"
        t2 = self.query_one("#recent-fires", DataTable)
        t2.add_columns("time UTC", "era", "mint", "score", "slot", "status", "exit",
                       "exit_ret", "dur", "net(pol)SOL",
                       "shred", "tip×")
        t2.zebra_stripes = True
        t2.cursor_type = "row"
        self.set_interval(self.refresh_s, self.refresh_all)
        self.refresh_all()

    def action_refresh(self):
        self.refresh_all()

    # ------------- data refresh -------------

    def refresh_all(self):
        try:
            self._render_status()
            self._render_risk()
            self._render_latency()
            self._render_recon()
            self._render_drift()
            self._render_open_positions()
            self._render_recent_fires()
        except Exception as e:
            # don't crash UI on transient errors
            self.query_one("#status-panel", Static).update(Text(f"refresh err: {e}", style="red"))

    def _render_status(self):
        status = load_json_safe(self.root/"bot_data"/"status.json") or {}
        stats = status.get("stats", {})
        broker = status.get("broker_kind", "?")
        live_or_paper = "LIVE" if broker == "JitoBroker" else "PAPER"
        dry_run_marker = "DRY_RUN" if status.get("dry_run", True) else None
        listener_source = status.get("listener_source", "?").upper()
        threshold = status.get("entry_threshold")
        if threshold is None:
            spec = load_json_safe(self.root/"bot_artifacts_K7V"/"model_spec.json") or {}
            threshold = spec.get("entry", {}).get("entry_threshold_top_decile", "?")
        uptime = status.get("uptime_s", 0)
        n_open = status.get("n_open_paper", 0)
        n_closed = status.get("n_closed", 0)
        mean_net = status.get("mean_net")    # fractional return per bet (proc/q_lam - 1)
        win_pct = status.get("win_pct")
        bet_sol = status.get("bet_sol")
        if bet_sol is None:
            try:
                import yaml
                cfg = yaml.safe_load((self.root/"config.yaml").read_text()) or {}
                bet_sol = float(cfg.get("bot", {}).get("bet_sol", 1.0))
            except Exception:
                bet_sol = 1.0
        # Convert fractional return to absolute SOL P&L per bet
        mean_sol = mean_net * bet_sol if isinstance(mean_net, (int, float)) else None

        header = Text()
        header.append("MODE ", style="bold")
        header.append(f" {live_or_paper} ", style="bold black on yellow" if live_or_paper == "PAPER"
                       else "bold white on red")
        if dry_run_marker:
            header.append("  DRY_RUN ", style="bold black on green")
        else:
            header.append("  LIVE_EXEC ", style="bold white on red")
        header.append(f"  {listener_source} ", style="bold cyan")
        header.append(f" uptime {fmt_age(uptime)}", style="cyan")
        header.append(f"  thr ", style="dim")
        thr_str = f"{threshold:.4f}" if isinstance(threshold, (int,float)) else str(threshold)
        header.append(thr_str, style="white")
        # Data freshness indicator. Shows how stale the data we're rendering is
        # by reading the mtime of shadow_run.jsonl (which the bot appends to on
        # every trade event). If this number grows >5s the dashboard is
        # rendering stale data — either the bot is not writing or the file is
        # being rotated.
        try:
            mtime = (self.root/"bot_data"/"shadow_run.jsonl").stat().st_mtime
            data_age = time.time() - mtime
            if   data_age < 2:   age_color = "green"
            elif data_age < 10:  age_color = "yellow"
            else:                age_color = "bold red"
            header.append(f"  data {data_age:.1f}s ago", style=age_color)
        except Exception:
            header.append("  data ?", style="dim")
        header.append("\n")
        header.append("activity: ", style="bold")
        header.append(f"evt={stats.get('events',0):,} ", style="white")
        header.append(f"trades={stats.get('trade_events',0):,} ", style="white")
        header.append(f"fresh={stats.get('fresh',0):,} ", style="white")
        header.append(f"k={stats.get('k_fires',0)} ", style="white")
        header.append(f"v={stats.get('v_fires',0)} ", style="white")
        header.append(f"ready={stats.get('both_ready',0)} ", style="bold cyan")
        header.append(f"fires={stats.get('entry_fire',0)}", style="bold magenta")
        header.append("\n")
        era = status.get("era") or {}
        model = status.get("model") or {}
        spec_full = load_json_safe(self.root/"bot_artifacts_K7V"/"model_spec.json") or {}
        hold = spec_full.get("holdout_result", {}) or {}
        header.append("model:    ", style="bold")
        art_name = Path(str(model.get("artifact") or "?")).name
        header.append(f"{art_name} ", style="bold white")
        n_feat = model.get("n_features")
        if n_feat:
            header.append(f"({n_feat}f{'/rich' if model.get('rich_entry') else ''}) ",
                          style="dim")
        cfg_pol = model.get("exit_policy")
        spec_pol = model.get("spec_exit_policy")
        if cfg_pol:
            header.append(f"exit={cfg_pol} ", style="cyan")
        if spec_pol and cfg_pol and spec_pol != cfg_pol:
            header.append(f" CONFIG!=SPEC({spec_pol}) ", style="bold white on red")
        if era.get("start_t"):
            header.append(f"era {fmt_age(time.time()-era['start_t'])}", style="dim")
        header.append("\n")
        header.append("this-run: ", style="bold")
        ready = era.get("ready") or 0
        fires = era.get("fires") or 0
        header.append(f"ready={ready} fires={fires} ", style="white")
        if ready:
            fr = 100.0 * fires / ready
            exp = hold.get("test_fire_rate")
            exp_s = f" vs spec {exp*100:.2f}%" if isinstance(exp, (int, float)) else ""
            fr_style = "cyan"
            if isinstance(exp, (int, float)) and exp > 0:
                ratio = (fr / 100.0) / exp
                fr_style = "green" if 0.4 <= ratio <= 2.5 else "bold red"
            header.append(f"[{fr:.2f}%{exp_s}] ", style=fr_style)
        closed_era = era.get("closed") or 0
        header.append(f"closed={closed_era} ", style="white")
        mpn = era.get("mean_policy_net")
        if isinstance(mpn, (int, float)):
            header.append("policy ")
            header.append_text(fmt_sol(mpn * bet_sol))
            header.append(" SOL/bet ")
            wp = era.get("win_pct_policy")
            if isinstance(wp, (int, float)):
                header.append(f"win {wp:.0f}% ", style="white")
        mbn = era.get("mean_book_net")
        if isinstance(mbn, (int, float)):
            header.append(f"(book {mbn*100:+.1f}%)", style="dim")
        header.append("\n")
        header.append("all-time: ", style="bold")
        header.append(f"open={n_open} closed={n_closed} ", style="white")
        header.append("book mean ")
        header.append_text(fmt_sol(mean_sol))
        if isinstance(mean_net, (int, float)):
            header.append(f" SOL/bet (={mean_net*100:+.1f}%)", style="dim")
        header.append(" win% ", style="dim")
        header.append(f"{win_pct:.0f}" if isinstance(win_pct, (int, float)) else "-",
                      style="white")
        header.append("  [book accounting, restored across model eras]", style="dim")
        self.query_one("#status-panel", Static).update(header)

    def _render_risk(self):
        status = load_json_safe(self.root/"bot_data"/"status.json") or {}
        stats = status.get("stats", {})
        # Load config to get limits
        cfg_path = self.root/"config.yaml"
        max_concurrent = 10; rate_limit = 6; daily_loss = -5.0
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            max_concurrent = cfg.get("risk", {}).get("max_concurrent_positions", 10)
            rate_limit = cfg.get("risk", {}).get("max_fires_per_minute", 6)
            daily_loss = cfg.get("risk", {}).get("daily_loss_limit_sol", -5.0)
        except Exception: pass

        n_open = status.get("n_open_paper", 0)
        bet_sol = status.get("bet_sol")
        if bet_sol is None:
            try:
                import yaml
                cfg = yaml.safe_load((self.root/"config.yaml").read_text()) or {}
                bet_sol = float(cfg.get("bot", {}).get("bet_sol", 1.0))
            except Exception:
                bet_sol = 1.0
        rets_total = None
        rets_alltime = None
        n_alltime_mints = 0
        # P&L: GROUND TRUTH from broker_jito.jsonl — the actual SOL the
        # live broker received minus the SOL paid in. This is what we ACTUALLY
        # made, not an analytical estimate from PaperBook's GREEN scheme nor a
        # replay (which would be missing the death-cut logic the harness applies).
        # Per mint: net_lam = sum(sell.expected_sol_out) - sum(buy.sol)
        # Multiply by bet_sol because broker logs use nominal sol=1.0 per buy
        # but the actual bet is bet_sol. Computed for BOTH last-24h (daily) and
        # since-inception (all-time) in a single pass over broker_jito.jsonl.
        cutoff = time.time() - 86400
        era_start_r = (status.get("era") or {}).get("start_t") or 0
        per_mint_24h = {}     # mint -> [buy_lam, sell_lam, first_t] for last 24h
        per_mint_all = {}     # mint -> [buy_lam, sell_lam] for ALL time
        try:
            with open(self.root/"logs"/"broker_jito.jsonl") as f:
                for ln in f:
                    if "DRY_RUN" not in ln and "LIVE_submit" not in ln: continue
                    try: r = json.loads(ln)
                    except Exception: continue
                    t = r.get("t", 0)
                    m = r.get("mint")
                    if not m: continue
                    op = r.get("op")
                    delta_b = int(r.get("sol", 0) * 1e9) if op == "buy" else 0
                    delta_s = int(r.get("expected_sol_out", 0)) if op in ("sell_slice", "sell_all") else 0
                    rec_all = per_mint_all.setdefault(m, [0, 0])
                    rec_all[0] += delta_b; rec_all[1] += delta_s
                    if t >= cutoff:
                        rec_24h = per_mint_24h.setdefault(m, [0, 0, t])
                        rec_24h[0] += delta_b; rec_24h[1] += delta_s
                        rec_24h[2] = min(rec_24h[2], t)
        except Exception: pass
        # Broker logs are in ACTUAL SOL: the harness passes sol=bet_sol per buy
        # and sells log the real position's expected_sol_out. The previous
        # "multiply by bet_sol" rescaling double-applied the bet size and
        # understated P&L 10x at bet=0.1. Completed roundtrips only: buys with
        # no sell yet are open positions, not realized losses.
        nets_real, n_open_rt = [], 0
        nets_era = []
        for m, (buy_lam, sell_lam, t0) in per_mint_24h.items():
            if buy_lam <= 0: continue
            if sell_lam <= 0:
                n_open_rt += 1
                continue
            net = (sell_lam - buy_lam) / 1e9
            nets_real.append(net)
            if t0 >= era_start_r:
                nets_era.append(net)
        if nets_real:
            rets_total = sum(nets_real)
        rets_era = sum(nets_era) if nets_era else None
        self._daily_pnl_coverage = (len(nets_real), n_open_rt)
        nets_all = []
        for m, (buy_lam, sell_lam) in per_mint_all.items():
            if buy_lam <= 0 or sell_lam <= 0: continue
            nets_all.append((sell_lam - buy_lam) / 1e9)
        if nets_all:
            rets_alltime = sum(nets_all)
            n_alltime_mints = len(nets_all)

        cb_active_count = stats.get("circuit_breaker_active", 0)
        refusals = sum([stats.get(k, 0) for k in
                        ("risk_refusal_max_concurrent", "risk_refusal_rate_limit",
                         "risk_refusal_daily_loss", "risk_refusal_failure_rate")])

        out = Text()
        out.append("RISK CIRCUITS\n", style="bold yellow")
        out.append("concurrent  ", style="white")
        col = "green" if n_open < max_concurrent * 0.8 else "yellow" if n_open < max_concurrent else "red"
        out.append(f"{n_open}/{max_concurrent}", style=f"bold {col}")
        out.append("\n")
        out.append("rate cap    ", style="white")
        out.append(f"≤{rate_limit} fires/min\n", style="dim")
        # THIS-ERA broker roundtrips are what the strategy is doing; the 24h
        # number blends every model era in the window (a broken yesterday is
        # not the deployed model's loss) so it is shown dim, uncolored, and
        # never against the loss limit.
        out.append("era P&L     ", style="white")
        out.append_text(fmt_sol(rets_era))
        out.append(f"  ({len(nets_era)} rt, broker) / ", style="dim")
        out.append(f"{daily_loss:.1f}", style="dim")
        out.append("\n")
        out.append("24h all-era ", style="white")
        out.append(f"{rets_total:+.4f}" if rets_total is not None else "-", style="dim")
        out.append("  [blends model eras; not a strategy signal]\n", style="dim")
        out.append("all-time    ", style="white")
        out.append_text(fmt_sol(rets_alltime))
        out.append(f"  (n={n_alltime_mints} roundtrips, actual SOL)\n", style="dim")
        era_r = status.get("era") or {}
        mpn_r = era_r.get("mean_policy_net")
        ncl_r = era_r.get("closed") or 0
        out.append("this-run    ", style="white")
        if isinstance(mpn_r, (int, float)) and ncl_r:
            out.append_text(fmt_sol(mpn_r * ncl_r * bet_sol))
            out.append(f"  ({ncl_r} closed, policy acct)\n", style="dim")
        else:
            out.append("-\n", style="dim")
        out.append("refusals    ", style="white")
        out.append(f"{refusals} total\n", style="yellow" if refusals > 0 else "dim")
        out.append("breaker     ", style="white")
        if cb_active_count > 0:
            out.append("ARMED", style="bold red blink")
        else:
            out.append("clear", style="green")
        self.query_one("#risk-panel", Static).update(out)

    def _render_latency(self):
        """Recent bundle assembly latency + scoring delay (from broker_jito.jsonl).
        Pulls last 500 bundles, then filters to those within the last 30 min so
        the percentiles reflect CURRENT execution behavior — not whatever the
        oldest 200 in the file happen to be. Bot fires ~few/hr so 30min gives a
        responsive but stable window."""
        raw = tail_jsonl(self.root/"logs"/"broker_jito.jsonl", 500)
        # Time-window filter
        t_now = time.time()
        WINDOW_SEC = 30 * 60   # 30 min rolling window
        bundles = [b for b in raw if b.get("t", 0) >= t_now - WINDOW_SEC]
        if not bundles:
            # fall back to whatever we have so panel isn't blank
            bundles = raw[-50:]
        out = Text("EXEC LATENCY\n", style="bold #00ff66")
        if not bundles:
            out.append("(no bundles yet)", style="dim")
            self.query_one("#latency-panel", Static).update(out)
            return
        # Strategy from config.yaml
        try:
            import yaml
            cfg = yaml.safe_load((self.root/"config.yaml").read_text()) or {}
            policy = cfg.get("exit", {}).get("policy", "?")
            br = cfg.get("broker", {})
            tip_lam = br.get("tip_lamports", 0)
            # Split slippage (2026-06-11); fall back to the legacy shared knob
            slip_legacy = br.get("slippage_bps", 0)
            slip_buy = br.get("slippage_bps_buy", slip_legacy)
            slip_sell = br.get("slippage_bps_sell", slip_legacy)
        except Exception:
            policy, tip_lam, slip_buy, slip_sell = "?", 0, 0, 0
        out.append("strategy   ", style="white")
        out.append(f"{policy}", style="bold cyan")
        # tip shown is the BASE; live buys add the adaptive contention bump
        # (tip floors 400k/1.5M by tier, outbid visible p90 x1.5, cap 5M) and
        # every tx also carries a ComputeBudget priority fee. Submission is
        # single-tx via Jito sendTransaction proxy, not sendBundle.
        out.append(f"  tip base {tip_lam:,} lam +adaptive  "
                   f"slip buy {slip_buy} / sell {slip_sell} bps\n", style="dim")
        # Stats on the recent bundles
        bh = [b.get("bh_age_ms", 0) for b in bundles if b.get("bh_age_ms") is not None]
        asm = [b.get("asm_ms", 0) for b in bundles if b.get("asm_ms") is not None]
        ops = {}
        for b in bundles: ops[b.get("op","?")] = ops.get(b.get("op","?"),0) + 1
        n_with_slot = sum(1 for b in bundles if b.get("slot") is not None)
        if bh:
            import statistics as s
            q = sorted(bh)
            p50 = q[len(q)//2]; p90 = q[int(len(q)*0.9)] if len(q) > 1 else q[-1]
            col = "green" if p90 < 500 else "yellow" if p90 < 1000 else "red"
            out.append("bh_age     ", style="white")
            out.append(f"p50 {p50:.0f}ms  p90 ", style="white")
            out.append(f"{p90:.0f}ms", style=f"bold {col}")
            out.append(f"  (n={len(bh)} / last {WINDOW_SEC//60}min)\n", style="dim")
        if asm:
            q = sorted(asm)
            p50 = q[len(q)//2]; p90 = q[int(len(q)*0.9)] if len(q) > 1 else q[-1]
            out.append("asm_ms     ", style="white")
            # asm_ms is float-ms (sub-ms is real and meaningful for DRY_RUN
            # where there's no signing/serialization on the sell paths)
            out.append(f"p50 {p50:.3f}ms  p90 {p90:.3f}ms\n",
                       style="white" if p90 < 50 else "yellow")
        out.append("txs        ", style="white")   # single-tx sendTransaction proxy (bundles retired 2026-06-11)
        out.append(" ".join(f"{k}:{v}" for k, v in ops.items()), style="white")
        out.append("\n")
        out.append("with slot  ", style="white")
        col = "green" if n_with_slot == len(bundles) else "yellow" if n_with_slot > 0 else "dim"
        out.append(f"{n_with_slot}/{len(bundles)} ", style=f"bold {col}")
        out.append("(gRPC source feeds tx.slot)", style="dim")
        self.query_one("#latency-panel", Static).update(out)

    def _render_recon(self):
        status = load_json_safe(self.root/"bot_data"/"status.json") or {}
        recon = status.get("recon", {})
        pending = status.get("pending_bundles", 0)
        # recent recon log tail
        recent = tail_jsonl(self.root/"logs"/"broker_recon.jsonl", 50)
        era_start_rc = (status.get("era") or {}).get("start_t") or 0
        recent = [r for r in recent if r.get("t", 0) >= era_start_rc]
        landed = [r for r in recent if r.get("kind") == "landed"]
        failed = [r for r in recent if r.get("kind") == "failed"]

        out = Text()
        out.append("RECON  ", style="bold cyan")
        if pending > 0:
            out.append(f"pending={pending}  ", style="bold yellow")
        else:
            out.append("pending=0  ", style="dim")
        n_outcomes = recon.get("n_outcomes", 0)
        if n_outcomes == 0:
            out.append("(DRY_RUN; no on-chain submissions)", style="dim")
        else:
            land_rate = recon.get("land_rate", 0)
            col = "green" if land_rate >= 0.8 else "yellow" if land_rate >= 0.5 else "red"
            out.append(f"land_rate ", style="white")
            out.append(f"{land_rate*100:.0f}%", style=f"bold {col}")
            out.append(f"  landed={recon.get('n_landed',0)}  failed={recon.get('n_failed',0)}  ", style="white")
            lat_p50 = recon.get("landing_latency_p50_s")
            if lat_p50 is not None:
                out.append(f"latency p50 {lat_p50:.1f}s  ", style="white")
            tip_p50 = recon.get("landed_tip_p50")
            if tip_p50 is not None:
                out.append(f"tip p50 {tip_p50:,.0f} lam", style="white")
        if failed:
            out.append("\nrecent failures: ", style="dim")
            for r in failed[-3:]:
                out.append(f"{r.get('reason','?')}({r.get('tip_lam','?')})  ", style="red")
        self.query_one("#recon-panel", Static).update(out)

    def _render_drift(self):
        drift = tail_jsonl(self.root/"logs"/"drift_log.jsonl", 5)
        out = Text()
        out.append("DRIFT  ", style="bold #ff9900")
        if not drift:
            out.append("no drift checks yet  ", style="dim")
            out.append("(install with `./scripts/pumpfun_ctl.sh install-drift-timer` for daily auto-checks)",
                       style="dim italic")
        else:
            last = drift[-1]
            flags = last.get("flags", [])
            t_ago = time.time() - last.get("t", 0)
            era_start_d = ((load_json_safe(self.root/"bot_data"/"status.json") or {})
                           .get("era") or {}).get("start_t") or 0
            if last.get("t", 0) < era_start_d:
                out.append("last check ", style="dim")
                out.append(fmt_age(t_ago), style="yellow")
                out.append(" ago — PRE-DEPLOY ERA, ignore (re-run drift check for the current model)",
                           style="dim yellow")
                if self._last_action_msg:
                    out.append("   |  ", style="dim")
                    out.append(self._last_action_msg, style="bold yellow")
                self.query_one("#drift-panel", Static).update(out)
                return
            # Color the age: green if < 26h (daily timer), yellow if stale, red if very stale
            age_style = "green" if t_ago < 26*3600 else "yellow" if t_ago < 48*3600 else "bold red"
            out.append(f"last check ", style="dim")
            out.append(fmt_age(t_ago), style=age_style)
            out.append(" ago  ", style="dim")
            if flags:
                out.append("ALERT: ", style="bold red")
                out.append(", ".join(flags), style="red")
            else:
                out.append("clean", style="green")
            wr = last.get("winner_rate_pct")
            if wr is not None:
                out.append(f"  winner rate {wr:.1f}%", style="white")
            if t_ago > 26*3600:
                out.append("  (timer not installed: run `install-drift-timer`)", style="yellow")
        # Action feedback: when user presses 'o' (open mint) or 'c' (copy URL),
        # append the result here. Otherwise show the key-binding hint.
        if self._last_action_msg:
            out.append("   |  ", style="dim")
            out.append(self._last_action_msg, style="bold yellow")
        else:
            out.append("   |  press 'o' to open focused mint, 'c' to copy URL", style="dim")
        self.query_one("#drift-panel", Static).update(out)

    def _render_open_positions(self):
        tbl = self.query_one("#open-positions", DataTable)
        saved_row = tbl.cursor_row
        saved_scroll_y = getattr(tbl, "scroll_y", 0)
        tbl.clear()
        # Open positions from the FULL-scan open/close cache built by
        # _render_recent_fires (refreshed every 5th tick). The previous
        # 5000-row tail of positions.jsonl was ~95% snaps, so an open event
        # older than the tail window made a live position invisible here.
        cache = getattr(self, "_fires_cache", None)
        latest_open = cache[3] if cache else {}
        latest_close = cache[2] if cache else {}
        active = []
        for m, o in latest_open.items():
            c = latest_close.get(m)
            if c is None or c.get("t", 0) < o.get("t", 0):
                active.append({"open": o, "snaps": [], "closed": None})
        # last-snap fallback for ret display still comes from the recent tail
        rows = tail_jsonl(self.root/"bot_data"/"positions.jsonl", 2000)
        snaps_by_mint: dict[str, dict] = {}
        for r in rows:
            if r.get("kind") == "snap" and r.get("mint"):
                snaps_by_mint[r["mint"]] = r
        for v in active:
            sn = snaps_by_mint.get(v["open"].get("mint"))
            if sn:
                v["snaps"].append(sn)
        # Walk shadow_run.jsonl to find: slice fire count + phase + latest path_snap state
        sr = tail_jsonl(self.root/"bot_data"/"shadow_run.jsonl", 20000)
        slice_count_by_mint = {}        # mint -> n slices fired live
        phase_by_mint = {}
        latest_path_by_mint = {}        # mint -> latest path_snap row
        for r in sr:
            m = r.get("mint")
            if not m: continue
            if r.get("kind") == "live_scale_slice":
                slice_count_by_mint[m] = slice_count_by_mint.get(m, 0) + 1
                phase_by_mint[m] = r.get("phase", "?")
            elif r.get("kind") == "path_snap":
                latest_path_by_mint[m] = r
        active.sort(key=lambda v: -(v["open"].get("t", 0)))
        self._row_mints_open = []
        for v in active[:20]:
            o = v["open"]
            full_mint = o["mint"]
            mint = _mint_link(full_mint)
            score = o.get("entry_score")
            age_s = time.time() - o.get("t", time.time())
            age = fmt_age(age_s)
            n_slices = slice_count_by_mint.get(o["mint"], 0)
            slice_disp = f"{n_slices}/8"
            # latest path_snap gives ground-truth current ret + run_max
            ps = latest_path_by_mint.get(o["mint"])
            if ps and isinstance(ps.get("path_feats"), dict):
                last_ret = ps["path_feats"].get("ret")
                rmax = ps["path_feats"].get("run_max_ret")
            else:
                last_ret = v["snaps"][-1].get("ret") if v["snaps"] else None
                rmax = None
            phase = phase_by_mint.get(o["mint"], "-")
            # If a position is older than ~10s and STILL has no path data, the
            # token went silent post-buy (typical sniper-bundle-then-vanish
            # pattern). Make this state explicit instead of showing "- - -"
            # which looks like a UI bug. The watchdog will close it at 5min.
            silent = (last_ret is None and rmax is None and age_s > 10)
            if silent:
                ret_disp = Text("silent", style="dim yellow")
                rmax_disp = Text("·", style="dim")
                phase_disp = Text("await_trades", style="dim yellow") if phase == "-" else phase
            else:
                ret_disp = (f"{last_ret:+.3f}" if isinstance(last_ret, (int,float)) else "-")
                rmax_disp = (f"{rmax:+.3f}" if isinstance(rmax, (int,float)) else "-")
                phase_disp = phase
            tbl.add_row(mint,
                        f"{score:.4f}" if isinstance(score, (int,float)) else "-",
                        age, slice_disp, ret_disp, rmax_disp, phase_disp)
            self._row_mints_open.append(full_mint)
        if not active:
            tbl.add_row("(none open)", "", "", "", "", "", "")
        try:
            if 0 <= saved_row < tbl.row_count:
                tbl.move_cursor(row=saved_row, animate=False)
            if saved_scroll_y:
                tbl.scroll_to(y=saved_scroll_y, animate=False)
        except Exception: pass

    def _render_recent_fires(self):
        tbl = self.query_one("#recent-fires", DataTable)
        # Preserve cursor + scroll across the 1s refresh tick. Without this
        # tbl.clear() resets cursor_row to 0 every second, kicking the user
        # back to the top of the table and making o/c always show the first
        # rows URL regardless of where they tried to scroll.
        saved_row = tbl.cursor_row
        saved_scroll_y = getattr(tbl, "scroll_y", 0)
        tbl.clear()
        # Full-file scan for fires (NOT tail_jsonl) — fires are rare and
        # tail_jsonl's 4MB byte window drops most of them when shadow_run.jsonl
        # is big and dominated by frequent entry_decision/path_snap events.
        # We line-grep for fire=true on every line; cheap because shadow_run.jsonl
        # is JSONL. Also keep last 10K path_snaps for the exit_ret fallback.
        self._scan_tick = getattr(self, "_scan_tick", 0) + 1
        cache = getattr(self, "_fires_cache", None)
        if cache is not None and self._scan_tick % 5 != 1:
            fires, sr, latest_close_by_mint, latest_open_by_mint = cache
            use_cache = True
        else:
            use_cache = False
        fires = [] if not use_cache else fires
        sr = [] if not use_cache else sr
        if not use_cache:
          try:
            with open(self.root/"bot_data"/"shadow_run.jsonl") as f:
                for ln in f:
                    if '"kind": "entry_decision"' in ln and '"fire": true' in ln:
                        try: fires.append(json.loads(ln))
                        except Exception: continue
                    elif '"kind": "path_snap"' in ln:
                        try: sr.append(json.loads(ln))
                        except Exception: continue
                        if len(sr) > 10000: sr = sr[-10000:]
          except Exception: pass
        if not use_cache:
          latest_close_by_mint = {}
          latest_open_by_mint = {}    # for duration: close.t - open.t
          # Full-file scan filtered to open/close (snaps dominate the file at ~95%,
          # so a line-tail would clobber close events for older fires — same bug
          # we hit in _render_risk's daily-PnL panel).
          try:
            with open(self.root/"bot_data"/"positions.jsonl") as f:
                for ln in f:
                    if '"kind": "close"' not in ln and '"kind": "open"' not in ln:
                        continue
                    try: r = json.loads(ln)
                    except Exception: continue
                    k = r.get("kind"); m = r.get("mint")
                    if not m: continue
                    if k == "close":
                        if r.get("t", 0) > latest_close_by_mint.get(m, {}).get("t", 0):
                            latest_close_by_mint[m] = r
                    elif k == "open":
                        if r.get("t", 0) > latest_open_by_mint.get(m, {}).get("t", 0):
                            latest_open_by_mint[m] = r
          except Exception:
            pass
          self._fires_cache = (fires, sr, latest_close_by_mint, latest_open_by_mint)
        # bet_sol for fractional -> absolute conversion
        try:
            import yaml
            cfg = yaml.safe_load((self.root/"config.yaml").read_text()) or {}
            bet_sol = float(cfg.get("bot", {}).get("bet_sol", 1.0))
        except Exception:
            bet_sol = 1.0
        fires.sort(key=lambda r: -r.get("t", 0))
        status_doc = load_json_safe(self.root/"bot_data"/"status.json") or {}
        era_start = (status_doc.get("era") or {}).get("start_t") or 0
        self._row_mints_fires = []
        # Show many more recent fires now that full-file scan finds them all.
        # 100 rows is comfortable on a normal terminal; older history is still
        # in the JSON file for offline analysis.
        for f in fires[:100]:
            full_mint = f.get("mint", "")
            mint = _mint_link(full_mint)
            self._row_mints_fires.append(full_mint)
            tstr = time.strftime("%H:%M:%S", time.gmtime(f.get("t", 0)))
            score = f.get("score")
            slot = f.get("ev_slot")
            close = latest_close_by_mint.get(f.get("mint"))
            # exit_ret comes from the close event when present (only logged for
            # fires after the exit_ret patch landed). For older closes / positions
            # where the close didn't include it, fall back to the last path_snap
            # ret we saw for this mint in shadow_run.jsonl (handled below).
            exit_ret_val = close.get("exit_ret") if close else None
            if close and isinstance(close.get("net_return"), (int, float)):
                status = "closed"
                # PaperBook kind ('rider','cut','hold','skip') alone is ambiguous —
                # 'hold' can mean "held to terminal then watchdog-closed at -35%".
                # Display kind + close reason together so it's clear what actually
                # happened. reason is 'stale' | 'shutdown' | 'slices_exhausted' |
                # 'death_cut' | 'runner_exit' | 'restart'.
                base_kind = close.get("exit_kind", "?")
                reason = close.get("reason", "")
                # Compact label: if reason adds info beyond kind, append.
                # kind=hold + reason=stale  -> "hold/stale"
                # kind=rider + reason=slices_exhausted -> "rider"
                # kind=rider + reason=shutdown -> "rider/shutdown"  (force-closed)
                # kind=cut + reason=death_cut -> "cut"
                if reason in ("slices_exhausted", "death_cut", "runner_exit", ""):
                    exit_kind = base_kind
                else:
                    exit_kind = f"{base_kind}/{reason}"
                # Prefer the LIVE policy's realized P&L; fall back to PaperBook
                # GREEN reference if not present (legacy close events written
                # before live_policy_net wiring landed).
                live = close.get("live_policy_net")
                net_frac = live if isinstance(live, (int, float)) else close["net_return"]
                net_abs = net_frac * bet_sol
                net_str = f"{net_abs:+.4f}"
            elif close:
                status = "closed"; exit_kind = close.get("exit_kind", "?"); net_str = "-"
            else:
                status = "open"; exit_kind = "-"; net_str = "-"
            # if we don't have a logged exit_ret, fall back to the last snap ret we
            # observed for this mint in shadow_run.jsonl (one pass per render is
            # fine — we already loaded sr above)
            if exit_ret_val is None:
                last_snap_ret = None
                tgt = f.get("mint")
                for r in sr:
                    if r.get("mint") == tgt and r.get("kind") == "path_snap":
                        pf = r.get("path_feats") or {}
                        if "ret" in pf: last_snap_ret = pf["ret"]
                exit_ret_val = last_snap_ret
            exit_ret_str = f"{exit_ret_val:+.3f}" if isinstance(exit_ret_val, (int, float)) else "-"
            # Position duration: close.t - open.t for closed rows; for still-open
            # rows show live age = now - open.t. Falls back to fire.t when an
            # open record isn't found (e.g. positions.jsonl truncated).
            o_rec = latest_open_by_mint.get(full_mint)
            open_t = (o_rec.get("t") if o_rec else None) or f.get("t")
            if open_t:
                if close and isinstance(close.get("t"), (int, float)):
                    dur_s = close["t"] - open_t
                else:
                    dur_s = time.time() - open_t
                dur_str = fmt_age(dur_s)
            else:
                dur_str = "-"
            # Shred signal columns. Read from the fire's entry_decision
            # record. Pre-shred-window-era fires won't have the field, so
            # default to "-/-" and "-". The tip-multiplier logic mirrors
            # the harness's tier decisions exactly.
            ss = f.get("shred_signal") or {}
            sn_500 = ss.get("shred_buy_500ms")
            sn_2k  = ss.get("shred_buy_2000ms")
            if isinstance(sn_500, int) and isinstance(sn_2k, int):
                shred_str = f"{sn_500}/{sn_2k}"
            else:
                shred_str = "-/-"
            # Mirror the tier logic from shadow_harness:
            #   tier 2 (4×): >=4 buys/500ms OR >=8 buys/2s
            #   tier 1 (2×): >=2 buys/500ms AND any jito-tipped
            tip_mult = "-"
            if isinstance(sn_500, int) and isinstance(sn_2k, int):
                jt = ss.get("shred_jito_tip_rate_2000ms", 0) or 0
                if sn_500 >= 4 or sn_2k >= 8:
                    tip_mult = "4×"
                elif sn_500 >= 2 and jt > 0:
                    tip_mult = "2×"
                else:
                    tip_mult = "1×"
            # Render shred column with subtle color if active
            if isinstance(sn_500, int) and sn_500 > 0:
                shred_cell = Text(shred_str, style="bold cyan")
            else:
                shred_cell = Text(shred_str, style="dim")
            if tip_mult in ("2×", "4×"):
                tip_cell = Text(tip_mult, style="bold yellow")
            else:
                tip_cell = Text(tip_mult, style="dim")
            era_cell = (Text("now", style="green") if f.get("t", 0) >= era_start
                        else Text("prev", style="dim"))
            tbl.add_row(tstr, era_cell, mint,
                        f"{score:.4f}" if isinstance(score, (int,float)) else "-",
                        f"{slot}" if slot is not None else "-",
                        status, exit_kind, exit_ret_str, dur_str, net_str,
                        shred_cell, tip_cell)
        if not fires:
            tbl.add_row("-", "", "(no fires yet)", "", "", "", "", "", "", "", "", "")
        # Restore cursor + scroll position
        try:
            if 0 <= saved_row < tbl.row_count:
                tbl.move_cursor(row=saved_row, animate=False)
            if saved_scroll_y:
                tbl.scroll_to(y=saved_scroll_y, animate=False)
        except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--refresh", type=float, default=1.0)
    args = ap.parse_args()
    PumpfunDashboard(Path(args.root), refresh_s=args.refresh).run()


if __name__ == "__main__":
    main()
