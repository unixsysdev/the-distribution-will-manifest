#!/usr/bin/env bash
# pumpfun_ctl.sh — control script for the pump.fun bot + gRPC capture on sol.
# Run as root on sol (the systemd units are system-level).
#
# Usage:
#   pumpfun_ctl.sh install        # copy unit files, daemon-reload (does NOT start)
#   pumpfun_ctl.sh enable         # enable both units to start on boot
#   pumpfun_ctl.sh start-capture  # start the gRPC training-data recorder
#   pumpfun_ctl.sh start-bot      # start the bot (PAPER mode)
#   pumpfun_ctl.sh stop-capture   # stop recorder
#   pumpfun_ctl.sh stop-bot       # stop bot
#   pumpfun_ctl.sh status         # systemd status + quick file stats
#   pumpfun_ctl.sh logs-bot       # follow bot journal
#   pumpfun_ctl.sh logs-capture   # follow capture journal
#   pumpfun_ctl.sh snapshot       # print bot status.json + capture disk usage
#   pumpfun_ctl.sh go-live        # PRINT the exact steps to flip live (does NOT do it)
set -euo pipefail

ROOT=/root/the-distribution-will-manifest
UNIT_SRC="$ROOT/systemd"
UNIT_DST=/etc/systemd/system
CAP=pumpfun-grpc-capture.service
BOT=pumpfun-bot.service

cmd="${1:-status}"

case "$cmd" in
  install)
    # data dirs MUST exist before systemd sets up ReadWritePaths namespaces
    mkdir -p "$ROOT/grpc_capture" "$ROOT/bot_data"
    cp "$UNIT_SRC/$CAP" "$UNIT_DST/$CAP"
    cp "$UNIT_SRC/$BOT" "$UNIT_DST/$BOT"
    systemctl daemon-reload
    echo "installed $CAP and $BOT; created data dirs; daemon-reloaded. (not started — use start-capture / start-bot)"
    ;;
  enable)
    systemctl enable "$CAP" "$BOT"
    echo "enabled $CAP and $BOT on boot"
    ;;
  disable)
    systemctl disable "$CAP" "$BOT" || true
    echo "disabled $CAP and $BOT"
    ;;
  start-capture) systemctl start "$CAP"; echo "started $CAP"; sleep 2; systemctl --no-pager status "$CAP" | head -6 || true ;;
  start-bot)     systemctl start "$BOT"; echo "started $BOT"; sleep 2; systemctl --no-pager status "$BOT" | head -6 || true ;;
  stop-capture)  systemctl stop "$CAP"; echo "stopped $CAP" ;;
  stop-bot)      systemctl stop "$BOT"; echo "stopped $BOT" ;;
  restart-capture) systemctl restart "$CAP"; echo "restarted $CAP" ;;
  restart-bot)     systemctl restart "$BOT"; echo "restarted $BOT" ;;
  status)
    echo "=== systemd ==="
    systemctl --no-pager status "$CAP" | head -4 || true
    echo
    systemctl --no-pager status "$BOT" | head -4 || true
    echo
    echo "=== capture dir ==="
    if [ -d "$ROOT/grpc_capture" ]; then
      du -sh "$ROOT/grpc_capture" 2>/dev/null || true
      ls -t "$ROOT/grpc_capture" 2>/dev/null | head -3
    else echo "(no capture dir yet)"; fi
    echo
    echo "=== bot data ==="
    if [ -f "$ROOT/bot_data/status.json" ]; then
      cat "$ROOT/bot_data/status.json"
    else echo "(no bot status.json yet)"; fi
    ;;
  logs-bot)      journalctl -u "$BOT" -f -n 50 ;;
  logs-capture)  journalctl -u "$CAP" -f -n 50 ;;
  snapshot)
    echo "--- bot status.json ---"
    [ -f "$ROOT/bot_data/status.json" ] && cat "$ROOT/bot_data/status.json" || echo "(none)"
    echo
    echo "--- capture disk ---"
    du -sh "$ROOT/grpc_capture" 2>/dev/null || echo "(none)"
    df -h "$ROOT" | tail -1
    ;;
  report)
    shift  # consume the 'report' arg so $@ holds only extra flags
    cd "$ROOT" && source venv/bin/activate && python tools/overnight_report.py "$@"
    ;;
  drift)
    shift
    cd "$ROOT" && source venv/bin/activate && python tools/drift_monitor.py "$@"
    ;;
  ab-replay)
    shift
    cd "$ROOT" && source venv/bin/activate && python tools/strategy_ab_replay.py "$@"
    ;;
  retrain-check)
    cd "$ROOT" && source venv/bin/activate && python tools/auto_retrain.py
    ;;
  retrain-now)
    cd "$ROOT" && source venv/bin/activate && python tools/auto_retrain.py --execute
    ;;
  dashboard)
    shift
    cd "$ROOT" && source venv/bin/activate && python tools/dashboard.py "$@"
    ;;
  rl-backtest)
    # Offline Fitted-Q-Iteration on May parquets. Trains a discrete-state MDP
    # policy and compares to K_combined. Best OOS uplift discovered so far
    # but tail-driven and bang-bang — see docs/notes/SHADOW_HARNESS_LOG.md for caveats.
    shift
    cd "$ROOT" && source venv/bin/activate && python tools/rl_backtest.py "$@"
    ;;
  ac-backtest)
    # Almgren-Chriss vs K_combined backtest on May parquets (full or OOS).
    # usage: ac-backtest                              # full TRAIN set, default kappa sweep
    #        ac-backtest --data-dir data/pumpfun_continuation_oos_K7_fresh
    #        ac-backtest --subset 1000                # quick sanity
    shift
    cd "$ROOT" && source venv/bin/activate && python tools/ac_backtest.py "$@"
    ;;
  gate-replay)
    # what did the two model heads (entry + recovery) say about ONE token?
    # usage: gate-replay <mint>                # specific mint
    #        gate-replay --last 1              # most recent fire
    #        gate-replay --last 5              # last 5 fires
    #        gate-replay --dumped              # only fires that died or lost >10%
    shift
    cd "$ROOT" && source venv/bin/activate && python tools/token_gate_replay.py "$@"
    ;;
  policy-check)
    # dry-run: prints what auto_policy would pick on the last N fires, no swap
    cd "$ROOT" && source venv/bin/activate && python tools/auto_policy.py
    ;;
  policy-now)
    # execute: swaps config.yaml + restarts bot if gates pass
    cd "$ROOT" && source venv/bin/activate && python tools/auto_policy.py --execute
    ;;
  install-drift-timer)
    cp "$UNIT_SRC/pumpfun-drift-monitor.service" "$UNIT_DST/"
    cp "$UNIT_SRC/pumpfun-drift-monitor.timer"   "$UNIT_DST/"
    systemctl daemon-reload
    systemctl enable --now pumpfun-drift-monitor.timer
    echo "installed and enabled pumpfun-drift-monitor.timer (daily)"
    systemctl list-timers pumpfun-drift-monitor.timer --no-pager 2>/dev/null | head -3
    ;;
  install-policy-timer)
    cp "$UNIT_SRC/pumpfun-auto-policy.service" "$UNIT_DST/"
    cp "$UNIT_SRC/pumpfun-auto-policy.timer"   "$UNIT_DST/"
    systemctl daemon-reload
    systemctl enable --now pumpfun-auto-policy.timer
    echo "installed and enabled pumpfun-auto-policy.timer (every 4h)"
    systemctl list-timers pumpfun-auto-policy.timer --no-pager 2>/dev/null | head -3
    ;;
  install-retrain-timer)
    cp "$UNIT_SRC/pumpfun-auto-retrain.service" "$UNIT_DST/"
    cp "$UNIT_SRC/pumpfun-auto-retrain.timer"   "$UNIT_DST/"
    systemctl daemon-reload
    systemctl enable --now pumpfun-auto-retrain.timer
    echo "installed and enabled pumpfun-auto-retrain.timer (weekly Sun 02:00 UTC)"
    echo "  (gates internally on capture archive >= 3 days; first firings will skip until ready)"
    systemctl list-timers pumpfun-auto-retrain.timer --no-pager 2>/dev/null | head -3
    ;;
  install-all-timers)
    "$0" install-drift-timer
    "$0" install-policy-timer
    "$0" install-retrain-timer
    ;;
  policy-impact)
    shift
    cd "$ROOT" && source venv/bin/activate && python tools/policy_impact.py "$@"
    ;;
  go-live)
    cat <<'EOF'
=== FLIP TO LIVE (manual, deliberate) ===
Live execution is double-gated. To actually trade you must do BOTH:

  1) Edit /etc/systemd/system/pumpfun-bot.service:
       - add `--live` to the ExecStart line
       - uncomment:  Environment=PUMPFUN_LIVE_OK=1
  2) systemctl daemon-reload && systemctl restart pumpfun-bot

Before doing that, confirm the configured wallet is the TEST wallet (no capital).
The JitoBroker still logs `status:not_yet_wired` until send_bundle is implemented,
so even in live mode it will not yet submit real bundles — that wiring is a
separate, explicitly-authorized step.
EOF
    ;;
  *) echo "unknown command: $cmd"; exit 1 ;;
esac
