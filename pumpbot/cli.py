"""Single CLI entry over the existing scripts (stage 2a: thin dispatch via
runpy so behavior is byte-identical to invoking the scripts directly; the
script mains move in here at stage 3).

    python -m pumpbot bot --artifact-dir bot_artifacts_K7V ...
    python -m pumpbot dashboard
    python -m pumpbot diag --since 1781047096
    python -m pumpbot health
    python -m pumpbot probe-shreds
    python -m pumpbot extract --out data/x.parquet --min-fwd 0
"""
import runpy
import sys

from ._paths import ROOT, bootstrap

COMMANDS = {
    "bot": "pumpfun_bot.py",
    "dashboard": "tools/dashboard.py",
    "diag": "tools/live_bucket_diag.py",
    "health": "tools/collector_health.py",
    "probe-shreds": "tools/shred_coverage_probe.py",
    "extract": "tools/extract_live_matched.py",
    "train-crossera": "tools/train_crossera_k3v03.py",
    "replay-exits": "tools/replay_exit_k3v03.py",
}

# Collector entry points are deliberately NOT exposed here: they are owned by
# systemd units in the frozen set; starting a second instance by hand would
# fight the running daemon (duplicate SHM ring writer, double capture files).


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        names = "\n  ".join(sorted(COMMANDS))
        print(f"usage: python -m pumpbot <command> [args...]\n\ncommands:\n  {names}")
        return 0
    cmd, rest = argv[0], argv[1:]
    script = COMMANDS.get(cmd)
    if script is None:
        print(f"unknown command {cmd!r}; try --help", file=sys.stderr)
        return 2
    bootstrap()
    path = ROOT / script
    sys.argv = [str(path)] + rest
    runpy.run_path(str(path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
