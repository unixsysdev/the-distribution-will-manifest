"""COLLECTOR FREEZE GUARD.

The five data collectors (grpc capture, grpc firehose, shred firehose, shred
intents, storagebox shipper) must never be broken by refactors: they have
Restart=always, so a module move that breaks their import closure turns the
next incidental crash into a capture-losing crashloop.

This test recomputes each entry point's local import closure and asserts
(1) every file exists and compiles, and (2) the closure matches the checked-in
manifest. Changing the manifest is a deliberate act that belongs in a planned
migration window (docs/notes/ARCHITECTURE.md stage 2b), never a side effect.
"""
import ast
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).parent / "collector_frozen_manifest.txt"
SEARCH = [ROOT, ROOT / "shred_bot", ROOT / "shred_bot/stubs", ROOT / "grpc_stubs",
          ROOT / "protos"]
ENTRIES = [
    "grpc_capture.py",
    "grpc_firehose.py",
    "shred_bot/raw_shred_firehose.py",
    "shred_bot/intent_recorder.py",
    "shred_bot/storagebox_shipper.py",
]


def _resolve(name: str):
    head = name.split(".")[0]
    for d in SEARCH:
        for cand in (d / f"{head}.py", d / head / "__init__.py"):
            if cand.exists():
                return cand
    return None


def closure() -> set[Path]:
    seen: set[Path] = set()
    todo = [ROOT / e for e in ENTRIES]
    while todo:
        f = todo.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for n in names:
                r = _resolve(n)
                if r and r not in seen:
                    todo.append(r)
    return seen


def test_all_entry_points_exist():
    for e in ENTRIES:
        assert (ROOT / e).exists(), f"collector entry missing: {e}"


def test_closure_compiles():
    for f in closure():
        py_compile.compile(str(f), doraise=True)


def test_closure_matches_manifest():
    current = sorted(str(p.relative_to(ROOT)) for p in closure())
    expected = [ln.strip() for ln in MANIFEST.read_text().splitlines() if ln.strip()]
    assert current == expected, (
        "collector import closure CHANGED. If intentional (planned migration "
        "window only), update tests/collector_frozen_manifest.txt in the same "
        "commit and restart-drill each collector. Diff:\n"
        f"  added:   {sorted(set(current) - set(expected))}\n"
        f"  removed: {sorted(set(expected) - set(current))}"
    )
