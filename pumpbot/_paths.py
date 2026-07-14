"""sys.path bootstrap for the flat-module era (removed at stage 2b)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bootstrap() -> Path:
    for p in (ROOT, ROOT / "shred_bot"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return ROOT
