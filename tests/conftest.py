"""Test bootstrap: trigger env must be set BEFORE feature_accum import
(module-level K_TRIGGER/V_TRIGGER read), and the flat-module tree needs
the project root plus shred_bot on sys.path until migration stage 2."""
import os
import sys
from pathlib import Path

os.environ.setdefault("K_TRIGGER", "3")
os.environ.setdefault("V_TRIGGER", "0.3")

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "shred_bot"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
