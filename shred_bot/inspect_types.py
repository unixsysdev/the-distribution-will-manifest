"""Quick type-breakdown of the active intent_capture file."""
import json
import sys
from collections import Counter
from pathlib import Path

path = sorted(Path("/root/the-distribution-will-manifest/shred_bot/intent_capture").glob("*.jsonl"))[-1]
rows = []
with open(path) as f:
    for ln in f:
        try: rows.append(json.loads(ln))
        except Exception: pass

print(f"file: {path.name}")
print(f"records: {len(rows)}")
types = Counter(r.get("type", "MISSING") for r in rows)
print("\ntype breakdown:")
for t, n in types.most_common():
    print(f"  {n:>5}  {t}")

# Show one sample of each non-buy/sell type if present
print("\nsamples (non-buy/sell types):")
seen = set()
for r in rows:
    t = r.get("type", "?")
    if t in ("buy", "sell") or t in seen: continue
    seen.add(t)
    print(f"\n--- type={t} ---")
    for k, v in r.items():
        if isinstance(v, str) and len(v) > 60: v = v[:60] + "..."
        if isinstance(v, list) and len(v) > 6: v = v[:6] + ["..."]
        print(f"  {k}: {v}")
