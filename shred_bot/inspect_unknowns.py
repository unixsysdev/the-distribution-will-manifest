"""Inspect remaining unknown pump.fun discriminators."""
import json
from collections import Counter
from pathlib import Path

path = sorted(Path("/root/the-distribution-will-manifest/shred_bot/intent_capture").glob("*.jsonl"))[-1]
unknowns = {"38fc74089edfcd5f", "7af3cc415e741d37", "253a237ebe35e4c5"}
samples = {d: [] for d in unknowns}
data_lens = {d: Counter() for d in unknowns}
n_accts = {d: Counter() for d in unknowns}
n_ix_in_tx = {d: Counter() for d in unknowns}

with open(path) as f:
    for ln in f:
        try: r = json.loads(ln)
        except Exception: continue
        d = r.get("ix_disc_hex")
        if d in unknowns:
            data_lens[d][r.get("ix_data_len", 0)] += 1
            n_accts[d][len(r.get("ix_accounts", []))] += 1
            n_ix_in_tx[d][r.get("n_ix", 0)] += 1
            if len(samples[d]) < 3:
                samples[d].append(r)

for d in unknowns:
    print(f"\n=== {d} ===")
    print(f"  data_len distribution: {dict(data_lens[d].most_common(5))}")
    print(f"  n_accs distribution:   {dict(n_accts[d].most_common(5))}")
    print(f"  n_ix_in_tx distrib:    {dict(n_ix_in_tx[d].most_common(5))}")
    if samples[d]:
        s = samples[d][0]
        hexdata = s.get("ix_data_hex", "")
        print(f"  sample data hex:       {hexdata[:80]}")
        accts = s.get("ix_accounts", [])
        print(f"  sample first 4 accts:  {accts[:4]}")
        progs = s.get("programs_touched", [])
        print(f"  sample n programs:     {len(progs)}")
