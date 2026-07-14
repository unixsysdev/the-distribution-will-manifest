#!/usr/bin/env python3
"""hazard_shred_extract.py — add per-snap SHRED sell-pressure to the recovery
snaps, so we can test whether pre-execution Jito/shred intent improves the
collapse-hazard AUC (the user's question). For each (mint, snap_ts) it computes
shred sell/buy intent counts + jito-tip stats in the 2s window BEFORE the snap,
joined by mint from the shred intent capture. Output:
data/recovery_snaps_shred_k3v03.parquet (recovery snaps + shred_* columns).
"""
import glob, gzip, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/root/the-distribution-will-manifest")
WIN_NS = 2_000_000_000

def main():
    sn = pd.read_parquet(ROOT/"data/recovery_snaps_k3v03.parquet")
    mints = set(sn.mint.unique())
    # per-mint sorted shred intents (recv_ns, is_buy, jito_tip_lam) for our mints
    by = {}
    n=0
    for path in sorted(glob.glob(str(ROOT/"shred_bot/intent_capture/intent-*.jsonl*"))):
        op=gzip.open if path.endswith(".gz") else open
        try: fh=op(path,"rt")
        except OSError: continue
        with fh:
            for ln in fh:
                if '"mint"' not in ln: continue
                try: r=json.loads(ln)
                except: continue
                m=r.get("mint")
                if m not in mints or r.get("type") not in ("buy","sell","buy_quote","buy_sol_in"): continue
                rn=r.get("recv_ns")
                if not rn: continue
                by.setdefault(m,[]).append((float(rn), bool(r.get("is_buy")), float(r.get("jito_tip_lam") or 0)))
                n+=1
    for m in by: by[m].sort()
    print(f"shred intents for our mints: {n:,} across {len(by):,} mints", flush=True)
    import bisect
    cols={k:[] for k in ("shred_sell_2s","shred_buy_2s","shred_sellfrac_2s","shred_tip_p90_2s")}
    recv={m:np.array([x[0] for x in by[m]]) for m in by}
    for row in sn.itertuples():
        m=row.mint; tns=float(row.snap_ts)*1e9
        arr=recv.get(m)
        if arr is None:
            for k in cols: cols[k].append(0.0)
            continue
        lo=bisect.bisect_left(arr, tns-WIN_NS); hi=bisect.bisect_right(arr, tns)
        seg=by[m][lo:hi]
        if not seg:
            for k in cols: cols[k].append(0.0)
            continue
        nb=sum(1 for _,b,_ in seg if b); ns=len(seg)-nb
        tips=[t for _,_,t in seg if t>0]
        cols["shred_sell_2s"].append(float(ns))
        cols["shred_buy_2s"].append(float(nb))
        cols["shred_sellfrac_2s"].append(ns/len(seg))
        cols["shred_tip_p90_2s"].append(float(np.percentile(tips,90)) if tips else 0.0)
    for k,v in cols.items(): sn[k]=v
    out=ROOT/"data/recovery_snaps_shred_k3v03.parquet"
    sn.to_parquet(out,index=False)
    cov=(sn.shred_sell_2s+sn.shred_buy_2s>0).mean()
    print(f"wrote {out}; snaps with any shred intent in 2s window: {cov:.0%}")

if __name__=="__main__": main()
