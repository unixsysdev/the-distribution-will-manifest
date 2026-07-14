"""Build and audit a causal June pump.fun entry model sweep.

This is an offline research harness. It deliberately does not touch the
production bot_artifacts_K7V symlink or the running paper bot.

Principles:
  * one candidate row per mint per fixed (K, V) trigger;
  * features are computed only from trades at or before the decision trade;
  * labels and fixed-TP returns are computed only from later trades;
  * train/validation/test are chronological by decision wall-clock time;
  * no inner joins. Intent/shred features are a separate overlap experiment.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import gzip
import json
import math
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score


ROOT = Path("/root/the-distribution-will-manifest")
CLASSIC_NAMES = [
    "win_ret",
    "dir_eff",
    "buy_frac",
    "uniq",
    "net_sol",
    "tot_sol",
    "single_actor_share",
    "trades_per_sec",
    "entry_sol",
    "win_drawup",
    "win_drawdown",
]

FEATURE_DENY = ("peak", "future", "label", "target", "tp", "terminal", "net_exit")


def buy_tokens(vs: float, vt: float, dsol: float) -> float:
    return vt - (vs * vt) / (vs + dsol)


def sell_sol(vs: float, vt: float, dtok: float) -> float:
    return vs - (vs * vt) / (vt + dtok)


def parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def capture_paths(capture_dir: Path, min_stem: str, include_active: bool) -> list[Path]:
    paths = sorted(capture_dir.glob("*.jsonl*"))
    out: list[Path] = []
    for p in paths:
        if min_stem and p.name < min_stem:
            continue
        if not include_active and p.suffix != ".gz":
            continue
        out.append(p)
    return out


def open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def is_classic_curve(vsol: float, rsol: float) -> bool:
    return abs(vsol - 30_000_000_000.0 - rsol) < 50_000_000.0


def fnum(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(y) or math.isinf(y):
        return default
    return y


def load_trades(
    capture_dir: Path,
    min_stem: str,
    include_active: bool,
    fresh_rsol_lam: int,
) -> tuple[dict[str, dict[str, list[Any]]], dict[str, Any]]:
    """Stream decoded gRPC capture into compact per-mint trade lists."""
    paths = capture_paths(capture_dir, min_stem, include_active)
    groups: dict[str, dict[str, list[Any]]] = {}
    dropped: set[str] = set()
    first_seen_rsol: dict[str, float] = {}
    stats = {
        "paths": [str(p) for p in paths],
        "n_paths": len(paths),
        "rows_with_vsol": 0,
        "rows_parsed": 0,
        "classic_rows": 0,
        "fresh_rows": 0,
        "dropped_mature_mints": 0,
        "bad_json": 0,
        "start_ts": None,
        "end_ts": None,
    }

    t0 = time.time()
    for pi, path in enumerate(paths, 1):
        file_rows = 0
        with open_jsonl(path) as fh:
            for ln in fh:
                if '"vsol"' not in ln:
                    continue
                stats["rows_with_vsol"] += 1
                try:
                    rec = json.loads(ln)
                except Exception:
                    stats["bad_json"] += 1
                    continue
                mint = rec.get("mint")
                if not mint or mint in dropped:
                    continue
                vsol = fnum(rec.get("vsol"))
                vtok = fnum(rec.get("vtok"))
                rsol = fnum(rec.get("rsol"))
                if vsol <= 0 or vtok <= 0:
                    continue
                stats["rows_parsed"] += 1
                if not is_classic_curve(vsol, rsol):
                    continue
                stats["classic_rows"] += 1
                if mint not in first_seen_rsol:
                    first_seen_rsol[mint] = rsol
                    if rsol >= fresh_rsol_lam:
                        dropped.add(mint)
                        stats["dropped_mature_mints"] += 1
                        continue
                ts = fnum(rec.get("t"), fnum(rec.get("ev_ts")))
                if ts <= 0:
                    continue
                ev_ts = fnum(rec.get("ev_ts"), ts)
                sig = rec.get("sig") or ""
                sol = fnum(rec.get("sol")) / 1e9
                is_buy = bool(rec.get("is_buy"))
                user = rec.get("user") or ""
                mid = vsol / vtok
                g = groups.get(mint)
                if g is None:
                    g = {
                        "ts": [], "ev_ts": [], "slot": [], "mid": [], "vsol": [], "vtok": [],
                        "rsol": [], "rtok": [], "sol": [], "is_buy": [], "user": [],
                        "fee_lam": [], "cu": [], "cu_limit": [], "priority_fee_micro": [],
                        "jito_tip_lam": [], "route_present": [], "n_inner_ix": [], "n_keys": [],
                        "failed": [], "sig": [],
                    }
                    groups[mint] = g
                g["ts"].append(ts)
                g["ev_ts"].append(ev_ts)
                g["slot"].append(int(fnum(rec.get("slot"))))
                g["mid"].append(mid)
                g["vsol"].append(vsol)
                g["vtok"].append(vtok)
                g["rsol"].append(rsol)
                g["rtok"].append(fnum(rec.get("rtok")))
                g["sol"].append(sol)
                g["is_buy"].append(is_buy)
                g["user"].append(user)
                g["fee_lam"].append(fnum(rec.get("fee_lam")))
                g["cu"].append(fnum(rec.get("cu")))
                g["cu_limit"].append(fnum(rec.get("cu_limit")))
                g["priority_fee_micro"].append(fnum(rec.get("priority_fee_micro")))
                g["jito_tip_lam"].append(fnum(rec.get("jito_tip_lam")))
                g["route_present"].append(1.0 if rec.get("route") else 0.0)
                g["n_inner_ix"].append(fnum(rec.get("n_inner_ix")))
                g["n_keys"].append(fnum(rec.get("n_keys")))
                g["failed"].append(1.0 if rec.get("failed") else 0.0)
                g["sig"].append(sig)
                file_rows += 1
                stats["fresh_rows"] += 1
                stats["start_ts"] = ts if stats["start_ts"] is None else min(stats["start_ts"], ts)
                stats["end_ts"] = ts if stats["end_ts"] is None else max(stats["end_ts"], ts)
        print(
            f"  [{pi:02d}/{len(paths):02d}] {path.name}: {file_rows:,} fresh classic trades | "
            f"{len(groups):,} mints | {time.time() - t0:.0f}s",
            flush=True,
        )
    return groups, stats


def percentile(vals: np.ndarray, q: float) -> float:
    if len(vals) == 0:
        return 0.0
    return float(np.percentile(vals, q))


def stats_for_values(vals: np.ndarray, prefix: str) -> dict[str, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_p50": percentile(vals, 50),
        f"{prefix}_p90": percentile(vals, 90),
        f"{prefix}_max": float(np.max(vals)),
    }


def snapshot_features(g: dict[str, list[Any]], idx: int, prefix: str) -> dict[str, float]:
    mids = np.asarray(g["mid"][: idx + 1], dtype=float)
    sol = np.asarray(g["sol"][: idx + 1], dtype=float)
    is_buy = np.asarray(g["is_buy"][: idx + 1], dtype=bool)
    ts = np.asarray(g["ts"][: idx + 1], dtype=float)
    users = g["user"][: idx + 1]
    n = len(mids)
    d = np.diff(mids)
    sa = float(np.abs(d).sum())
    win_ret = float(mids[-1] / mids[0] - 1.0) if mids[0] > 0 else 0.0
    dir_eff = abs(float(d.sum())) / sa if sa > 0 else 0.0
    span = max(1e-6, float(ts[-1] - ts[0]))
    by_user: dict[str, float] = defaultdict(float)
    for u, s in zip(users, sol):
        by_user[u] += float(s)
    tot_sol = float(sol.sum())
    top_share = max(by_user.values()) / tot_sol if tot_sol > 0 and by_user else 0.0
    rel = mids / mids[0] - 1.0 if mids[0] > 0 else np.zeros_like(mids)
    buy_sol = sol[is_buy]
    sell_sol_v = sol[~is_buy]
    iat = np.diff(ts)
    feats = {
        f"{prefix}_win_ret": win_ret,
        f"{prefix}_dir_eff": dir_eff,
        f"{prefix}_buy_frac": float(is_buy.mean()) if n else 0.0,
        f"{prefix}_uniq": float(len(set(users))),
        f"{prefix}_net_sol": float(buy_sol.sum() - sell_sol_v.sum()),
        f"{prefix}_tot_sol": tot_sol,
        f"{prefix}_single_actor_share": top_share,
        f"{prefix}_trades_per_sec": n / span,
        f"{prefix}_entry_sol": float(sol[0]) if n else 0.0,
        f"{prefix}_win_drawup": float(rel.max()) if n else 0.0,
        f"{prefix}_win_drawdown": float(rel.min()) if n else 0.0,
        f"{prefix}_n_trades": float(n),
        f"{prefix}_age_sec": span,
        f"{prefix}_cum_buy_sol": float(buy_sol.sum()),
        f"{prefix}_cum_sell_sol": float(sell_sol_v.sum()),
        f"{prefix}_last_sol": float(sol[-1]) if n else 0.0,
        f"{prefix}_last_is_buy": float(is_buy[-1]) if n else 0.0,
        f"{prefix}_iat_mean": float(iat.mean()) if len(iat) else 0.0,
        f"{prefix}_iat_p50": percentile(iat, 50),
        f"{prefix}_iat_p90": percentile(iat, 90),
        f"{prefix}_iat_min": float(iat.min()) if len(iat) else 0.0,
    }
    feats.update(stats_for_values(buy_sol, f"{prefix}_buy_sol"))
    feats.update(stats_for_values(sell_sol_v, f"{prefix}_sell_sol"))
    return feats


def window_features(g: dict[str, list[Any]], idx: int, n_last: int, prefix: str) -> dict[str, float]:
    lo = max(0, idx + 1 - n_last)
    mids = np.asarray(g["mid"][lo : idx + 1], dtype=float)
    sol = np.asarray(g["sol"][lo : idx + 1], dtype=float)
    is_buy = np.asarray(g["is_buy"][lo : idx + 1], dtype=bool)
    ts = np.asarray(g["ts"][lo : idx + 1], dtype=float)
    users = g["user"][lo : idx + 1]
    n = len(mids)
    span = max(1e-6, float(ts[-1] - ts[0])) if n else 1e-6
    buy_sol = sol[is_buy] if n else np.asarray([], dtype=float)
    sell_sol_v = sol[~is_buy] if n else np.asarray([], dtype=float)
    by_user: dict[str, float] = defaultdict(float)
    for u, s in zip(users, sol):
        by_user[u] += float(s)
    tot = float(sol.sum()) if n else 0.0
    ret = float(mids[-1] / mids[0] - 1.0) if n and mids[0] > 0 else 0.0
    return {
        f"{prefix}_n": float(n),
        f"{prefix}_span_sec": span if n > 1 else 0.0,
        f"{prefix}_ret": ret,
        f"{prefix}_buy_frac": float(is_buy.mean()) if n else 0.0,
        f"{prefix}_net_sol": float(buy_sol.sum() - sell_sol_v.sum()),
        f"{prefix}_tot_sol": tot,
        f"{prefix}_uniq": float(len(set(users))),
        f"{prefix}_top_share": max(by_user.values()) / tot if tot > 0 and by_user else 0.0,
        f"{prefix}_trades_per_sec": n / span,
        f"{prefix}_max_buy_sol": float(buy_sol.max()) if len(buy_sol) else 0.0,
        f"{prefix}_max_sell_sol": float(sell_sol_v.max()) if len(sell_sol_v) else 0.0,
    }


def execution_features(g: dict[str, list[Any]], idx: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for col, name in [
        ("fee_lam", "fee_lam"),
        ("cu", "cu"),
        ("cu_limit", "cu_limit"),
        ("priority_fee_micro", "priority_fee_micro"),
        ("jito_tip_lam", "jito_tip_lam"),
        ("n_inner_ix", "n_inner_ix"),
        ("n_keys", "n_keys"),
    ]:
        out.update(stats_for_values(np.asarray(g[col][: idx + 1], dtype=float), f"exec_{name}"))
    tip = np.asarray(g["jito_tip_lam"][: idx + 1], dtype=float)
    out["exec_jito_tip_rate"] = float((tip > 0).mean()) if len(tip) else 0.0
    out["exec_jito_tip_sum_lam"] = float(tip.sum()) if len(tip) else 0.0
    out["exec_route_rate"] = float(np.mean(g["route_present"][: idx + 1])) if idx >= 0 else 0.0
    out["exec_failed_rate"] = float(np.mean(g["failed"][: idx + 1])) if idx >= 0 else 0.0
    return out


def exit_return(
    g: dict[str, list[Any]],
    decision_idx: int,
    tp_level: float,
    horizon_sec: float,
    q_sol: float,
    cost_bps: float,
    fee_per_tx_sol: float,
) -> tuple[float, float, float, float, int, float]:
    entry_vs = float(g["vsol"][decision_idx])
    entry_vt = float(g["vtok"][decision_idx])
    entry_mid = float(g["mid"][decision_idx])
    entry_ts = float(g["ts"][decision_idx])
    q_lam = q_sol * 1e9
    pos_tok = buy_tokens(entry_vs, entry_vt, q_lam)
    best_mid = entry_mid
    best_ret = 0.0
    exit_vs = entry_vs
    exit_vt = entry_vt
    exit_ret = 0.0
    hit = 0
    future_n = 0
    future_sec = 0.0
    for j in range(decision_idx + 1, len(g["mid"])):
        ts = float(g["ts"][j])
        if ts - entry_ts > horizon_sec:
            break
        future_n += 1
        future_sec = ts - entry_ts
        mid = float(g["mid"][j])
        ret = mid / entry_mid - 1.0 if entry_mid > 0 else 0.0
        if mid > best_mid:
            best_mid = mid
            best_ret = ret
        exit_vs = float(g["vsol"][j])
        exit_vt = float(g["vtok"][j])
        exit_ret = ret
        if hit == 0 and ret >= tp_level:
            hit = 1
            exit_vs = float(g["vsol"][j])
            exit_vt = float(g["vtok"][j])
            exit_ret = ret
            break
    proceeds = sell_sol(exit_vs, exit_vt, pos_tok)
    net = proceeds / q_lam - 1.0 - cost_bps / 1e4 - (fee_per_tx_sol * 2.0) / q_sol
    return float(net), float(best_ret), float(exit_ret), float(hit), int(future_n), float(future_sec)


def trigger_indices(g: dict[str, list[Any]], k: int, v_sol: float) -> tuple[int | None, int | None, int | None]:
    if len(g["mid"]) < max(k, 3):
        return None, None, None
    k_idx = k - 1
    cum_buy = 0.0
    v_idx: int | None = None
    for i, (is_buy, sol) in enumerate(zip(g["is_buy"], g["sol"])):
        if is_buy:
            cum_buy += float(sol)
        if i + 1 >= 3 and cum_buy >= v_sol:
            v_idx = i
            break
    if v_idx is None:
        return k_idx, None, None
    return k_idx, v_idx, max(k_idx, v_idx)


def load_intents(intent_dir: Path) -> tuple[dict[str, dict[str, list[Any]]], dict[str, Any]]:
    paths = sorted(intent_dir.glob("intent-*.jsonl*"))
    groups: dict[str, dict[str, list[Any]]] = {}
    stats = {"n_paths": len(paths), "rows": 0, "trade_rows": 0, "start_ts": None, "end_ts": None}
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:
            for ln in fh:
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                stats["rows"] += 1
                typ = rec.get("type")
                if typ not in ("buy", "sell", "buy_quote", "buy_sol_in"):
                    continue
                mint = rec.get("mint")
                if not mint:
                    continue
                recv_ns = fnum(rec.get("recv_ns"))
                if recv_ns <= 0:
                    continue
                ts = recv_ns / 1e9
                g = groups.get(mint)
                if g is None:
                    g = {
                        "ts": [], "is_buy": [], "sol_limit_sol": [], "priority_fee_micro": [],
                        "cu_limit": [], "jito_tip_lam": [], "probable_spoof": [], "signer": [],
                    }
                    groups[mint] = g
                is_buy = bool(rec.get("is_buy"))
                g["ts"].append(ts)
                g["is_buy"].append(is_buy)
                g["sol_limit_sol"].append(fnum(rec.get("sol_limit_sol")))
                g["priority_fee_micro"].append(fnum(rec.get("priority_fee_micro")))
                g["cu_limit"].append(fnum(rec.get("cu_limit")))
                g["jito_tip_lam"].append(fnum(rec.get("jito_tip_lam")))
                g["probable_spoof"].append(1.0 if rec.get("probable_spoof") else 0.0)
                g["signer"].append(rec.get("signer") or rec.get("user") or "")
                stats["trade_rows"] += 1
                stats["start_ts"] = ts if stats["start_ts"] is None else min(stats["start_ts"], ts)
                stats["end_ts"] = ts if stats["end_ts"] is None else max(stats["end_ts"], ts)
    for g in groups.values():
        order = np.argsort(np.asarray(g["ts"], dtype=float), kind="mergesort")
        for k in list(g.keys()):
            g[k] = [g[k][int(i)] for i in order]
    return groups, stats


def intent_features(
    intents: dict[str, dict[str, list[Any]]],
    mint: str,
    decision_ts: float,
    windows: tuple[float, ...] = (0.5, 2.0, 5.0),
) -> dict[str, float]:
    g = intents.get(mint)
    out: dict[str, float] = {}
    for w in windows:
        p = f"intent_{str(w).replace('.', 'p')}s"
        if g is None:
            out.update({
                f"{p}_present": 0.0,
                f"{p}_n": 0.0, f"{p}_buy": 0.0, f"{p}_sell": 0.0,
                f"{p}_buy_frac": 0.0, f"{p}_net_limit_sol": 0.0,
                f"{p}_uniq_signers": 0.0, f"{p}_tip_rate": 0.0,
                f"{p}_tip_max_lam": 0.0, f"{p}_priority_p90": 0.0,
                f"{p}_spoof_rate": 0.0,
            })
            continue
        ts = g["ts"]
        hi = bisect.bisect_right(ts, decision_ts)
        lo = bisect.bisect_left(ts, decision_ts - w)
        n = hi - lo
        if n <= 0:
            out.update({
                f"{p}_present": 0.0,
                f"{p}_n": 0.0, f"{p}_buy": 0.0, f"{p}_sell": 0.0,
                f"{p}_buy_frac": 0.0, f"{p}_net_limit_sol": 0.0,
                f"{p}_uniq_signers": 0.0, f"{p}_tip_rate": 0.0,
                f"{p}_tip_max_lam": 0.0, f"{p}_priority_p90": 0.0,
                f"{p}_spoof_rate": 0.0,
            })
            continue
        is_buy = np.asarray(g["is_buy"][lo:hi], dtype=bool)
        lim = np.asarray(g["sol_limit_sol"][lo:hi], dtype=float)
        tip = np.asarray(g["jito_tip_lam"][lo:hi], dtype=float)
        pri = np.asarray(g["priority_fee_micro"][lo:hi], dtype=float)
        spoof = np.asarray(g["probable_spoof"][lo:hi], dtype=float)
        signers = g["signer"][lo:hi]
        out.update({
            f"{p}_present": 1.0,
            f"{p}_n": float(n),
            f"{p}_buy": float(is_buy.sum()),
            f"{p}_sell": float((~is_buy).sum()),
            f"{p}_buy_frac": float(is_buy.mean()),
            f"{p}_net_limit_sol": float(lim[is_buy].sum() - lim[~is_buy].sum()),
            f"{p}_uniq_signers": float(len(set(signers))),
            f"{p}_tip_rate": float((tip > 0).mean()),
            f"{p}_tip_max_lam": float(tip.max()) if len(tip) else 0.0,
            f"{p}_priority_p90": percentile(pri, 90),
            f"{p}_spoof_rate": float(spoof.mean()) if len(spoof) else 0.0,
        })
    return out


def build_candidates(
    groups: dict[str, dict[str, list[Any]]],
    intents: dict[str, dict[str, list[Any]]] | None,
    ks: list[int],
    vs: list[float],
    tp_levels: list[float],
    horizon_sec: float,
    maturity_sec: float,
    q_sol: float,
    cost_bps: float,
    fee_per_tx_sol: float,
    global_end_ts: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mi, (mint, g) in enumerate(groups.items(), 1):
        if mi % 5000 == 0:
            print(f"  candidate build: {mi:,}/{len(groups):,} mints -> {len(rows):,} rows", flush=True)
        if len(g["mid"]) < min(ks):
            continue
        snap_cache: dict[tuple[str, int], dict[str, float]] = {}
        for k in ks:
            for v in vs:
                k_idx, v_idx, decision_idx = trigger_indices(g, k, v)
                if decision_idx is None or k_idx is None or v_idx is None:
                    continue
                decision_ts = float(g["ts"][decision_idx])
                if global_end_ts - decision_ts < maturity_sec:
                    continue
                row: dict[str, Any] = {
                    "mint": mint,
                    "k": int(k),
                    "v_sol": float(v),
                    "first_ts": float(g["ts"][0]),
                    "decision_ts": decision_ts,
                    "decision_slot": int(g["slot"][decision_idx]),
                    "decision_idx": int(decision_idx),
                    "k_idx": int(k_idx),
                    "v_idx": int(v_idx),
                    "n_total_trades_seen": int(len(g["mid"])),
                    "entry_vsol": float(g["vsol"][decision_idx]),
                    "entry_vtok": float(g["vtok"][decision_idx]),
                    "entry_rsol": float(g["rsol"][decision_idx]),
                    "entry_mid": float(g["mid"][decision_idx]),
                    "entry_fill_k": max(0.0, min(1.0, (float(g["vsol"][decision_idx]) / 1e9 - 30.0) / 85.0)),
                    "k_wait_to_v_sec": float(g["ts"][decision_idx] - g["ts"][k_idx]),
                    "v_wait_to_k_sec": float(g["ts"][decision_idx] - g["ts"][v_idx]),
                }
                for name, idx, prefix in (("k", k_idx, "k"), ("v", v_idx, "v"), ("d", decision_idx, "d")):
                    key = (name, idx)
                    if key not in snap_cache:
                        snap_cache[key] = snapshot_features(g, idx, prefix)
                    row.update(snap_cache[key])
                row.update(window_features(g, decision_idx, 3, "w3"))
                row.update(window_features(g, decision_idx, 5, "w5"))
                row.update(window_features(g, decision_idx, 10, "w10"))
                row.update(execution_features(g, decision_idx))
                for tp in tp_levels:
                    net, peak, exit_ret, hit, nfut, fsec = exit_return(
                        g, decision_idx, tp, horizon_sec, q_sol, cost_bps, fee_per_tx_sol
                    )
                    tag = f"tp{int(tp * 100)}"
                    row[f"{tag}_net"] = net
                    row[f"{tag}_hit"] = hit
                    row[f"{tag}_exit_ret"] = exit_ret
                    row[f"{tag}_future_n"] = nfut
                    row[f"{tag}_future_sec"] = fsec
                    row[f"peak_ret_h{int(horizon_sec)}"] = peak
                if intents is not None:
                    row.update(intent_features(intents, mint, decision_ts))
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["decision_ts", "mint", "k", "v_sol"], kind="mergesort").reset_index(drop=True)
    return df


def split_chrono(df: pd.DataFrame, train_frac: float = 0.60, val_frac: float = 0.20):
    order = np.argsort(df["decision_ts"].values, kind="mergesort")
    n = len(order)
    a = int(n * train_frac)
    b = int(n * (train_frac + val_frac))
    return order[:a], order[a:b], order[b:]


def feature_columns(df: pd.DataFrame, include_intent: bool) -> list[str]:
    meta = {
        "mint", "first_ts", "decision_ts", "decision_slot", "decision_idx", "k_idx", "v_idx",
        "n_total_trades_seen", "k", "v_sol",
    }
    cols: list[str] = []
    for c in df.columns:
        if c in meta:
            continue
        if any(x in c for x in FEATURE_DENY):
            continue
        if c.startswith("intent_") and not include_intent:
            continue
        if not c.startswith("intent_") and include_intent:
            cols.append(c)
            continue
        cols.append(c)
    bad = [c for c in cols if any(x in c for x in FEATURE_DENY)]
    if bad:
        raise RuntimeError(f"leaky feature names: {bad[:20]}")
    return cols


def eval_selected(y_true, scores, realized, threshold: float) -> dict[str, float]:
    m = scores >= threshold
    n = int(m.sum())
    if n == 0:
        return {
            "n": 0, "fire_rate": 0.0, "precision": math.nan, "mean_net": math.nan,
            "median_net": math.nan, "p25_net": math.nan, "win_rate": math.nan,
            "es10_net": math.nan,
        }
    r = np.asarray(realized[m], dtype=float)
    y = np.asarray(y_true[m], dtype=int)
    k_es = max(1, int(math.ceil(0.10 * n)))
    return {
        "n": n,
        "fire_rate": float(n / len(scores)),
        "precision": float(y.mean()),
        "mean_net": float(np.mean(r)),
        "median_net": float(np.median(r)),
        "p25_net": percentile(r, 25),
        "win_rate": float((r > 0).mean()),
        "es10_net": float(np.sort(r)[:k_es].mean()),
    }


def choose_threshold(y_val, s_val, r_val, min_fires: int) -> tuple[float, dict[str, float]]:
    candidates = []
    for pct in (0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30):
        thr = float(np.quantile(s_val, 1.0 - pct / 100.0))
        ev = eval_selected(y_val, s_val, r_val, thr)
        if ev["n"] < min_fires:
            continue
        # Conservative validation score: reward mean return but penalize weak tails.
        score = ev["mean_net"] + 0.25 * ev["p25_net"] + 0.10 * ev["es10_net"]
        candidates.append((score, thr, ev, pct))
    if not candidates:
        thr = float(np.quantile(s_val, 0.95))
        return thr, eval_selected(y_val, s_val, r_val, thr)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def train_sweep(df: pd.DataFrame, out_dir: Path, include_intent: bool, seed: int = 42) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] = {}
    target_defs = [
        ("peak_ge_25", "peak_ret_h300", 0.25, "tp50_net"),
        ("peak_ge_50", "peak_ret_h300", 0.50, "tp50_net"),
        ("peak_ge_100", "peak_ret_h300", 1.00, "tp100_net"),
        ("peak_ge_200", "peak_ret_h300", 2.00, "tp200_net"),
    ]
    # The horizon is encoded in the label column name at build time.
    peak_cols = [c for c in df.columns if c.startswith("peak_ret_h")]
    if peak_cols:
        for i, tup in enumerate(target_defs):
            target_defs[i] = (tup[0], peak_cols[0], tup[2], tup[3])

    feats = feature_columns(df, include_intent=include_intent)
    for c in feats:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise RuntimeError(f"non-numeric feature: {c}")

    leakage_audit = {
        "include_intent": include_intent,
        "n_features": len(feats),
        "features": feats,
        "denied_tokens": FEATURE_DENY,
    }

    for k in sorted(df["k"].unique()):
        for v in sorted(df["v_sol"].unique()):
            sub = df[(df["k"] == k) & (df["v_sol"] == v)].copy()
            if include_intent:
                # Do not let zero-filled pre-intent rows create a time-era model.
                intent_cols = [c for c in feats if c.startswith("intent_")]
                if intent_cols:
                    has_any_intent = sub[[c for c in intent_cols if c.endswith("_n")]].sum(axis=1) > 0
                    sub = sub[has_any_intent].copy()
            if len(sub) < 800:
                continue
            tr, va, te = split_chrono(sub)
            if len(tr) < 400 or len(va) < 100 or len(te) < 100:
                continue
            X = sub[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
            split_info = {
                "train_start": float(sub.iloc[tr]["decision_ts"].min()),
                "train_end": float(sub.iloc[tr]["decision_ts"].max()),
                "val_start": float(sub.iloc[va]["decision_ts"].min()),
                "val_end": float(sub.iloc[va]["decision_ts"].max()),
                "test_start": float(sub.iloc[te]["decision_ts"].min()),
                "test_end": float(sub.iloc[te]["decision_ts"].max()),
            }
            for target_name, peak_col, peak_thr, realized_col in target_defs:
                if realized_col not in sub.columns:
                    continue
                y = (sub[peak_col].values >= peak_thr).astype(int)
                if y[tr].min() == y[tr].max() or y[va].min() == y[va].max() or y[te].min() == y[te].max():
                    continue
                clf = HistGradientBoostingClassifier(
                    max_iter=220,
                    learning_rate=0.045,
                    max_depth=3,
                    l2_regularization=5.0,
                    random_state=seed,
                )
                clf.fit(X[tr], y[tr])
                s_tr = clf.predict_proba(X[tr])[:, 1]
                s_va = clf.predict_proba(X[va])[:, 1]
                s_te = clf.predict_proba(X[te])[:, 1]
                r = sub[realized_col].values.astype(float)
                min_fires = max(20, int(0.01 * len(va)))
                thr, val_ev = choose_threshold(y[va], s_va, r[va], min_fires=min_fires)
                test_ev = eval_selected(y[te], s_te, r[te], thr)
                val_selection_score = (
                    val_ev["mean_net"] + 0.25 * val_ev["p25_net"] + 0.10 * val_ev["es10_net"]
                    if not math.isnan(val_ev["mean_net"]) else -9.0
                )
                train_auc = roc_auc_score(y[tr], s_tr)
                val_auc = roc_auc_score(y[va], s_va)
                test_auc = roc_auc_score(y[te], s_te)
                row = {
                    "kind": "classifier",
                    "include_intent": include_intent,
                    "k": int(k),
                    "v_sol": float(v),
                    "target": target_name,
                    "realized_col": realized_col,
                    "n": int(len(sub)),
                    "n_train": int(len(tr)),
                    "n_val": int(len(va)),
                    "n_test": int(len(te)),
                    "train_pos": float(y[tr].mean()),
                    "val_pos": float(y[va].mean()),
                    "test_pos": float(y[te].mean()),
                    "train_auc": float(train_auc),
                    "val_auc": float(val_auc),
                    "test_auc": float(test_auc),
                    "threshold": float(thr),
                    "val_selection_score": float(val_selection_score),
                    **{f"val_{kk}": vv for kk, vv in val_ev.items()},
                    **{f"test_{kk}": vv for kk, vv in test_ev.items()},
                    **split_info,
                }
                rows.append(row)
                if val_ev["n"] >= min_fires and (not best or val_selection_score > best["score"]):
                    best = {
                        "score": val_selection_score,
                        "model": clf,
                        "features": feats,
                        "row": row,
                        "leakage_audit": leakage_audit,
                        "sub": sub,
                        "splits": (tr, va, te),
                        "X": X,
                        "y": y,
                        "target": target_name,
                        "realized_col": realized_col,
                    }

            # Direct return regressor, evaluated against tp50_net. This is a comparator,
            # not the default artifact unless it wins on holdout.
            if "tp50_net" in sub.columns:
                yret = sub["tp50_net"].values.astype(float)
                reg = HistGradientBoostingRegressor(
                    max_iter=220,
                    learning_rate=0.045,
                    max_depth=3,
                    l2_regularization=5.0,
                    random_state=seed,
                )
                reg.fit(X[tr], yret[tr])
                s_va = reg.predict(X[va])
                s_te = reg.predict(X[te])
                thr, val_ev = choose_threshold((yret[va] > 0).astype(int), s_va, yret[va], min_fires=max(20, int(0.01 * len(va))))
                test_ev = eval_selected((yret[te] > 0).astype(int), s_te, yret[te], thr)
                val_selection_score = (
                    val_ev["mean_net"] + 0.25 * val_ev["p25_net"] + 0.10 * val_ev["es10_net"]
                    if not math.isnan(val_ev["mean_net"]) else -9.0
                )
                row = {
                    "kind": "regressor",
                    "include_intent": include_intent,
                    "k": int(k),
                    "v_sol": float(v),
                    "target": "tp50_net",
                    "realized_col": "tp50_net",
                    "n": int(len(sub)),
                    "n_train": int(len(tr)),
                    "n_val": int(len(va)),
                    "n_test": int(len(te)),
                    "train_pos": float((yret[tr] > 0).mean()),
                    "val_pos": float((yret[va] > 0).mean()),
                    "test_pos": float((yret[te] > 0).mean()),
                    "train_auc": math.nan,
                    "val_auc": math.nan,
                    "test_auc": math.nan,
                    "threshold": float(thr),
                    "val_selection_score": float(val_selection_score),
                    **{f"val_{kk}": vv for kk, vv in val_ev.items()},
                    **{f"test_{kk}": vv for kk, vv in test_ev.items()},
                    **split_info,
                }
                rows.append(row)

    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(["val_selection_score", "test_mean_net", "test_n"], ascending=[False, False, False]).reset_index(drop=True)
    return res, best


def null_check(best: dict[str, Any], seed: int = 123) -> dict[str, float]:
    if not best:
        return {}
    X = best["X"]
    y = best["y"].copy()
    sub = best["sub"]
    tr, va, te = best["splits"]
    rng = np.random.default_rng(seed)
    y_perm = y.copy()
    shuffled_train = y_perm[tr].copy()
    rng.shuffle(shuffled_train)
    y_perm[tr] = shuffled_train
    clf = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.045,
        max_depth=3,
        l2_regularization=5.0,
        random_state=seed,
    )
    clf.fit(X[tr], y_perm[tr])
    s_va = clf.predict_proba(X[va])[:, 1]
    s_te = clf.predict_proba(X[te])[:, 1]
    realized = sub[best["realized_col"]].values.astype(float)
    thr, val_ev = choose_threshold(y[va], s_va, realized[va], min_fires=max(20, int(0.01 * len(va))))
    test_ev = eval_selected(y[te], s_te, realized[te], thr)
    out = {
        "null_threshold": float(thr),
        "null_val_mean_net": float(val_ev["mean_net"]) if not math.isnan(val_ev["mean_net"]) else math.nan,
        "null_test_mean_net": float(test_ev["mean_net"]) if not math.isnan(test_ev["mean_net"]) else math.nan,
        "null_test_n": float(test_ev["n"]),
    }
    try:
        out["null_val_auc"] = float(roc_auc_score(y[va], s_va))
        out["null_test_auc"] = float(roc_auc_score(y[te], s_te))
    except Exception:
        out["null_val_auc"] = math.nan
        out["null_test_auc"] = math.nan
    return out


def fmt_ts(x: float | None) -> str:
    if x is None or math.isnan(float(x)):
        return "n/a"
    return dt.datetime.fromtimestamp(float(x), tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def write_report(
    out_dir: Path,
    capture_stats: dict[str, Any],
    intent_stats: dict[str, Any] | None,
    df: pd.DataFrame,
    results: pd.DataFrame,
    best: dict[str, Any],
    null: dict[str, float],
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return ""
        text = frame.copy()
        for c in text.columns:
            if pd.api.types.is_float_dtype(text[c]):
                text[c] = text[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
            else:
                text[c] = text[c].map(lambda x: "" if pd.isna(x) else str(x))
        widths = {c: max(len(str(c)), *(len(str(v)) for v in text[c].tolist())) for c in text.columns}
        header = "| " + " | ".join(str(c).ljust(widths[c]) for c in text.columns) + " |"
        sep = "| " + " | ".join("-" * widths[c] for c in text.columns) + " |"
        body = [
            "| " + " | ".join(str(row[c]).ljust(widths[c]) for c in text.columns) + " |"
            for _, row in text.iterrows()
        ]
        return "\n".join([header, sep, *body])

    lines: list[str] = []
    lines.append("# June causal entry sweep")
    lines.append("")
    lines.append(f"created_at_utc: {fmt_ts(time.time())}")
    lines.append(f"sklearn: {sklearn.__version__}")
    lines.append("")
    lines.append("## Data")
    lines.append(f"- decoded gRPC files: {capture_stats['n_paths']}")
    lines.append(f"- decoded gRPC rows with vsol: {capture_stats['rows_with_vsol']:,}")
    lines.append(f"- fresh classic trades used: {capture_stats['fresh_rows']:,}")
    lines.append(f"- mints after fresh/classic gate: {df['mint'].nunique() if not df.empty else 0:,}")
    lines.append(f"- candidate rows: {len(df):,}")
    lines.append(f"- time span: {fmt_ts(capture_stats['start_ts'])} to {fmt_ts(capture_stats['end_ts'])}")
    if intent_stats:
        lines.append(f"- intent trade rows loaded: {intent_stats['trade_rows']:,}")
        lines.append(f"- intent time span: {fmt_ts(intent_stats['start_ts'])} to {fmt_ts(intent_stats['end_ts'])}")
    lines.append("")
    lines.append("## Leakage controls")
    lines.append("- one row per mint per fixed K/V trigger")
    lines.append("- feature columns deny target-like names: " + ", ".join(FEATURE_DENY))
    lines.append("- chronological train/validation/test split by decision wall-clock time")
    lines.append("- labels use trades strictly after the decision index")
    lines.append("- no inner joins; intent is handled as a separate overlap experiment")
    lines.append("")
    lines.append("## Top results")
    if results.empty:
        lines.append("No model met the minimum sample constraints.")
    else:
        cols = [
            "kind", "include_intent", "k", "v_sol", "target", "realized_col", "n",
            "test_auc", "threshold", "val_selection_score", "val_mean_net", "val_p25_net", "val_n",
            "test_mean_net", "test_p25_net", "test_es10_net", "test_win_rate",
            "test_precision", "test_n", "test_fire_rate",
        ]
        show = results[cols].head(20).copy()
        lines.append(markdown_table(show))
    if best:
        lines.append("")
        lines.append("## Selected offline artifact")
        row = best["row"]
        lines.append(json.dumps({k: v for k, v in row.items() if k in (
            "kind", "include_intent", "k", "v_sol", "target", "realized_col",
            "test_auc", "threshold", "test_mean_net", "test_p25_net",
            "test_es10_net", "test_win_rate", "test_precision", "test_n",
            "test_fire_rate", "train_end", "val_start", "val_end", "test_start",
            "test_end",
        )}, indent=2, sort_keys=True))
    if null:
        lines.append("")
        lines.append("## Shuffled-label null check")
        lines.append(json.dumps(null, indent=2, sort_keys=True))
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def save_artifact(out_dir: Path, best: dict[str, Any], null: dict[str, float]) -> None:
    if not best:
        return
    art = out_dir / "offline_best_artifact"
    art.mkdir(parents=True, exist_ok=True)
    pickle.dump(best["model"], open(art / "entry_model.pkl", "wb"))
    spec = {
        "artifact_kind": "offline_research_only",
        "sklearn_version": sklearn.__version__,
        "created_at_utc": fmt_ts(time.time()),
        "entry": {
            "features": best["features"],
            "threshold": best["row"]["threshold"],
            "target": best["target"],
            "realized_col_for_selection": best["realized_col"],
            "k": best["row"]["k"],
            "v_sol": best["row"]["v_sol"],
            "include_intent": best["row"]["include_intent"],
        },
        "holdout_result": best["row"],
        "null_check": null,
        "warning": (
            "Not loadable by current ModelServer unless the paper bot is patched to "
            "compute these exact feature columns at the decision time."
        ),
    }
    (art / "model_spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", default=str(ROOT / "grpc_capture"))
    ap.add_argument("--intent-dir", default=str(ROOT / "shred_bot/intent_capture"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/june_causal_sweep"))
    ap.add_argument("--min-stem", default="capture_20260609T043337Z",
                    help="first decoded capture file to include; defaults to current rich-schema era")
    ap.add_argument("--include-active", action="store_true")
    ap.add_argument("--k", default="3,5,7,9")
    ap.add_argument("--v", default="0.3,0.5,0.8,1.0")
    ap.add_argument("--tp", default="0.5,1.0,2.0")
    ap.add_argument("--horizon-sec", type=float, default=300.0)
    ap.add_argument("--maturity-sec", type=float, default=300.0)
    ap.add_argument("--q-sol", type=float, default=0.1)
    ap.add_argument("--cost-bps", type=float, default=250.0)
    ap.add_argument("--fee-per-tx-sol", type=float, default=0.0015)
    ap.add_argument("--skip-intent", action="store_true")
    ap.add_argument("--candidates-in", default="", help="reuse an existing candidates.parquet and skip raw parsing")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = parse_csv_ints(args.k)
    vs = parse_csv_floats(args.v)
    tp_levels = parse_csv_floats(args.tp)

    intent_stats = None
    if args.candidates_in:
        cand_path = Path(args.candidates_in)
        print(f"=== reusing causal candidates: {cand_path} ===", flush=True)
        df = pd.read_parquet(cand_path)
        capture_stats = {
            "n_paths": 0,
            "rows_with_vsol": 0,
            "fresh_rows": 0,
            "start_ts": float(df["first_ts"].min()) if len(df) else None,
            "end_ts": float(df["decision_ts"].max()) if len(df) else None,
        }
        if any(c.startswith("intent_") for c in df.columns):
            intent_stats = {
                "trade_rows": 0,
                "start_ts": float(df["decision_ts"].min()) if len(df) else None,
                "end_ts": float(df["decision_ts"].max()) if len(df) else None,
            }
    else:
        print("=== loading decoded gRPC trades ===", flush=True)
        groups, capture_stats = load_trades(
            Path(args.capture_dir),
            min_stem=args.min_stem,
            include_active=args.include_active,
            fresh_rsol_lam=3_000_000_000,
        )
        if not groups:
            raise SystemExit("no trades loaded")
        global_end_ts = float(capture_stats["end_ts"])

        intents = None
        if not args.skip_intent:
            print("\n=== loading intent capture ===", flush=True)
            intents, intent_stats = load_intents(Path(args.intent_dir))
            print(f"  intent trade rows: {intent_stats['trade_rows']:,} | mints {len(intents):,}", flush=True)

        print("\n=== building causal candidates ===", flush=True)
        df = build_candidates(
            groups=groups,
            intents=intents,
            ks=ks,
            vs=vs,
            tp_levels=tp_levels,
            horizon_sec=args.horizon_sec,
            maturity_sec=args.maturity_sec,
            q_sol=args.q_sol,
            cost_bps=args.cost_bps,
            fee_per_tx_sol=args.fee_per_tx_sol,
            global_end_ts=global_end_ts,
        )
        cand_path = out_dir / "candidates.parquet"
        df.to_parquet(cand_path, index=False)
        print(f"  wrote {cand_path}: {len(df):,} rows, {df['mint'].nunique() if not df.empty else 0:,} mints", flush=True)

    all_results = []
    best_all: dict[str, Any] = {}
    for include_intent in (False, True):
        if include_intent and args.skip_intent:
            continue
        print(f"\n=== training sweep include_intent={include_intent} ===", flush=True)
        res, best = train_sweep(df, out_dir, include_intent=include_intent)
        res_path = out_dir / ("results_intent.csv" if include_intent else "results_base.csv")
        res.to_csv(res_path, index=False)
        print(f"  wrote {res_path}: {len(res):,} rows", flush=True)
        all_results.append(res)
        if best:
            score = best["score"]
            if not best_all or score > best_all["score"]:
                best_all = best

    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    if not results.empty:
        results = results.sort_values(["val_selection_score", "test_mean_net", "test_n"], ascending=[False, False, False]).reset_index(drop=True)
        results.to_csv(out_dir / "results_all.csv", index=False)

    null = null_check(best_all) if best_all else {}
    save_artifact(out_dir, best_all, null)
    write_report(out_dir, capture_stats, intent_stats, df, results, best_all, null)
    print(f"\n=== done ===\n  report: {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
