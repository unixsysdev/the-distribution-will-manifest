"""Rich causal entry features for the June K/V sweep artifact.

This module mirrors the feature formulas used by
tools/train_june_causal_sweep.py. Keep the names and formulas aligned: the
runtime calls this only for artifacts whose model_spec lists these rich feature
columns.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def fnum(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    if np.isnan(y) or np.isinf(y):
        return default
    return y


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


def trade_row_from_event(ev) -> dict[str, Any]:
    extras = getattr(ev, "grpc_extras", None) or {}
    vsol = float(ev.virtual_sol_reserves)
    vtok = float(ev.virtual_token_reserves)
    return {
        "ts": float(ev.timestamp),
        "slot": int(getattr(ev, "slot", 0) or 0),
        "mid": vsol / vtok if vtok > 0 else 0.0,
        "vsol": vsol,
        "vtok": vtok,
        "rsol": float(ev.real_sol_reserves),
        "sol": float(ev.sol),
        "is_buy": bool(ev.is_buy),
        "user": ev.user or "",
        "fee_lam": fnum(extras.get("fee_lam")),
        "cu": fnum(extras.get("cu")),
        "cu_limit": fnum(extras.get("cu_limit")),
        "priority_fee_micro": fnum(extras.get("priority_fee_micro")),
        "jito_tip_lam": fnum(extras.get("jito_tip_lam")),
        "route_present": 1.0 if extras.get("route") else 0.0,
        "n_inner_ix": fnum(extras.get("n_inner_ix")),
        "n_keys": fnum(extras.get("n_keys")),
        "failed": 1.0 if extras.get("failed") else 0.0,
    }


def trigger_indices(rows: list[dict[str, Any]], k: int, v_sol: float) -> tuple[int | None, int | None, int | None]:
    if len(rows) < max(k, 3):
        return None, None, None
    k_idx = k - 1
    cum_buy = 0.0
    v_idx = None
    for i, row in enumerate(rows):
        if row["is_buy"]:
            cum_buy += float(row["sol"])
        if i + 1 >= 3 and cum_buy >= v_sol:
            v_idx = i
            break
    if v_idx is None:
        return k_idx, None, None
    return k_idx, v_idx, max(k_idx, v_idx)


def snapshot_features(rows: list[dict[str, Any]], idx: int, prefix: str) -> dict[str, float]:
    part = rows[: idx + 1]
    mids = np.asarray([r["mid"] for r in part], dtype=float)
    sol = np.asarray([r["sol"] for r in part], dtype=float)
    is_buy = np.asarray([r["is_buy"] for r in part], dtype=bool)
    ts = np.asarray([r["ts"] for r in part], dtype=float)
    users = [r["user"] for r in part]
    n = len(mids)
    d = np.diff(mids)
    sa = float(np.abs(d).sum())
    win_ret = float(mids[-1] / mids[0] - 1.0) if n and mids[0] > 0 else 0.0
    dir_eff = abs(float(d.sum())) / sa if sa > 0 else 0.0
    span = max(1e-6, float(ts[-1] - ts[0])) if n else 1e-6
    by_user: dict[str, float] = defaultdict(float)
    for u, s in zip(users, sol):
        by_user[u] += float(s)
    tot_sol = float(sol.sum())
    top_share = max(by_user.values()) / tot_sol if tot_sol > 0 and by_user else 0.0
    rel = mids / mids[0] - 1.0 if n and mids[0] > 0 else np.zeros_like(mids)
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


def window_features(rows: list[dict[str, Any]], idx: int, n_last: int, prefix: str) -> dict[str, float]:
    part = rows[max(0, idx + 1 - n_last) : idx + 1]
    mids = np.asarray([r["mid"] for r in part], dtype=float)
    sol = np.asarray([r["sol"] for r in part], dtype=float)
    is_buy = np.asarray([r["is_buy"] for r in part], dtype=bool)
    ts = np.asarray([r["ts"] for r in part], dtype=float)
    users = [r["user"] for r in part]
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


def execution_features(rows: list[dict[str, Any]], idx: int) -> dict[str, float]:
    part = rows[: idx + 1]
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
        out.update(stats_for_values(np.asarray([r[col] for r in part], dtype=float), f"exec_{name}"))
    tip = np.asarray([r["jito_tip_lam"] for r in part], dtype=float)
    out["exec_jito_tip_rate"] = float((tip > 0).mean()) if len(tip) else 0.0
    out["exec_jito_tip_sum_lam"] = float(tip.sum()) if len(tip) else 0.0
    out["exec_route_rate"] = float(np.mean([r["route_present"] for r in part])) if part else 0.0
    out["exec_failed_rate"] = float(np.mean([r["failed"] for r in part])) if part else 0.0
    return out


def build_entry_features(
    rows: list[dict[str, Any]],
    *,
    k: int,
    v_sol: float,
    expected_features: list[str],
    intent_features: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    k_idx, v_idx, decision_idx = trigger_indices(rows, k, v_sol)
    if k_idx is None or v_idx is None or decision_idx is None:
        raise ValueError(f"rich trigger not ready for k={k} v={v_sol}")
    row = rows[decision_idx]
    feats: dict[str, float] = {
        "entry_vsol": float(row["vsol"]),
        "entry_vtok": float(row["vtok"]),
        "entry_rsol": float(row["rsol"]),
        "entry_mid": float(row["mid"]),
        "entry_fill_k": max(0.0, min(1.0, (float(row["vsol"]) / 1e9 - 30.0) / 85.0)),
        "k_wait_to_v_sec": float(rows[decision_idx]["ts"] - rows[k_idx]["ts"]),
        "v_wait_to_k_sec": float(rows[decision_idx]["ts"] - rows[v_idx]["ts"]),
    }
    feats.update(snapshot_features(rows, k_idx, "k"))
    feats.update(snapshot_features(rows, v_idx, "v"))
    feats.update(snapshot_features(rows, decision_idx, "d"))
    feats.update(window_features(rows, decision_idx, 3, "w3"))
    feats.update(window_features(rows, decision_idx, 5, "w5"))
    feats.update(window_features(rows, decision_idx, 10, "w10"))
    feats.update(execution_features(rows, decision_idx))
    if intent_features:
        feats.update(intent_features)
    missing = [f for f in expected_features if f not in feats]
    if missing:
        raise KeyError(f"missing rich entry features: {missing[:12]}")
    return {f: float(feats[f]) for f in expected_features}, feats


def decision_path_features(entry_mid: float, run_max_ret: float, vsol: float, vtok: float, pf: dict) -> tuple[dict, float]:
    mid = vsol / vtok if vtok else 0.0
    ret = mid / entry_mid - 1.0 if entry_mid > 0 else 0.0
    run_max_ret = max(run_max_ret, ret)
    dd = (mid / (entry_mid * (1.0 + run_max_ret)) - 1.0) if run_max_ret > -1 else 0.0
    out = dict(pf)
    out["ret"] = ret
    out["run_max_ret"] = run_max_ret
    out["dd"] = dd
    out["mid"] = mid
    return out, run_max_ret
