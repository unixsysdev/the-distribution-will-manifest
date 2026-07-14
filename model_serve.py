"""Load the V+K7 production pickles (bot_artifacts_K7V) and serve scoring decisions.

Entry model: 22-feature V+K7 stacked head trained on inner-joined K7+V tokens.
  features = features_k7 (11) + features_v (11)
  fire if predict_proba >= entry_threshold (top-decile in training)

Recovery model: 20-feature head (9 path + 11 K-features). NOT the V-features — the
recovery model was trained on K-anchored path snapshots only. Death-cut on score < 0.10.

Hot path: tens of microseconds per call. Loads once, scores per token.
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


class ModelServer:
    def __init__(self, artifact_dir: str | Path = "bot_artifacts_K7V"):
        p = Path(artifact_dir)
        self.entry    = pickle.load(open(p / "entry_model.pkl", "rb"))
        self.spec     = json.load(open(p / "model_spec.json"))
        self.entry_features    = self.spec["entry"]["features"]            # 22
        self.entry_features_k7 = self.spec["entry"].get("features_k7", self.entry_features[:11])
        self.entry_features_v  = self.spec["entry"].get("features_v",  self.entry_features[11:])
        self.entry_threshold   = float(
            self.spec["entry"].get("entry_threshold_top_decile",
                                   self.spec["entry"].get("threshold")))
        self.entry_k = int(self.spec["entry"].get("k", self.spec["entry"].get("K_WINDOW", 5)))
        self.entry_v_sol = float(self.spec["entry"].get("v_sol", 0.5))
        if "rich_features" in self.spec["entry"]:
            # explicit flag wins: the legacy 22-feat names include entry_sol,
            # which the heuristic below would misread as the rich path
            self.rich_entry = bool(self.spec["entry"]["rich_features"])
        else:
            self.rich_entry = any(
                f.startswith("entry_") or f.startswith("intent_") for f in self.entry_features)
        rec_spec = self.spec.get("recovery", {})
        self.recovery_disabled = bool(rec_spec.get("disabled", False))
        rec_path = p / "recovery_model.pkl"
        if self.recovery_disabled or not rec_path.exists():
            self.recovery = None
            self.recovery_features = rec_spec.get("features", [])
            self.death_threshold = float(rec_spec.get("death_cut_threshold", -1.0))
            self.recovery_disabled = True
        else:
            self.recovery = pickle.load(open(rec_path, "rb"))
            self.recovery_features = rec_spec["features"]         # 20
            self.death_threshold = float(rec_spec["death_cut_threshold"])
        # sklearn version warning
        import sklearn
        if sklearn.__version__ != self.spec.get("sklearn_version", ""):
            import warnings
            warnings.warn(
                f"sklearn version mismatch: serving on {sklearn.__version__}, "
                f"trained on {self.spec.get('sklearn_version')}. Pin to match in production.")

    def score_entry(self, entry_feats: Dict[str, float]) -> Tuple[float, bool]:
        """entry_feats must contain BOTH K-features (unsuffixed) AND V-features (_v suffix).
        Returns (entry_score, fire_entry)."""
        x = np.array([[entry_feats[f] for f in self.entry_features]], dtype=float)
        s = float(self.entry.predict_proba(x)[0, 1])
        return s, s >= self.entry_threshold

    def score_recovery(self, k_entry_feats: Dict[str, float],
                       path_feats: Dict[str, float]) -> Tuple[float, bool]:
        """k_entry_feats: 11 K-features (unsuffixed). path_feats: 9 path features.
        Returns (p_recover, fire_death_cut). Fire cut if P(recover) < death_threshold."""
        if self.recovery_disabled or self.recovery is None:
            return 1.0, False
        merged = {**path_feats, **k_entry_feats}
        x = np.array([[merged[f] for f in self.recovery_features]], dtype=float)
        p = float(self.recovery.predict_proba(x)[0, 1])
        return p, p < self.death_threshold

    def __repr__(self) -> str:
        return (f"<ModelServer V+K7 entry_thr={self.entry_threshold:.4f} "
                f"death_thr={self.death_threshold:.4f} n_entry_feats={len(self.entry_features)} "
                f"n_recovery_feats={len(self.recovery_features)} "
                f"sklearn={self.spec.get('sklearn_version')}>")


if __name__ == "__main__":
    # Smoke test: load the V+K7 server and score a few real K7+V joined tokens.
    import pandas as pd
    srv = ModelServer()
    print(srv)
    tk = pd.read_parquet("data/pumpfun_continuation_K7/token_level.parquet")
    tv = pd.read_parquet("data/pumpfun_continuation_V05/token_level.parquet")
    K_NAMES = srv.entry_features_k7
    V_NAMES = srv.entry_features_v
    K_TO_V = {k: v for k, v in zip(K_NAMES, V_NAMES)}
    tv_feats = tv[["mint"] + K_NAMES].rename(columns=K_TO_V)
    joined = tk.merge(tv_feats, on="mint", how="inner").head(5)
    print(f"\nSmoke: scoring {len(joined)} real V+K7 joined entry vectors:")
    for _, row in joined.iterrows():
        ef = {f: float(row[f]) for f in srv.entry_features}
        score, fire = srv.score_entry(ef)
        print(f"  mint={row.mint[:18]}.. score={score:.4f} fire={fire} "
              f"peak_ret={row.peak_ret:+.2f} terminal_ret={row.terminal_ret:+.2f}")
    # Recovery smoke on a path snap
    snap = pd.read_parquet("data/pumpfun_continuation_K7/path_snapshots.parquet").iloc[0]
    tok_row = joined.iloc[0]
    kef = {f: float(tok_row[f]) for f in K_NAMES}
    pf  = {"ret":float(snap.ret), "run_max_ret":float(snap.run_max_ret),
           "dd":float(snap.dd), "fill_k":float(snap.fill_k),
           "buy_frac_w":float(snap.buy_frac_w), "nsell_w":float(snap.nsell_w),
           "solo_sell_w":float(snap.solo_sell_w), "vel_w":float(snap.vel_w),
           "dts":float(snap.dts)}
    p_rec, cut = srv.score_recovery(kef, pf)
    print(f"\nRecovery smoke: p_rec={p_rec:.4f} death_cut={cut}")
