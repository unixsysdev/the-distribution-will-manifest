"""ModelServer artifact contract.

Locks in the 2026-06-10 rich-path fix: the legacy 22-feature set contains
`entry_sol`, which the startswith("entry_") heuristic misreads as the rich
feature path. An explicit rich_features flag in model_spec must win; the
heuristic stays only as the fallback for spec files written before the flag.
"""
import json
import pickle

import numpy as np
import pytest
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from feature_accum import ENTRY_FEATURE_NAMES
from model_serve import ModelServer

N_FEAT = len(ENTRY_FEATURE_NAMES)


def _make_artifact(tmp_path, spec_entry_extra: dict, threshold_key="threshold"):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, N_FEAT))
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 0).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X, y)
    with open(tmp_path / "entry_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    spec = {
        "sklearn_version": sklearn.__version__,
        "entry": {
            "features": list(ENTRY_FEATURE_NAMES),
            threshold_key: 0.5,
            "k": 3,
            "v_sol": 0.3,
            **spec_entry_extra,
        },
        "recovery": {"disabled": True, "death_cut_threshold": -1.0, "features": []},
    }
    (tmp_path / "model_spec.json").write_text(json.dumps(spec))
    return tmp_path


def test_explicit_rich_false_wins_over_entry_sol_heuristic(tmp_path):
    art = _make_artifact(tmp_path, {"rich_features": False})
    srv = ModelServer(art)
    assert srv.rich_entry is False  # regression: entry_sol must not trip the rich path


def test_legacy_spec_without_flag_falls_back_to_heuristic(tmp_path):
    art = _make_artifact(tmp_path, {})
    srv = ModelServer(art)
    # Documented fallback: legacy names include entry_sol -> heuristic says rich.
    # Any new artifact MUST therefore write the explicit flag.
    assert srv.rich_entry is True


def test_threshold_parsing_both_keys(tmp_path):
    art = _make_artifact(tmp_path, {"rich_features": False},
                         threshold_key="entry_threshold_top_decile")
    assert ModelServer(art).entry_threshold == pytest.approx(0.5)
    art2 = _make_artifact(tmp_path, {"rich_features": False}, threshold_key="threshold")
    assert ModelServer(art2).entry_threshold == pytest.approx(0.5)


def test_recovery_disabled_never_cuts(tmp_path):
    srv = ModelServer(_make_artifact(tmp_path, {"rich_features": False}))
    p, cut = srv.score_recovery({}, {})
    assert p == 1.0 and cut is False


def test_score_entry_threshold_semantics(tmp_path):
    srv = ModelServer(_make_artifact(tmp_path, {"rich_features": False}))
    feats = {name: 0.0 for name in ENTRY_FEATURE_NAMES}
    score, fire = srv.score_entry(feats)
    assert 0.0 <= score <= 1.0
    assert fire == (score >= srv.entry_threshold)
