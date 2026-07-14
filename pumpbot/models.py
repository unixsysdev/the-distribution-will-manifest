"""Model serving (canonical: model_serve.py). Artifact contract:
bot_artifacts_*/entry_model.pkl (bare sklearn) + model_spec.json."""
from ._lazy import make_lazy

make_lazy(__name__, {
    "ModelServer": ("model_serve", "ModelServer"),
})
