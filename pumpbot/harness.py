"""Trading harness surface (canonical: shadow_harness.py, paper_book.py,
position_store.py). Importing attributes reads config.yaml (bot_config);
run from the repo root."""
from ._lazy import make_lazy

make_lazy(__name__, {
    "ShadowHarness": ("shadow_harness", "ShadowHarness"),
    "PaperBook": ("paper_book", "PaperBook"),
    "PositionStore": ("position_store", "PositionStore"),
})
