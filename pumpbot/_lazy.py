"""PEP 562 lazy re-export helper.

make_lazy(__name__, {"TokenState": ("feature_accum", "TokenState"), ...})
installs a module-level __getattr__ that imports the flat module on first
attribute access. Keeps `import pumpbot.harness` free of side effects
(config.yaml reads, env-var trigger resolution) until something is used.
"""
import importlib
import sys

from ._paths import bootstrap


def make_lazy(pkg_module_name: str, exports: dict[str, tuple[str, str]]):
    mod = sys.modules[pkg_module_name]

    def __getattr__(name: str):
        if name not in exports:
            raise AttributeError(f"module {pkg_module_name!r} has no attribute {name!r}")
        bootstrap()
        flat_mod, flat_attr = exports[name]
        target = importlib.import_module(flat_mod)
        val = getattr(target, flat_attr)
        setattr(mod, name, val)  # cache
        return val

    def __dir__():
        return sorted(exports)

    mod.__getattr__ = __getattr__
    mod.__dir__ = __dir__
    mod.__all__ = sorted(exports)
