"""Exit-policy registry surface (canonical: exit_policies/)."""
from ._lazy import make_lazy

make_lazy(__name__, {
    "get_policy": ("exit_policies.base", "get_policy"),
    "register": ("exit_policies.base", "register"),
    "ExitPolicy": ("exit_policies.base", "ExitPolicy"),
    "ExitDecision": ("exit_policies.base", "ExitDecision"),
    "HarnessConsts": ("exit_policies.base", "HarnessConsts"),
})
