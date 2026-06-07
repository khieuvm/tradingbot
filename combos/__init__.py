from combos.base import BaseCombo
from combos.cb import CBCombo
from combos.nr4 import NR4Combo

COMBO_REGISTRY: dict[str, type[BaseCombo]] = {
    "CB": CBCombo,
    "NR4": NR4Combo,
}


def get_combo(name: str) -> BaseCombo:
    """Get combo instance by name."""
    cls = COMBO_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown combo: {name}. Available: {list(COMBO_REGISTRY.keys())}")
    return cls()
