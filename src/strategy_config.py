"""Single source of truth: loads strategy_config.yaml once, provides typed access."""

import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "strategy_config.yaml"
_cache: dict | None = None


def get_config() -> dict:
    global _cache
    if _cache is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _cache = yaml.safe_load(f)
    return _cache


def reload_config() -> dict:
    global _cache
    _cache = None
    return get_config()


def get_combo_config(combo_name: str) -> dict:
    """Get per-combo config section from combos.<name>."""
    cfg = get_config()
    combos = cfg.get("combos", {})
    return combos.get(combo_name, {})


def get_session_params() -> dict:
    """Get shared session exit params (AM/PM SL, trail, max_hold)."""
    cfg = get_config()
    return cfg.get("session_params", {
        "AM": {"sl_atr_mult": 1.5, "trail_activate_pts": 7.0, "trail_atr_mult": 2.5, "max_hold_bars": 30,
                "be_trigger_pts": 3.0, "be_atr_min": 3.5, "be_partial_pts": 1.0},
        "PM": {"sl_atr_mult": 1.2, "trail_activate_pts": 6.0, "trail_atr_mult": 1.0, "max_hold_bars": 8,
                "be_trigger_pts": 3.0, "be_atr_min": 3.5, "be_partial_pts": 1.0},
    })


def get_adaptive_exit() -> dict:
    """Get adaptive exit config."""
    cfg = get_config()
    return cfg.get("adaptive_exit", {})
