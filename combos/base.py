from abc import ABC, abstractmethod
import pandas as pd

from src.strategy_config import get_combo_config, get_session_params


class BaseCombo(ABC):
    """Abstract base for all trading combos. Loads params from strategy_config.yaml."""

    name: str
    timeframe: str

    def __init__(self):
        self._cfg = get_combo_config(self.name)
        self._session_params = get_session_params()

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> dict | None:
        """Detect signal on latest bar. Return signal dict or None."""
        ...

    def get_session_params(self, session: str) -> dict:
        """Return exit params for AM/PM from shared config."""
        defaults = {"sl_atr_mult": 1.5, "trail_activate_pts": 7.0, "trail_atr_mult": 2.5, "max_hold_bars": 30,
                    "be_trigger_pts": 3.0, "be_atr_min": 3.5, "be_partial_pts": 1.0}
        return self._session_params.get(session, defaults)

    @property
    def dedup_bars(self) -> int:
        """Minimum bars between signals."""
        return self._cfg.get("dedup_bars", 5)
