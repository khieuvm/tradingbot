"""Role Reversal strategy from PDF.

When support is broken, it becomes resistance (sell on retest).
When resistance is broken, it becomes support (buy on retest).
Uses recent swing highs/lows as S/R levels.
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy
from strategies.utils.swing import detect_swing_points


class RoleReversalStrategy(BaseStrategy):
    """Broken S/R flip — buy at old resistance turned support, sell at old support turned resistance."""

    name = "role_reversal"
    exit_type = "mean_reversion"

    _TOLERANCE_ATR = 0.3  # how close price must be to level (in ATR units)
    _MIN_BREAK_BARS = 3   # min bars since break to confirm role reversal

    def prepare(self, df):
        """Pre-compute swing points."""
        self._swing_highs, self._swing_lows = detect_swing_points(
            df['high'].values, df['low'].values, lookback=5)

    def detect(self, df, idx):
        if idx < 30:
            return 0

        atr = df['atr'].iloc[idx]
        if pd.isna(atr) or atr < 1.0:
            return 0

        close = df['close'].iloc[idx]
        tolerance = self._TOLERANCE_ATR * atr

        # Find recent swing highs (potential resistance turned support)
        sh = self._swing_highs[:idx]
        sh_list = [(i, sh[i]) for i in range(len(sh)) if not np.isnan(sh[i])]

        # Find recent swing lows (potential support turned resistance)
        sl = self._swing_lows[:idx]
        sl_list = [(i, sl[i]) for i in range(len(sl)) if not np.isnan(sl[i])]

        # BUY: old resistance turned support
        # Price broke above a swing high, came back to retest it from above
        for sh_idx, sh_val in reversed(sh_list[-10:]):
            bars_since = idx - sh_idx
            if bars_since < self._MIN_BREAK_BARS:
                continue
            if bars_since > 50:
                break
            # Was resistance (price was below), now price is above = broken resistance
            # Price is retesting from above (close near the level, approaching from above)
            if close > sh_val and abs(close - sh_val) < tolerance:
                # Confirm: price was above level for at least 2 bars (broken through)
                if df['close'].iloc[idx - 1] > sh_val and df['close'].iloc[idx - 2] > sh_val:
                    return 1

        # SELL: old support turned resistance
        # Price broke below a swing low, came back to retest it from below
        for sl_idx, sl_val in reversed(sl_list[-10:]):
            bars_since = idx - sl_idx
            if bars_since < self._MIN_BREAK_BARS:
                continue
            if bars_since > 50:
                break
            # Was support (price was above), now price is below = broken support
            # Price retesting from below
            if close < sl_val and abs(sl_val - close) < tolerance:
                if df['close'].iloc[idx - 1] < sl_val and df['close'].iloc[idx - 2] < sl_val:
                    return -1

        return 0
