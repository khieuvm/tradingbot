"""Bollinger Band Squeeze strategy from PDF.

When BB width contracts (squeeze), wait for breakout.
BUY: full candle closes above BB middle after squeeze
SELL: full candle closes below BB middle after squeeze
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from strategies.base import BaseStrategy


class BBSqueezeStrategy(BaseStrategy):
    """Bollinger Band squeeze into breakout."""

    name = "bb_squeeze"
    exit_type = "breakout"

    _SQUEEZE_LOOKBACK = 20
    _SQUEEZE_PERCENTILE = 25  # BB width in lowest 25th percentile = squeeze

    def prepare(self, df):
        """Pre-compute BB width and squeeze detection."""
        bb = ta.bbands(df['close'], length=20, std=2.0)
        if bb is not None and len(bb.columns) >= 3:
            self._bb_lower = bb.iloc[:, 0].values
            self._bb_mid = bb.iloc[:, 1].values
            self._bb_upper = bb.iloc[:, 2].values
            self._bb_width = (self._bb_upper - self._bb_lower) / self._bb_mid
        else:
            self._bb_width = np.full(len(df), np.nan)
            self._bb_mid = np.full(len(df), np.nan)
            self._bb_upper = np.full(len(df), np.nan)
            self._bb_lower = np.full(len(df), np.nan)

    def detect(self, df, idx):
        if idx < 40:
            return 0

        width = self._bb_width[idx]
        if np.isnan(width):
            return 0

        # Check if in squeeze: current width in lowest percentile of recent history
        lookback_start = max(0, idx - 100)
        recent_widths = self._bb_width[lookback_start:idx]
        recent_widths = recent_widths[~np.isnan(recent_widths)]
        if len(recent_widths) < 20:
            return 0

        threshold = np.percentile(recent_widths, self._SQUEEZE_PERCENTILE)

        # Need squeeze on prev bar, expansion starting on current bar
        prev_width = self._bb_width[idx - 1]
        if np.isnan(prev_width):
            return 0

        was_squeeze = prev_width <= threshold
        expanding = width > prev_width

        if not (was_squeeze and expanding):
            return 0

        close = df['close'].iloc[idx]
        bb_mid = self._bb_mid[idx]

        # BUY: close above BB middle
        if close > bb_mid and df['close'].iloc[idx - 1] <= self._bb_mid[idx - 1]:
            return 1

        # SELL: close below BB middle
        if close < bb_mid and df['close'].iloc[idx - 1] >= self._bb_mid[idx - 1]:
            return -1

        return 0
