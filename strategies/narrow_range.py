"""Narrow Range strategy (NR4/NR7) from PDF.

Identify the narrowest range candle of last 4 or 7 bars.
Breakout above/below that candle triggers entry.
Similar to CB compression but uses single-bar range comparison.
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class NarrowRangeStrategy(BaseStrategy):
    """NR4 + NR7 narrow range breakout.

    Signal when current bar has narrowest range of last 7 bars,
    and next bar breaks above/below.
    """

    name = "narrow_range"
    exit_type = "breakout"

    _NR_PERIOD = 7

    def prepare(self, df):
        """Pre-compute NR7 flags."""
        ranges = (df['high'] - df['low']).values
        n = len(ranges)
        self._is_nr = np.zeros(n, dtype=bool)

        for i in range(self._NR_PERIOD, n):
            window = ranges[i - self._NR_PERIOD:i + 1]
            if ranges[i] <= np.min(window[:-1]):
                self._is_nr[i] = True

    def detect(self, df, idx):
        if idx < self._NR_PERIOD + 1:
            return 0

        # Previous bar must be NR7
        if not self._is_nr[idx - 1]:
            return 0

        nr_high = df['high'].iloc[idx - 1]
        nr_low = df['low'].iloc[idx - 1]
        close = df['close'].iloc[idx]

        # Breakout above NR bar high
        if close > nr_high:
            return 1

        # Breakout below NR bar low
        if close < nr_low:
            return -1

        return 0
