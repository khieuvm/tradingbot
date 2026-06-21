"""RSI 2-Period strategy from PDF.

Very aggressive mean-reversion strategy using 2-period RSI.
BUY: RSI(2) drops below 10 (extreme oversold)
SELL: RSI(2) rises above 90 (extreme overbought)
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from strategies.base import BaseStrategy


class RSI2Strategy(BaseStrategy):
    """RSI(2) extreme reversal strategy."""

    name = "rsi2"
    exit_type = "mean_reversion"

    def prepare(self, df):
        """Pre-compute RSI(2)."""
        self._rsi2 = ta.rsi(df['close'], length=2).values

    def detect(self, df, idx):
        if idx < 5:
            return 0

        rsi2 = self._rsi2[idx]
        rsi2_prev = self._rsi2[idx - 1]

        if np.isnan(rsi2) or np.isnan(rsi2_prev):
            return 0

        # BUY: RSI(2) crosses up from below 10
        if rsi2 > 10 and rsi2_prev <= 10:
            return 1

        # SELL: RSI(2) crosses down from above 90
        if rsi2 < 90 and rsi2_prev >= 90:
            return -1

        return 0
