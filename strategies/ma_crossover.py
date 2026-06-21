"""Moving Average Crossover strategy (20/60/100) from PDF.

Fast MA(20) crossing slow MA(60), with MA(100) as trend filter.
BUY: MA20 crosses above MA60, price above MA100
SELL: MA20 crosses below MA60, price below MA100
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from strategies.base import BaseStrategy


class MACrossoverStrategy(BaseStrategy):
    """Triple MA crossover (20/60/100) with trend confirmation."""

    name = "ma_crossover"
    exit_type = "trend"

    def prepare(self, df):
        """Pre-compute MAs."""
        self._ma20 = ta.ema(df['close'], length=20).values
        self._ma60 = ta.ema(df['close'], length=60).values
        self._ma100 = ta.ema(df['close'], length=100).values

    def detect(self, df, idx):
        if idx < 101:
            return 0

        ma20 = self._ma20[idx]
        ma20_prev = self._ma20[idx - 1]
        ma60 = self._ma60[idx]
        ma60_prev = self._ma60[idx - 1]
        ma100 = self._ma100[idx]
        close = df['close'].iloc[idx]

        if np.isnan(ma20) or np.isnan(ma60) or np.isnan(ma100):
            return 0

        cross_up = ma20 > ma60 and ma20_prev <= ma60_prev
        cross_down = ma20 < ma60 and ma20_prev >= ma60_prev

        if cross_up and close > ma100:
            return 1

        if cross_down and close < ma100:
            return -1

        return 0
