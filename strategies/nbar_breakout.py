"""N-bar Breakout Strategy.

Compute N-bar high/low (excluding current bar) from a consolidation window.
BUY when close breaks above N-bar high, SELL when close breaks below N-bar low.
Requires consolidation: N-bar range < 2.5 * ATR.
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class NBarBreakoutStrategy(BaseStrategy):
    name = "nbar_breakout"
    exit_type = "breakout"

    def __init__(self, n=10, consolidation_mult=2.5, offset=0.1):
        self.n = n
        self.consolidation_mult = consolidation_mult
        self.offset = offset

    def detect(self, df, idx):
        n = self.n
        # Need n bars before the current bar
        if idx < n:
            return 0

        # Window: the N bars immediately preceding the current bar (exclusive)
        window = df.iloc[idx - n:idx]

        n_high = window['high'].max()
        n_low = window['low'].min()
        n_range = n_high - n_low

        atr = df['atr'].iloc[idx]
        if pd.isna(atr) or atr <= 0:
            return 0

        # Only trade if the preceding N bars show consolidation
        if n_range >= self.consolidation_mult * atr:
            return 0

        curr_close = df['close'].iloc[idx]

        if curr_close > n_high + self.offset:
            return 1
        if curr_close < n_low - self.offset:
            return -1

        return 0
