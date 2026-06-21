"""Heikin-Ashi + Stochastic strategy from PDF.

Combines HA reversal pattern (2 consecutive candles in new direction)
with Stochastic filter for confirmation.
BUY: 2 green HA candles after red series + Stochastic oversold
SELL: 2 red HA candles after green series + Stochastic overbought
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from strategies.base import BaseStrategy


class HAStochStrategy(BaseStrategy):
    """Heikin-Ashi reversal confirmed by Stochastic oscillator."""

    name = "ha_stoch"
    exit_type = "reversal"

    _MIN_PRIOR_CANDLES = 3  # min candles in prior trend direction

    def prepare(self, df):
        """Pre-compute HA candles and Stochastic."""
        closes = df['close'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)

        # Compute Heikin-Ashi
        ha_close = (opens + highs + lows + closes) / 4.0
        ha_open = np.zeros(n)
        ha_open[0] = (opens[0] + closes[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0

        self._ha_green = ha_close > ha_open
        self._ha_red = ha_close < ha_open

        # Stochastic (14,3,3)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None and len(stoch) == n:
            self._stoch_k = stoch.iloc[:, 0].values
        else:
            self._stoch_k = np.full(n, 50.0)

    def detect(self, df, idx):
        if idx < 10:
            return 0

        stoch_k = self._stoch_k[idx]
        if np.isnan(stoch_k):
            return 0

        # Check for 2 consecutive green HA after series of red
        if self._ha_green[idx] and self._ha_green[idx - 1]:
            # Count prior red candles
            red_count = 0
            for j in range(idx - 2, max(idx - 15, -1), -1):
                if self._ha_red[j]:
                    red_count += 1
                else:
                    break
            if red_count >= self._MIN_PRIOR_CANDLES and stoch_k < 30:
                return 1

        # Check for 2 consecutive red HA after series of green
        if self._ha_red[idx] and self._ha_red[idx - 1]:
            green_count = 0
            for j in range(idx - 2, max(idx - 15, -1), -1):
                if self._ha_green[j]:
                    green_count += 1
                else:
                    break
            if green_count >= self._MIN_PRIOR_CANDLES and stoch_k > 70:
                return -1

        return 0
