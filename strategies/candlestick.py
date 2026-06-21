"""Candlestick Pattern Strategy.

Detected patterns (manual, not pandas_ta):
  - Bullish Engulfing: prev bearish, curr bullish, curr body fully covers prev body
  - Bearish Engulfing: prev bullish, curr bearish, curr body fully covers prev body
  - Hammer: lower wick > 2x body, tiny upper wick, after 3+ bar downmove
  - Shooting Star: upper wick > 2x body, tiny lower wick, after 3+ bar upmove

All patterns require a preceding trend of at least 3 bars in the opposite direction.
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class CandlestickStrategy(BaseStrategy):
    name = "candlestick"
    exit_type = "reversal"

    def __init__(self, trend_bars=3, wick_body_ratio=2.0, upper_wick_max_ratio=0.5):
        self.trend_bars = trend_bars
        self.wick_body_ratio = wick_body_ratio
        # For hammer/shooting star: upper/lower wick must be < this fraction of body
        self.upper_wick_max_ratio = upper_wick_max_ratio

    def _is_bullish(self, o, c):
        return c > o

    def _is_bearish(self, o, c):
        return c < o

    def _preceding_downtrend(self, df, idx, n):
        """True if closes have been declining over the n bars before idx."""
        if idx < n:
            return False
        closes = [df['close'].iloc[idx - i - 1] for i in range(n)]
        # At least n-1 of the n bars should close lower than the one before them
        declining = sum(1 for i in range(len(closes) - 1) if closes[i] < closes[i + 1])
        return declining >= n - 1

    def _preceding_uptrend(self, df, idx, n):
        """True if closes have been rising over the n bars before idx."""
        if idx < n:
            return False
        closes = [df['close'].iloc[idx - i - 1] for i in range(n)]
        rising = sum(1 for i in range(len(closes) - 1) if closes[i] > closes[i + 1])
        return rising >= n - 1

    def detect(self, df, idx):
        if idx < self.trend_bars + 1:
            return 0

        # Current and previous bar
        curr = df.iloc[idx]
        prev = df.iloc[idx - 1]

        c_o, c_h, c_l, c_c = curr['open'], curr['high'], curr['low'], curr['close']
        p_o, p_h, p_l, p_c = prev['open'], prev['high'], prev['low'], prev['close']

        # Body and wick measures for current bar
        c_body = abs(c_c - c_o)
        c_upper_wick = c_h - max(c_o, c_c)
        c_lower_wick = min(c_o, c_c) - c_l

        # Body and wick measures for previous bar
        p_body = abs(p_c - p_o)

        # Avoid division by zero for tiny-body bars
        min_body = 0.1

        # --- Bullish Engulfing (BUY) ---
        # prev: bearish, curr: bullish, curr body completely engulfs prev body
        if (self._is_bearish(p_o, p_c) and
                self._is_bullish(c_o, c_c) and
                c_o < p_c and c_c > p_o and
                self._preceding_downtrend(df, idx, self.trend_bars)):
            return 1

        # --- Bearish Engulfing (SELL) ---
        # prev: bullish, curr: bearish, curr body completely engulfs prev body
        if (self._is_bullish(p_o, p_c) and
                self._is_bearish(c_o, c_c) and
                c_o > p_c and c_c < p_o and
                self._preceding_uptrend(df, idx, self.trend_bars)):
            return -1

        # --- Hammer (BUY) ---
        # Lower wick > 2x body, small upper wick, after downtrend
        if (c_body >= min_body and
                c_lower_wick >= self.wick_body_ratio * c_body and
                c_upper_wick <= c_body * self.upper_wick_max_ratio and
                self._preceding_downtrend(df, idx, self.trend_bars)):
            return 1

        # --- Shooting Star (SELL) ---
        # Upper wick > 2x body, small lower wick, after uptrend
        if (c_body >= min_body and
                c_upper_wick >= self.wick_body_ratio * c_body and
                c_lower_wick <= c_body * self.upper_wick_max_ratio and
                self._preceding_uptrend(df, idx, self.trend_bars)):
            return -1

        return 0
