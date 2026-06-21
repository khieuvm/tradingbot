"""Trendline Strategy.

Fit ascending trendlines through recent swing lows (BUY on touch).
Fit descending trendlines through recent swing highs (SELL on touch).
Requires positive slope for ascending, negative for descending.
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy
from strategies.utils.swing import detect_swing_points, get_recent_swings


class TrendlineStrategy(BaseStrategy):
    name = "trendlines"
    exit_type = "trend"

    def __init__(self, swing_lookback=5, touch_tol=0.5, window_bars=60,
                 min_swing_pts=2):
        self.swing_lookback = swing_lookback
        self.touch_tol = touch_tol
        self.window_bars = window_bars
        self.min_swing_pts = min_swing_pts  # minimum swings to fit a line

    def _fit_trendline(self, swing_list):
        """Fit a line through swing points. Returns (slope, intercept) or None."""
        if len(swing_list) < 2:
            return None
        xs = np.array([s[0] for s in swing_list], dtype=float)
        ys = np.array([s[1] for s in swing_list], dtype=float)
        try:
            coeffs = np.polyfit(xs, ys, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None
        return coeffs  # (slope, intercept)

    def detect(self, df, idx):
        lb = self.swing_lookback
        start = max(0, idx - self.window_bars)
        window = df.iloc[start:idx + 1]

        if len(window) < 2 * lb + 1:
            return 0

        highs = window['high'].values
        lows = window['low'].values
        local_idx = len(window) - 1  # index of current bar within window

        sh, sl = detect_swing_points(highs, lows, lookback=lb)
        sh_list, sl_list = get_recent_swings(sh, sl, n_recent=6)

        curr_low = df['low'].iloc[idx]
        curr_high = df['high'].iloc[idx]
        curr_close = df['close'].iloc[idx]

        # --- Ascending trendline from swing lows (BUY signal) ---
        if len(sl_list) >= self.min_swing_pts:
            result = self._fit_trendline(sl_list)
            if result is not None:
                slope, intercept = result
                if slope > 0:  # must be ascending
                    trendline_val = slope * local_idx + intercept
                    touched = abs(curr_low - trendline_val) <= self.touch_tol
                    if touched and curr_close > trendline_val:
                        return 1

        # --- Descending trendline from swing highs (SELL signal) ---
        if len(sh_list) >= self.min_swing_pts:
            result = self._fit_trendline(sh_list)
            if result is not None:
                slope, intercept = result
                if slope < 0:  # must be descending
                    trendline_val = slope * local_idx + intercept
                    touched = abs(curr_high - trendline_val) <= self.touch_tol
                    if touched and curr_close < trendline_val:
                        return -1

        return 0
