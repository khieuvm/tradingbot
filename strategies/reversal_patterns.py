"""Reversal Patterns Strategy: Double Top and Double Bottom.

Double Top: two swing highs within 1.0 pts, at least 5 bars apart.
  SELL when close breaks below the neckline (lowest point between peaks) - 0.1.

Double Bottom: two swing lows within 1.0 pts, at least 5 bars apart.
  BUY when close breaks above the neckline (highest point between troughs) + 0.1.
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy
from strategies.utils.swing import detect_swing_points, get_recent_swings


class ReversalPatternsStrategy(BaseStrategy):
    name = "reversal_patterns"
    exit_type = "reversal"

    def __init__(self, swing_lookback=5, peak_tol=1.0, min_bar_sep=5,
                 window_bars=80, breakout_offset=0.1):
        self.swing_lookback = swing_lookback
        self.peak_tol = peak_tol
        self.min_bar_sep = min_bar_sep
        self.window_bars = window_bars
        self.breakout_offset = breakout_offset

    def detect(self, df, idx):
        lb = self.swing_lookback
        start = max(0, idx - self.window_bars)
        window = df.iloc[start:idx + 1]

        if len(window) < 2 * lb + 1:
            return 0

        highs = window['high'].values
        lows = window['low'].values
        closes = window['close'].values

        sh, sl = detect_swing_points(highs, lows, lookback=lb)
        sh_list, sl_list = get_recent_swings(sh, sl, n_recent=6)

        curr_close = df['close'].iloc[idx]

        # --- Double Top → SELL ---
        if len(sh_list) >= 2:
            # Check all pairs of swing highs for double-top condition
            for k in range(len(sh_list) - 1, 0, -1):
                peak2_idx, peak2_val = sh_list[k]
                peak1_idx, peak1_val = sh_list[k - 1]

                # Must be separated by enough bars
                bar_sep = peak2_idx - peak1_idx
                if bar_sep < self.min_bar_sep:
                    continue

                # Peaks must be within tolerance of each other
                if abs(peak2_val - peak1_val) > self.peak_tol:
                    continue

                # Neckline: lowest close between the two peaks
                between = closes[peak1_idx:peak2_idx + 1]
                if len(between) == 0:
                    continue
                neckline = np.min(between)

                # SELL when current close breaks below neckline
                if curr_close < neckline - self.breakout_offset:
                    return -1

        # --- Double Bottom → BUY ---
        if len(sl_list) >= 2:
            for k in range(len(sl_list) - 1, 0, -1):
                trough2_idx, trough2_val = sl_list[k]
                trough1_idx, trough1_val = sl_list[k - 1]

                bar_sep = trough2_idx - trough1_idx
                if bar_sep < self.min_bar_sep:
                    continue

                if abs(trough2_val - trough1_val) > self.peak_tol:
                    continue

                # Neckline: highest close between the two troughs
                between = closes[trough1_idx:trough2_idx + 1]
                if len(between) == 0:
                    continue
                neckline = np.max(between)

                # BUY when current close breaks above neckline
                if curr_close > neckline + self.breakout_offset:
                    return 1

        return 0
