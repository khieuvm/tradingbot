"""Strategy 22: Gann Fan / Angles.

Identifies significant swing low and swing high over the last 50 bars, then
projects a 1×1 Gann angle (scaled to ATR) from each anchor.

  Ascending angle from significant low:
    level(t) = sig_low + bars_since_low * (ATR / 3)
    BUY: current bar's low touches the angle (within 0.5 pts) AND close > angle
         (price bounced off the ascending support angle)

  Descending angle from significant high:
    level(t) = sig_high - bars_since_high * (ATR / 3)
    SELL: current bar's high touches the angle (within 0.5 pts) AND close < angle
          (price rejected by the descending resistance angle)

Minimum 5 bars from the anchor point are required for the angle to be meaningful.
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class GannStrategy(BaseStrategy):
    name = "gann"
    exit_type = "mean_reversion"

    def __init__(self, lookback=50, angle_divisor=3.0, touch_tolerance=0.5,
                 min_bars_from_anchor=5):
        """
        Args:
            lookback: bars to search for the significant low/high
            angle_divisor: ATR divisor for the 1x1 angle slope (level per bar)
            touch_tolerance: max distance (pts) between bar extreme and angle level
            min_bars_from_anchor: minimum bars that must have elapsed since anchor
        """
        self.lookback = lookback
        self.angle_divisor = angle_divisor
        self.touch_tolerance = touch_tolerance
        self.min_bars_from_anchor = min_bars_from_anchor

    def detect(self, df, idx):
        if idx < self.lookback:
            return 0

        atr = df['atr'].iloc[idx]
        if pd.isna(atr) or atr <= 0:
            return 0

        cur_close = df['close'].iloc[idx]
        cur_low = df['low'].iloc[idx]
        cur_high = df['high'].iloc[idx]

        # Window excludes the current bar so the anchor is in the past
        window = df.iloc[idx - self.lookback: idx]
        lows_w = window['low'].values
        highs_w = window['high'].values
        n_w = len(lows_w)   # = self.lookback

        sig_low_widx = int(np.argmin(lows_w))
        sig_high_widx = int(np.argmax(highs_w))

        sig_low = lows_w[sig_low_widx]
        sig_high = highs_w[sig_high_widx]

        # bars_since = distance from the anchor to the current bar
        bars_since_low = n_w - sig_low_widx       # always >= 1
        bars_since_high = n_w - sig_high_widx

        slope = atr / self.angle_divisor

        # --- Ascending angle from significant low ---
        if bars_since_low >= self.min_bars_from_anchor:
            asc_level = sig_low + bars_since_low * slope
            # Bar's low touched the ascending angle AND close held above it
            if abs(cur_low - asc_level) <= self.touch_tolerance and cur_close > asc_level:
                return 1

        # --- Descending angle from significant high ---
        if bars_since_high >= self.min_bars_from_anchor:
            desc_level = sig_high - bars_since_high * slope
            # Bar's high touched the descending angle AND close fell below it
            if abs(cur_high - desc_level) <= self.touch_tolerance and cur_close < desc_level:
                return -1

        return 0
