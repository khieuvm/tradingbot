"""Strategy 21: Elliott Wave (Simplified 5-wave count).

Builds a zigzag from swing points over the last 40 bars and looks for a
completed wave-4 correction — signalling the start of wave 5.

Bullish pattern (L-H-L-H-L):
  Points: L0, H1, L2, H3, L4
  Wave 1 = H1 - L0          (must be > 0)
  Wave 2 retrace = (H1-L2)/(H1-L0)   (must be < 100%)
  Wave 3 = H3 - L2          (must be >= wave1, cannot be shortest)
  Wave 4 retrace = (H3-L4)/(H3-L2)   (must be 38.2%–61.8%)
  Signal: L4 is recent and current close > L4  ->  BUY

Bearish pattern (H-L-H-L-H):
  Points: H0, L1, H2, L3, H4
  Wave 1 = H0 - L1          (must be > 0)
  Wave 2 retrace = (H2-L1)/(H0-L1)   (must be < 100%)
  Wave 3 = H2 - L3          (must be >= wave1)
  Wave 4 retrace = (H4-L3)/(H2-L3)   (must be 38.2%–61.8%)
  Signal: H4 is recent and current close < H4  ->  SELL
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy
from strategies.utils.swing import detect_swing_points, build_zigzag


class ElliottWaveStrategy(BaseStrategy):
    name = "elliott_wave"
    exit_type = "trend"

    def __init__(self, lookback=40, swing_lookback=2, recency_bars=6,
                 w4_retrace_lo=0.382, w4_retrace_hi=0.618):
        """
        Args:
            lookback: bars of history to build zigzag from
            swing_lookback: bars on each side for swing detection (smaller = more sensitive)
            recency_bars: wave-4 point must be within this many bars of current bar
            w4_retrace_lo: minimum wave-4 retracement of wave 3
            w4_retrace_hi: maximum wave-4 retracement of wave 3
        """
        self.lookback = lookback
        self.swing_lookback = swing_lookback
        self.recency_bars = recency_bars
        self.w4_retrace_lo = w4_retrace_lo
        self.w4_retrace_hi = w4_retrace_hi

    def _validate_bullish(self, l0, h1, l2, h3, l4):
        """Return True if L0-H1-L2-H3-L4 satisfies Elliott wave rules."""
        w1 = h1 - l0
        w3 = h3 - l2
        if w1 <= 0 or w3 <= 0:
            return False
        # Wave 2 must not retrace 100%+ of wave 1 (L2 must stay above L0)
        w2_retrace = (h1 - l2) / w1
        if w2_retrace >= 1.0:
            return False
        # Wave 3 cannot be shorter than wave 1
        if w3 < w1:
            return False
        # Wave 4 must retrace 38.2%–61.8% of wave 3
        w4_retrace = (h3 - l4) / w3
        if not (self.w4_retrace_lo <= w4_retrace <= self.w4_retrace_hi):
            return False
        return True

    def _validate_bearish(self, h0, l1, h2, l3, h4):
        """Return True if H0-L1-H2-L3-H4 satisfies Elliott wave rules."""
        w1 = h0 - l1
        w3 = h2 - l3
        if w1 <= 0 or w3 <= 0:
            return False
        w2_retrace = (h2 - l1) / w1
        if w2_retrace >= 1.0:
            return False
        if w3 < w1:
            return False
        w4_retrace = (h4 - l3) / w3
        if not (self.w4_retrace_lo <= w4_retrace <= self.w4_retrace_hi):
            return False
        return True

    def detect(self, df, idx):
        if idx < self.lookback + self.swing_lookback:
            return 0

        window = df.iloc[idx - self.lookback: idx + 1]
        highs = window['high'].values
        lows = window['low'].values
        n = len(window)

        sh, sl = detect_swing_points(highs, lows, lookback=self.swing_lookback)
        zigzag = build_zigzag(sh, sl)

        if len(zigzag) < 5:
            return 0

        cur_close = df['close'].iloc[idx]
        # Examine the last 5 zigzag points
        pts = zigzag[-5:]

        # --- Bullish: L-H-L-H-L ---
        if (pts[0][0] == 'L' and pts[1][0] == 'H' and pts[2][0] == 'L'
                and pts[3][0] == 'H' and pts[4][0] == 'L'):
            l4_widx = pts[4][1]   # index inside the window
            # Wave-4 point must be recent
            if (n - 1 - l4_widx) <= self.recency_bars:
                l0, h1, l2, h3, l4 = (pts[0][2], pts[1][2], pts[2][2],
                                       pts[3][2], pts[4][2])
                if self._validate_bullish(l0, h1, l2, h3, l4):
                    # Price is bouncing up from the wave-4 low
                    if cur_close > l4:
                        return 1

        # --- Bearish: H-L-H-L-H ---
        if (pts[0][0] == 'H' and pts[1][0] == 'L' and pts[2][0] == 'H'
                and pts[3][0] == 'L' and pts[4][0] == 'H'):
            h4_widx = pts[4][1]
            if (n - 1 - h4_widx) <= self.recency_bars:
                h0, l1, h2, l3, h4 = (pts[0][2], pts[1][2], pts[2][2],
                                       pts[3][2], pts[4][2])
                if self._validate_bearish(h0, l1, h2, l3, h4):
                    if cur_close < h4:
                        return -1

        return 0
