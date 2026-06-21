"""Strategy 16: RSI/MACD Divergence.

Regular Bullish Divergence: price makes Lower Low, RSI makes Higher Low -> BUY
Regular Bearish Divergence: price makes Higher High, RSI makes Lower High -> SELL

Swing points are detected in price; RSI values are read at those same price swing
point indices (not independently detected RSI swings).
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy
from strategies.utils.swing import detect_swing_points, get_recent_swings


class DivergenceStrategy(BaseStrategy):
    name = "divergence"
    exit_type = "reversal"

    def __init__(self, lookback=20, swing_n=3, min_bars_between=5):
        """
        Args:
            lookback: bars of history to analyse
            swing_n: lookback for detect_swing_points (bars on each side)
            min_bars_between: minimum bar gap required between two swing points
        """
        self.lookback = lookback
        self.swing_n = swing_n
        self.min_bars_between = min_bars_between

    def detect(self, df, idx):
        if idx < self.lookback + self.swing_n:
            return 0

        window = df.iloc[idx - self.lookback: idx + 1]
        highs = window['high'].values
        lows = window['low'].values
        rsi = window['rsi'].values
        n = len(window)

        # Require valid RSI across the window
        if np.any(np.isnan(rsi)):
            return 0

        sh_price, sl_price = detect_swing_points(highs, lows, lookback=self.swing_n)
        sh_list, sl_list = get_recent_swings(sh_price, sl_price, n_recent=3)

        # Recency threshold: last swing point must be within this many bars of current
        recency_threshold = self.swing_n + 2

        # --- Bearish divergence: price HH but RSI LH -> SELL ---
        if len(sh_list) >= 2:
            last_sh = sh_list[-1]   # (window_idx, price_value)
            prev_sh = sh_list[-2]

            # Last swing high must be recent
            if (n - 1 - last_sh[0]) <= recency_threshold:
                # Minimum separation between the two swing highs
                if (last_sh[0] - prev_sh[0]) >= self.min_bars_between:
                    # Price makes Higher High
                    if last_sh[1] > prev_sh[1]:
                        # RSI at last SH must be lower than RSI at prev SH
                        if rsi[last_sh[0]] < rsi[prev_sh[0]]:
                            return -1

        # --- Bullish divergence: price LL but RSI HL -> BUY ---
        if len(sl_list) >= 2:
            last_sl = sl_list[-1]
            prev_sl = sl_list[-2]

            if (n - 1 - last_sl[0]) <= recency_threshold:
                if (last_sl[0] - prev_sl[0]) >= self.min_bars_between:
                    # Price makes Lower Low
                    if last_sl[1] < prev_sl[1]:
                        # RSI at last SL must be higher than RSI at prev SL
                        if rsi[last_sl[0]] > rsi[prev_sl[0]]:
                            return 1

        return 0
