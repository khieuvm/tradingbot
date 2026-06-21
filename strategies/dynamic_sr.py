"""Dynamic S/R Strategy using EMA50 and Bollinger Bands.

Primary signals: price touches EMA50 with trend confirmation.
Alt signals: price touches outer Bollinger Band and reverses.
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class DynamicSRStrategy(BaseStrategy):
    name = "dynamic_sr"
    exit_type = "mean_reversion"

    def __init__(self, tolerance=0.3):
        self.tolerance = tolerance

    def detect(self, df, idx):
        row = df.iloc[idx]

        ema8 = row['ema8']
        ema21 = row['ema21']
        ema50 = row['ema50']
        bb_upper = row['bb_upper']
        bb_lower = row['bb_lower']
        curr_open = row['open']
        curr_high = row['high']
        curr_low = row['low']
        curr_close = row['close']

        # Skip if any required indicator is missing
        if pd.isna(ema50) or pd.isna(ema8) or pd.isna(ema21):
            return 0
        if pd.isna(bb_upper) or pd.isna(bb_lower):
            return 0

        tol = self.tolerance

        # --- BUY signals ---
        # Primary: low touches EMA50 from above, closes above EMA50, EMA8 > EMA21 (uptrend)
        ema50_touch_buy = curr_low <= ema50 + tol and curr_low >= ema50 - tol * 3
        if ema50_touch_buy and curr_close > ema50 and ema8 > ema21:
            return 1

        # Alt: low touches lower Bollinger Band, closes above it
        bb_lower_touch = curr_low <= bb_lower + tol and curr_low >= bb_lower - tol * 3
        if bb_lower_touch and curr_close > bb_lower:
            return 1

        # --- SELL signals ---
        # Primary: high touches EMA50 from below, closes below EMA50, EMA8 < EMA21 (downtrend)
        ema50_touch_sell = curr_high >= ema50 - tol and curr_high <= ema50 + tol * 3
        if ema50_touch_sell and curr_close < ema50 and ema8 < ema21:
            return -1

        # Alt: high touches upper Bollinger Band, closes below it
        bb_upper_touch = curr_high >= bb_upper - tol and curr_high <= bb_upper + tol * 3
        if bb_upper_touch and curr_close < bb_upper:
            return -1

        return 0
