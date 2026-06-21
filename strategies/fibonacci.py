"""Fibonacci Retracement Strategy.

BUY: price pulls back to 61.8% retracement level from above (uptrend context).
SELL: price rallies to 61.8% retracement level from below (downtrend context).
"""
import numpy as np
from strategies.base import BaseStrategy
from strategies.utils.swing import detect_swing_points, get_recent_swings
from strategies.utils.levels import compute_fibonacci


class FibonacciStrategy(BaseStrategy):
    name = "fibonacci"
    exit_type = "mean_reversion"

    def __init__(self, swing_lookback=15, min_swing=3.0, tolerance=0.5):
        self.swing_lookback = swing_lookback
        self.min_swing = min_swing
        self.tolerance = tolerance

    def detect(self, df, idx):
        lb = self.swing_lookback
        # Need at least 2*lookback bars for swing detection
        start = max(0, idx - 100)
        window = df.iloc[start:idx + 1]

        if len(window) < 2 * lb + 1:
            return 0

        highs = window['high'].values
        lows = window['low'].values

        sh, sl = detect_swing_points(highs, lows, lookback=lb)
        sh_list, sl_list = get_recent_swings(sh, sl, n_recent=3)

        if not sh_list or not sl_list:
            return 0

        last_sh_idx, last_sh_val = sh_list[-1]
        last_sl_idx, last_sl_val = sl_list[-1]

        swing_size = last_sh_val - last_sl_val
        if swing_size < self.min_swing:
            return 0

        fib = compute_fibonacci(last_sh_val, last_sl_val)
        fib_618 = fib['0.618']  # swing_high - 0.618 * range

        curr = df.iloc[idx]
        curr_low = curr['low']
        curr_high = curr['high']
        curr_close = curr['close']

        # BUY: uptrend context — last swing high came after last swing low.
        # Price pulled back to 61.8% level, touching from above, closing above it.
        if last_sh_idx > last_sl_idx:
            touched = curr_low <= fib_618 + self.tolerance
            closed_above = curr_close > fib_618
            if touched and closed_above:
                return 1

        # SELL: downtrend context — last swing low came after last swing high.
        # Price rallied to 61.8% level, touching from below, closing below it.
        if last_sl_idx > last_sh_idx:
            touched = curr_high >= fib_618 - self.tolerance
            closed_below = curr_close < fib_618
            if touched and closed_below:
                return -1

        return 0
