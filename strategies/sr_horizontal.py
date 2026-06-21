"""Support & Resistance Horizontal Levels Strategy.

Cluster swing highs (resistance) and swing lows (support) over the last 50 bars.
BUY at support levels (min 2 touches), SELL at resistance levels (min 2 touches).
"""
import numpy as np
from strategies.base import BaseStrategy
from strategies.utils.swing import detect_swing_points, get_recent_swings
from strategies.utils.levels import cluster_levels


class SRHorizontalStrategy(BaseStrategy):
    name = "sr_horizontal"
    exit_type = "mean_reversion"

    def __init__(self, lookback_bars=50, cluster_tol=1.0, touch_tol=0.5,
                 min_touches=2, swing_lookback=5):
        self.lookback_bars = lookback_bars
        self.cluster_tol = cluster_tol
        self.touch_tol = touch_tol
        self.min_touches = min_touches
        self.swing_lookback = swing_lookback

    def detect(self, df, idx):
        lb = self.swing_lookback
        start = max(0, idx - self.lookback_bars)
        window = df.iloc[start:idx + 1]

        if len(window) < 2 * lb + 1:
            return 0

        highs = window['high'].values
        lows = window['low'].values

        sh, sl = detect_swing_points(highs, lows, lookback=lb)

        # Collect raw swing high and swing low prices (excluding NaN)
        res_levels = [highs[i] for i in range(len(sh)) if not np.isnan(sh[i])]
        sup_levels = [lows[i] for i in range(len(sl)) if not np.isnan(sl[i])]

        if not res_levels and not sup_levels:
            return 0

        curr = df.iloc[idx]
        curr_low = curr['low']
        curr_high = curr['high']
        curr_close = curr['close']

        # Cluster support levels (swing lows) and check for BUY
        if sup_levels:
            sup_clusters = cluster_levels(sup_levels, tolerance=self.cluster_tol)
            for level_price, touch_count in sup_clusters:
                if touch_count < self.min_touches:
                    continue
                # Price must be near or slightly below the support level
                if curr_low <= level_price + self.touch_tol and curr_close > level_price:
                    return 1

        # Cluster resistance levels (swing highs) and check for SELL
        if res_levels:
            res_clusters = cluster_levels(res_levels, tolerance=self.cluster_tol)
            for level_price, touch_count in res_clusters:
                if touch_count < self.min_touches:
                    continue
                # Price must be near or slightly above the resistance level
                if curr_high >= level_price - self.touch_tol and curr_close < level_price:
                    return -1

        return 0
