"""Fair Value Gap (FVG) strategy — mean reversion back into unfilled gaps."""
import numpy as np
from strategies.base import BaseStrategy


class FVGStrategy(BaseStrategy):
    """Detect 3-bar Fair Value Gaps and trade when price re-enters unfilled zones.

    Bullish FVG: high[j] < low[j+2] — gap between bar-2 high and current low.
    Bearish FVG: low[j] > high[j+2] — inverse gap.
    Signal triggers when price revisits an unfilled gap on the current bar.
    """

    name = "fvg"
    exit_type = "mean_reversion"

    def detect(self, df, idx):
        if idx < 10:
            return 0

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        cur_low = lows[idx]
        cur_high = highs[idx]
        cur_close = closes[idx]

        # Scan FVG formations: 3-bar pattern at positions (j, j+1, j+2)
        # j+2 must be < idx so the formation is complete before the current bar.
        # range(a, idx-2) gives j up to idx-3, so j+2 up to idx-1. ✓
        scan_start = max(0, idx - 22)

        buy_signal = False
        sell_signal = False

        for j in range(scan_start, idx - 2):
            j2 = j + 2  # third bar of the formation

            # ── Bullish FVG ──────────────────────────────────────────────────
            # Gap zone: bottom = highs[j], top = lows[j2]
            b_bottom = highs[j]
            b_top = lows[j2]
            if b_top > b_bottom and (b_top - b_bottom) >= 0.3:
                # Unfilled: no intermediate bar (j+3 … idx-1) entered the zone
                unfilled = True
                for k in range(j + 3, idx):
                    if lows[k] <= b_top and highs[k] >= b_bottom:
                        unfilled = False
                        break
                if unfilled and cur_low <= b_top and cur_close > b_bottom:
                    buy_signal = True

            # ── Bearish FVG ──────────────────────────────────────────────────
            # Gap zone: bottom = highs[j2], top = lows[j]
            s_bottom = highs[j2]
            s_top = lows[j]
            if s_top > s_bottom and (s_top - s_bottom) >= 0.3:
                unfilled = True
                for k in range(j + 3, idx):
                    if lows[k] <= s_top and highs[k] >= s_bottom:
                        unfilled = False
                        break
                if unfilled and cur_high >= s_bottom and cur_close < s_top:
                    sell_signal = True

        if buy_signal:
            return 1
        if sell_signal:
            return -1
        return 0
