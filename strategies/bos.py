"""Break of Structure (BOS) strategy — momentum breakout past last swing extreme."""
import numpy as np
from strategies.base import BaseStrategy
from strategies.utils.swing import detect_swing_points, build_zigzag


class BOSStrategy(BaseStrategy):
    """Signal when price breaks cleanly through the last swing high/low in trend direction.

    Trend determined from last two swing high/low pairs:
      HH + HL -> uptrend.   BOS-BUY: close > last swing high + 0.1
      LH + LL -> downtrend. BOS-SELL: close < last swing low  - 0.1

    Requires the break to be fresh (previous bar did not break the level).
    """

    name = "bos"
    exit_type = "breakout"

    _MIN_STRUCT_PTS = 0.5
    _BOS_BUFFER = 0.1

    def prepare(self, df):
        """Pre-compute swing points and zigzag once."""
        highs = df['high'].values
        lows = df['low'].values
        self._swing_highs, self._swing_lows = detect_swing_points(highs, lows, lookback=5)
        self._zigzag = build_zigzag(self._swing_highs, self._swing_lows)

    def detect(self, df, idx):
        if idx < 25:
            return 0

        # Get zigzag points up to current bar
        zz = [s for s in self._zigzag if s[1] <= idx]
        sh_list = [(s[1], s[2]) for s in zz if s[0] == 'H']
        sl_list = [(s[1], s[2]) for s in zz if s[0] == 'L']

        if len(sh_list) < 2 or len(sl_list) < 2:
            return 0

        last_sh = sh_list[-1]
        prev_sh = sh_list[-2]
        last_sl = sl_list[-1]
        prev_sl = sl_list[-2]

        min_d = self._MIN_STRUCT_PTS
        is_uptrend = (last_sh[1] > prev_sh[1] + min_d and
                      last_sl[1] > prev_sl[1] + min_d)
        is_downtrend = (last_sh[1] < prev_sh[1] - min_d and
                        last_sl[1] < prev_sl[1] - min_d)

        closes = df['close'].values
        cur_close = closes[idx]
        prev_close = closes[idx - 1]
        buf = self._BOS_BUFFER

        if is_uptrend:
            level = last_sh[1] + buf
            if cur_close > level and prev_close <= level:
                return 1

        if is_downtrend:
            level = last_sl[1] - buf
            if cur_close < level and prev_close >= level:
                return -1

        return 0
