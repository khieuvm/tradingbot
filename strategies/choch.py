"""Change of Character (CHoCH) strategy — early reversal signal against established trend."""
import numpy as np
from strategies.base import BaseStrategy
from strategies.utils.swing import detect_swing_points, build_zigzag


class CHoCHStrategy(BaseStrategy):
    """Signal when an established trend is broken by a CHoCH (Change of Character).

    Requires a minimum of 2 legs in the prior trend to confirm it was established:
      Downtrend (2+ lower highs): CHoCH-BUY when close breaks above last lower high.
      Uptrend   (2+ higher lows): CHoCH-SELL when close breaks below last higher low.

    Fresh break required: previous bar must not have crossed the level.
    """

    name = "choch"
    exit_type = "reversal"

    _CHOCH_BUFFER = 0.1

    def prepare(self, df):
        """Pre-compute swing points and zigzag once."""
        highs = df['high'].values
        lows = df['low'].values
        self._swing_highs, self._swing_lows = detect_swing_points(highs, lows, lookback=5)
        self._zigzag = build_zigzag(self._swing_highs, self._swing_lows)

    def detect(self, df, idx):
        if idx < 30:
            return 0

        adx = df['adx'].iloc[idx]
        if np.isnan(adx) or adx < 15:
            return 0

        zz = [s for s in self._zigzag if s[1] <= idx]
        sh_list = [(s[1], s[2]) for s in zz if s[0] == 'H']
        sl_list = [(s[1], s[2]) for s in zz if s[0] == 'L']

        if len(sh_list) < 3 or len(sl_list) < 3:
            return 0

        closes = df['close'].values
        cur_close = closes[idx]
        prev_close = closes[idx - 1]
        buf = self._CHOCH_BUFFER

        # CHoCH-BUY: downtrend established (2+ lower highs)
        has_two_lower_highs = (sh_list[-3][1] > sh_list[-2][1] > sh_list[-1][1])
        if has_two_lower_highs:
            last_lh = sh_list[-1][1]
            level = last_lh + buf
            if cur_close > level and prev_close <= level:
                return 1

        # CHoCH-SELL: uptrend established (2+ higher lows)
        has_two_higher_lows = (sl_list[-3][1] < sl_list[-2][1] < sl_list[-1][1])
        if has_two_higher_lows:
            last_hl = sl_list[-1][1]
            level = last_hl - buf
            if cur_close < level and prev_close >= level:
                return -1

        return 0
