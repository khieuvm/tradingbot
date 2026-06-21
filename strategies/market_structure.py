"""Market Structure strategy — trade with confirmed HH/HL or LH/LL trend structure."""
import numpy as np
from strategies.base import BaseStrategy
from strategies.utils.swing import detect_swing_points, get_recent_swings


class MarketStructureStrategy(BaseStrategy):
    """Trade in the direction of confirmed market structure.

    BUY:  uptrend (HH+HL) with a recent higher low confirmed within last 7 bars.
    SELL: downtrend (LH+LL) with a recent lower high confirmed within last 7 bars.
    """

    name = "market_structure"
    exit_type = "trend"

    _MAX_SWING_AGE = 7
    _MIN_STRUCT_PTS = 0.5

    def prepare(self, df):
        """Pre-compute swing points once for the entire dataset."""
        highs = df['high'].values
        lows = df['low'].values
        self._swing_highs, self._swing_lows = detect_swing_points(highs, lows, lookback=5)

    def detect(self, df, idx):
        if idx < 20:
            return 0

        sh = self._swing_highs[:idx + 1]
        sl = self._swing_lows[:idx + 1]

        sh_list = [(i, sh[i]) for i in range(len(sh)) if not np.isnan(sh[i])]
        sl_list = [(i, sl[i]) for i in range(len(sl)) if not np.isnan(sl[i])]

        if len(sh_list) < 2 or len(sl_list) < 2:
            return 0

        last_sh, prev_sh = sh_list[-1], sh_list[-2]
        last_sl, prev_sl = sl_list[-1], sl_list[-2]

        min_d = self._MIN_STRUCT_PTS
        is_hh = last_sh[1] > prev_sh[1] + min_d
        is_hl = last_sl[1] > prev_sl[1] + min_d
        is_lh = last_sh[1] < prev_sh[1] - min_d
        is_ll = last_sl[1] < prev_sl[1] - min_d

        closes = df['close'].values

        if is_hh and is_hl:
            if 'session' in df.columns and df['session'].iloc[idx] == 'PM':
                return 0
            sl_age = idx - last_sl[0]
            if sl_age <= self._MAX_SWING_AGE and closes[idx] > last_sl[1]:
                return 1

        elif is_lh and is_ll:
            sh_age = idx - last_sh[0]
            if sh_age <= self._MAX_SWING_AGE and closes[idx] < last_sh[1]:
                return -1

        return 0
