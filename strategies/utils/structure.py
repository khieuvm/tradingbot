"""Market structure classification - HH/HL/LH/LL detection."""
import numpy as np
from strategies.utils.swing import detect_swing_points, get_recent_swings


def classify_market_structure(highs, lows, swing_lookback=5, min_pts=0.5):
    """Classify market structure based on swing sequence.

    Returns:
        structure: 'uptrend', 'downtrend', or 'ranging'
        last_sh: (idx, val) of last swing high
        last_sl: (idx, val) of last swing low
        prev_sh: (idx, val) of previous swing high
        prev_sl: (idx, val) of previous swing low
    """
    swing_highs, swing_lows = detect_swing_points(highs, lows, lookback=swing_lookback)
    sh_list, sl_list = get_recent_swings(swing_highs, swing_lows, n_recent=3)

    if len(sh_list) < 2 or len(sl_list) < 2:
        return 'ranging', None, None, None, None

    prev_sh, last_sh = sh_list[-2], sh_list[-1]
    prev_sl, last_sl = sl_list[-2], sl_list[-1]

    is_hh = last_sh[1] > prev_sh[1] + min_pts
    is_hl = last_sl[1] > prev_sl[1] + min_pts
    is_lh = last_sh[1] < prev_sh[1] - min_pts
    is_ll = last_sl[1] < prev_sl[1] - min_pts

    if is_hh and is_hl:
        structure = 'uptrend'
    elif is_lh and is_ll:
        structure = 'downtrend'
    else:
        structure = 'ranging'

    return structure, last_sh, last_sl, prev_sh, prev_sl
