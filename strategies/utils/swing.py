"""Swing point detection utility - used by 12+ strategies."""
import numpy as np


def detect_swing_points(highs, lows, lookback=5):
    """Detect swing highs and lows using N-bar lookback.

    Args:
        highs: array of high prices
        lows: array of low prices
        lookback: N bars on each side to confirm swing

    Returns:
        swing_highs: array with swing high values (NaN elsewhere)
        swing_lows: array with swing low values (NaN elsewhere)
    """
    n = len(highs)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback:i + lookback + 1]
        if highs[i] == np.max(window_h):
            swing_highs[i] = highs[i]

        window_l = lows[i - lookback:i + lookback + 1]
        if lows[i] == np.min(window_l):
            swing_lows[i] = lows[i]

    return swing_highs, swing_lows


def get_recent_swings(swing_highs, swing_lows, n_recent=5):
    """Get most recent N swing highs and lows as lists of (index, value)."""
    sh_list = [(i, swing_highs[i]) for i in range(len(swing_highs))
               if not np.isnan(swing_highs[i])]
    sl_list = [(i, swing_lows[i]) for i in range(len(swing_lows))
               if not np.isnan(swing_lows[i])]
    return sh_list[-n_recent:], sl_list[-n_recent:]


def build_zigzag(swing_highs, swing_lows):
    """Build alternating zigzag from swing points."""
    swings = []
    for i in range(len(swing_highs)):
        if not np.isnan(swing_highs[i]):
            swings.append(('H', i, swing_highs[i]))
        if not np.isnan(swing_lows[i]):
            swings.append(('L', i, swing_lows[i]))

    swings.sort(key=lambda x: x[1])

    if not swings:
        return []

    cleaned = [swings[0]]
    for s in swings[1:]:
        if s[0] == cleaned[-1][0]:
            if s[0] == 'H' and s[2] > cleaned[-1][2]:
                cleaned[-1] = s
            elif s[0] == 'L' and s[2] < cleaned[-1][2]:
                cleaned[-1] = s
        else:
            cleaned.append(s)

    return cleaned
