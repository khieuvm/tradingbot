"""Level clustering and Fibonacci computation utilities."""
import numpy as np


def cluster_levels(levels, tolerance=1.0):
    """Cluster nearby price levels within tolerance.

    Returns list of (level_price, touch_count) sorted by price.
    """
    if not levels:
        return []

    levels_sorted = sorted(levels)
    clusters = []
    current_cluster = [levels_sorted[0]]

    for lvl in levels_sorted[1:]:
        if lvl - current_cluster[-1] <= tolerance:
            current_cluster.append(lvl)
        else:
            clusters.append((np.mean(current_cluster), len(current_cluster)))
            current_cluster = [lvl]
    clusters.append((np.mean(current_cluster), len(current_cluster)))

    return clusters


def compute_fibonacci(swing_high, swing_low):
    """Compute Fibonacci retracement levels between swing high and low.

    Returns dict of level_name: price for retracements from high.
    """
    diff = swing_high - swing_low
    return {
        '0.0': swing_high,
        '0.236': swing_high - 0.236 * diff,
        '0.382': swing_high - 0.382 * diff,
        '0.5': swing_high - 0.5 * diff,
        '0.618': swing_high - 0.618 * diff,
        '0.786': swing_high - 0.786 * diff,
        '1.0': swing_low,
        '1.272': swing_low - 0.272 * diff,
        '1.618': swing_low - 0.618 * diff,
    }
