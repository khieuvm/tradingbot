"""Strategy 19: Moon Phases (Calendar / Statistical).

Hardcoded new moon and full moon dates for 2025-2026.

  New moon (±1 day window): BUY signal at first bar of AM session
  Full moon (±1 day window): SELL signal at first bar of AM session

Only fires once per moon event even when the ±1 day window spans multiple
trading days.  A triggered-set keyed on the exact moon date prevents
repeated signals for the same event.

Expected frequency: ~2-4 signals per month.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from strategies.base import BaseStrategy


# ---------------------------------------------------------------------------
# Precomputed moon dates 2025-2026
# ---------------------------------------------------------------------------
NEW_MOONS = [
    date(2025, 1, 29), date(2025, 2, 28), date(2025, 3, 29), date(2025, 4, 27),
    date(2025, 5, 27), date(2025, 6, 25), date(2025, 7, 24), date(2025, 8, 23),
    date(2025, 9, 21), date(2025, 10, 21), date(2025, 11, 20), date(2025, 12, 20),
    date(2026, 1, 18), date(2026, 2, 17), date(2026, 3, 18), date(2026, 4, 17),
    date(2026, 5, 16), date(2026, 6, 15), date(2026, 7, 14), date(2026, 8, 12),
    date(2026, 9, 11), date(2026, 10, 10), date(2026, 11, 9), date(2026, 12, 8),
]

FULL_MOONS = [
    date(2025, 1, 13), date(2025, 2, 12), date(2025, 3, 14), date(2025, 4, 12),
    date(2025, 5, 12), date(2025, 6, 11), date(2025, 7, 10), date(2025, 8, 9),
    date(2025, 9, 7), date(2025, 10, 7), date(2025, 11, 5), date(2025, 12, 5),
    date(2026, 1, 3), date(2026, 2, 2), date(2026, 3, 3), date(2026, 4, 2),
    date(2026, 5, 1), date(2026, 5, 31), date(2026, 6, 29), date(2026, 7, 29),
    date(2026, 8, 28), date(2026, 9, 26), date(2026, 10, 26), date(2026, 11, 24),
    date(2026, 12, 24),
]


class MoonPhasesStrategy(BaseStrategy):
    name = "moon_phases"
    exit_type = "trend"

    def __init__(self, window_days=1):
        """
        Args:
            window_days: days on each side of exact moon date that are valid
        """
        self.window_days = window_days
        # Track which moon events have already fired (keyed on exact moon date)
        self._triggered_new: set = set()
        self._triggered_full: set = set()

    def detect(self, df, idx):
        row = df.iloc[idx]
        session = str(row.get('session', ''))
        mins = int(row['mins'])

        # Only fire in AM session
        if session != 'AM':
            return 0

        # Only at first two bars of the valid AM window (9:15–9:20)
        # run_backtest already filters AM to [555, 645]; we narrow to the opening bars
        if not (555 <= mins <= 560):
            return 0

        bar_date = row['date']
        if isinstance(bar_date, str):
            bar_date = date.fromisoformat(str(bar_date)[:10])

        # Check new moons first
        for nm in NEW_MOONS:
            if abs((bar_date - nm).days) <= self.window_days:
                if nm not in self._triggered_new:
                    self._triggered_new.add(nm)
                    return 1   # BUY on new moon

        # Then full moons
        for fm in FULL_MOONS:
            if abs((bar_date - fm).days) <= self.window_days:
                if fm not in self._triggered_full:
                    self._triggered_full.add(fm)
                    return -1  # SELL on full moon

        return 0
