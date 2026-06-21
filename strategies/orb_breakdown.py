"""Opening Range Breakdown strategy for shadow scanning.

Rule: Wait 15 min (09:00-09:14), record High/Low. SELL when price breaks below Low.
Filter: Mon+Wed only (WR 60%, PF 3.25 on 33d backtest).
Exit: SL=3.0xATR, hold to session end.
"""
import numpy as np
import pandas as pd


class ORBBreakdownStrategy:
    """Opening Range Breakdown — SELL on break below 15-min opening range."""

    name = 'orb_breakdown'

    def __init__(self, or_bars: int = 15, weekday_filter: list[int] | None = None):
        self.or_bars = or_bars
        self.weekday_filter = weekday_filter or [0, 2]  # Mon=0, Wed=2
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._or_date = None
        self._triggered_today = False

    def _compute_opening_range(self, df: pd.DataFrame, idx: int) -> bool:
        """Compute opening range from first N bars of AM session."""
        row = df.iloc[idx]
        date = row['date']
        mins = row['mins']

        if self._or_date == date:
            return self._or_high is not None

        # Need enough bars into the session to have opening range
        if mins < 540 + self.or_bars:
            return False

        # Get today's first N bars
        am_today = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] < 540 + self.or_bars)]
        if len(am_today) < self.or_bars - 2:
            return False

        self._or_high = am_today['high'].max()
        self._or_low = am_today['low'].min()
        self._or_date = date
        self._triggered_today = False
        return True

    @property
    def trigger_price(self) -> float | None:
        return self._or_low

    def detect(self, df: pd.DataFrame, idx: int) -> int:
        """Return -1 for SELL signal, 0 for no signal. Never returns +1 (SELL only)."""
        if idx < 30:
            return 0

        row = df.iloc[idx]

        # Session filter: AM only
        if row.get('session', '') != 'AM':
            return 0

        # Weekday filter
        weekday = row['time'].weekday() if hasattr(row.get('time', None), 'weekday') else -1
        if weekday not in self.weekday_filter:
            return 0

        # Time window: only after opening range is set, before 11:25
        mins = row['mins']
        if mins < 540 + self.or_bars or mins > 685:
            return 0

        # Compute opening range if not done for today
        if not self._compute_opening_range(df, idx):
            return 0

        # Already triggered today
        if self._triggered_today:
            return 0

        # SELL trigger: close breaks below opening range low
        if row['close'] < self._or_low:
            self._triggered_today = True
            return -1

        return 0
