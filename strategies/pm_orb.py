"""PM Opening Range Breakout strategy for shadow scanning.

Rule: Wait 15 min (13:00-13:14), record High/Low.
  BUY when price breaks above High.
  SELL when price breaks below Low.
Filter: Weekday-dependent direction (from PM opening analysis, 33d backtest).
  Mon PM: SELL (PF 6.52, WR 75%, N=4)
  Tue PM: BUY (PF 1.84, WR 40%, N=5)
  Wed PM: BUY (PF 2.30, WR 50%, N=6)
  Fri PM: SELL (PF 4.30, WR 33%, N=3)
Exit: SL=3.0xATR, hold to session end (14:25).
"""
import pandas as pd


class PMORBStrategy:
    """PM Opening Range Breakout — trade on break of 13:00-13:14 range."""

    name = 'pm_orb'

    def __init__(self, or_bars: int = 15, direction: int = 0,
                 weekday_filter: list[int] | None = None):
        """
        Args:
            or_bars: Number of 1m bars for opening range (13:00-13:14 = 15 bars).
            direction: 1=BUY only, -1=SELL only, 0=both.
            weekday_filter: Which weekdays to trade (0=Mon..4=Fri).
        """
        self.or_bars = or_bars
        self.direction = direction
        self.weekday_filter = weekday_filter
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._or_date = None
        self._triggered_today = False
        self._trigger_price: float | None = None

    def _compute_opening_range(self, df: pd.DataFrame, idx: int) -> bool:
        """Compute PM opening range from first N bars (13:00-13:14)."""
        row = df.iloc[idx]
        date = row['date']
        mins = row['mins']

        if self._or_date == date:
            return self._or_high is not None

        # PM starts at 780 (13:00). Need at least or_bars into session.
        if mins < 780 + self.or_bars:
            return False

        pm_open = df[(df['date'] == date) & (df['mins'] >= 780) & (df['mins'] < 780 + self.or_bars)]
        if len(pm_open) < self.or_bars - 2:
            return False

        self._or_high = pm_open['high'].max()
        self._or_low = pm_open['low'].min()
        self._or_date = date
        self._triggered_today = False
        self._trigger_price = None
        return True

    @property
    def trigger_price(self) -> float | None:
        return self._trigger_price

    def detect(self, df: pd.DataFrame, idx: int) -> int:
        """Return 1 for BUY, -1 for SELL, 0 for no signal."""
        if idx < 30:
            return 0

        row = df.iloc[idx]

        # Session filter: PM only
        if row.get('session', '') != 'PM':
            return 0

        # Weekday filter
        if self.weekday_filter is not None:
            weekday = row['time'].weekday() if hasattr(row.get('time', None), 'weekday') else -1
            if weekday not in self.weekday_filter:
                return 0

        mins = row['mins']
        # After opening range is set, before 14:15 (855)
        if mins < 780 + self.or_bars or mins > 855:
            return 0

        if not self._compute_opening_range(df, idx):
            return 0

        if self._triggered_today:
            return 0

        # BUY trigger: close breaks above opening range high
        if self.direction in (0, 1) and row['close'] > self._or_high:
            self._triggered_today = True
            self._trigger_price = float(self._or_high)
            return 1

        # SELL trigger: close breaks below opening range low
        if self.direction in (0, -1) and row['close'] < self._or_low:
            self._triggered_today = True
            self._trigger_price = float(self._or_low)
            return -1

        return 0
