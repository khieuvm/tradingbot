"""Crabel Stretch strategy for shadow scanning.

Rule: Entry = AM open - SMA(5 daily ranges) × 50%. Stop = today's open.
Filter: Tue+Wed only (100% WR on 33d backtest, PF >> 1).
Exit: SL = today's open, hold to session end.
"""
import numpy as np
import pandas as pd


class CrabelStretchStrategy:
    """Crabel Stretch SELL — short when price drops below open minus stretch."""

    name = 'crabel_stretch'

    def __init__(self, stretch_sma: int = 5, stretch_pct: float = 0.5,
                 weekday_filter: list[int] | None = None):
        self.stretch_sma = stretch_sma
        self.stretch_pct = stretch_pct
        self.weekday_filter = weekday_filter or [1, 2]  # Tue=1, Wed=2
        self._daily_ranges: dict = {}
        self._today_open: float | None = None
        self._today_trigger: float | None = None
        self._today_date = None
        self._triggered_today = False

    def _setup_today(self, df: pd.DataFrame, idx: int) -> bool:
        """Compute today's open and stretch trigger from prior daily ranges."""
        row = df.iloc[idx]
        date = row['date']

        if self._today_date == date:
            return self._today_trigger is not None

        mins = row['mins']
        if mins < 545:
            return False

        all_dates = sorted(df['date'].unique())
        for d in all_dates:
            if d in self._daily_ranges:
                continue
            day_df = df[(df['date'] == d) & (df['mins'] >= 540) & (df['mins'] <= 690)]
            if len(day_df) >= 10:
                self._daily_ranges[d] = day_df['high'].max() - day_df['low'].min()

        prior_dates = [d for d in sorted(self._daily_ranges.keys()) if d < date]
        if len(prior_dates) < self.stretch_sma:
            return False

        recent = [self._daily_ranges[d] for d in prior_dates[-self.stretch_sma:]]
        stretch = np.mean(recent)

        am_today = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 545)]
        if len(am_today) == 0:
            am_today = df[(df['date'] == date) & (df['mins'] >= 540)]
            if len(am_today) == 0:
                return False

        self._today_open = am_today['open'].iloc[0]
        self._today_trigger = self._today_open - stretch * self.stretch_pct
        self._today_date = date
        self._triggered_today = False
        return True

    @property
    def sl_price(self) -> float | None:
        return self._today_open

    @property
    def trigger_price(self) -> float | None:
        return self._today_trigger

    def detect(self, df: pd.DataFrame, idx: int) -> int:
        """Return -1 for SELL signal, 0 for no signal."""
        if idx < 30:
            return 0

        row = df.iloc[idx]

        if row.get('session', '') != 'AM':
            return 0

        weekday = row['time'].weekday() if hasattr(row.get('time', None), 'weekday') else -1
        if weekday not in self.weekday_filter:
            return 0

        mins = row['mins']
        if mins < 545 or mins > 685:
            return 0

        if not self._setup_today(df, idx):
            return 0

        if self._triggered_today:
            return 0

        if row['low'] < self._today_trigger:
            self._triggered_today = True
            return -1

        return 0
