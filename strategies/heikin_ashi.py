"""Strategy 18: Heikin Ashi Candle Reversal.

Convert OHLC to Heikin Ashi and detect colour-reversal setups:
  BUY:  first green HA bar after 3+ consecutive red HA bars, body >= 50% of range
  SELL: first red HA bar after 3+ consecutive green HA bars, body >= 50% of range

HA formula:
  HA_close = (O + H + L + C) / 4
  HA_open  = (prev_HA_open + prev_HA_close) / 2   (first bar: (O + C) / 2)
  HA_high  = max(H, HA_open, HA_close)
  HA_low   = min(L, HA_open, HA_close)
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class HeikinAshiStrategy(BaseStrategy):
    name = "heikin_ashi"
    exit_type = "trend"

    def __init__(self, consec_bars=3, body_ratio=0.5, lookback=60):
        """
        Args:
            consec_bars: consecutive bars of opposite colour required before reversal
            body_ratio: minimum body-to-range ratio on the reversal bar
            lookback: bars of history to compute HA (more = better HA_open warmup)
        """
        self.consec_bars = consec_bars
        self.body_ratio = body_ratio
        self.lookback = lookback

    def _compute_ha(self, opens, highs, lows, closes):
        """Return HA open, high, low, close arrays."""
        n = len(opens)
        ha_close = (opens + highs + lows + closes) / 4.0
        ha_open = np.empty(n)
        ha_open[0] = (opens[0] + closes[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high = np.maximum(highs, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(lows, np.minimum(ha_open, ha_close))
        return ha_open, ha_high, ha_low, ha_close

    def detect(self, df, idx):
        # Need enough warmup bars for HA_open to stabilise
        min_bars = self.consec_bars + max(self.lookback // 4, 10)
        if idx < min_bars:
            return 0

        start = max(0, idx - self.lookback)
        window = df.iloc[start: idx + 1]

        ha_open, ha_high, ha_low, ha_close = self._compute_ha(
            window['open'].values,
            window['high'].values,
            window['low'].values,
            window['close'].values,
        )

        n = len(ha_open)
        cur = n - 1

        # Colour: True = green (bullish), False = red (bearish)
        colors = ha_close > ha_open

        cur_green = bool(colors[cur])
        prev_green = bool(colors[cur - 1])

        # Body and range of the current (reversal candidate) HA bar
        body = abs(ha_close[cur] - ha_open[cur])
        bar_range = ha_high[cur] - ha_low[cur]
        body_ok = (bar_range > 0) and (body / bar_range >= self.body_ratio)

        # BUY: current green, previous red, 3+ consecutive reds before
        if cur_green and not prev_green and body_ok:
            consec_reds = 0
            for k in range(cur - 1, -1, -1):
                if not colors[k]:
                    consec_reds += 1
                else:
                    break
            if consec_reds >= self.consec_bars:
                return 1

        # SELL: current red, previous green, 3+ consecutive greens before
        if not cur_green and prev_green and body_ok:
            # PM SELL only after 13:45 — early PM has no confirmed trend to reverse
            if 'session' in df.columns and 'mins' in df.columns:
                session = df['session'].iloc[idx]
                mins = df['mins'].iloc[idx]
                if session == 'PM' and mins < 825:
                    return 0
            consec_greens = 0
            for k in range(cur - 1, -1, -1):
                if colors[k]:
                    consec_greens += 1
                else:
                    break
            if consec_greens >= self.consec_bars:
                return -1

        return 0
