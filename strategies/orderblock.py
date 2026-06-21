"""Order Block (Supply & Demand) strategy — mean reversion to institutional zones."""
import numpy as np
from strategies.base import BaseStrategy


class OrderBlockStrategy(BaseStrategy):
    """Detect supply/demand order blocks created by impulse moves.

    Demand zone: last bearish candle before a bullish impulse → [low, open].
    Supply zone: last bullish candle before a bearish impulse → [close, high].
    Zone max age: 30 bars. Invalidated if price breaks through the zone completely.
    """

    name = "orderblock"
    exit_type = "mean_reversion"

    # Maximum bars to search backward for the preceding candle relative to impulse.
    _PRECEDE_LOOKBACK = 10
    # Impulse bar body threshold as multiple of ATR.
    _IMPULSE_MULT = 1.5
    # Zone age limit in bars.
    _MAX_AGE = 30

    def detect(self, df, idx):
        if idx < 20:
            return 0

        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        atrs = df['atr'].values

        demand_zones = []  # list of (bottom, top)
        supply_zones = []
        seen = set()  # deduplicate zones by rounded coordinates

        scan_start = max(20, idx - 35)

        for i in range(scan_start, idx):
            atr_i = atrs[i]
            if np.isnan(atr_i) or atr_i <= 0:
                continue
            body = abs(closes[i] - opens[i])
            if body < self._IMPULSE_MULT * atr_i:
                continue  # not an impulse bar

            is_bull_impulse = closes[i] > opens[i]
            is_bear_impulse = closes[i] < opens[i]

            # ── Demand zone (bullish impulse) ────────────────────────────────
            if is_bull_impulse:
                lb = max(0, i - self._PRECEDE_LOOKBACK)
                for j in range(i - 1, lb - 1, -1):
                    age = idx - j
                    if age > self._MAX_AGE:
                        break  # j only gets older as we go back
                    if closes[j] >= opens[j]:
                        continue  # need bearish candle
                    zb, zt = lows[j], opens[j]
                    if zt <= zb:
                        break
                    key = (round(zb, 1), round(zt, 1), 'D')
                    if key in seen:
                        break
                    # Invalidated if any bar's low broke below zone bottom
                    valid = all(lows[k] >= zb for k in range(j + 1, idx))
                    if valid:
                        demand_zones.append((zb, zt))
                        seen.add(key)
                    break  # only use nearest preceding candle

            # ── Supply zone (bearish impulse) ────────────────────────────────
            if is_bear_impulse:
                lb = max(0, i - self._PRECEDE_LOOKBACK)
                for j in range(i - 1, lb - 1, -1):
                    age = idx - j
                    if age > self._MAX_AGE:
                        break
                    if closes[j] <= opens[j]:
                        continue  # need bullish candle
                    zb, zt = closes[j], highs[j]
                    if zt <= zb:
                        break
                    key = (round(zb, 1), round(zt, 1), 'S')
                    if key in seen:
                        break
                    # Invalidated if any bar's high broke above zone top
                    valid = all(highs[k] <= zt for k in range(j + 1, idx))
                    if valid:
                        supply_zones.append((zb, zt))
                        seen.add(key)
                    break

        cur_low = lows[idx]
        cur_high = highs[idx]
        cur_close = closes[idx]

        # BUY: price returns to demand zone — low touches zone top, close above bottom
        for zb, zt in demand_zones:
            if cur_low <= zt and cur_close > zb:
                return 1

        # SELL: price returns to supply zone — high touches zone bottom, close below top
        for zb, zt in supply_zones:
            if cur_high >= zb and cur_close < zt:
                return -1

        return 0
