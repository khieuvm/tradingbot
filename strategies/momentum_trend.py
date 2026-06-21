"""Momentum/Trend strategy — ADX + DI divergence + EMA cross (primary) or MACD (fallback)."""
import numpy as np
from strategies.base import BaseStrategy


class MomentumTrendStrategy(BaseStrategy):
    """Combined trend-following signals using ADX, DI spread, EMA cross, and MACD.

    Primary signal (stricter):
      BUY:  ADX > 25, DI+ > DI-, EMA8 crossed above EMA21 on this or previous bar.
      SELL: ADX > 25, DI- > DI+, EMA8 crossed below EMA21 on this or previous bar.

    Fallback signal (no fresh cross required):
      BUY:  ADX > 30, DI spread > 10, MACD line > MACD signal.
      SELL: ADX > 30, DI spread < -10, MACD line < MACD signal.

    Primary takes priority; fallback fires only when no primary signal is present.
    """

    name = "momentum_trend"
    exit_type = "trend"
    _use_tight_trail_am_sell = True

    _ADX_MIN_PRIMARY = 25.0
    _ADX_MIN_FALLBACK = 30.0
    _DI_SPREAD_FALLBACK = 10.0

    def get_exit_params(self, session):
        from strategies.base import EXIT_PRESETS
        if session == 'AM' and self._use_tight_trail_am_sell:
            preset = EXIT_PRESETS['trend_tight']
        else:
            preset = EXIT_PRESETS[self.exit_type]
        max_hold = preset['max_hold_am'] if session == 'AM' else preset['max_hold_pm']
        return {
            'sl_mult': preset['sl_mult'],
            'trail_pts': preset['trail_pts'],
            'trail_mult': preset['trail_mult'],
            'tp_mult': preset['tp_mult'],
            'max_hold': max_hold,
        }

    def detect(self, df, idx):
        if idx < 3:
            return 0

        adx = df['adx'].iloc[idx]
        di_plus = df['di_plus'].iloc[idx]
        di_minus = df['di_minus'].iloc[idx]
        ema8 = df['ema8'].iloc[idx]
        ema21 = df['ema21'].iloc[idx]
        ema8_p1 = df['ema8'].iloc[idx - 1]
        ema21_p1 = df['ema21'].iloc[idx - 1]
        ema8_p2 = df['ema8'].iloc[idx - 2]
        ema21_p2 = df['ema21'].iloc[idx - 2]
        macd_line = df['macd_line'].iloc[idx]
        macd_sig = df['macd_signal'].iloc[idx]

        # Guard against NaN in core indicators
        core = [adx, di_plus, di_minus, ema8, ema21, ema8_p1, ema21_p1, ema8_p2, ema21_p2]
        if any(np.isnan(v) for v in core):
            return 0

        # EMA cross detection: cross on current bar or on the previous bar
        cross_up_now = ema8 > ema21 and ema8_p1 <= ema21_p1
        cross_up_prev = ema8_p1 > ema21_p1 and ema8_p2 <= ema21_p2
        cross_dn_now = ema8 < ema21 and ema8_p1 >= ema21_p1
        cross_dn_prev = ema8_p1 < ema21_p1 and ema8_p2 >= ema21_p2

        fresh_cross_up = cross_up_now or cross_up_prev
        fresh_cross_dn = cross_dn_now or cross_dn_prev

        mins = df['mins'].iloc[idx]
        session = df['session'].iloc[idx]

        # ── Primary signal ───────────────────────────────────────────────────
        if adx > self._ADX_MIN_PRIMARY:
            if di_plus > di_minus and fresh_cross_up:
                return 1
            if di_minus > di_plus and fresh_cross_dn:
                # PM SELL: only after 13:45 (trend has established itself)
                if session == 'PM' and mins < 825:
                    return 0
                return -1

        # ── Fallback signal (no cross requirement) ───────────────────────────
        if adx > self._ADX_MIN_FALLBACK and not np.isnan(macd_line) and not np.isnan(macd_sig):
            di_spread = di_plus - di_minus
            if di_spread > self._DI_SPREAD_FALLBACK and macd_line > macd_sig:
                return 1
            if di_spread < -self._DI_SPREAD_FALLBACK and macd_line < macd_sig:
                if session == 'PM' and mins < 825:
                    return 0
                return -1

        return 0
