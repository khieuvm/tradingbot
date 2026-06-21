"""Strategy 17: Volume Indicators.

Primary signal — Volume spike:
  BUY:  volume > 2x rolling mean AND close > open (bullish bar) AND range > 0.5*ATR
  SELL: volume > 2x rolling mean AND close < open (bearish bar) AND range > 0.5*ATR

Secondary signal — OBV divergence:
  BUY:  current price near 10-bar close low, but OBV higher than at that low
  SELL: current price near 10-bar close high, but OBV lower than at that high

Volume spike takes priority; OBV divergence is checked only when no spike is found.
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class VolumeStrategy(BaseStrategy):
    name = "volume"
    exit_type = "trend_tight"

    def __init__(self, vol_mult=2.0, vol_window=20, obv_window=10, obv_proximity_mult=0.5):
        self.vol_mult = vol_mult
        self.vol_window = vol_window
        self.obv_window = obv_window
        self.obv_proximity_mult = obv_proximity_mult

    def get_exit_params(self, session):
        from strategies.base import EXIT_PRESETS
        preset = EXIT_PRESETS['trend_tight']
        max_hold = preset['max_hold_am'] if session == 'AM' else preset['max_hold_pm']
        return {
            'sl_mult': preset['sl_mult'],
            'trail_pts': preset['trail_pts'],
            'trail_mult': preset['trail_mult'],
            'tp_mult': preset['tp_mult'],
            'max_hold': max_hold,
        }

    def detect(self, df, idx):
        required = max(self.vol_window, self.obv_window) + 1
        if idx < required:
            return 0

        close = df['close'].iloc[idx]
        open_ = df['open'].iloc[idx]
        high = df['high'].iloc[idx]
        low = df['low'].iloc[idx]
        volume = df['volume'].iloc[idx]
        atr = df['atr'].iloc[idx]

        if pd.isna(atr) or atr <= 0:
            return 0

        # PM SELL requires ADX >= 25 for directional confirmation
        adx = df['adx'].iloc[idx]
        session = df['session'].iloc[idx] if 'session' in df.columns else None
        mins = df['mins'].iloc[idx] if 'mins' in df.columns else 0

        bar_range = high - low

        # --- Primary: volume spike ---
        vol_slice = df['volume'].iloc[idx - self.vol_window: idx]
        vol_mean = vol_slice.mean()

        if vol_mean > 0 and volume > self.vol_mult * vol_mean and bar_range > 0.5 * atr:
            if close > open_:
                return 1
            elif close < open_:
                # PM SELL: require ADX >= 25 (strong directional move)
                if session == 'PM' and (pd.isna(adx) or adx < 25):
                    return 0
                return -1

        # --- Secondary: OBV divergence ---
        w = df.iloc[idx - self.obv_window: idx + 1]
        closes_w = w['close'].values
        obv_w = w['obv'].values

        hist_closes = closes_w[:-1]
        hist_obv = obv_w[:-1]
        cur_obv = obv_w[-1]
        prox = self.obv_proximity_mult * atr

        # BUY: price near 10-bar close low but OBV higher than at that low
        low_idx = int(np.argmin(hist_closes))
        if abs(close - hist_closes[low_idx]) < prox:
            if cur_obv > hist_obv[low_idx]:
                return 1

        # SELL: price near 10-bar close high but OBV lower than at that high
        high_idx = int(np.argmax(hist_closes))
        if abs(close - hist_closes[high_idx]) < prox:
            if cur_obv < hist_obv[high_idx]:
                if session == 'PM' and (pd.isna(adx) or adx < 25):
                    return 0
                return -1

        return 0
