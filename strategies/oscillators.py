"""Oscillators strategy — combined RSI + Stochastic + CCI mean-reversion signals."""
import numpy as np
from strategies.base import BaseStrategy


class OscillatorsStrategy(BaseStrategy):
    """Multi-oscillator mean-reversion: fires on extreme oversold/overbought confluence.

    Primary (strict, requires all three oscillators):
      BUY:  RSI < 30, stoch_k < 20, CCI < -100, AND RSI turning up.
      SELL: RSI > 70, stoch_k > 80, CCI > 100, AND RSI turning down.

    Fallback A — single extreme RSI (less strict):
      BUY:  RSI < 25 with RSI turning up.
      SELL: RSI > 75 with RSI turning down.

    Fallback B — stochastic cross at extreme (less strict):
      BUY:  stoch_k crosses above stoch_d while both are below 20.
      SELL: stoch_k crosses below stoch_d while both are above 80.

    Primary takes priority, then fallback A, then fallback B.
    """

    name = "oscillators"
    exit_type = "mean_reversion"

    def detect(self, df, idx):
        if idx < 2:
            return 0

        rsi = df['rsi'].iloc[idx]
        rsi_prev = df['rsi'].iloc[idx - 1]
        stoch_k = df['stoch_k'].iloc[idx]
        stoch_d = df['stoch_d'].iloc[idx]
        stoch_k_prev = df['stoch_k'].iloc[idx - 1]
        stoch_d_prev = df['stoch_d'].iloc[idx - 1]
        cci = df['cci'].iloc[idx]

        # Guard NaN — all indicators must be valid
        indicators = [rsi, rsi_prev, stoch_k, stoch_d, stoch_k_prev, stoch_d_prev, cci]
        if any(np.isnan(v) for v in indicators):
            return 0

        rsi_turning_up = rsi > rsi_prev
        rsi_turning_dn = rsi < rsi_prev

        # ── Primary: all three oscillators at extreme ────────────────────────
        if rsi < 30 and stoch_k < 20 and cci < -100 and rsi_turning_up:
            return 1
        if rsi > 70 and stoch_k > 80 and cci > 100 and rsi_turning_dn:
            return -1

        # ── Fallback A: single extreme RSI ───────────────────────────────────
        if rsi < 25 and rsi_turning_up:
            return 1
        if rsi > 75 and rsi_turning_dn:
            return -1

        # ── Fallback B: stochastic cross at extreme ───────────────────────────
        # BUY: stoch_k crosses above stoch_d while stoch_k < 20
        if stoch_k < 20 and stoch_k > stoch_d and stoch_k_prev <= stoch_d_prev:
            return 1
        # SELL: stoch_k crosses below stoch_d while stoch_k > 80
        if stoch_k > 80 and stoch_k < stoch_d and stoch_k_prev >= stoch_d_prev:
            return -1

        return 0
