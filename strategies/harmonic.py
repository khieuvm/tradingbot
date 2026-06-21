"""Harmonic Patterns Strategy: Gartley XABCD.

Gartley ratios (per spec):
  AB/XA   ≈ 0.618 (±0.05)
  BC/AB   ∈ [0.382, 0.886]
  CD/XA   ≈ 0.786 (±0.05)

Bullish Gartley: X=Low, A=High, B=Low, C=High, D=Low → BUY at D completion.
Bearish Gartley: X=High, A=Low, B=High, C=Low, D=High → SELL at D completion.

D is not yet a confirmed swing — we detect when current price is near the expected
D level computed from the prior four zigzag points (X, A, B, C).
Signals are rare by design.
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy
from strategies.utils.swing import detect_swing_points, get_recent_swings, build_zigzag


class HarmonicStrategy(BaseStrategy):
    name = "harmonic"
    exit_type = "reversal"

    def __init__(self, swing_lookback=5, window_bars=100,
                 ab_xa_target=0.618, ab_xa_tol=0.05,
                 bc_ab_lo=0.382, bc_ab_hi=0.886,
                 cd_xa_target=0.786, cd_xa_tol=0.05,
                 d_price_tol=1.0, min_xa=3.0):
        self.swing_lookback = swing_lookback
        self.window_bars = window_bars
        self.ab_xa_target = ab_xa_target
        self.ab_xa_tol = ab_xa_tol
        self.bc_ab_lo = bc_ab_lo
        self.bc_ab_hi = bc_ab_hi
        self.cd_xa_target = cd_xa_target
        self.cd_xa_tol = cd_xa_tol
        self.d_price_tol = d_price_tol
        self.min_xa = min_xa  # minimum XA leg size in points

    def _check_ratios(self, xa, ab, bc):
        """Verify AB/XA and BC/AB ratios. xa, ab, bc are absolute distances."""
        if xa < self.min_xa or ab <= 0 or bc <= 0:
            return False
        ab_xa = ab / xa
        if not (self.ab_xa_target - self.ab_xa_tol <= ab_xa <= self.ab_xa_target + self.ab_xa_tol):
            return False
        bc_ab = bc / ab
        if not (self.bc_ab_lo <= bc_ab <= self.bc_ab_hi):
            return False
        return True

    def detect(self, df, idx):
        lb = self.swing_lookback
        start = max(0, idx - self.window_bars)
        window = df.iloc[start:idx + 1]

        if len(window) < 2 * lb + 1:
            return 0

        highs = window['high'].values
        lows = window['low'].values

        sh, sl = detect_swing_points(highs, lows, lookback=lb)
        zz = build_zigzag(sh, sl)

        # Need at least 4 confirmed zigzag points to form X, A, B, C
        if len(zz) < 4:
            return 0

        # Use last 4 zigzag points as X, A, B, C
        X_type, X_idx, X_val = zz[-4]
        A_type, A_idx, A_val = zz[-3]
        B_type, B_idx, B_val = zz[-2]
        C_type, C_idx, C_val = zz[-1]

        xa = abs(A_val - X_val)
        ab = abs(B_val - A_val)
        bc = abs(C_val - B_val)

        if not self._check_ratios(xa, ab, bc):
            return 0

        curr_close = df['close'].iloc[idx]

        # --- Bullish Gartley: X=L, A=H, B=L, C=H → D is a new Low ---
        # D_expected = C - 0.786 * XA  (C is a high, D completes below)
        if X_type == 'L' and A_type == 'H' and B_type == 'L' and C_type == 'H':
            D_expected = C_val - self.cd_xa_target * xa
            cd = C_val - curr_close  # curr_close should be below C (it's the D Low)
            if cd <= 0:
                return 0
            cd_xa = cd / xa
            ratio_ok = abs(cd_xa - self.cd_xa_target) <= self.cd_xa_tol
            price_near_d = abs(curr_close - D_expected) <= self.d_price_tol
            if ratio_ok and price_near_d:
                return 1

        # --- Bearish Gartley: X=H, A=L, B=H, C=L → D is a new High ---
        # D_expected = C + 0.786 * XA  (C is a low, D completes above)
        if X_type == 'H' and A_type == 'L' and B_type == 'H' and C_type == 'L':
            D_expected = C_val + self.cd_xa_target * xa
            cd = curr_close - C_val  # curr_close should be above C (it's the D High)
            if cd <= 0:
                return 0
            cd_xa = cd / xa
            ratio_ok = abs(cd_xa - self.cd_xa_target) <= self.cd_xa_tol
            price_near_d = abs(curr_close - D_expected) <= self.d_price_tol
            if ratio_ok and price_near_d:
                return -1

        return 0
