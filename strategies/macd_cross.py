"""MACD Crossover strategy — enhanced with trend + momentum filters.

From PDF analysis + VN30F1M backtest optimization:
Best config: MACD cross + EMA50 trend + ADX>20 + DI confirmation
"""
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class MACDCrossStrategy(BaseStrategy):
    """MACD signal line crossover with multi-filter confirmation.

    BUY:  MACD crosses above signal + price>EMA50 + ADX>20 + DI+>DI-
    SELL: MACD crosses below signal + price<EMA50 + ADX>20 + DI->DI+
    """

    name = "macd_cross"
    exit_type = "trend"

    def detect(self, df, idx):
        if idx < 30:
            return 0

        macd = df['macd_line'].iloc[idx]
        macd_prev = df['macd_line'].iloc[idx - 1]
        signal = df['macd_signal'].iloc[idx]
        signal_prev = df['macd_signal'].iloc[idx - 1]

        if pd.isna(macd) or pd.isna(signal) or pd.isna(macd_prev) or pd.isna(signal_prev):
            return 0

        cross_up = macd > signal and macd_prev <= signal_prev
        cross_down = macd < signal and macd_prev >= signal_prev

        if not cross_up and not cross_down:
            return 0

        ema50 = df['ema50'].iloc[idx]
        adx = df['adx'].iloc[idx]
        di_plus = df['di_plus'].iloc[idx]
        di_minus = df['di_minus'].iloc[idx]
        close = df['close'].iloc[idx]

        if pd.isna(ema50) or pd.isna(adx) or pd.isna(di_plus) or pd.isna(di_minus):
            return 0

        if adx < 20:
            return 0

        if cross_up and close > ema50 and di_plus > di_minus:
            return 1

        if cross_down and close < ema50 and di_minus > di_plus:
            return -1

        return 0
