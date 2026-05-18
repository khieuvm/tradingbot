"""
Strategy module - Định nghĩa và quản lý chiến lược trading
"""
from abc import ABC, abstractmethod
import pandas as pd
import pandas_ta as ta


class BaseStrategy(ABC):
    """Base class cho tất cả strategies"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo tín hiệu mua/bán từ dữ liệu OHLCV

        Returns:
            DataFrame với cột 'signal': 1 = BUY, -1 = SELL, 0 = HOLD
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Trả về tham số hiện tại của strategy"""
        pass


class MACrossStrategy(BaseStrategy):
    """Chiến lược giao cắt đường trung bình (Moving Average Crossover)

    - BUY: MA ngắn cắt lên MA dài
    - SELL: MA ngắn cắt xuống MA dài
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        super().__init__(name=f"MA_Cross_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ma_fast"] = ta.sma(df["close"], length=self.fast_period)
        df["ma_slow"] = ta.sma(df["close"], length=self.slow_period)

        df["signal"] = 0
        # MA fast cắt lên MA slow -> BUY
        df.loc[
            (df["ma_fast"] > df["ma_slow"]) & (df["ma_fast"].shift(1) <= df["ma_slow"].shift(1)),
            "signal",
        ] = 1
        # MA fast cắt xuống MA slow -> SELL
        df.loc[
            (df["ma_fast"] < df["ma_slow"]) & (df["ma_fast"].shift(1) >= df["ma_slow"].shift(1)),
            "signal",
        ] = -1

        return df

    def get_params(self) -> dict:
        return {"fast_period": self.fast_period, "slow_period": self.slow_period}


class RSIStrategy(BaseStrategy):
    """Chiến lược RSI

    - BUY: RSI < oversold (mặc định 30)
    - SELL: RSI > overbought (mặc định 70)
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(name=f"RSI_{period}_{oversold}_{overbought}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rsi"] = ta.rsi(df["close"], length=self.period)

        df["signal"] = 0
        # RSI vượt lên từ vùng oversold -> BUY
        df.loc[
            (df["rsi"] > self.oversold) & (df["rsi"].shift(1) <= self.oversold),
            "signal",
        ] = 1
        # RSI vượt xuống từ vùng overbought -> SELL
        df.loc[
            (df["rsi"] < self.overbought) & (df["rsi"].shift(1) >= self.overbought),
            "signal",
        ] = -1

        return df

    def get_params(self) -> dict:
        return {"period": self.period, "oversold": self.oversold, "overbought": self.overbought}


class MACDStrategy(BaseStrategy):
    """Chiến lược MACD

    - BUY: MACD cắt lên Signal line
    - SELL: MACD cắt xuống Signal line
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(name=f"MACD_{fast}_{slow}_{signal}")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        macd = ta.macd(df["close"], fast=self.fast, slow=self.slow, signal=self.signal_period)
        df["macd"] = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 2]
        df["macd_hist"] = macd.iloc[:, 1]

        df["signal"] = 0
        # MACD cắt lên Signal -> BUY
        df.loc[
            (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1)),
            "signal",
        ] = 1
        # MACD cắt xuống Signal -> SELL
        df.loc[
            (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1)),
            "signal",
        ] = -1

        return df

    def get_params(self) -> dict:
        return {"fast": self.fast, "slow": self.slow, "signal": self.signal_period}
