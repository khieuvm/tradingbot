"""
Data fetcher module - Lấy dữ liệu từ DNSE API và vnstock
Hỗ trợ: Cổ phiếu, Phái sinh (Futures/VN30F), Index, ETF
Providers: KBS (default, fast), VCI (longer history ~3 years)
"""
import pandas as pd
from vnstock import Market, Quote, Reference
from dnse.client import DnseClient
from dnse.models import BoardId

from config import Config


class DataFetcher:
    """Lấy dữ liệu lịch sử và realtime từ DNSE/vnstock"""

    # Số nến tối đa mỗi ngày giao dịch theo interval
    _CANDLES_PER_DAY = {
        "1m": 270, "3m": 90, "5m": 54, "15m": 18, "30m": 9,
        "1H": 5, "1D": 1, "1W": 0.2, "1M": 0.05,
    }

    # KBS: ~8 months max. VCI: ~3 years max.
    _MAX_COUNT = {"KBS": 15000, "VCI": 500000}

    def __init__(self, provider: str = "KBS"):
        """
        Args:
            provider: "KBS" (default, fast, ~8 months) or "VCI" (slower, ~3 years history)
        """
        self._provider = provider.upper()
        self.market = Market()
        self.reference = Reference()

    @property
    def provider(self) -> str:
        return self._provider

    @staticmethod
    def _calc_count(start: str, end: str, interval: str, provider: str = "KBS") -> int:
        """Tính số lượng nến cần fetch dựa vào khoảng thời gian và interval."""
        from datetime import datetime
        days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days + 1
        cpd = DataFetcher._CANDLES_PER_DAY.get(interval, 1)
        count = int(days * 0.72 * cpd) + 50
        max_count = DataFetcher._MAX_COUNT.get(provider, 15000)
        return min(max(count, 100), max_count)

    def get_historical_ohlcv(self, symbol: str, start: str, end: str, interval: str = "1D") -> pd.DataFrame:
        """Lấy dữ liệu OHLCV lịch sử từ vnstock

        Args:
            symbol: Mã cổ phiếu (e.g., "HPG", "VNM")
            start: Ngày bắt đầu "YYYY-MM-DD"
            end: Ngày kết thúc "YYYY-MM-DD"
            interval: Khung thời gian ("1D", "1W", "1M")
        """
        count = self._calc_count(start, end, interval, self._provider)
        df = self.market.equity(symbol).ohlcv(
            start=start, end=end, interval=interval, count=count,
            source=self._provider.lower(),
        )
        if self._provider == "VCI" and df is not None and not df.empty:
            df = self._filter_trading_hours(df)
        return df

    # === PHÁI SINH (FUTURES) ===

    def get_futures_list(self) -> pd.DataFrame:
        """Lấy danh sách các hợp đồng phái sinh đang giao dịch"""
        return self.reference.futures.list()

    def get_futures_ohlcv(self, symbol: str, start: str, end: str, interval: str = "1D") -> pd.DataFrame:
        """Lấy dữ liệu OHLCV phái sinh (VN30F1M, VN30F2M, VN30F1Q, VN30F2Q)

        Args:
            symbol: Mã hợp đồng phái sinh (e.g., "VN30F1M")
            start: Ngày bắt đầu "YYYY-MM-DD"
            end: Ngày kết thúc "YYYY-MM-DD"
            interval: Khung thời gian ("1m","3m","5m","15m","1D",...).
                      Nếu interval không được hỗ trợ bởi API (ví dụ "3m"),
                      sẽ tự động fetch "1m" rồi resample.
        """
        _RESAMPLE_FROM_1M = {"3m", "2m", "4m", "10m"}
        if interval in _RESAMPLE_FROM_1M:
            tf_min = int(interval.rstrip("m"))
            df1m = self.get_futures_ohlcv(symbol, start, end, interval="1m")
            return self._resample_ohlcv(df1m, tf_min)
        count = self._calc_count(start, end, interval, self._provider)
        df = self.market.futures(symbol).ohlcv(
            start=start, end=end, interval=interval, count=count,
            source=self._provider.lower(),
        )
        if self._provider == "VCI" and df is not None and not df.empty and interval != "1D":
            df = self._filter_trading_hours(df)
        return df

    @staticmethod
    def _resample_ohlcv(df1m: pd.DataFrame, tf_min: int) -> pd.DataFrame:
        """Resample 1m OHLCV to N-minute bars, keeping only VN session hours."""
        df = df1m.copy()
        df["time"] = pd.to_datetime(df["time"])
        resampled = (
            df.set_index("time")
            .resample(f"{tf_min}min", label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna(subset=["open"])
            .reset_index()
        )
        # Keep only VN trading session bars (AM: 09:00-11:30, PM: 13:00-14:30)
        h = resampled["time"].dt.hour
        m = resampled["time"].dt.minute
        mins = h * 60 + m
        in_session = ((mins >= 9 * 60) & (mins < 11 * 60 + 30)) | \
                     ((mins >= 13 * 60) & (mins < 14 * 60 + 30))
        return resampled[in_session].reset_index(drop=True)

    @staticmethod
    def _filter_trading_hours(df: pd.DataFrame) -> pd.DataFrame:
        """Filter VCI data to VN trading hours only (VCI returns 24h data)."""
        if df is None or df.empty:
            return df
        df = df.copy()
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            mins = df["time"].dt.hour * 60 + df["time"].dt.minute
        else:
            return df
        in_session = ((mins >= 9 * 60) & (mins < 11 * 60 + 30)) | \
                     ((mins >= 13 * 60) & (mins < 14 * 60 + 30))
        return df[in_session].reset_index(drop=True)

    def get_futures_info(self, symbol: str = "VN30F1M") -> dict:
        """Lấy thông tin chi tiết hợp đồng phái sinh"""
        return self.reference.futures.info()

    # === INDEX ===

    def get_index_ohlcv(self, symbol: str, start: str, end: str, interval: str = "1D") -> pd.DataFrame:
        """Lấy dữ liệu OHLCV chỉ số (VNINDEX, VN30, HNX30, ...)

        Args:
            symbol: Mã chỉ số (e.g., "VNINDEX", "VN30", "HNX30")
        """
        count = self._calc_count(start, end, interval, self._provider)
        df = self.market.index(symbol).ohlcv(
            start=start, end=end, interval=interval, count=count,
            source=self._provider.lower(),
        )
        return df

    def get_index_list(self) -> pd.DataFrame:
        """Lấy danh sách các chỉ số thị trường"""
        return self.reference.index.list()

    def get_intraday(self, symbol: str) -> pd.DataFrame:
        """Lấy dữ liệu intraday (tick-by-tick)"""
        quote = Quote(symbol=symbol, source=self._provider)
        return quote.intraday(symbol=symbol, page_size=10_000, show_log=False)

    def get_price_board(self, symbols: list[str]) -> pd.DataFrame:
        """Lấy bảng giá realtime"""
        from vnstock import Trading
        return Trading(source="KBS").price_board(symbols)

    def get_security_info(self, symbol: str):
        """Lấy thông tin chứng khoán (giá trần, sàn, tham chiếu) từ DNSE"""
        with DnseClient(api_key=Config.DNSE_API_KEY, api_secret=Config.DNSE_API_SECRET) as client:
            secs = client.market.security_info(symbol, board_id=BoardId.ROUND_LOT)
            return secs[0] if secs else None

    def get_multiple_stocks_history(self, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        """Lấy dữ liệu lịch sử cho nhiều mã cùng lúc"""
        result = {}
        for symbol in symbols:
            try:
                df = self.get_historical_ohlcv(symbol, start, end)
                result[symbol] = df
            except Exception as e:
                print(f"[WARNING] Cannot fetch {symbol}: {e}")
        return result


class RealtimeStream:
    """WebSocket streaming realtime data từ DNSE"""

    def __init__(self, on_trade_callback=None, on_quote_callback=None):
        from dnse.stream import DnseMarketStream
        self.stream = DnseMarketStream(
            api_key=Config.DNSE_API_KEY,
            api_secret=Config.DNSE_API_SECRET,
        )
        self._on_trade_callback = on_trade_callback
        self._on_quote_callback = on_quote_callback

    def subscribe(self, symbols: list[str]):
        """Subscribe realtime data cho danh sách mã"""
        if self._on_trade_callback:
            self.stream.subscribe_trades(symbols, self._on_trade_callback)
        if self._on_quote_callback:
            self.stream.subscribe_quotes(symbols, self._on_quote_callback)

    def run(self):
        """Bắt đầu stream (blocking)"""
        self.stream.run()
