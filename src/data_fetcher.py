"""
Data fetcher module - Lấy dữ liệu từ DNSE API và vnstock
Hỗ trợ: Cổ phiếu, Phái sinh (Futures/VN30F), Index, ETF
"""
import pandas as pd
from vnstock import Market, Quote, Reference
from dnse import DnseClient, DnseMarketStream, BoardId

from config import Config


class DataFetcher:
    """Lấy dữ liệu lịch sử và realtime từ DNSE/vnstock"""

    # Số nến tối đa mỗi ngày giao dịch theo interval
    _CANDLES_PER_DAY = {
        "1m": 270, "3m": 90, "5m": 54, "15m": 18, "30m": 9,
        "1H": 5, "1D": 1, "1W": 0.2, "1M": 0.05,
    }

    def __init__(self):
        self.market = Market()
        self.reference = Reference()

    @staticmethod
    def _calc_count(start: str, end: str, interval: str) -> int:
        """Tính số lượng nến cần fetch dựa vào khoảng thời gian và interval."""
        from datetime import datetime
        days = (datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")).days + 1
        cpd = DataFetcher._CANDLES_PER_DAY.get(interval, 1)
        # Trading days ~ 70% of calendar days
        count = int(days * 0.72 * cpd) + 50
        return min(max(count, 100), 5000)

    def get_historical_ohlcv(self, symbol: str, start: str, end: str, interval: str = "1D") -> pd.DataFrame:
        """Lấy dữ liệu OHLCV lịch sử từ vnstock

        Args:
            symbol: Mã cổ phiếu (e.g., "HPG", "VNM")
            start: Ngày bắt đầu "YYYY-MM-DD"
            end: Ngày kết thúc "YYYY-MM-DD"
            interval: Khung thời gian ("1D", "1W", "1M")
        """
        count = self._calc_count(start, end, interval)
        df = self.market.equity(symbol).ohlcv(start=start, end=end, interval=interval, count=count)
        return df

    # === PHÁI SINH (FUTURES) ===

    def get_futures_list(self) -> pd.DataFrame:
        """Lấy danh sách các hợp đồng phái sinh đang giao dịch"""
        return self.reference.futures.list()

    def get_futures_ohlcv(self, symbol: str, start: str, end: str, interval: str = "1D") -> pd.DataFrame:
        """Lấy dữ liệu OHLCV phái sinh (VN30F1M, VN30F2M, VN30F1Q, VN30F2Q)

        Args:
            symbol: Mã hợp đồng phái sinh (e.g., "VN30F1M", "VN30F2M")
            start: Ngày bắt đầu "YYYY-MM-DD"
            end: Ngày kết thúc "YYYY-MM-DD"
            interval: Khung thời gian ("1D", "1W")
        """
        count = self._calc_count(start, end, interval)
        df = self.market.futures(symbol).ohlcv(start=start, end=end, interval=interval, count=count)
        return df

    def get_futures_info(self, symbol: str = "VN30F1M") -> dict:
        """Lấy thông tin chi tiết hợp đồng phái sinh"""
        return self.reference.futures.info()

    # === INDEX ===

    def get_index_ohlcv(self, symbol: str, start: str, end: str, interval: str = "1D") -> pd.DataFrame:
        """Lấy dữ liệu OHLCV chỉ số (VNINDEX, VN30, HNX30, ...)

        Args:
            symbol: Mã chỉ số (e.g., "VNINDEX", "VN30", "HNX30")
        """
        count = self._calc_count(start, end, interval)
        df = self.market.index(symbol).ohlcv(start=start, end=end, interval=interval, count=count)
        return df

    def get_index_list(self) -> pd.DataFrame:
        """Lấy danh sách các chỉ số thị trường"""
        return self.reference.index.list()

    def get_intraday(self, symbol: str) -> pd.DataFrame:
        """Lấy dữ liệu intraday (tick-by-tick)"""
        quote = Quote(symbol=symbol, source="KBS")
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
