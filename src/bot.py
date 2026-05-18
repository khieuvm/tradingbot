"""
Bot module - Logic chính của trading bot
"""
from dnse import DnseClient, PlaceOrderRequest, BoardId

from config import Config
from src.strategy import BaseStrategy
from src.notifier import TelegramNotifier


class TradingBot:
    """Bot tự động trading qua DNSE"""

    def __init__(self, strategy: BaseStrategy, account_no: str = ""):
        self.strategy = strategy
        self.account_no = account_no or Config.ACCOUNT_NO
        self.notifier = TelegramNotifier()
        self._trading_token_valid = False

    def authenticate(self, otp: str, otp_type: str = "smart_otp"):
        """Xác thực OTP để có thể đặt lệnh

        Args:
            otp: Mã OTP
            otp_type: "email_otp" hoặc "smart_otp"
        """
        with DnseClient(api_key=Config.DNSE_API_KEY, api_secret=Config.DNSE_API_SECRET) as client:
            client.registration.send_otp()
            client.registration.verify_otp(otp, otp_type=otp_type)
            self._trading_token_valid = True
            print("[OK] Trading token authenticated")

    def get_account_info(self) -> dict:
        """Lấy thông tin tài khoản"""
        with DnseClient(api_key=Config.DNSE_API_KEY, api_secret=Config.DNSE_API_SECRET) as client:
            accounts = client.accounts.list()
            return accounts

    def get_balance(self) -> dict:
        """Lấy số dư tài khoản"""
        with DnseClient(api_key=Config.DNSE_API_KEY, api_secret=Config.DNSE_API_SECRET) as client:
            return client.accounts.balances(self.account_no)

    def place_order(self, symbol: str, side: str, quantity: int, price: float, order_type: str = "LO"):
        """Đặt lệnh mua/bán

        Args:
            symbol: Mã cổ phiếu
            side: "NB" (mua) hoặc "NS" (bán)
            quantity: Số lượng (bội số 100)
            price: Giá đặt lệnh
            order_type: "LO" (limit), "MP" (market), "ATO", "ATC"
        """
        with DnseClient(api_key=Config.DNSE_API_KEY, api_secret=Config.DNSE_API_SECRET) as client:
            # Verify OTP first
            client.registration.send_otp()
            otp = input("Enter OTP: ")
            client.registration.verify_otp(otp)

            order = client.orders.place(PlaceOrderRequest(
                account_no=self.account_no,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
            ))
            print(f"[ORDER] {side} {quantity} {symbol} @ {price:,.0f} -> ID: {order.id}")
            self.notifier.send_order_executed(symbol, side, quantity, price)
            return order

    def cancel_order(self, order_id: int):
        """Hủy lệnh"""
        with DnseClient(api_key=Config.DNSE_API_KEY, api_secret=Config.DNSE_API_SECRET) as client:
            client.registration.send_otp()
            otp = input("Enter OTP: ")
            client.registration.verify_otp(otp)
            client.orders.cancel(self.account_no, order_id)
            print(f"[CANCELLED] Order {order_id}")

    def check_signal(self, df, symbol: str) -> int:
        """Kiểm tra tín hiệu từ strategy với dữ liệu mới nhất

        Returns:
            1 = BUY, -1 = SELL, 0 = HOLD
        """
        df_signals = self.strategy.generate_signals(df)
        last_signal = df_signals["signal"].iloc[-1]

        if last_signal != 0:
            signal_name = "BUY" if last_signal == 1 else "SELL"
            price = df_signals["close"].iloc[-1]
            print(f"[SIGNAL] {signal_name} {symbol} @ {price:,.0f}")
            self.notifier.send_signal(symbol, signal_name, price, self.strategy.name)

        return last_signal
