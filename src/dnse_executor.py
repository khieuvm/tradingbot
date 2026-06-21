"""
DNSE Order Executor — Place real orders via DNSE API.

Handles:
- Symbol mapping (VN30F1M → KRX format like 41I1G6000)
- Place limit/market orders
- Cancel orders
- Query positions
- Auto-close positions
"""

from datetime import datetime, timedelta, timezone

from dnse.client import DnseClient
from dnse.models import BoardId
from dnse.resources.orders import PlaceOrderRequest

from config import Config
from src.notifier import TelegramNotifier

VN_TZ = timezone(timedelta(hours=7))
MARKET_TYPE = "DERIVATIVE"
ORDER_CATEGORY = "NORMAL"

# VN30F1M KRX symbol — resolved via DNSE security_info
VN30F1M_KRX = "41I1G6000"


class DnseExecutor:
    """Execute real trades via DNSE API."""

    def __init__(self, client: DnseClient, notifier: TelegramNotifier):
        self.client = client
        self.notifier = notifier
        self.account_no = Config.ACCOUNT_NO
        self._contract_symbol: str = VN30F1M_KRX

    @property
    def contract_symbol(self) -> str:
        """Get current front-month contract symbol (KRX format)."""
        return self._contract_symbol

    def refresh_contract_symbol(self) -> str:
        """Re-resolve contract symbol from DNSE (for monthly rollover)."""
        try:
            info = self.client.market.security_info(VN30F1M_KRX, board_id=BoardId.ROUND_LOT)
            if info:
                self._contract_symbol = info[0].symbol
                return self._contract_symbol
        except Exception:
            pass
        return self._contract_symbol

    def place_order(self, direction: int, quantity: int = 1,
                    price: float | None = None,
                    order_type: str = "LO") -> dict | None:
        """Place an order on DNSE.

        Args:
            direction: 1=BUY, -1=SELL
            quantity: Number of contracts
            price: Limit price (required for LO orders)
            order_type: "LO" (Limit) or "MTL" (Market-to-Limit)

        Returns:
            Order response dict or None on failure.
        """
        side = "NB" if direction == 1 else "NS"
        symbol = self.contract_symbol

        request = PlaceOrderRequest(
            account_no=self.account_no,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )

        try:
            response = self.client.orders.place(
                request,
                market_type=MARKET_TYPE,
                order_category=ORDER_CATEGORY,
            )
            dir_str = "BUY" if direction == 1 else "SELL"
            price_str = f"{price:,.1f}" if price else "MARKET"
            self.notifier.send(
                f"\U0001f4b0 <b>Order Placed</b>\n"
                f"{dir_str} {quantity}x {symbol} @ {price_str}\n"
                f"Type: {order_type}"
            )
            print(f"  [ORDER] {dir_str} {quantity}x {symbol} @ {price_str} ({order_type})")
            return {"order_id": response.id, "status": response.order_status}
        except Exception as e:
            self.notifier.send(f"\u274c <b>Order FAILED</b>\n{e}")
            print(f"  [ORDER ERROR] {e}")
            return None

    def cancel_order(self, order_id: int) -> bool:
        """Cancel an existing order."""
        try:
            self.client.orders.cancel(
                self.account_no, order_id,
                market_type=MARKET_TYPE,
                order_category=ORDER_CATEGORY,
            )
            print(f"  [ORDER] Cancelled #{order_id}")
            return True
        except Exception as e:
            print(f"  [ORDER ERROR] Cancel failed: {e}")
            return False

    def get_positions(self) -> list:
        """Get current open derivative positions."""
        try:
            response = self.client.deals.list(
                self.account_no, market_type=MARKET_TYPE
            )
            return response.deals if hasattr(response, 'deals') else []
        except Exception as e:
            print(f"  [POSITIONS ERROR] {e}")
            return []

    def get_pending_orders(self) -> list:
        """Get current pending orders."""
        try:
            response = self.client.orders.list(
                self.account_no,
                market_type=MARKET_TYPE,
                order_category=ORDER_CATEGORY,
            )
            return response.orders if hasattr(response, 'orders') else []
        except Exception as e:
            print(f"  [ORDERS ERROR] {e}")
            return []

    def close_position(self, direction: int, quantity: int = 1,
                       price: float | None = None) -> dict | None:
        """Close an existing position (place opposite order).

        Args:
            direction: Direction of EXISTING position (1=long → sell to close)
            quantity: Contracts to close
            price: Limit price (None for market)
        """
        close_direction = -direction  # Opposite to close
        order_type = "MTL" if price is None else "LO"
        return self.place_order(close_direction, quantity, price, order_type)
