"""
Notifier module - Send alerts via Telegram
"""
import asyncio
import urllib.request
import urllib.parse
import json
from config import Config


class TelegramNotifier:
    """Send trading alerts via Telegram (lightweight, no python-telegram-bot dependency)"""

    def __init__(self):
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID

    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, message: str):
        """Send message via Telegram Bot API (sync, no external deps)"""
        if not self.is_configured():
            print(f"[TELEGRAM NOT CONFIGURED] {message}")
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
        }).encode("utf-8")
        try:
            urllib.request.urlopen(url, data, timeout=10)
        except Exception as e:
            print(f"[TELEGRAM ERROR] {e}")

    def send_signal_alert(self, symbol: str, signal: str, price: float,
                          conditions_fired: list[str], confidence: int,
                          combo_name: str = "", sl: float = 0, tp: float = 0,
                          extra: dict = None):
        """Send detailed signal alert with conditions breakdown.

        Args:
            symbol: e.g. "VN30F1M"
            signal: "BUY" or "SELL"
            price: current price
            conditions_fired: list of condition labels that triggered
            confidence: signal confidence (1-3)
            combo_name: active combo preset name
            sl: stop loss level
            tp: take profit level
            extra: dict with additional info (rsi, ema_slope, adx, etc.)
        """
        direction = "BUY" if signal == "BUY" else "SELL"
        header = f"{'=' * 20}\n<b>{direction} SIGNAL</b>\n{'=' * 20}"

        lines = [
            header,
            f"<b>Symbol:</b> <code>{symbol}</code>",
            f"<b>Price:</b> <code>{price:,.1f}</code>",
            f"<b>Confidence:</b> {'*' * confidence} ({confidence}/3)",
        ]

        if combo_name:
            lines.append(f"<b>Strategy:</b> {combo_name}")

        if sl and tp:
            lines.append(f"<b>SL:</b> <code>{sl:,.1f}</code> | <b>TP:</b> <code>{tp:,.1f}</code>")
            rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 0 else 0
            lines.append(f"<b>R:R:</b> 1:{rr:.1f}")

        lines.append("")
        lines.append("<b>Conditions met:</b>")
        for cond in conditions_fired:
            lines.append(f"  - {cond}")

        if extra:
            lines.append("")
            lines.append("<b>Context:</b>")
            for k, v in extra.items():
                lines.append(f"  {k}: <code>{v}</code>")

        self.send("\n".join(lines))

    def send_signal(self, symbol: str, signal: str, price: float, strategy: str):
        """Simple signal alert (legacy)"""
        direction = "BUY" if signal == "BUY" else "SELL"
        msg = (
            f"<b>{direction} Signal</b>\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Price: <code>{price:,.0f}</code>\n"
            f"Strategy: {strategy}\n"
        )
        self.send(msg)

    def send_error(self, error: str):
        """Send error notification"""
        msg = f"<b>Error</b>\n<code>{error}</code>"
        self.send(msg)
