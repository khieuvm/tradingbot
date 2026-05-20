"""
Position Manager - Track open positions, trailing SL, TP management.
"""
from datetime import datetime, timedelta, timezone

from src.notifier import TelegramNotifier
from src.trade_logger import TradeLogger

VN_TZ = timezone(timedelta(hours=7))


class PositionManager:
    """Manages simulated positions with trailing SL and tiered TP."""

    def __init__(self, notifier: TelegramNotifier, logger: TradeLogger,
                 sl_atr_mult: float = 1.5, tp_atr_mult: float = 3.0):
        self.notifier = notifier
        self.logger = logger
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.positions: dict = {}  # symbol -> position dict

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def open_position(self, symbol: str, direction: int, entry_price: float,
                      sl: float, tp: float, atr: float, combo: str,
                      timeframe: str, n_combos: int, score: int):
        """Open a new tracked position."""
        now = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
        self.positions[symbol] = {
            "direction": direction,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "atr": atr,
            "combo": combo,
            "timeframe": timeframe,
            "n_combos": n_combos,
            "score": score,
            "opened_at": now,
            "highest_pnl": 0.0,
            "tp1_hit": False,
        }
        self.logger.log_entry(
            symbol=symbol, direction=direction, entry_price=entry_price,
            sl=sl, tp=tp, atr=atr, combo=combo, timeframe=timeframe,
            n_combos=n_combos, score=score, timestamp=now,
        )
        dir_str = "BUY" if direction == 1 else "SELL"
        self.notifier.send(
            f"📍 <b>Position Opened</b>\n"
            f"{dir_str} {symbol} @ <code>{entry_price:,.1f}</code>\n"
            f"SL: <code>{sl:,.1f}</code> | TP: <code>{tp:,.1f}</code>\n"
            f"Combo: {combo} | TF: {timeframe}\n"
            f"Combos agreeing: {n_combos} | Score: {score}"
        )
        print(f"  [POSITION] Opened {dir_str} {symbol} @ {entry_price:.1f} "
              f"(SL={sl:.1f}, TP={tp:.1f})")

    def update(self, symbol: str, current_price: float, current_atr: float):
        """Update position: check SL/TP hit, apply trailing SL."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        direction = pos["direction"]
        entry = pos["entry_price"]

        # Calculate current PnL
        if direction == 1:  # BUY
            pnl_pts = current_price - entry
        else:  # SELL
            pnl_pts = entry - current_price

        # Track highest PnL for trailing
        pos["highest_pnl"] = max(pos["highest_pnl"], pnl_pts)

        # --- Check TP1 hit (1x ATR profit) → move SL to breakeven ---
        tp1_level = pos["atr"]  # 1x ATR in points
        if not pos["tp1_hit"] and pnl_pts >= tp1_level:
            pos["tp1_hit"] = True
            pos["sl"] = entry  # Move SL to breakeven
            self.notifier.send(
                f"🎯 <b>TP1 Hit - SL → Breakeven</b>\n"
                f"{symbol}: +{pnl_pts:.1f} pts | SL moved to {entry:,.1f}"
            )
            print(f"  [POSITION] TP1 hit, SL moved to BE @ {entry:.1f}")

        # --- Trailing SL after TP1 (trail by 1x ATR from highest) ---
        if pos["tp1_hit"] and current_atr > 0:
            if direction == 1:
                trail_sl = current_price - current_atr
                if trail_sl > pos["sl"]:
                    pos["sl"] = trail_sl
            else:
                trail_sl = current_price + current_atr
                if trail_sl < pos["sl"]:
                    pos["sl"] = trail_sl

        # --- Check SL hit ---
        sl_hit = False
        if direction == 1 and current_price <= pos["sl"]:
            sl_hit = True
        elif direction == -1 and current_price >= pos["sl"]:
            sl_hit = True

        if sl_hit:
            reason = "SL (trailing)" if pos["tp1_hit"] else "SL (initial)"
            self._close_position(symbol, current_price, reason, pnl_pts)
            return

        # --- Check TP (final) hit ---
        tp_hit = False
        if direction == 1 and current_price >= pos["tp"]:
            tp_hit = True
        elif direction == -1 and current_price <= pos["tp"]:
            tp_hit = True

        if tp_hit:
            self._close_position(symbol, current_price, "TP", pnl_pts)

    def _close_position(self, symbol: str, exit_price: float,
                        reason: str, pnl_pts: float):
        """Close position and log/notify."""
        pos = self.positions.pop(symbol)
        now = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
        direction = pos["direction"]
        entry = pos["entry_price"]

        self.logger.log_exit(
            symbol=symbol, direction=direction, entry_price=entry,
            exit_price=exit_price, reason=reason, pnl_pts=pnl_pts,
            timestamp=now,
        )

        pnl_vnd = pnl_pts * 100_000
        icon = "✅" if pnl_pts > 0 else "❌"
        dir_str = "BUY" if direction == 1 else "SELL"
        self.notifier.send(
            f"{icon} <b>Position Closed - {reason}</b>\n"
            f"{dir_str} {symbol}: {entry:,.1f} → {exit_price:,.1f}\n"
            f"<b>PnL: {pnl_pts:+.1f} pts ({pnl_vnd:+,.0f} VND)</b>\n"
            f"Combo: {pos['combo']} | Duration: {pos['opened_at']} → {now}"
        )
        print(f"  [POSITION] Closed {dir_str} {symbol} @ {exit_price:.1f} "
              f"({reason}, PnL={pnl_pts:+.1f} pts)")
