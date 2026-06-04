"""
Position Manager - Track open positions, trailing SL, TP management.
Per-combo trailing config: some combos use fixed TP/SL, others trail after TP2.
"""
from datetime import datetime, timedelta, timezone

from src.notifier import TelegramNotifier
from src.trade_logger import TradeLogger

VN_TZ = timezone(timedelta(hours=7))

# Per-combo trailing configuration (backtest-optimized)
# "fixed" = no trailing, just hard SL/TP
# "trail_tp2" = activate trailing after +2xATR profit, trail by trail_atr * ATR
COMBO_TRAIL_CONFIG = {
    "P": {"mode": "trail_tp2", "activate_atr": 2.0, "trail_atr": 1.5},
    "R": {"mode": "trail_tp2", "activate_atr": 2.0, "trail_atr": 1.5},
    # All others default to "fixed"
}


class PositionManager:
    """Manages simulated positions with per-combo trailing SL and tiered TP."""

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

        # Determine trailing mode from combo short name
        combo_short = combo.split(":")[0].strip() if ":" in combo else combo[:4]
        trail_cfg = COMBO_TRAIL_CONFIG.get(combo_short, {"mode": "fixed"})

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
            "trail_activated": False,
            "trail_mode": trail_cfg["mode"],
            "trail_activate_atr": trail_cfg.get("activate_atr", 2.0),
            "trail_distance_atr": trail_cfg.get("trail_atr", 1.5),
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
        """Update position: check SL/TP hit, apply per-combo trailing SL."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        direction = pos["direction"]
        entry = pos["entry_price"]
        atr = pos["atr"]

        # Calculate current PnL
        if direction == 1:  # BUY
            pnl_pts = current_price - entry
        else:  # SELL
            pnl_pts = entry - current_price

        # Track highest PnL for trailing
        pos["highest_pnl"] = max(pos["highest_pnl"], pnl_pts)

        # --- Trailing logic based on combo config ---
        trail_mode = pos.get("trail_mode", "fixed")

        if trail_mode == "fixed":
            # No trailing — just check hard SL/TP
            pass
        elif trail_mode == "trail_tp2":
            # Activate trailing after profit >= activate_atr * ATR
            activate_dist = pos["trail_activate_atr"] * atr
            trail_dist = pos["trail_distance_atr"] * (current_atr if current_atr > 0 else atr)

            if not pos["trail_activated"] and pos["highest_pnl"] >= activate_dist:
                pos["trail_activated"] = True
                # Lock profit: move SL to entry + 1xATR
                lock_sl = entry + direction * 1.0 * atr
                pos["sl"] = lock_sl
                self.notifier.send(
                    f"🎯 <b>Trail Activated (+{pos['highest_pnl']:.1f}pts)</b>\n"
                    f"{symbol}: SL locked at {lock_sl:,.1f} (+1×ATR)"
                )
                print(f"  [POSITION] Trail activated, SL locked @ {lock_sl:.1f}")

            if pos["trail_activated"]:
                if direction == 1:
                    new_trail = current_price - trail_dist
                    if new_trail > pos["sl"]:
                        pos["sl"] = new_trail
                else:
                    new_trail = current_price + trail_dist
                    if new_trail < pos["sl"]:
                        pos["sl"] = new_trail

        # --- Check SL hit ---
        sl_hit = False
        if direction == 1 and current_price <= pos["sl"]:
            sl_hit = True
        elif direction == -1 and current_price >= pos["sl"]:
            sl_hit = True

        if sl_hit:
            if pos.get("trail_activated"):
                reason = "SL (trailing)"
            elif pos.get("tp1_hit"):
                reason = "SL (breakeven)"
            else:
                reason = "SL (initial)"
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
