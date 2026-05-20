"""
Portfolio Manager - Manage multiple contracts with portfolio-level constraints.

Rules:
- Max N contracts simultaneously (all same direction)
- Only leaders (G, G+) can change portfolio direction
- Flip cooldown after direction reversal
- 15m signals have priority over 5m
- Per-combo SL/TP from strategy_config.yaml
"""

from datetime import datetime, timedelta, timezone

import yaml
from pathlib import Path

from src.notifier import TelegramNotifier
from src.trade_logger import TradeLogger

VN_TZ = timezone(timedelta(hours=7))
POINT_VALUE = 100_000
COMMISSION = 0.47  # pts per side

CONFIG_PATH = Path(__file__).parent.parent / "strategy_config.yaml"


class PortfolioManager:
    """Manages multiple positions with portfolio-level direction lock."""

    def __init__(self, notifier: TelegramNotifier, logger: TradeLogger,
                 max_contracts: int = 3, flip_cooldown: int = 3):
        self.notifier = notifier
        self.logger = logger
        self.max_contracts = max_contracts
        self.flip_cooldown = flip_cooldown

        self.positions: list[dict] = []  # list of open positions
        self.current_direction: int = 0  # 0=flat, 1=long, -1=short
        self.cooldown_remaining: int = 0
        self._next_pos_id: int = 1

        # Load per-combo risk from config
        self.combo_risk = self._load_combo_risk()

    def _load_combo_risk(self) -> dict:
        """Load per-combo SL/TP multipliers from strategy_config.yaml."""
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("combo_risk", {})
        except Exception:
            return {}

    def get_risk(self, combo: str) -> tuple[float, float]:
        """Get SL/TP multipliers for a combo."""
        risk = self.combo_risk.get(combo, {})
        sl = risk.get("sl_atr_mult", 1.5)
        tp = risk.get("tp_atr_mult", 4.0)
        return sl, tp

    @property
    def is_flat(self) -> bool:
        return len(self.positions) == 0

    @property
    def n_open(self) -> int:
        return len(self.positions)

    @property
    def has_capacity(self) -> bool:
        return len(self.positions) < self.max_contracts

    @property
    def in_cooldown(self) -> bool:
        return self.cooldown_remaining > 0

    def tick(self):
        """Call once per bar to decrease cooldown timer."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def can_open(self, direction: int) -> bool:
        """Check if a new position can be opened in given direction."""
        if self.in_cooldown:
            return False
        if not self.has_capacity:
            return False
        if self.current_direction != 0 and direction != self.current_direction:
            return False  # wrong direction, need flip first
        return True

    def should_flip(self, direction: int, signal_tf: str, confidence: int) -> bool:
        """Determine if portfolio should flip direction for this signal."""
        if self.current_direction == 0:
            return False  # no flip needed, flat
        if direction == self.current_direction:
            return False  # same direction
        if self.in_cooldown:
            return False  # in cooldown, can't flip

        # Only strong signals can flip:
        # - 15m signal always can flip
        # - 5m signal needs conf >= 2
        if signal_tf == "15m":
            return True
        if confidence >= 2:
            return True
        return False

    def execute_flip(self, current_price: float, reason: str = "FLIP"):
        """Close all positions at market price (direction reversal)."""
        now = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
        closed_pnl = 0.0

        for pos in self.positions:
            pnl = pos["direction"] * (current_price - pos["entry"]) - 2 * COMMISSION
            closed_pnl += pnl
            self.logger.log_exit(
                symbol=pos["symbol"],
                direction=pos["direction"],
                entry_price=pos["entry"],
                exit_price=current_price,
                reason=reason,
                pnl_pts=pnl,
                combo=pos["combo"],
                timeframe=pos["tf"],
                pos_id=pos["pos_id"],
                timestamp=now,
            )

        n_closed = len(self.positions)
        self.positions.clear()
        self.current_direction = 0
        self.cooldown_remaining = self.flip_cooldown

        pnl_vnd = closed_pnl * POINT_VALUE
        self.logger.log_event("FLIP", {
            "positions_closed": n_closed,
            "pnl_pts": closed_pnl,
            "pnl_vnd": pnl_vnd,
            "reason": reason,
        }, timestamp=now)

        icon = "\U0001f504"
        self.notifier.send(
            f"{icon} <b>FLIP - {n_closed} positions closed</b>\n"
            f"Exit @ <code>{current_price:,.1f}</code>\n"
            f"PnL: {closed_pnl:+.1f} pts ({pnl_vnd:+,.0f} VND)\n"
            f"Cooldown: {self.flip_cooldown} bars"
        )
        print(f"  [FLIP] Closed {n_closed} positions @ {current_price:.1f} "
              f"(PnL={closed_pnl:+.1f} pts)")

    def open_position(self, symbol: str, direction: int, entry_price: float,
                      atr: float, combo: str, timeframe: str,
                      confidence: int = 0):
        """Open a new position in the portfolio."""
        if not self.can_open(direction):
            return None

        sl_mult, tp_mult = self.get_risk(combo)
        sl = entry_price - direction * sl_mult * atr
        tp = entry_price + direction * tp_mult * atr
        now = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
        pos_id = self._next_pos_id
        self._next_pos_id += 1

        pos = {
            "pos_id": pos_id,
            "symbol": symbol,
            "direction": direction,
            "entry": entry_price,
            "sl": sl,
            "tp": tp,
            "atr": atr,
            "combo": combo,
            "tf": timeframe,
            "confidence": confidence,
            "opened_at": now,
            "tp1_hit": False,
        }
        self.positions.append(pos)
        self.current_direction = direction

        self.logger.log_entry(
            symbol=symbol, direction=direction, entry_price=entry_price,
            sl=sl, tp=tp, atr=atr, combo=combo, timeframe=timeframe,
            confidence=confidence, pos_id=pos_id, timestamp=now,
        )

        dir_str = "BUY" if direction == 1 else "SELL"
        slot_str = f"{self.n_open}/{self.max_contracts}"
        self.notifier.send(
            f"\U0001f4cd <b>Entry [{slot_str}]</b>\n"
            f"{dir_str} {symbol} @ <code>{entry_price:,.1f}</code>\n"
            f"SL: <code>{sl:,.1f}</code> ({sl_mult}x ATR)\n"
            f"TP: <code>{tp:,.1f}</code> ({tp_mult}x ATR)\n"
            f"Combo: {combo}({timeframe}) | Conf: {confidence}"
        )
        print(f"  [ENTRY {slot_str}] {dir_str} {symbol} @ {entry_price:.1f} "
              f"(SL={sl:.1f}, TP={tp:.1f}) {combo}({timeframe})")
        return pos

    def update_prices(self, symbol: str, high: float, low: float,
                      close: float, atr: float):
        """
        Update all positions with latest price bar.
        Check SL/TP hits. Apply trailing SL after TP1.
        """
        now = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
        to_remove = []

        for i, pos in enumerate(self.positions):
            if pos["symbol"] != symbol:
                continue

            direction = pos["direction"]
            entry = pos["entry"]

            # Current PnL
            if direction == 1:
                pnl_pts = close - entry
            else:
                pnl_pts = entry - close

            # --- TP1 check (1x ATR profit) → move SL to breakeven ---
            if not pos["tp1_hit"] and pnl_pts >= pos["atr"]:
                pos["tp1_hit"] = True
                pos["sl"] = entry  # breakeven
                self.notifier.send(
                    f"\U0001f3af <b>TP1 Hit - SL\u2192BE</b>\n"
                    f"#{pos['pos_id']} {pos['combo']}({pos['tf']}): "
                    f"+{pnl_pts:.1f} pts"
                )

            # --- Trailing SL after TP1 (trail by 1x ATR) ---
            if pos["tp1_hit"] and atr > 0:
                if direction == 1:
                    trail_sl = close - atr
                    if trail_sl > pos["sl"]:
                        pos["sl"] = trail_sl
                else:
                    trail_sl = close + atr
                    if trail_sl < pos["sl"]:
                        pos["sl"] = trail_sl

            # --- Check SL hit ---
            sl_hit = False
            if direction == 1 and low <= pos["sl"]:
                sl_hit = True
                exit_price = pos["sl"]
            elif direction == -1 and high >= pos["sl"]:
                sl_hit = True
                exit_price = pos["sl"]

            if sl_hit:
                pnl = direction * (exit_price - entry) - 2 * COMMISSION
                reason = "SL (trailing)" if pos["tp1_hit"] else "SL"
                self._close_one(i, exit_price, reason, pnl, now)
                to_remove.append(i)
                continue

            # --- Check TP hit ---
            tp_hit = False
            if direction == 1 and high >= pos["tp"]:
                tp_hit = True
                exit_price = pos["tp"]
            elif direction == -1 and low <= pos["tp"]:
                tp_hit = True
                exit_price = pos["tp"]

            if tp_hit:
                pnl = direction * (exit_price - entry) - 2 * COMMISSION
                self._close_one(i, exit_price, "TP", pnl, now)
                to_remove.append(i)

        # Remove closed positions (reverse order to keep indices valid)
        for i in sorted(to_remove, reverse=True):
            self.positions.pop(i)

        if not self.positions:
            self.current_direction = 0

    def _close_one(self, idx: int, exit_price: float, reason: str,
                   pnl_pts: float, timestamp: str):
        """Close a single position and log/notify."""
        pos = self.positions[idx]
        pnl_vnd = pnl_pts * POINT_VALUE
        dir_str = "BUY" if pos["direction"] == 1 else "SELL"

        self.logger.log_exit(
            symbol=pos["symbol"],
            direction=pos["direction"],
            entry_price=pos["entry"],
            exit_price=exit_price,
            reason=reason,
            pnl_pts=pnl_pts,
            combo=pos["combo"],
            timeframe=pos["tf"],
            pos_id=pos["pos_id"],
            timestamp=timestamp,
        )

        icon = "\u2705" if pnl_pts > 0 else "\u274c"
        self.notifier.send(
            f"{icon} <b>{reason}</b> #{pos['pos_id']}\n"
            f"{dir_str} {pos['combo']}({pos['tf']}): "
            f"{pos['entry']:,.1f} \u2192 {exit_price:,.1f}\n"
            f"<b>{pnl_pts:+.1f} pts ({pnl_vnd:+,.0f} VND)</b>"
        )
        print(f"  [{reason}] #{pos['pos_id']} {dir_str} {pos['combo']}({pos['tf']}) "
              f"@ {exit_price:.1f} PnL={pnl_pts:+.1f} pts")

    def close_all(self, current_price: float, reason: str = "EOD"):
        """Close all open positions (end of day)."""
        if not self.positions:
            return

        now = datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
        total_pnl = 0.0
        for pos in self.positions:
            pnl = pos["direction"] * (current_price - pos["entry"]) - 2 * COMMISSION
            total_pnl += pnl
            self.logger.log_exit(
                symbol=pos["symbol"],
                direction=pos["direction"],
                entry_price=pos["entry"],
                exit_price=current_price,
                reason=reason,
                pnl_pts=pnl,
                combo=pos["combo"],
                timeframe=pos["tf"],
                pos_id=pos["pos_id"],
                timestamp=now,
            )

        pnl_vnd = total_pnl * POINT_VALUE
        n = len(self.positions)
        self.positions.clear()
        self.current_direction = 0

        self.notifier.send(
            f"\U0001f551 <b>{reason} - {n} positions closed</b>\n"
            f"Exit @ <code>{current_price:,.1f}</code>\n"
            f"<b>PnL: {total_pnl:+.1f} pts ({pnl_vnd:+,.0f} VND)</b>"
        )
        print(f"  [{reason}] Closed {n} positions @ {current_price:.1f} "
              f"(Total PnL={total_pnl:+.1f} pts)")

    def status_str(self) -> str:
        """Get current portfolio status string."""
        if not self.positions:
            return "FLAT"
        dir_str = "LONG" if self.current_direction == 1 else "SHORT"
        combos = ", ".join(f"{p['combo']}({p['tf']})" for p in self.positions)
        return f"{dir_str} x{self.n_open}: {combos}"
