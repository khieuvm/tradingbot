"""
Portfolio Manager - Manage multiple contracts with portfolio-level constraints.

Rules:
- Max N contracts simultaneously (all same direction)
- Only leaders (G, G+) can change portfolio direction
- Flip cooldown after direction reversal
- 15m signals have priority over 5m
- Per-combo SL/TP from strategy_config.yaml
- AM/PM session-aware trailing (validated backtest: +6.34/d)
- Adaptive exit: AM + pre_move/ATR > 0.8 → exit@4pts
"""

from datetime import datetime, timedelta, timezone

import yaml
from pathlib import Path

from src.notifier import TelegramNotifier
from src.trade_logger import TradeLogger
from src.strategy_config import get_config, get_session_params, get_combo_config

VN_TZ = timezone(timedelta(hours=7))
POINT_VALUE = 100_000
COMMISSION = 0.47  # pts per side (0.46 actual, rounded up)

# Fallback defaults — overridden by strategy_config.yaml session_params
_DEFAULT_SESSION_PARAMS = {
    "AM": {"sl_atr_mult": 1.5, "trail_activate_pts": 7.0, "trail_atr_mult": 2.5, "max_hold_bars": 30,
            "be_trigger_pts": 3.0, "be_atr_min": 3.5, "be_partial_pts": 1.0},
    "PM": {"sl_atr_mult": 1.2, "trail_activate_pts": 6.0, "trail_atr_mult": 1.0, "max_hold_bars": 8,
            "be_trigger_pts": 3.0, "be_atr_min": 3.5, "be_partial_pts": 1.0},
}


class PortfolioManager:
    """Manages multiple positions with portfolio-level direction lock."""

    def __init__(self, notifier: TelegramNotifier, logger: TradeLogger,
                 max_contracts: int = 3, flip_cooldown: int = 3,
                 executor=None):
        self.notifier = notifier
        self.logger = logger
        self.max_contracts = max_contracts
        self.flip_cooldown = flip_cooldown
        self.executor = executor  # Optional DnseExecutor; None = signal-only mode

        self.positions: list[dict] = []  # list of open positions
        self.current_direction: int = 0  # 0=flat, 1=long, -1=short
        self.cooldown_remaining: int = 0
        self._next_pos_id: int = 1
        self.on_close_callback = None  # callback(combo_short, direction_str, pnl_pts)

        # Load all params from strategy_config.yaml (single source of truth)
        self._load_config()

    def _load_config(self):
        """Load session params, adaptive exit, and combo risk from strategy_config.yaml."""
        cfg = get_config()

        # Session-specific trailing/SL params
        sp = cfg.get("session_params", {})
        self.session_params = {}
        for sess, defaults in _DEFAULT_SESSION_PARAMS.items():
            self.session_params[sess] = {k: sp.get(sess, {}).get(k, v) for k, v in defaults.items()}

        # Adaptive exit params
        ae = cfg.get("adaptive_exit", {})
        self.adaptive_exit_enabled = ae.get("enabled", True)
        self.adaptive_exit_session = ae.get("session", "AM")
        self.adaptive_exit_ratio = ae.get("pre_move_ratio_threshold", 0.8)
        self.adaptive_exit_mfe = ae.get("mfe_trigger_pts", 2.0)
        self.adaptive_exit_pts = ae.get("exit_target_pts", 3.2)

    def get_tp_mult(self, combo: str) -> float:
        """Get TP multiplier for a combo."""
        combo_cfg = get_combo_config(combo)
        return combo_cfg.get("tp_atr_mult", 4.0)

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
            if self.executor is not None:
                self.executor.close_position(pos["direction"], quantity=1)

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

    @staticmethod
    def _detect_session() -> str:
        """Detect current trading session: AM or PM."""
        now = datetime.now(VN_TZ)
        current_mins = now.hour * 60 + now.minute
        if current_mins < 11 * 60 + 30:
            return "AM"
        return "PM"

    def open_position(self, symbol: str, direction: int, entry_price: float,
                      atr: float, combo: str, timeframe: str,
                      confidence: int = 0, pre_move_ratio: float = 0.0):
        """Open a new position in the portfolio with session-aware SL."""
        if not self.can_open(direction):
            return None

        session = self._detect_session()
        session_params = self.session_params[session]

        sl_mult = session_params["sl_atr_mult"]
        tp_mult = self.get_tp_mult(combo)
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
            "session": session,
            "pre_move_ratio": pre_move_ratio,
            "tp1_hit": False,
            "bars_held": 0,
        }
        self.positions.append(pos)
        self.current_direction = direction

        self.logger.log_entry(
            symbol=symbol, direction=direction, entry_price=entry_price,
            sl=sl, tp=tp, atr=atr, combo=combo, timeframe=timeframe,
            confidence=confidence, pos_id=pos_id, timestamp=now,
        )
        if self.executor is not None:
            self.executor.place_order(direction, quantity=1,
                                      price=entry_price, order_type="LO")

        dir_str = "BUY" if direction == 1 else "SELL"
        slot_str = f"{self.n_open}/{self.max_contracts}"
        self.notifier.send(
            f"\U0001f4cd <b>Entry [{slot_str}]</b>\n"
            f"{dir_str} {symbol} @ <code>{entry_price:,.1f}</code>\n"
            f"SL: <code>{sl:,.1f}</code> ({sl_mult}x ATR)\n"
            f"TP: <code>{tp:,.1f}</code> ({tp_mult}x ATR)\n"
            f"Combo: {combo}({timeframe}) | {session} | PMR: {pre_move_ratio:.2f}"
        )
        print(f"  [ENTRY {slot_str}] {dir_str} {symbol} @ {entry_price:.1f} "
              f"(SL={sl:.1f}, TP={tp:.1f}) {combo}({timeframe}) [{session}]")
        return pos

    def update_prices(self, symbol: str, high: float, low: float,
                      close: float, atr: float, regime: str = "NORMAL"):
        """
        Update all positions with latest price bar.
        Session-aware trailing (AM/PM split params from validated backtest).
        Adaptive exit: AM + pre_move_ratio > 0.8 → exit@4pts.
        Breakeven at +4pts when ATR>=3.5.
        Trail tightening at +8pts when ATR>=3.5.
        Session-aware tightening: PM only, from 14:15 (14 mins before close).
        Max hold bars enforcement.
        """
        now_dt = datetime.now(VN_TZ)
        now = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        mins_to_close = self._minutes_until_session_end(now_dt)
        to_remove = []

        for i, pos in enumerate(self.positions):
            if pos["symbol"] != symbol:
                continue

            direction = pos["direction"]
            entry = pos["entry"]
            entry_atr = pos.get("atr", atr)
            session = pos.get("session", "AM")
            pre_move_ratio = pos.get("pre_move_ratio", 0.0)
            session_params = self.session_params[session]

            pos["bars_held"] = pos.get("bars_held", 0) + 1

            # Current PnL
            if direction == 1:
                pnl_pts = close - entry
                mfe = high - entry
            else:
                pnl_pts = entry - close
                mfe = entry - low

            # Track max favorable excursion
            pos["mfe"] = max(pos.get("mfe", 0), mfe)

            # --- Adaptive exit: AM + pre_move_ratio > threshold → exit early ---
            if (session == self.adaptive_exit_session
                    and pre_move_ratio > self.adaptive_exit_ratio
                    and pos["mfe"] >= self.adaptive_exit_mfe and not pos.get("adapt_exited")):
                pnl = direction * (close - entry) - 2 * COMMISSION
                if pnl_pts >= self.adaptive_exit_pts:
                    self._close_one(i, close, "ADAPT_EXIT", pnl, now)
                    to_remove.append(i)
                    continue
                else:
                    pos["adapt_exited"] = True
                    if direction == 1:
                        pos["sl"] = max(pos["sl"], entry + 2.0)
                    else:
                        pos["sl"] = min(pos["sl"], entry - 2.0)

            # --- Session-time exits (match backtest: AM 11:25, PM 14:25) ---
            current_mins = now_dt.hour * 60 + now_dt.minute
            if session == "AM" and current_mins >= 11 * 60 + 25:
                pnl = direction * (close - entry) - 2 * COMMISSION
                self._close_one(i, close, "SESSION", pnl, now)
                to_remove.append(i)
                continue
            if session == "PM" and current_mins >= 14 * 60 + 25:
                pnl = direction * (close - entry) - 2 * COMMISSION
                self._close_one(i, close, "SESSION", pnl, now)
                to_remove.append(i)
                continue
            # AM position persisting past lunch break (should never happen, safety net)
            if session == "AM" and current_mins >= 13 * 60:
                pnl = direction * (close - entry) - 2 * COMMISSION
                self._close_one(i, close, "SESSION", pnl, now)
                to_remove.append(i)
                continue

            # --- Max hold (time-based, matches backtest: max_hold_bars × 5 min per bar) ---
            max_hold_bars = session_params["max_hold_bars"]
            try:
                opened_dt = datetime.strptime(
                    pos["opened_at"], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=VN_TZ)
                elapsed_min = (now_dt - opened_dt).total_seconds() / 60
            except Exception:
                elapsed_min = 0
            if elapsed_min >= max_hold_bars * 5:
                pnl = direction * (close - entry) - 2 * COMMISSION
                self._close_one(i, close, "MAX_HOLD", pnl, now)
                to_remove.append(i)
                continue

            # --- Breakeven when MFE >= be_trigger and entry ATR >= be_atr_min ---
            trail_activate = session_params["trail_activate_pts"]
            trail_atr_mult = session_params["trail_atr_mult"]
            be_trigger = session_params.get("be_trigger_pts", 3.0)
            be_atr_min = session_params.get("be_atr_min", 3.5)
            be_partial = session_params.get("be_partial_pts", 1.0)
            if not pos.get("be_done") and entry_atr >= be_atr_min and pos["mfe"] >= be_trigger:
                pos["be_done"] = True
                new_sl = entry + direction * be_partial
                if direction == 1 and new_sl > pos["sl"]:
                    pos["sl"] = new_sl
                elif direction == -1 and new_sl < pos["sl"]:
                    pos["sl"] = new_sl

            # --- TP1 notification: pnl >= 1×ATR (UX only, not a gate) ---
            be_threshold = 0.7 * entry_atr if regime == "RANGING" else entry_atr
            if not pos["tp1_hit"] and pnl_pts >= be_threshold:
                pos["tp1_hit"] = True
                self.notifier.send(
                    f"\U0001f3af <b>TP1 Hit</b>\n"
                    f"#{pos['pos_id']} {pos['combo']}({pos['tf']}): "
                    f"+{pnl_pts:.1f} pts [{session}]"
                )

            # --- Trail: anchored to best_price (matches backtest), entry ATR fixed ---
            # best_price = entry + direction * mfe (running max high / min low)
            if pos["mfe"] >= trail_activate and entry_atr > 0:
                best_price = entry + direction * pos["mfe"]
                # Tighten at +8pts when entry ATR >= be_atr_min
                if entry_atr >= be_atr_min and pos["mfe"] >= 8.0:
                    trail_dist = 1.2 * entry_atr
                else:
                    trail_dist = trail_atr_mult * entry_atr
                if direction == 1:
                    trail_sl = best_price - trail_dist
                    if trail_sl > pos["sl"]:
                        pos["sl"] = trail_sl
                else:
                    trail_sl = best_price + trail_dist
                    if trail_sl < pos["sl"]:
                        pos["sl"] = trail_sl

            # --- Session-aware exit: tighten SL from 14:15 in PM only ---
            if session == "PM" and mins_to_close <= 14 and atr > 0:
                tight_dist = 0.5 * atr
                if direction == 1:
                    tight_sl = close - tight_dist
                    if tight_sl > pos["sl"]:
                        pos["sl"] = tight_sl
                else:
                    tight_sl = close + tight_dist
                    if tight_sl < pos["sl"]:
                        pos["sl"] = tight_sl

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

    @staticmethod
    def _minutes_until_session_end(now: datetime) -> int:
        """Minutes until position must be closed (AM 11:25, PM 14:29)."""
        current_mins = now.hour * 60 + now.minute
        if current_mins < 11 * 60 + 30:
            return (11 * 60 + 25) - current_mins  # AM: close by 11:25
        elif current_mins >= 13 * 60:
            return (14 * 60 + 29) - current_mins  # PM: close by 14:29
        return 999  # lunch break

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
        if self.executor is not None:
            self.executor.close_position(pos["direction"], quantity=1)

        icon = "\u2705" if pnl_pts > 0 else "\u274c"
        self.notifier.send(
            f"{icon} <b>{reason}</b> #{pos['pos_id']}\n"
            f"{dir_str} {pos['combo']}({pos['tf']}): "
            f"{pos['entry']:,.1f} \u2192 {exit_price:,.1f}\n"
            f"<b>{pnl_pts:+.1f} pts ({pnl_vnd:+,.0f} VND)</b>"
        )
        print(f"  [{reason}] #{pos['pos_id']} {dir_str} {pos['combo']}({pos['tf']}) "
              f"@ {exit_price:.1f} PnL={pnl_pts:+.1f} pts")

        if self.on_close_callback:
            self.on_close_callback(pos.get("combo", ""), dir_str, pnl_pts)

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
            if self.executor is not None:
                self.executor.close_position(pos["direction"], quantity=1)

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
