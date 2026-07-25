"""
ORB (Opening Range Breakout) — PM Session Strategy
====================================================
Hyperopt-validated: OR=6 bars (30min), PM only, hold 8 bars
Best config: WR 71-83%, Net +5.11 to +8.44/trade, PF 6.78-29.6

Concept:
  - Define opening range = first 6 bars (30 min) of PM session
  - Trade breakout above OR_high or below OR_low
  - Volume filter confirms real vs fake breakout
  - SL = 0.7 × OR range (tight, breakout should not retrace into range)

Usage in scanner.py:
    orb = ORBSignal(notifier=notifier)
    # Each scan cycle:
    signals = orb.check(df_5m, current_time)
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategy_config import get_config

VN_TZ = timezone(timedelta(hours=7))


class ORBSignal:
    """Opening Range Breakout signal — PM session only."""

    def __init__(self, notifier=None):
        cfg = get_config().get("orb", {})
        self.enabled = cfg.get("enabled", False)
        self.shadow_mode = cfg.get("shadow_mode", True)
        self.notifier = notifier
        self._log_path = Path("logs/orb_shadow.jsonl")

        # Hyperopt-validated params
        self.or_bars = cfg.get("or_bars", 6)          # 30 min opening range
        self.buffer = cfg.get("buffer", 0.3)           # breakout buffer (pts)
        self.hold_bars = cfg.get("hold_bars", 8)       # exit after 8 bars (40 min)
        self.sl_mult = cfg.get("sl_mult", 0.7)         # SL = 0.7 × OR range
        self.vol_min = cfg.get("vol_min", 0.8)         # min volume ratio
        self.session_filter = cfg.get("session", "PM") # PM only (hyperopt result)

        # State
        self._or_high = None
        self._or_low = None
        self._or_range = None
        self._or_defined = False
        self._signaled_today = False
        self._pm_bar_count = 0

    def reset(self):
        """Reset at EOD."""
        self._or_high = None
        self._or_low = None
        self._or_range = None
        self._or_defined = False
        self._signaled_today = False
        self._pm_bar_count = 0

    def check(self, df_5m: pd.DataFrame, now: datetime) -> list[dict]:
        """Check for ORB breakout signal. Returns list of signal dicts."""
        if not self.enabled or df_5m is None or len(df_5m) < 20:
            return []

        signals = []
        mins = now.hour * 60 + now.minute

        # Only PM session (13:00 - 14:29)
        if self.session_filter == "PM" and (mins < 13 * 60 or mins >= 14 * 60 + 29):
            return []

        if self._signaled_today:
            return []

        # Get today's PM bars
        df = df_5m.copy()
        df["time"] = pd.to_datetime(df["time"])
        df["mins"] = df["time"].dt.hour * 60 + df["time"].dt.minute
        today = now.date()
        today_pm = df[(df["time"].dt.date == today) & (df["mins"] >= 13 * 60) & (df["mins"] < 14 * 60 + 30)]

        if len(today_pm) < self.or_bars:
            return []

        # Define OR from first N bars of PM
        if not self._or_defined:
            or_bars = today_pm.head(self.or_bars)
            self._or_high = float(or_bars["high"].max())
            self._or_low = float(or_bars["low"].min())
            self._or_range = self._or_high - self._or_low
            self._or_defined = True

            if self._or_range < 0.5:
                # Too tight range — skip
                self._signaled_today = True
                return []

        # Only check bars after OR period
        post_or = today_pm.iloc[self.or_bars:]
        if len(post_or) == 0:
            return []

        # Don't scan too late
        if len(post_or) > 8:
            return []

        # Volume SMA for comparison
        vol_sma = df["volume"].tail(20).mean()
        if vol_sma <= 0:
            vol_sma = 1

        # Check latest bar for breakout
        last_bar = post_or.iloc[-1]
        c = float(last_bar["close"])
        v = float(last_bar["volume"])
        vol_ratio = v / vol_sma

        if vol_ratio < self.vol_min:
            return []

        direction = 0
        signal_type = ""

        if c > self._or_high + self.buffer:
            direction = 1
            signal_type = "ORB_BREAKOUT_UP"
        elif c < self._or_low - self.buffer:
            direction = -1
            signal_type = "ORB_BREAKDOWN_DN"

        if direction == 0:
            return []

        self._signaled_today = True

        sl_dist = self.sl_mult * self._or_range
        sl = c - direction * sl_dist

        signal = {
            "type": signal_type,
            "direction": direction,
            "direction_str": "BUY" if direction == 1 else "SELL",
            "entry": c,
            "sl": sl,
            "sl_dist": sl_dist,
            "or_high": self._or_high,
            "or_low": self._or_low,
            "or_range": self._or_range,
            "vol_ratio": vol_ratio,
            "time": str(last_bar["time"]),
            "session": "PM",
            "label": f"ORB_{signal_type}",
        }

        signals.append(signal)

        # Log
        self._log_signal(signal)

        # Telegram alert
        if self.notifier:
            dir_icon = "\U0001f7e2" if direction == 1 else "\U0001f534"
            mode_tag = " (SHADOW)" if self.shadow_mode else ""
            msg = (
                f"{dir_icon} <b>[ORB{mode_tag}] {signal['direction_str']} — PM</b>\n"
                f"{'─' * 24}\n"
                f"Opening Range: {self._or_low:.1f} — {self._or_high:.1f} "
                f"(range={self._or_range:.1f})\n"
                f"Breakout: {c:.1f} | Vol: {vol_ratio:.1f}x\n"
                f"SL: {sl:.1f} ({self.sl_mult}×OR = {sl_dist:.1f})\n"
                f"Hold: {self.hold_bars} bars (40 min)\n"
                f"<i>Hyperopt: WR 71%, Net +7.21, PF 29.6</i>"
            )
            self.notifier.send(msg)

        return signals

    def _log_signal(self, signal: dict):
        """Append signal to JSONL log."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(signal, default=str) + "\n")
        except Exception:
            pass
