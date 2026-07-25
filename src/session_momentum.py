"""
Session Momentum — 45-min Opening Pulse Continuation (SELL)
============================================================
Hyperopt-validated: After 45 min, if session drops ≥ 0.3×ATR → SELL continuation
Best configs:
  - pulse>0.3, SK≥60, hold=8:  WR 86%, Net +6.70, PF 36.7 (n=7)
  - pulse>0.3, no filter, hold=6: WR 67%, Net +3.84, PF 7.46 (n=21)

Concept:
  - Measure opening pulse: close[bar 9] - open[bar 0] for each session
  - If pulse/ATR < -threshold → momentum SELL
  - Optional StochRSI filter: SK > 60 (not oversold yet = room to fall)
  - Entry at close of opening period
  - Hold for N bars then exit

Usage in scanner.py:
    sess_mom = SessionMomentumSignal(notifier=notifier)
    # Each scan cycle:
    signals = sess_mom.check(df_5m, current_time)
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.strategy_config import get_config

VN_TZ = timezone(timedelta(hours=7))


class SessionMomentumSignal:
    """Session momentum continuation signal — primarily SELL after down-pulse."""

    def __init__(self, notifier=None):
        cfg = get_config().get("session_momentum", {})
        self.enabled = cfg.get("enabled", False)
        self.shadow_mode = cfg.get("shadow_mode", True)
        self.notifier = notifier
        self._log_path = Path("logs/session_momentum_shadow.jsonl")

        # Hyperopt-validated params
        self.or_period = cfg.get("or_period", 9)         # 9 bars = 45 min
        self.pulse_threshold = cfg.get("pulse_threshold", 0.3)  # min |pulse/ATR|
        self.sk_min = cfg.get("sk_min", 60)              # StochRSI must be above this for SELL
        self.hold_bars = cfg.get("hold_bars", 8)         # hold 8 bars (40 min)
        self.direction_filter = cfg.get("direction", "SELL")  # SELL only (hyperopt)

        # State per session
        self._am_signaled = False
        self._pm_signaled = False

    def reset(self):
        """Reset at EOD."""
        self._am_signaled = False
        self._pm_signaled = False

    def check(self, df_5m: pd.DataFrame, now: datetime) -> list[dict]:
        """Check for session momentum signal. Returns list of signal dicts."""
        if not self.enabled or df_5m is None or len(df_5m) < 30:
            return []

        signals = []
        mins = now.hour * 60 + now.minute

        # Determine current session
        if 9 * 60 <= mins < 11 * 60 + 30:
            session = "AM"
            if self._am_signaled:
                return []
        elif 13 * 60 <= mins < 14 * 60 + 29:
            session = "PM"
            if self._pm_signaled:
                return []
        else:
            return []

        # Get today's session bars
        df = df_5m.copy()
        df["time"] = pd.to_datetime(df["time"])
        df["mins"] = df["time"].dt.hour * 60 + df["time"].dt.minute
        today = now.date()

        if session == "AM":
            sess_df = df[(df["time"].dt.date == today) &
                        (df["mins"] >= 9 * 60) & (df["mins"] < 11 * 60 + 30)]
        else:
            sess_df = df[(df["time"].dt.date == today) &
                        (df["mins"] >= 13 * 60) & (df["mins"] < 14 * 60 + 30)]

        # Need exactly or_period bars to define opening, plus some buffer
        if len(sess_df) < self.or_period + 1:
            return []

        # Only trigger once at the end of opening period (or_period to or_period+2 bars)
        if len(sess_df) > self.or_period + 3:
            return []

        # Opening period
        opening = sess_df.head(self.or_period)
        op_open = float(opening.iloc[0]["open"])
        op_close = float(opening.iloc[-1]["close"])
        op_move = op_close - op_open

        # ATR from recent data
        atr_series = ta.atr(df["high"].astype(float), df["low"].astype(float),
                           df["close"].astype(float), length=14)
        if atr_series is None or len(atr_series) == 0:
            return []

        atr_idx = opening.index[-1]
        atr = float(atr_series.iloc[atr_idx]) if atr_idx < len(atr_series) else None
        if atr is None or np.isnan(atr) or atr <= 0:
            return []

        pulse_atr = op_move / atr

        # StochRSI at end of opening
        stochrsi = ta.stochrsi(df["close"].astype(float), length=14,
                              rsi_length=14, k=3, d=3)
        sk = None
        if stochrsi is not None and len(stochrsi) > atr_idx:
            sk_val = stochrsi.iloc[atr_idx, 0]
            if not np.isnan(sk_val):
                sk = float(sk_val)

        # Direction logic
        if self.direction_filter == "SELL":
            if pulse_atr > -self.pulse_threshold:
                return []  # Not enough down pulse
            if self.sk_min > 0 and (sk is None or sk < self.sk_min):
                return []  # StochRSI too low (already oversold, no room to fall)
            direction = -1
            direction_str = "SELL"
        elif self.direction_filter == "BUY":
            if pulse_atr < self.pulse_threshold:
                return []
            direction = 1
            direction_str = "BUY"
        else:
            if abs(pulse_atr) < self.pulse_threshold:
                return []
            direction = 1 if pulse_atr > 0 else -1
            direction_str = "BUY" if direction == 1 else "SELL"

        # Mark signaled
        if session == "AM":
            self._am_signaled = True
        else:
            self._pm_signaled = True

        entry = op_close  # Entry at close of opening period
        sl = entry + direction * 2.0 * atr  # Wide SL: 2×ATR (opposite direction)

        signal = {
            "type": "SESSION_MOMENTUM",
            "direction": direction,
            "direction_str": direction_str,
            "entry": entry,
            "sl": sl,
            "pulse": op_move,
            "pulse_atr": pulse_atr,
            "atr": atr,
            "sk": sk,
            "or_period_min": self.or_period * 5,
            "hold_bars": self.hold_bars,
            "time": str(opening.iloc[-1]["time"]),
            "session": session,
            "label": f"SESS_MOM_{direction_str}",
        }

        signals.append(signal)

        # Log
        self._log_signal(signal)

        # Telegram
        if self.notifier:
            dir_icon = "\U0001f534" if direction == -1 else "\U0001f7e2"
            mode_tag = " (SHADOW)" if self.shadow_mode else ""
            sk_str = f"SK={sk:.0f}" if sk is not None else "SK=N/A"
            msg = (
                f"{dir_icon} <b>[SessMom{mode_tag}] {direction_str} — {session}</b>\n"
                f"{'─' * 24}\n"
                f"Opening {self.or_period * 5}min: pulse={op_move:+.1f} "
                f"({pulse_atr:+.2f}×ATR)\n"
                f"{sk_str} | ATR={atr:.2f}\n"
                f"Entry: {entry:.1f} | SL: {sl:.1f}\n"
                f"Hold: {self.hold_bars} bars ({self.hold_bars * 5} min)\n"
                f"<i>Hyperopt: WR 67-86%, Net +3.84–6.70</i>"
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
