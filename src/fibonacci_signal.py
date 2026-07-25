"""
Fibonacci Retracement Signals — VN30F1M Intraday
=================================================
Two Fibonacci-based strategies validated on 205 days of VN30 data:

1. FIB+EMA TREND (PM): EMA8<EMA21<EMA50 bearish alignment + price at Fib 23.6%
   retracement → SELL with wide TP (5×ATR)
   - N=28, WR=39.3%, Net=+3.33, PF=2.56, Pts/day=+0.45

2. SESSION FIB ZONES (AM SELL): First 4 bars form OR, price bounces to Fib 78.6%
   of OR range → SELL
   - N=129, WR=58.1%, Net=+1.07, PF=1.34, Pts/day=+0.68

3. FIB MOMENTUM (SELL): Price at Fib 38.2% + RSI drops from >70 to <65 → SELL
   - N=10, WR=80.0%, Net=+3.33, PF=5.19 (rare signal)

Cost: 0.96 pts/trade
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.strategy_config import get_config

VN_TZ = timezone(timedelta(hours=7))


class FibSignal:
    """Fibonacci retracement-based signals for VN30F1M intraday."""

    def __init__(self, notifier=None):
        cfg = get_config().get("fibonacci", {})
        self.enabled = cfg.get("enabled", False)
        self.shadow_mode = cfg.get("shadow_mode", True)
        self.notifier = notifier
        self._log_path = Path("logs/fibonacci_shadow.jsonl")

        # --- Fib+EMA params (PM) ---
        ema_cfg = cfg.get("fib_ema", {})
        self.ema_enabled = ema_cfg.get("enabled", True)
        self.ema_lookback = ema_cfg.get("lookback", 10)
        self.ema_fib_level = ema_cfg.get("fib_level", 0.236)
        self.ema_hold = ema_cfg.get("hold_bars", 6)
        self.ema_sl_mult = ema_cfg.get("sl_mult", 0.8)
        self.ema_tp_mult = ema_cfg.get("tp_mult", 5.0)
        self.ema_session = ema_cfg.get("session", "PM")
        self.ema_tolerance = ema_cfg.get("tolerance", 0.3)  # ATR fraction

        # --- Session Fib Zones params (AM SELL) ---
        zone_cfg = cfg.get("session_fib", {})
        self.zone_enabled = zone_cfg.get("enabled", True)
        self.zone_or_bars = zone_cfg.get("or_bars", 4)
        self.zone_fib_level = zone_cfg.get("fib_level", 0.786)
        self.zone_hold = zone_cfg.get("hold_bars", 15)
        self.zone_sl_mult = zone_cfg.get("sl_mult", 1.0)
        self.zone_tp_ext = zone_cfg.get("tp_extension", 1.272)
        self.zone_session = zone_cfg.get("session", "AM")
        self.zone_direction = zone_cfg.get("direction", "SELL")
        self.zone_min_or = zone_cfg.get("min_or_range", 1.5)

        # --- Fib Momentum params (SELL) ---
        mom_cfg = cfg.get("fib_momentum", {})
        self.mom_enabled = mom_cfg.get("enabled", True)
        self.mom_lookback = mom_cfg.get("lookback", 5)
        self.mom_fib_level = mom_cfg.get("fib_level", 0.382)
        self.mom_rsi_from = mom_cfg.get("rsi_from", 70)
        self.mom_rsi_to = mom_cfg.get("rsi_to", 65)
        self.mom_hold = mom_cfg.get("hold_bars", 8)
        self.mom_sl_mult = mom_cfg.get("sl_mult", 1.2)
        self.mom_tp_mult = mom_cfg.get("tp_mult", 2.0)

        # State tracking per session
        self._ema_signaled_am = False
        self._ema_signaled_pm = False
        self._zone_signaled_am = False
        self._zone_signaled_pm = False
        self._mom_signaled = False

    def reset(self):
        """Reset at EOD."""
        self._ema_signaled_am = False
        self._ema_signaled_pm = False
        self._zone_signaled_am = False
        self._zone_signaled_pm = False
        self._mom_signaled = False

    def check(self, df_5m: pd.DataFrame, now: datetime) -> list[dict]:
        """Check for Fibonacci signals. Returns list of signal dicts."""
        if not self.enabled or df_5m is None or len(df_5m) < 30:
            return []

        signals = []
        mins = now.hour * 60 + now.minute

        # Determine session
        if 9 * 60 <= mins < 11 * 60 + 30:
            session = "AM"
        elif 13 * 60 <= mins < 14 * 60 + 29:
            session = "PM"
        else:
            return []

        df = df_5m.copy()
        df["time"] = pd.to_datetime(df["time"])
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # Compute indicators
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["rsi"] = ta.rsi(df["close"], length=14)
        df["ema8"] = ta.ema(df["close"], length=8)
        df["ema21"] = ta.ema(df["close"], length=21)
        df["ema50"] = ta.ema(df["close"], length=50)

        # ── 1. Fib+EMA Trend ─────────────────────────────────────────────
        if self.ema_enabled and not self._is_ema_signaled(session):
            if self.ema_session == "BOTH" or self.ema_session == session:
                sig = self._check_fib_ema(df, session)
                if sig:
                    signals.append(sig)

        # ── 2. Session Fib Zones ──────────────────────────────────────────
        if self.zone_enabled and not self._is_zone_signaled(session):
            if self.zone_session == "BOTH" or self.zone_session == session:
                sig = self._check_session_fib(df, now, session)
                if sig:
                    signals.append(sig)

        # ── 3. Fib Momentum ──────────────────────────────────────────────
        if self.mom_enabled and not self._mom_signaled:
            sig = self._check_fib_momentum(df, session)
            if sig:
                signals.append(sig)

        # Log & notify
        for sig in signals:
            self._log_signal(sig)
            if self.notifier:
                dir_emoji = "🔴" if sig["direction"] == -1 else "🟢"
                msg = (
                    f"{dir_emoji} FIB {sig['sub_type']} {sig['direction_str']} "
                    f"@ {sig['entry']:.1f}\n"
                    f"SL: {sig['sl']:.1f} | TP: {sig['tp']:.1f}\n"
                    f"ATR: {sig.get('atr', 0):.2f} | Session: {sig['session']}\n"
                    f"{'SHADOW' if self.shadow_mode else 'LIVE'}"
                )
                try:
                    self.notifier.send(msg)
                except Exception:
                    pass

        return signals

    # ── Internal checks ───────────────────────────────────────────────────

    def _check_fib_ema(self, df: pd.DataFrame, session: str) -> dict | None:
        """Fib+EMA Trend Confluence: aligned EMAs + price at Fib retracement."""
        if len(df) < self.ema_lookback + 5:
            return None

        last = df.iloc[-1]
        atr = last.get("atr")
        ema8 = last.get("ema8")
        ema21 = last.get("ema21")
        ema50 = last.get("ema50")
        close = last["close"]

        if any(v is None or (isinstance(v, float) and np.isnan(v))
               for v in [atr, ema8, ema21, ema50]):
            return None
        if atr < 1.0:
            return None

        # Determine trend
        bullish = ema8 > ema21 > ema50
        bearish = ema8 < ema21 < ema50
        if not bullish and not bearish:
            return None

        # Find recent swing
        lb = min(self.ema_lookback, len(df) - 1)
        recent = df.iloc[-lb:]
        recent_high = recent["high"].max()
        recent_low = recent["low"].min()
        swing = recent_high - recent_low
        if swing < 1.5:
            return None

        if bullish:
            fib_price = recent_high - swing * self.ema_fib_level
            if abs(close - fib_price) > atr * self.ema_tolerance:
                return None
            direction = 1
            direction_str = "BUY"
            entry = close
            sl = entry - atr * self.ema_sl_mult
            tp = entry + atr * self.ema_tp_mult
        else:
            fib_price = recent_low + swing * self.ema_fib_level
            if abs(close - fib_price) > atr * self.ema_tolerance:
                return None
            direction = -1
            direction_str = "SELL"
            entry = close
            sl = entry + atr * self.ema_sl_mult
            tp = entry - atr * self.ema_tp_mult

        self._set_ema_signaled(session)

        return {
            "type": "FIBONACCI",
            "sub_type": "FIB_EMA",
            "direction": direction,
            "direction_str": direction_str,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "atr": atr,
            "fib_price": fib_price,
            "fib_level": self.ema_fib_level,
            "swing": swing,
            "ema8": ema8,
            "ema21": ema21,
            "ema50": ema50,
            "session": session,
            "hold_bars": self.ema_hold,
        }

    def _check_session_fib(self, df: pd.DataFrame, now: datetime, session: str) -> dict | None:
        """Session Fib Zones: OR range → Fib retracement entry."""
        today = now.date()
        df_mins = df["time"].dt.hour * 60 + df["time"].dt.minute

        if session == "AM":
            sess_mask = (df["time"].dt.date == today) & (df_mins >= 9*60) & (df_mins < 11*60+30)
        else:
            sess_mask = (df["time"].dt.date == today) & (df_mins >= 13*60) & (df_mins < 14*60+30)

        sess_df = df[sess_mask]
        if len(sess_df) < self.zone_or_bars + 2:
            return None

        # Only check after OR is formed, within first 15 bars
        bars_elapsed = len(sess_df)
        if bars_elapsed < self.zone_or_bars + 1 or bars_elapsed > self.zone_or_bars + 15:
            return None

        # Compute OR
        or_df = sess_df.head(self.zone_or_bars)
        or_high = or_df["high"].max()
        or_low = or_df["low"].min()
        or_range = or_high - or_low
        if or_range < self.zone_min_or:
            return None

        last = sess_df.iloc[-1]
        close = last["close"]
        atr_val = last.get("atr")
        if atr_val is None or np.isnan(atr_val) or atr_val < 1.0:
            return None

        if self.zone_direction == "SELL":
            fib_price = or_low + or_range * self.zone_fib_level
            if close < fib_price - 0.3:
                return None  # Not near fib level
            direction = -1
            direction_str = "SELL"
            entry = close
            sl = entry + or_range * self.zone_sl_mult
            tp = or_low - or_range * (self.zone_tp_ext - 1.0)
        else:
            fib_price = or_high - or_range * self.zone_fib_level
            if close > fib_price + 0.3:
                return None
            direction = 1
            direction_str = "BUY"
            entry = close
            sl = entry - or_range * self.zone_sl_mult
            tp = or_high + or_range * (self.zone_tp_ext - 1.0)

        self._set_zone_signaled(session)

        return {
            "type": "FIBONACCI",
            "sub_type": "SESSION_FIB",
            "direction": direction,
            "direction_str": direction_str,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "atr": atr_val,
            "or_high": or_high,
            "or_low": or_low,
            "or_range": or_range,
            "fib_price": fib_price,
            "fib_level": self.zone_fib_level,
            "session": session,
            "hold_bars": self.zone_hold,
        }

    def _check_fib_momentum(self, df: pd.DataFrame, session: str) -> dict | None:
        """Fib Momentum: RSI drop from overbought + price at Fib level → SELL."""
        if len(df) < self.mom_lookback + 5:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = last.get("atr")
        rsi_now = last.get("rsi")
        rsi_prev = prev.get("rsi")

        if any(v is None or (isinstance(v, float) and np.isnan(v))
               for v in [atr, rsi_now, rsi_prev]):
            return None
        if atr < 1.0:
            return None

        # RSI cross condition: prev >= from, now <= to
        if not (rsi_prev >= self.mom_rsi_from and rsi_now <= self.mom_rsi_to):
            return None

        # Fib level check
        lb = min(self.mom_lookback, len(df) - 1)
        recent = df.iloc[-lb:]
        recent_high = recent["high"].max()
        recent_low = recent["low"].min()
        swing = recent_high - recent_low
        if swing < 2.0:
            return None

        close = last["close"]
        fib_resistance = recent_low + swing * self.mom_fib_level
        if abs(close - fib_resistance) > atr * 0.5:
            return None

        self._mom_signaled = True

        entry = close
        sl = entry + atr * self.mom_sl_mult
        tp = entry - atr * self.mom_tp_mult

        return {
            "type": "FIBONACCI",
            "sub_type": "FIB_MOMENTUM",
            "direction": -1,
            "direction_str": "SELL",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "atr": atr,
            "rsi_now": rsi_now,
            "rsi_prev": rsi_prev,
            "fib_price": fib_resistance,
            "fib_level": self.mom_fib_level,
            "swing": swing,
            "session": session,
            "hold_bars": self.mom_hold,
        }

    # ── State helpers ─────────────────────────────────────────────────────

    def _is_ema_signaled(self, session: str) -> bool:
        return self._ema_signaled_am if session == "AM" else self._ema_signaled_pm

    def _set_ema_signaled(self, session: str):
        if session == "AM":
            self._ema_signaled_am = True
        else:
            self._ema_signaled_pm = True

    def _is_zone_signaled(self, session: str) -> bool:
        return self._zone_signaled_am if session == "AM" else self._zone_signaled_pm

    def _set_zone_signaled(self, session: str):
        if session == "AM":
            self._zone_signaled_am = True
        else:
            self._zone_signaled_pm = True

    def _log_signal(self, sig: dict):
        """Log signal to JSONL."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.now(VN_TZ).isoformat(),
                **{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in sig.items()},
            }
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass
