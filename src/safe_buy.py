"""
Safe BUY Signal — Price-action based BUY with trap avoidance
=============================================================
Detects high-probability BUY setups using Opening Range Breakout + Higher Low
pattern, filtered by previous session context and trap avoidance.

Research (682 days, 5m):
  safe_buy_v2: prev session DN + OR breakout UP + higher low + no wick rejection
    FULL: WR 56.4%, PF 1.76, n=250
    1Y:   WR 71.1%, PF 3.56, n=76   (IMPROVING)
    6M:   WR 74.3%, PF 4.46, n=35   (IMPROVING)

  or_break + higher_low + not_overext (more robust):
    FULL: WR 55.3%, PF 1.64, n=606
    1Y:   WR 60.2%, PF 2.28, n=176
    6M:   WR 58.9%, PF 2.28, n=73

Entry: After S1+S2 (9:15-9:45), enter BUY when conditions met.
Exit: TP=+10pts, SL=S1_low or -1.5×ATR, trail after +5pts.

Usage in scanner.py:
    safe_buy = SafeBuySignal()
    signal = safe_buy.check(df_5m, current_time)
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategy_config import get_config

VN_TZ = timezone(timedelta(hours=7))


class SafeBuySignal:
    """Detect safe BUY entries using price action + trap avoidance."""

    def __init__(self, notifier=None):
        cfg = get_config().get("safe_buy", {})
        self.enabled = cfg.get("enabled", False)
        self.shadow_mode = cfg.get("shadow_mode", True)
        self.variant = cfg.get("variant", "v2")  # "v2" (strict) or "robust"
        self.tp_pts = cfg.get("tp_pts", 10.0)
        self.sl_atr_mult = cfg.get("sl_atr_mult", 1.5)
        self.trail_trigger_pts = cfg.get("trail_trigger_pts", 5.0)
        self.trail_atr_mult = cfg.get("trail_atr_mult", 2.0)

        self.notifier = notifier
        self._log_path = Path("logs/safe_buy_shadow.jsonl")

        self._reset_state()

    def _reset_state(self):
        self._today_date = ""
        self._signals_fired = {}  # {session: True} — max 1 per session
        self._prev_session_ret = None
        self._prev_session_computed = False

    def reset(self):
        self._reset_state()

    def check(self, df_5m: pd.DataFrame, current_time: datetime = None) -> dict | None:
        """Check for safe BUY signal. Called each scan cycle (~60s).

        Returns signal dict or None.
        Signal is emitted once per session, between 9:15-10:45 (AM) or 13:15-14:15 (PM).
        """
        if not self.enabled or df_5m is None or len(df_5m) < 30:
            return None

        if current_time is None:
            current_time = datetime.now(VN_TZ)

        today_str = current_time.strftime("%Y-%m-%d")
        if self._today_date != today_str:
            self._reset_state()
            self._today_date = today_str

        hhmm = current_time.hour * 60 + current_time.minute

        # Determine session
        if 9 * 60 + 15 <= hhmm <= 10 * 60 + 45:
            session = "AM"
        elif 13 * 60 + 15 <= hhmm <= 14 * 60 + 15:
            session = "PM"
        else:
            return None

        # One signal per session
        if session in self._signals_fired:
            return None

        df = df_5m.copy()
        df["time"] = pd.to_datetime(df["time"])
        df["mins"] = df["time"].dt.hour * 60 + df["time"].dt.minute

        today_bars = df[df["time"].dt.strftime("%Y-%m-%d") == today_str].copy()
        if len(today_bars) < 6:
            return None

        # Get session bars
        if session == "AM":
            sess_bars = today_bars[(today_bars["mins"] >= 9 * 60) & (today_bars["mins"] < 11 * 60 + 30)]
        else:
            sess_bars = today_bars[(today_bars["mins"] >= 13 * 60) & (today_bars["mins"] < 14 * 60 + 30)]

        if len(sess_bars) < 4:
            return None

        # S1 (first 3 bars = 15 min) and S2 (next 6 bars = 30 min)
        s1_bars = sess_bars.iloc[:3]
        s2_end_idx = min(9, len(sess_bars))
        s2_bars = sess_bars.iloc[3:s2_end_idx] if s2_end_idx > 3 else pd.DataFrame()

        if s1_bars.empty or s2_bars.empty:
            return None

        # ── Compute features ──
        s1_open = float(s1_bars.iloc[0]["open"])
        s1_close = float(s1_bars.iloc[-1]["close"])
        s1_high = float(s1_bars["high"].max())
        s1_low = float(s1_bars["low"].min())
        s1_ret = s1_close - s1_open

        s2_close = float(s2_bars.iloc[-1]["close"])
        s2_low = float(s2_bars["low"].min())
        s2_high = float(s2_bars["high"].max())
        s2_vol = float(s2_bars["volume"].sum())
        s1_vol = float(s1_bars["volume"].sum())

        # Opening Range
        or_high = s1_high
        or_low = s1_low
        or_breakout_up = s2_high > or_high

        # Higher low
        higher_low = s2_low > s1_low

        # S1 wick rejection (upper wick > body = selling pressure)
        s1_body = abs(s1_close - s1_open)
        s1_upper_wick = s1_high - max(s1_close, s1_open)
        no_wick_rejection = s1_upper_wick <= s1_body

        # ATR
        try:
            import pandas_ta as ta
            atr_s = ta.atr(df["high"], df["low"], df["close"], length=14)
            atr = float(atr_s.iloc[-1]) if atr_s is not None and not atr_s.empty else 3.5
        except Exception:
            atr = 3.5

        # Not overextended
        not_overext = s1_ret <= 2 * atr

        # Previous session return
        prev_ret = self._get_prev_session_return(df, today_str, session)
        prev_down = prev_ret is not None and prev_ret < 0

        # ── Apply filter based on variant ──
        if self.variant == "v2":
            # safe_buy_v2: prev DN + OR break + higher low + no wick
            # WR 74.3% (6M), PF 4.46
            signal_ok = (
                prev_down and
                or_breakout_up and
                higher_low and
                no_wick_rejection
            )
            variant_label = "Safe BUY v2"
        else:
            # robust: OR break + higher low + not overextended
            # WR 58.9% (6M), PF 2.28, n=73 (more trades)
            signal_ok = (
                or_breakout_up and
                higher_low and
                not_overext
            )
            variant_label = "Safe BUY Robust"

        if not signal_ok:
            return None

        # Signal detected!
        self._signals_fired[session] = True

        entry_price = float(sess_bars.iloc[-1]["close"])  # enter at current price
        sl = min(s1_low, entry_price - self.sl_atr_mult * atr)
        tp = entry_price + self.tp_pts

        signal = {
            "direction": "BUY",
            "direction_int": 1,
            "combo": "SAFE_BUY",
            "variant": variant_label,
            "session": session,
            "entry": entry_price,
            "sl": sl,
            "tp": tp,
            "atr": atr,
            "or_high": or_high,
            "or_low": or_low,
            "s1_ret": s1_ret,
            "s2_close": s2_close,
            "higher_low": higher_low,
            "no_wick": no_wick_rejection,
            "prev_ret": prev_ret,
            "time": current_time.strftime("%H:%M:%S"),
        }

        # Log
        self._log_signal(current_time, signal)

        # Print
        print(f"  [SAFE_BUY] {variant_label} detected [{session}]")
        print(f"    OR break: {or_high:.1f} | HL: S2_low={s2_low:.1f} > S1_low={s1_low:.1f}")
        print(f"    Prev ret: {prev_ret:+.1f} | S1 wick ok: {no_wick_rejection}")
        print(f"    Entry: {entry_price:.1f} | SL: {sl:.1f} | TP: {tp:.1f}")

        # Telegram
        if self.notifier:
            dir_icon = "\U0001f7e2"
            msg = (
                f"{dir_icon} <b>[{variant_label}] BUY — VN30F1M</b>\n"
                f"{'─' * 24}\n"
                f"Session: {session} | ATR: {atr:.2f}\n"
                f"OR Breakout: {or_high:.1f} ✓ | Higher Low: {s2_low:.1f}>{s1_low:.1f} ✓\n"
                f"Prev session: {prev_ret:+.1f} pts | Wick ok: {'✓' if no_wick_rejection else '✗'}\n"
                f"\n<b>Entry:</b> <code>{entry_price:,.1f}</code>\n"
                f"SL: <code>{sl:,.1f}</code> | TP: <code>{tp:,.1f}</code>\n"
                f"R:R = {self.tp_pts / (entry_price - sl):.1f}:1\n"
            )
            if self.shadow_mode:
                msg += "<i>Shadow mode — not trading</i>"
            self.notifier.send(msg)

        return signal

    def _get_prev_session_return(self, df: pd.DataFrame, today_str: str, session: str) -> float | None:
        """Get previous session return (pts)."""
        if self._prev_session_computed and self._prev_session_ret is not None:
            return self._prev_session_ret

        df_before = df[df["time"].dt.strftime("%Y-%m-%d") <= today_str].copy()
        if len(df_before) < 20:
            return None

        df_before["session"] = "AM"
        df_before.loc[df_before["mins"] >= 13 * 60, "session"] = "PM"
        df_before["date_str"] = df_before["time"].dt.strftime("%Y-%m-%d")

        # Find previous session
        sessions = []
        for (d, s), grp in df_before.groupby(["date_str", "session"]):
            if len(grp) >= 3:
                sess_open = float(grp.iloc[0]["open"])
                sess_close = float(grp.iloc[-1]["close"])
                sessions.append({"date": d, "session": s, "return": sess_close - sess_open})

        if len(sessions) < 2:
            return None

        # Current session index
        current_key = (today_str, session)
        # Find the last session before current
        prev = None
        for s in reversed(sessions):
            if (s["date"], s["session"]) < current_key:
                prev = s
                break

        if prev is None:
            return None

        self._prev_session_ret = prev["return"]
        self._prev_session_computed = True
        return self._prev_session_ret

    def _log_signal(self, current_time: datetime, signal: dict):
        """Log signal to JSONL."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"timestamp": current_time.isoformat(), **signal}
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  [SAFE_BUY] Log error: {e}")
