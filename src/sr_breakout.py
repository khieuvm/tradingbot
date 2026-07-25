"""
S/R Breakout + Volume — Support/Resistance breakout & rejection signals
========================================================================
Monitors key S/R levels (PDH/PDL/PDC, Opening Range, Day Open) and detects:
1. Wick Rejection at resistance → SHORT (AM: WR 85.7%, PF 18.24)
2. Volume Breakout UP (HiVol→HiVol) → LONG (WR 77.8%, PF 9.20)
3. Volume Breakdown DN (Vol≥1.5x) → SHORT (WR 75.0%, PF 37.67)

Research (26 days, 329 S/R events):
  - Volume is the #1 factor separating real vs fake breakouts
  - AM = rejection territory, PM = breakout territory
  - HiVol→HiVol = real breakout, HiVol→LoVol = fake breakout
  - Wick rejection (upper wick > body at resistance) = strongest reversal signal

Usage in scanner.py:
    sr_breakout = SRBreakoutSignal(notifier=notifier)
    # Each scan cycle:
    signals = sr_breakout.check(df_5m, current_time)
    # Returns list of signal dicts or empty list
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.strategy_config import get_config

VN_TZ = timezone(timedelta(hours=7))

# S/R touch detection thresholds
TOUCH_DIST = 1.5     # within 1.5 pts of level = "touching"
BREAK_CONFIRM = 0.5  # close beyond level + 0.5 = confirmed breakout
DEDUP_MINUTES = 20   # min minutes between signals at same level


class SRBreakoutSignal:
    """Detect S/R breakout and rejection signals with volume confirmation."""

    def __init__(self, notifier=None):
        cfg = get_config().get("sr_breakout", {})
        self.enabled = cfg.get("enabled", False)
        self.shadow_mode = cfg.get("shadow_mode", True)
        self.notifier = notifier
        self._log_path = Path("logs/sr_breakout_shadow.jsonl")

        # Thresholds from config
        self.vol_surge = cfg.get("vol_surge_mult", 1.2)
        self.vol_spike = cfg.get("vol_spike_mult", 1.5)
        self.wick_min_pct = cfg.get("wick_min_pct", 0.50)

        # StochRSI extreme exit thresholds
        self.stochrsi_ob = cfg.get("stochrsi_ob", 80)
        self.stochrsi_os = cfg.get("stochrsi_os", 10)
        self.stochrsi_dur_min = cfg.get("stochrsi_dur_min", 3)
        self.stochrsi_dur_max = cfg.get("stochrsi_dur_max", 10)

        self._reset_state()

    def _reset_state(self):
        self._today_date = ""
        self._levels = {}        # {"PDH": price, "PDL": price, ...}
        self._levels_set = False
        self._vol_mean = 0.0     # today's avg volume (running)
        self._recent_signals = []  # dedup tracking
        self._prev_day_data = None  # (high, low, close) from previous day
        # StochRSI tracking
        self._in_ob = False      # currently in overbought zone
        self._ob_start_bar = 0   # bar count when entered OB
        self._in_os = False
        self._os_start_bar = 0
        self._bar_count = 0      # running bar counter for today
        self._stochrsi_signaled_ob = False  # already fired OB exit signal today
        self._stochrsi_signaled_os = False

    def reset(self):
        """Reset for new trading day."""
        self._reset_state()

    def _set_levels(self, df_5m: pd.DataFrame, today_date: str):
        """Compute S/R levels for today from 5m data."""
        if self._levels_set and self._today_date == today_date:
            return

        self._today_date = today_date
        df = df_5m.copy()
        df["time"] = pd.to_datetime(df["time"])
        df["date"] = df["time"].dt.date.astype(str)

        # Get previous day's data
        dates = sorted(df["date"].unique())
        today_idx = None
        for i, d in enumerate(dates):
            if d == today_date:
                today_idx = i
                break

        if today_idx is not None and today_idx > 0:
            prev_date = dates[today_idx - 1]
            prev_df = df[df["date"] == prev_date]
            if len(prev_df) > 0:
                self._levels["PDH"] = float(prev_df["high"].max())
                self._levels["PDL"] = float(prev_df["low"].min())
                self._levels["PDC"] = float(prev_df.iloc[-1]["close"])

        # Today's data
        today_df = df[df["date"] == today_date].sort_values("time")
        if len(today_df) < 3:
            return

        # Opening Range (first 3 bars = 15 min)
        mins = pd.to_datetime(today_df["time"]).dt.hour * 60 + pd.to_datetime(today_df["time"]).dt.minute
        am_df = today_df[mins < 11 * 60 + 30]
        if len(am_df) >= 3:
            or_bars = am_df.head(3)
            self._levels["OR_H"] = float(or_bars["high"].max())
            self._levels["OR_L"] = float(or_bars["low"].min())

        # Day open
        self._levels["DAY_OPEN"] = float(today_df.iloc[0]["open"])

        self._levels_set = True

    def _compute_vol_mean(self, today_bars: pd.DataFrame) -> float:
        """Compute average volume for today's bars so far."""
        if len(today_bars) < 3:
            return 0.0
        return float(today_bars["volume"].mean())

    def _is_wick_rejection_down(self, o, h, l, c) -> tuple[bool, float]:
        """Check if bar has upper wick rejection (failed to break above).
        Returns (is_rejection, wick_pct)."""
        total_range = h - l
        if total_range <= 0.1:
            return False, 0.0
        upper_wick = h - max(c, o)
        wick_pct = upper_wick / total_range
        body = abs(c - o)
        # Wick rejection = upper wick > body AND wick >= min threshold
        return (upper_wick > body and wick_pct >= self.wick_min_pct), wick_pct

    def _is_wick_rejection_up(self, o, h, l, c) -> tuple[bool, float]:
        """Check if bar has lower wick rejection (bounced from below).
        Returns (is_rejection, wick_pct)."""
        total_range = h - l
        if total_range <= 0.1:
            return False, 0.0
        lower_wick = min(c, o) - l
        wick_pct = lower_wick / total_range
        body = abs(c - o)
        return (lower_wick > body and wick_pct >= self.wick_min_pct), wick_pct

    def _dedup_ok(self, level_name: str, signal_time: datetime) -> bool:
        """Check dedup: no signal at same level within DEDUP_MINUTES."""
        cutoff = signal_time - timedelta(minutes=DEDUP_MINUTES)
        for sig in self._recent_signals:
            if sig["level"] == level_name and sig["time"] > cutoff:
                return False
        return True

    def check(self, df_5m: pd.DataFrame, current_time: datetime) -> list[dict]:
        """
        Check for S/R breakout/rejection signals.
        Returns list of signal dicts (may be empty).
        Each signal: {type, direction, level, level_price, entry, vol_ratio, ...}
        """
        if df_5m is None or len(df_5m) < 10:
            return []

        today_str = current_time.strftime("%Y-%m-%d")
        self._set_levels(df_5m, today_str)

        if not self._levels:
            return []

        # Get today's bars
        df = df_5m.copy()
        df["time"] = pd.to_datetime(df["time"])
        df["date"] = df["time"].dt.date.astype(str)
        today_df = df[df["date"] == today_str].sort_values("time")

        if len(today_df) < 5:
            return []

        # Current time check (only scan during trading hours)
        mins = current_time.hour * 60 + current_time.minute
        if mins < 9 * 60 + 15:  # too early, OR not formed yet
            return []
        if mins >= 14 * 60 + 25:  # too late
            return []

        session = "AM" if mins < 11 * 60 + 30 else "PM"
        vol_mean = self._compute_vol_mean(today_df)
        if vol_mean <= 0:
            return []

        # Analyze the last 2 bars (current + previous for confirmation)
        signals = []
        bar = today_df.iloc[-1]  # most recent bar
        prev_bar = today_df.iloc[-2] if len(today_df) >= 2 else None

        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        o = float(bar["open"])
        vol = float(bar["volume"])
        vol_ratio = vol / vol_mean

        # Next bar volume (use prev bar for "previous vol" context)
        prev_vol_ratio = float(prev_bar["volume"]) / vol_mean if prev_bar is not None else 1.0

        bar_time = pd.to_datetime(bar["time"])

        for level_name, level_price in self._levels.items():
            if not self._dedup_ok(level_name, current_time):
                continue

            # Skip levels too far from current price
            if abs(c - level_price) > 10:
                continue

            # ── Test from BELOW (resistance) ────────────────────────
            if h >= level_price - TOUCH_DIST and l < level_price and o < level_price:

                # --- WICK REJECT DOWN (AM dominant) ---
                is_wick, wick_pct = self._is_wick_rejection_down(o, h, l, c)
                if is_wick and c < level_price:
                    # Research: AM wick reject WR 85.7%, PF 18.24
                    # Best when: AM session, vol not too high (<1.5x)
                    sig = {
                        "type": "WICK_REJECT",
                        "direction": -1,
                        "direction_str": "SHORT",
                        "level": level_name,
                        "level_price": level_price,
                        "entry": c,
                        "vol_ratio": round(vol_ratio, 2),
                        "wick_pct": round(wick_pct * 100, 1),
                        "session": session,
                        "time": bar_time,
                        "label": f"SR_WickReject_{level_name}",
                    }
                    # Quality score based on research
                    quality = 0
                    if session == "AM":
                        quality += 2  # AM wick reject WR 85.7%
                    if vol_ratio < 1.5:
                        quality += 1  # low vol rejection = real
                    if wick_pct >= 0.65:
                        quality += 1  # wick 65-80% best
                    sig["quality"] = quality
                    signals.append(sig)

                # --- BREAKOUT UP + Volume ---
                elif c > level_price + BREAK_CONFIRM:
                    # Volume confirmation check
                    hi_vol = vol_ratio >= self.vol_surge
                    hi_vol_sustained = hi_vol and prev_vol_ratio >= self.vol_surge

                    if hi_vol:
                        sig = {
                            "type": "VOL_BREAKOUT" if hi_vol_sustained else "BREAK_UP",
                            "direction": 1,
                            "direction_str": "LONG",
                            "level": level_name,
                            "level_price": level_price,
                            "entry": c,
                            "vol_ratio": round(vol_ratio, 2),
                            "prev_vol_ratio": round(prev_vol_ratio, 2),
                            "sustained": hi_vol_sustained,
                            "session": session,
                            "time": bar_time,
                            "label": f"SR_{'VolBreak' if hi_vol_sustained else 'Break'}_{level_name}",
                        }
                        quality = 0
                        if hi_vol_sustained:
                            quality += 3  # HiVol→HiVol WR 77.8%
                        elif vol_ratio >= self.vol_spike:
                            quality += 2  # Vol spike
                        else:
                            quality += 1
                        if session == "PM":
                            quality += 1  # PM breakout better
                        sig["quality"] = quality
                        signals.append(sig)

            # ── Test from ABOVE (support) ───────────────────────────
            elif l <= level_price + TOUCH_DIST and h > level_price and o > level_price:

                # --- BREAKDOWN DN + Volume ---
                if c < level_price - BREAK_CONFIRM:
                    hi_vol = vol_ratio >= self.vol_surge

                    if hi_vol:
                        hi_vol_sustained = prev_vol_ratio >= self.vol_surge
                        sig = {
                            "type": "VOL_BREAKDOWN" if hi_vol_sustained else "BREAK_DN",
                            "direction": -1,
                            "direction_str": "SHORT",
                            "level": level_name,
                            "level_price": level_price,
                            "entry": c,
                            "vol_ratio": round(vol_ratio, 2),
                            "prev_vol_ratio": round(prev_vol_ratio, 2),
                            "sustained": hi_vol_sustained,
                            "session": session,
                            "time": bar_time,
                            "label": f"SR_{'VolBreak' if hi_vol_sustained else 'Break'}DN_{level_name}",
                        }
                        quality = 0
                        if vol_ratio >= self.vol_spike:
                            quality += 2  # Vol≥1.5x WR 75%
                        else:
                            quality += 1
                        if hi_vol_sustained:
                            quality += 1
                        sig["quality"] = quality
                        signals.append(sig)

                # --- WICK REJECT UP (bounce at support) ---
                else:
                    is_wick, wick_pct = self._is_wick_rejection_up(o, h, l, c)
                    if is_wick and c > level_price:
                        sig = {
                            "type": "WICK_BOUNCE",
                            "direction": 1,
                            "direction_str": "LONG",
                            "level": level_name,
                            "level_price": level_price,
                            "entry": c,
                            "vol_ratio": round(vol_ratio, 2),
                            "wick_pct": round(wick_pct * 100, 1),
                            "session": session,
                            "time": bar_time,
                            "label": f"SR_WickBounce_{level_name}",
                        }
                        # Wick bounce at support less reliable than wick reject
                        quality = 1
                        if level_name in ("OR_L",):
                            quality += 1  # OR_L wick bounce WR 71.4%
                        sig["quality"] = quality
                        signals.append(sig)

        # Process signals: dedup, notify, log
        for sig in signals:
            self._recent_signals.append({
                "level": sig["level"],
                "time": current_time,
            })
            self._send_alert(sig)
            self._log_signal(sig)

        # ── StochRSI extreme exit signals ───────────────────────────
        stochrsi_sigs = self._check_stochrsi_exit(today_df, session, current_time)
        for sig in stochrsi_sigs:
            self._send_alert(sig)
            self._log_signal(sig)
        signals.extend(stochrsi_sigs)

        return signals

    def _check_stochrsi_exit(self, today_df: pd.DataFrame, session: str,
                              current_time: datetime) -> list[dict]:
        """
        Detect StochRSI exiting extreme zones after medium duration (3-10 bars).
        Research: Exit OB(>80) after 3-5 bars → SHORT WR 83.3%
                  Exit OS(<10) after 3-5 bars → LONG WR 73.3%
        """
        signals = []
        if len(today_df) < 16:
            return signals

        # Compute StochRSI on today's data (need enough history)
        closes = today_df["close"].astype(float)
        if len(closes) < 20:
            return signals

        stochrsi = ta.stochrsi(closes, length=14, rsi_length=14, k=3, d=3)
        if stochrsi is None or len(stochrsi) < 2:
            return signals

        sk_col = stochrsi.columns[0]

        # Get last 2 bar values
        sk_curr = stochrsi[sk_col].iloc[-1]
        sk_prev = stochrsi[sk_col].iloc[-2]

        if pd.isna(sk_curr) or pd.isna(sk_prev):
            return signals

        close = float(today_df.iloc[-1]["close"])
        bar_time = pd.to_datetime(today_df.iloc[-1]["time"])
        self._bar_count += 1

        # ── Track OB zone entry/exit ────────────────────────────────
        if sk_curr > self.stochrsi_ob and not self._in_ob:
            self._in_ob = True
            self._ob_start_bar = self._bar_count

        elif sk_curr <= self.stochrsi_ob and self._in_ob:
            self._in_ob = False
            duration = self._bar_count - self._ob_start_bar

            if (self.stochrsi_dur_min <= duration <= self.stochrsi_dur_max
                    and not self._stochrsi_signaled_ob):
                self._stochrsi_signaled_ob = True
                quality = 3 if self.stochrsi_dur_min <= duration <= 5 else 2
                signals.append({
                    "type": "STOCHRSI_OB_EXIT",
                    "direction": -1,
                    "direction_str": "SHORT",
                    "level": f"StochRSI_OB{self.stochrsi_ob}",
                    "level_price": close,
                    "entry": close,
                    "vol_ratio": 0,
                    "session": session,
                    "time": bar_time,
                    "label": f"StochRSI_ExitOB_{duration}bars",
                    "quality": quality,
                    "stochrsi_duration": duration,
                })

        # ── Track OS zone entry/exit ────────────────────────────────
        if sk_curr < self.stochrsi_os and not self._in_os:
            self._in_os = True
            self._os_start_bar = self._bar_count

        elif sk_curr >= self.stochrsi_os and self._in_os:
            self._in_os = False
            duration = self._bar_count - self._os_start_bar

            if (self.stochrsi_dur_min <= duration <= self.stochrsi_dur_max
                    and not self._stochrsi_signaled_os):
                self._stochrsi_signaled_os = True
                quality = 3 if self.stochrsi_dur_min <= duration <= 5 else 2
                signals.append({
                    "type": "STOCHRSI_OS_EXIT",
                    "direction": 1,
                    "direction_str": "LONG",
                    "level": f"StochRSI_OS{self.stochrsi_os}",
                    "level_price": close,
                    "entry": close,
                    "vol_ratio": 0,
                    "session": session,
                    "time": bar_time,
                    "label": f"StochRSI_ExitOS_{duration}bars",
                    "quality": quality,
                    "stochrsi_duration": duration,
                })

        return signals

    def should_filter(self, direction_int: int) -> bool:
        """
        Optional: can be used to filter other strategies.
        If we just detected a wick rejection at resistance, block LONG entries.
        """
        # Check if any recent high-quality rejection signal opposes direction
        cutoff = datetime.now(VN_TZ) - timedelta(minutes=15)
        for sig in self._recent_signals:
            if sig["time"] > cutoff:
                # Don't filter based on just dedup data
                pass
        return False  # Don't filter by default, shadow mode

    def _send_alert(self, sig: dict):
        """Send Telegram alert for shadow signal."""
        if not self.notifier:
            return

        dir_emoji = "🔴" if sig["direction"] == -1 else "🟢"
        quality_stars = "⭐" * min(sig.get("quality", 1), 4)

        msg = (
            f"{dir_emoji} SR Signal: {sig['type']}\n"
            f"Direction: {sig['direction_str']}\n"
            f"Level: {sig['level']} @ {sig['level_price']:.1f}\n"
            f"Entry: {sig['entry']:.1f}\n"
            f"Vol ratio: {sig['vol_ratio']:.2f}x\n"
            f"Session: {sig['session']}\n"
            f"Quality: {quality_stars}\n"
            f"Mode: {'SHADOW' if self.shadow_mode else 'LIVE'}"
        )

        if sig["type"] == "WICK_REJECT":
            msg += f"\nWick: {sig['wick_pct']:.0f}%"
        elif sig["type"] in ("VOL_BREAKOUT", "VOL_BREAKDOWN"):
            msg += f"\nSustained vol: {sig.get('prev_vol_ratio', 0):.2f}x → {sig['vol_ratio']:.2f}x"
        elif sig["type"] in ("STOCHRSI_OB_EXIT", "STOCHRSI_OS_EXIT"):
            msg += f"\nDuration in zone: {sig.get('stochrsi_duration', 0)} bars"

        self.notifier.send(msg)
        print(f"  [SR] {sig['label']} {sig['direction_str']} @ {sig['entry']:.1f} "
              f"(vol={sig['vol_ratio']:.2f}x, q={sig.get('quality', 0)}) "
              f"{'[SHADOW]' if self.shadow_mode else '[LIVE]'}")

    def _log_signal(self, sig: dict):
        """Log signal to JSONL."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(VN_TZ).isoformat(),
                "type": sig["type"],
                "direction": sig["direction"],
                "level": sig["level"],
                "level_price": sig["level_price"],
                "entry": sig["entry"],
                "vol_ratio": sig["vol_ratio"],
                "session": sig["session"],
                "quality": sig.get("quality", 0),
                "shadow": self.shadow_mode,
            }
            if "wick_pct" in sig:
                record["wick_pct"] = sig["wick_pct"]
            if "prev_vol_ratio" in sig:
                record["prev_vol_ratio"] = sig["prev_vol_ratio"]

            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
