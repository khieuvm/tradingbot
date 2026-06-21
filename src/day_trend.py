"""
Day Trend Predictor — Opening Pulse (9:00-9:30) + Retracement Check (9:45)
==========================================================================
Rule-based predictor using opening 30-min pulse magnitude, direction,
day-of-week effects, and retracement speed to determine intraday trend bias.

Dual function:
  1. Bias filter: block CB/NR4 entries opposing predicted direction
  2. Independent signal: entry at 9:45 when pulse is large + retrace shallow
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.strategy_config import get_config

VN_TZ = timezone(timedelta(hours=7))
DOW_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

# Historical probabilities from 680 trading days (Sep 2023 - Jun 2026)
# P(day ends in pulse direction | pulse direction, day of week)
HIST_DOW_PROB = {
    # dow: {UP: P(day UP | pulse UP), DOWN: P(day DOWN | pulse DOWN)}
    0: {"UP": 63.3, "DOWN": 70.3},   # Mon — DOWN strongest
    1: {"UP": 68.6, "DOWN": 54.5},   # Tue — UP better
    2: {"UP": 73.1, "DOWN": 63.2},   # Wed — UP strongest
    3: {"UP": 67.2, "DOWN": 65.2},   # Thu — balanced
    4: {"UP": 61.7, "DOWN": 60.9},   # Fri — weakest
}

# P(day continues) by pulse magnitude bucket
HIST_SIZE_PROB = {
    "SMALL":  {"hr_am": 57.6, "hr_day": 51.9},   # |pulse| < 1.4
    "MEDIUM": {"hr_am": 73.9, "hr_day": 62.2},   # 1.4 - 3.5
    "LARGE":  {"hr_am": 86.5, "hr_day": 79.6},   # > 3.5
}

# Overall stats (unused currently, kept for reference)
SIZE_THRESHOLDS = (1.4, 3.5)


class DayTrendPredictor:

    def __init__(self, notifier=None):
        cfg = get_config().get("day_trend", {})
        self.enabled = cfg.get("enabled", False)
        self.shadow_mode = cfg.get("shadow_mode", True)
        self.enable_bias_filter = cfg.get("enable_bias_filter", True)
        self.enable_signal = cfg.get("enable_signal", True)

        pulse_cfg = cfg.get("pulse", {})
        self.pulse_min_pts = pulse_cfg.get("min_pts", 3.5)
        self.pulse_bar_hhmm = pulse_cfg.get("compute_bar_hhmm", 565)

        ret_cfg = cfg.get("retracement", {})
        self.retrace_check_hhmm = ret_cfg.get("check_at_hhmm", 585)
        self.threshold_high_pct = ret_cfg.get("threshold_high_pct", 30)
        self.threshold_medium_pct = ret_cfg.get("threshold_medium_pct", 50)

        self.dow_adjustments = cfg.get("dow_adjustments", {})

        sig_cfg = cfg.get("signal", {})
        self.tp_mult = sig_cfg.get("tp_mult", 2.0)

        pm_cfg = cfg.get("pm_reversal", {})
        self.pm_reversal_enabled = pm_cfg.get("enabled", True)

        self.notifier = notifier
        self._log_path = Path("logs/day_trend_shadow.jsonl")

        self._reset_state()

    def _reset_state(self):
        self._state = "WAITING"
        self._today_date = ""
        self._day_open = 0.0
        self._pulse_close = 0.0
        self._pulse_pts = 0.0
        self._pulse_direction = 0
        self._retrace_pct = 0.0
        self._confidence = "LOW"
        self._signal_fired = False
        self._pm_alert_sent = False
        self._entry_price = 0.0
        self._am_max_adverse = 0.0

    def reset(self):
        self._reset_state()

    def update(self, df_5m: pd.DataFrame, current_time: datetime = None):
        if not self.enabled or df_5m is None or len(df_5m) < 6:
            return

        if current_time is None:
            current_time = datetime.now(VN_TZ)

        today_str = current_time.strftime("%Y-%m-%d")
        if self._today_date != today_str:
            self._reset_state()
            self._today_date = today_str

        hhmm = current_time.hour * 60 + current_time.minute

        if self._state == "WAITING" and hhmm >= self.pulse_bar_hhmm + 5:
            self._compute_pulse(df_5m, current_time)

        if self._state == "PULSE_COMPUTED" and hhmm >= self.retrace_check_hhmm:
            self._check_retracement(df_5m, current_time)

    def _compute_pulse(self, df_5m: pd.DataFrame, current_time: datetime):
        today_str = current_time.strftime("%Y-%m-%d")
        today_bars = df_5m[df_5m["time"].astype(str).str.startswith(today_str)].copy()
        if len(today_bars) < 6:
            return

        self._day_open = float(today_bars.iloc[0]["open"])

        pulse_bar_time = self.pulse_bar_hhmm
        pulse_bar = None
        for i in range(len(today_bars)):
            t = pd.to_datetime(today_bars.iloc[i]["time"])
            bar_hhmm = t.hour * 60 + t.minute
            if bar_hhmm >= pulse_bar_time:
                pulse_bar = today_bars.iloc[i]
                break

        if pulse_bar is None:
            pulse_bar = today_bars.iloc[min(5, len(today_bars) - 1)]

        self._pulse_close = float(pulse_bar["close"])
        self._pulse_pts = self._pulse_close - self._day_open
        self._pulse_direction = 1 if self._pulse_pts > 0 else (-1 if self._pulse_pts < 0 else 0)
        self._state = "PULSE_COMPUTED"

        self._send_pulse_alert(current_time)

    def _check_retracement(self, df_5m: pd.DataFrame, current_time: datetime):
        if abs(self._pulse_pts) < 0.1:
            self._confidence = "LOW"
            self._state = "PREDICTION_ACTIVE"
            return

        today_str = current_time.strftime("%Y-%m-%d")
        today_bars = df_5m[df_5m["time"].astype(str).str.startswith(today_str)].copy()

        pulse_bar_hhmm = self.pulse_bar_hhmm
        post_pulse_bars = []
        for i in range(len(today_bars)):
            t = pd.to_datetime(today_bars.iloc[i]["time"])
            bar_hhmm = t.hour * 60 + t.minute
            if bar_hhmm > pulse_bar_hhmm:
                post_pulse_bars.append(today_bars.iloc[i])

        if not post_pulse_bars:
            self._confidence = "LOW"
            self._state = "PREDICTION_ACTIVE"
            self._send_retrace_alert(current_time)
            return

        max_adverse = 0.0
        for bar in post_pulse_bars:
            if self._pulse_direction == 1:
                adverse = self._pulse_close - float(bar["low"])
            else:
                adverse = float(bar["high"]) - self._pulse_close
            max_adverse = max(max_adverse, adverse)

        self._retrace_pct = (max_adverse / abs(self._pulse_pts)) * 100 if abs(self._pulse_pts) > 0 else 0
        self._am_max_adverse = max_adverse

        base_confidence = self._calc_base_confidence()
        self._confidence = self._apply_dow_adjustment(base_confidence, current_time)
        self._state = "PREDICTION_ACTIVE"

        self._entry_price = float(post_pulse_bars[-1]["close"])
        self._send_retrace_alert(current_time)

    def _calc_base_confidence(self) -> str:
        if abs(self._pulse_pts) >= self.pulse_min_pts:
            if self._retrace_pct < self.threshold_high_pct:
                return "HIGH"
            elif self._retrace_pct < self.threshold_medium_pct:
                return "MEDIUM"
        return "LOW"

    def _apply_dow_adjustment(self, base: str, current_time: datetime) -> str:
        dow = current_time.weekday()
        adjustments = self.dow_adjustments.get(str(dow), self.dow_adjustments.get(dow, {}))
        if not adjustments:
            return base

        dir_str = "UP" if self._pulse_direction == 1 else "DOWN"
        adj = adjustments.get(dir_str, 0)

        tiers = ["LOW", "MEDIUM", "HIGH"]
        idx = tiers.index(base)
        new_idx = max(0, min(2, idx + adj))
        return tiers[new_idx]

    def get_prediction(self) -> dict | None:
        if self._state != "PREDICTION_ACTIVE":
            return None
        return {
            "direction": self._pulse_direction,
            "direction_str": "BUY" if self._pulse_direction == 1 else "SELL",
            "confidence": self._confidence,
            "pulse_pts": self._pulse_pts,
            "retrace_pct": self._retrace_pct,
            "state": self._state,
            "action": self._get_action(),
        }

    def _get_action(self) -> str:
        if self._confidence == "HIGH":
            return "BUY_ONLY" if self._pulse_direction == 1 else "SELL_ONLY"
        elif self._confidence == "MEDIUM":
            return "BUY_ONLY" if self._pulse_direction == 1 else "SELL_ONLY"
        return "NEUTRAL"

    def should_filter(self, direction: int) -> bool:
        if not self.enabled or not self.enable_bias_filter:
            return False
        if self._state != "PREDICTION_ACTIVE":
            return False
        if self._confidence == "LOW":
            return False
        return direction != self._pulse_direction

    def get_signal(self) -> dict | None:
        if not self.enabled or not self.enable_signal:
            return None
        if self._state != "PREDICTION_ACTIVE":
            return None
        if self._signal_fired:
            return None
        if self._confidence != "HIGH":
            return None
        if self._pulse_direction == 0:
            return None
        if abs(self._pulse_pts) < self.pulse_min_pts:
            return None

        self._signal_fired = True
        direction = self._pulse_direction
        entry = self._entry_price if self._entry_price > 0 else self._pulse_close
        sl = self._day_open
        tp = entry + direction * self.tp_mult * abs(self._pulse_pts)

        signal = {
            "combo": "DT",
            "direction": direction,
            "direction_str": "BUY" if direction == 1 else "SELL",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "atr": abs(self._pulse_pts),
            "session": "AM",
            "confidence": self._confidence,
            "pulse_pts": self._pulse_pts,
            "retrace_pct": self._retrace_pct,
            "timeframe": "5m",
        }

        self._log_shadow_signal(signal)
        return signal

    def check_pm_reversal(self, df_5m: pd.DataFrame) -> dict | None:
        if not self.enabled or not self.pm_reversal_enabled:
            return None
        if self._state != "PREDICTION_ACTIVE":
            return None
        if self._pm_alert_sent:
            return None
        return self._check_pm_reversal_internal(df_5m)

    def _check_pm_reversal_internal(self, df_5m: pd.DataFrame) -> dict | None:
        if self._pm_alert_sent or self._pulse_direction == 0:
            return None

        today_str = self._today_date
        today_bars = df_5m[df_5m["time"].astype(str).str.startswith(today_str)].copy()

        pm_bars = []
        for i in range(len(today_bars)):
            t = pd.to_datetime(today_bars.iloc[i]["time"])
            if t.hour >= 13:
                pm_bars.append(today_bars.iloc[i])

        if not pm_bars:
            return None

        pm_max_adverse = 0.0
        for bar in pm_bars:
            if self._pulse_direction == 1:
                adverse = self._pulse_close - float(bar["low"])
            else:
                adverse = float(bar["high"]) - self._pulse_close
            pm_max_adverse = max(pm_max_adverse, adverse)

        pm_retrace_pct = (pm_max_adverse / abs(self._pulse_pts)) * 100 if abs(self._pulse_pts) > 0 else 0

        if pm_max_adverse > self._am_max_adverse and pm_retrace_pct > 100:
            self._pm_alert_sent = True
            msg = (
                f"\u26a0\ufe0f <b>PM REVERSAL ALERT</b>\n"
                f"{'─' * 20}\n"
                f"AM pulse: {self._pulse_pts:+.1f} pts "
                f"({'BUY' if self._pulse_direction == 1 else 'SELL'})\n"
                f"PM retrace: {pm_retrace_pct:.0f}% (exceeds AM)\n"
                f"<b>Bias filter suspended</b>"
            )
            if self.notifier:
                self.notifier.send(msg)
            self._confidence = "LOW"
            return {"message": msg, "pm_retrace_pct": pm_retrace_pct}

        return None

    def _get_size_bucket(self) -> str:
        mag = abs(self._pulse_pts)
        if mag < SIZE_THRESHOLDS[0]:
            return "SMALL"
        elif mag < SIZE_THRESHOLDS[1]:
            return "MEDIUM"
        return "LARGE"

    def _lookup_hist_prob(self, dow: int) -> dict:
        """Look up historical probabilities for current pulse."""
        dir_str = "UP" if self._pulse_direction == 1 else "DOWN"
        bucket = self._get_size_bucket()

        dow_prob = HIST_DOW_PROB.get(dow, {}).get(dir_str, 65.0)
        size_prob = HIST_SIZE_PROB.get(bucket, {})

        return {
            "dow_prob": dow_prob,
            "size_hr_am": size_prob.get("hr_am", 65.0),
            "size_hr_day": size_prob.get("hr_day", 65.0),
            "bucket": bucket,
            "dir_str": dir_str,
        }

    def _send_pulse_alert(self, current_time: datetime):
        if not self.notifier:
            return
        dow = current_time.weekday()
        dow_name = DOW_NAMES.get(dow, "?")
        direction_str = "BUY" if self._pulse_direction == 1 else "SELL"
        icon = "\U0001f7e2" if self._pulse_direction == 1 else "\U0001f534"
        mag = abs(self._pulse_pts)

        hist = self._lookup_hist_prob(dow)
        bucket = hist["bucket"]

        if bucket == "LARGE":
            strength = "STRONG"
            strength_icon = "\U0001f4aa"
            guidance = (
                f"Large pulse + {dow_name} {hist['dir_str']}: "
                f"<b>{hist['dow_prob']:.0f}%</b> day continues\n"
                f"If retrace &lt;30% by 9:45 → trend day (100% hist)\n"
                f"<b>Action:</b> Prepare to follow {direction_str}"
            )
        elif bucket == "MEDIUM":
            strength = "MODERATE"
            strength_icon = "\u2796"
            guidance = (
                f"Medium pulse: {hist['size_hr_day']:.0f}% day continues\n"
                f"{dow_name} {hist['dir_str']}: {hist['dow_prob']:.0f}%\n"
                f"<b>Action:</b> Wait for 9:45 retrace confirmation"
            )
        else:
            strength = "WEAK"
            strength_icon = "\u26a0\ufe0f"
            guidance = (
                f"Small pulse ({mag:.1f} pts) — near coin-flip ({hist['size_hr_day']:.0f}%)\n"
                f"<b>Action:</b> No directional edge. Treat as noise"
            )

        msg = (
            f"{icon} <b>[DayTrend] Opening Pulse — 9:30</b>\n"
            f"{'─' * 24}\n"
            f"Pulse: <b>{self._pulse_pts:+.1f} pts</b> ({direction_str})\n"
            f"Open: {self._day_open:.1f} → 9:25: {self._pulse_close:.1f}\n"
            f"Strength: {strength_icon} {strength} (|{mag:.1f}| pts)\n"
            f"\n<b>Historical Edge ({dow_name}):</b>\n"
            f"{guidance}\n"
            f"\n<i>Next: 9:45 retracement check...</i>"
        )
        self.notifier.send(msg)

    def _send_retrace_alert(self, current_time: datetime):
        if not self.notifier:
            return
        dow = current_time.weekday()
        dow_name = DOW_NAMES.get(dow, "?")
        direction_str = "BUY" if self._pulse_direction == 1 else "SELL"
        hist = self._lookup_hist_prob(dow)

        if self._confidence == "HIGH":
            conf_icon = "\u2705"
            verdict = f"Trend day likely — {hist['dow_prob']:.0f}% hist ({dow_name} {hist['dir_str']})"
        elif self._confidence == "MEDIUM":
            conf_icon = "\U0001f7e1"
            verdict = f"Moderate edge — {hist['size_hr_day']:.0f}% continuation"
        else:
            conf_icon = "\u26aa"
            verdict = "No clear edge — stay neutral"

        # Retrace zone context
        ret = self._retrace_pct
        if ret < 30:
            zone_info = "Shallow pullback (0-30%): 100% hist continuation"
        elif ret < 50:
            zone_info = "Normal pullback (30-50%): 85% hist continuation"
        elif ret < 70:
            zone_info = "Deep pullback (50-70%): 87% hist continuation"
        elif ret < 100:
            zone_info = "Major retrace (70-100%): 73% hist continuation"
        else:
            zone_info = "Full reversal (&gt;100%): only 36% continuation"

        action = self._get_action()
        signal_line = ""
        if self._confidence == "HIGH" and self.enable_signal:
            entry = self._entry_price if self._entry_price > 0 else self._pulse_close
            sl = self._day_open
            tp = entry + self._pulse_direction * self.tp_mult * abs(self._pulse_pts)
            mode = "SHADOW" if self.shadow_mode else "LIVE"
            signal_line = (
                f"\n<b>Signal [{mode}]:</b> DT {direction_str}\n"
                f"  Entry: {entry:.1f} | SL: {sl:.1f} | TP: {tp:.1f}"
            )

        msg = (
            f"{conf_icon} <b>[DayTrend] 9:45 Confirmation</b>\n"
            f"{'─' * 24}\n"
            f"Pulse: {self._pulse_pts:+.1f} pts ({direction_str})\n"
            f"Retrace: <b>{ret:.0f}%</b> of pulse\n"
            f"{zone_info}\n"
            f"\nConfidence: <b>{self._confidence}</b>\n"
            f"{verdict}\n"
            f"Bias: <b>{action}</b>"
            f"{signal_line}"
        )
        self.notifier.send(msg)

    def _log_shadow_signal(self, signal: dict):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "date": self._today_date,
            "direction": signal["direction_str"],
            "pulse_pts": signal["pulse_pts"],
            "retrace_pct": signal["retrace_pct"],
            "confidence": signal["confidence"],
            "entry": signal["entry"],
            "sl": signal["sl"],
            "tp": signal["tp"],
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
