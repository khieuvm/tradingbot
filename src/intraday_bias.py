"""
Intraday Bias Tracker — Dynamic prediction updates throughout trading day
=========================================================================
Monitors VN30F1M momentum at 5 checkpoints to update directional bias.

Research (682 days):
  - CP2 (9:45): S1+S2 strong move (>3pts) → rest of day continuation WR 69% (1Y)
  - CP3 (10:30): S1+S2+S3 all same dir → S4+PM continuation WR 65-70%
  - Gap DN days: AM hammered, but PM reversal 58.8% → gap = AM signal only
  - Day high/low forms in S6 (PM body) ~40% of time

Checkpoints:
  CP1 (9:15): opening 3 bars → early direction read
  CP2 (9:45): 9 bars done → strongest self-predictive signal
  CP3 (10:30): AM trend established → predict rest of AM + PM
  CP4 (13:00): PM open → AM result feeds PM prediction
  CP5 (13:30): PM direction established → predict rest of PM

Usage in scanner.py:
    intraday_bias = IntradayBiasTracker(notifier=notifier)
    # Every scan cycle:
    intraday_bias.update(df_5m, current_time)
    bias = intraday_bias.get_bias()       # -1, 0, +1
    strength = intraday_bias.get_strength()  # 0.0-1.0
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.strategy_config import get_config

VN_TZ = timezone(timedelta(hours=7))

# Slot definitions (minutes from midnight)
SLOTS = {
    "S1": (9 * 60, 9 * 60 + 15),       # 9:00-9:15
    "S2": (9 * 60 + 15, 9 * 60 + 45),  # 9:15-9:45
    "S3": (9 * 60 + 45, 10 * 60 + 30),  # 9:45-10:30
    "S4": (10 * 60 + 30, 11 * 60 + 30), # 10:30-11:30
    "S5": (13 * 60, 13 * 60 + 30),      # 13:00-13:30
    "S6": (13 * 60 + 30, 14 * 60 + 30), # 13:30-14:30
}

# Checkpoint definitions: when to evaluate, what slots to look at
CHECKPOINTS = {
    "CP1": {"time_hhmm": 9 * 60 + 16, "slots_done": ["S1"],
            "label": "Opening Read (9:15)"},
    "CP2": {"time_hhmm": 9 * 60 + 46, "slots_done": ["S1", "S2"],
            "label": "Early AM (9:45)"},
    "CP3": {"time_hhmm": 10 * 60 + 31, "slots_done": ["S1", "S2", "S3"],
            "label": "Mid AM (10:30)"},
    "CP4": {"time_hhmm": 13 * 60 + 1, "slots_done": ["S1", "S2", "S3", "S4"],
            "label": "PM Open (13:00)"},
    "CP5": {"time_hhmm": 13 * 60 + 31, "slots_done": ["S1", "S2", "S3", "S4", "S5"],
            "label": "PM Direction (13:30)"},
}

# Research-backed thresholds
STRONG_MOVE_PTS = 3.0   # |move| > 3pts = strong (from checkpoint analysis)
ALL_SAME_DIR_MIN_SLOTS = 3  # S1+S2+S3 all same dir → continuation signal


class IntradayBiasTracker:
    """Track F1M intraday momentum and update bias at checkpoints."""

    def __init__(self, notifier=None):
        cfg = get_config().get("intraday_bias", {})
        self.enabled = cfg.get("enabled", False)
        self.shadow_mode = cfg.get("shadow_mode", True)
        self.strong_move_pts = cfg.get("strong_move_pts", STRONG_MOVE_PTS)

        self.notifier = notifier
        self._log_path = Path("logs/intraday_bias.jsonl")

        self._reset_state()

    def _reset_state(self):
        self._today_date = ""
        self._bias = 0           # -1=SHORT, 0=neutral, 1=LONG
        self._strength = 0.0     # 0.0-1.0 confidence
        self._slot_returns = {}  # {slot_name: return_pts}
        self._checkpoints_done = set()
        self._day_open = 0.0
        self._current_cp = ""    # last checkpoint evaluated
        self._am_return = 0.0
        self._history = []       # checkpoint history for today

    def reset(self):
        self._reset_state()

    @property
    def bias(self) -> int:
        return self._bias

    def get_bias(self) -> int:
        return self._bias

    def get_strength(self) -> float:
        return self._strength

    def get_current_checkpoint(self) -> str:
        return self._current_cp

    def get_state(self) -> dict:
        return {
            "date": self._today_date,
            "bias": self._bias,
            "strength": self._strength,
            "checkpoint": self._current_cp,
            "slot_returns": self._slot_returns.copy(),
            "am_return": self._am_return,
            "history": self._history.copy(),
        }

    def update(self, df_5m: pd.DataFrame, current_time: datetime = None):
        """Update bias based on current 5m data. Called every scan cycle."""
        if not self.enabled or df_5m is None or len(df_5m) < 3:
            return

        if current_time is None:
            current_time = datetime.now(VN_TZ)

        today_str = current_time.strftime("%Y-%m-%d")

        # New day → reset
        if self._today_date != today_str:
            self._reset_state()
            self._today_date = today_str

        hhmm = current_time.hour * 60 + current_time.minute

        # Only during trading hours
        if hhmm < 9 * 60 or (11 * 60 + 30 <= hhmm < 13 * 60) or hhmm >= 14 * 60 + 30:
            return

        # Get today's bars
        df_5m = df_5m.copy()
        df_5m["time"] = pd.to_datetime(df_5m["time"])
        today_bars = df_5m[df_5m["time"].dt.strftime("%Y-%m-%d") == today_str]
        if today_bars.empty:
            return

        # Set day open
        if self._day_open == 0.0:
            self._day_open = float(today_bars.iloc[0]["open"])

        # Compute slot returns
        self._compute_slot_returns(today_bars)

        # Check each checkpoint
        for cp_name, cp_def in CHECKPOINTS.items():
            if cp_name in self._checkpoints_done:
                continue
            if hhmm >= cp_def["time_hhmm"]:
                # Check all required slots have data
                all_done = all(s in self._slot_returns for s in cp_def["slots_done"])
                if all_done:
                    self._evaluate_checkpoint(cp_name, cp_def, current_time)

    def _compute_slot_returns(self, today_bars: pd.DataFrame):
        """Compute return (pts) for each completed slot."""
        for slot_name, (start_min, end_min) in SLOTS.items():
            if slot_name in self._slot_returns:
                continue  # already computed

            bars_in_slot = today_bars[
                (today_bars["time"].dt.hour * 60 + today_bars["time"].dt.minute >= start_min) &
                (today_bars["time"].dt.hour * 60 + today_bars["time"].dt.minute < end_min)
            ]

            if bars_in_slot.empty:
                continue

            # Check if slot is complete (last bar's time is near end)
            last_bar_min = (bars_in_slot.iloc[-1]["time"].hour * 60 +
                            bars_in_slot.iloc[-1]["time"].minute)
            # Slot complete if we have bars AND current time past slot end
            # (allow 1 bar = 5 min tolerance)
            if last_bar_min >= end_min - 5:
                slot_open = float(bars_in_slot.iloc[0]["open"])
                slot_close = float(bars_in_slot.iloc[-1]["close"])
                slot_high = float(bars_in_slot["high"].max())
                slot_low = float(bars_in_slot["low"].min())
                slot_vol = float(bars_in_slot["volume"].sum())

                self._slot_returns[slot_name] = {
                    "return": slot_close - slot_open,
                    "open": slot_open,
                    "close": slot_close,
                    "high": slot_high,
                    "low": slot_low,
                    "volume": slot_vol,
                    "direction": 1 if slot_close > slot_open else (-1 if slot_close < slot_open else 0),
                }

    def _evaluate_checkpoint(self, cp_name: str, cp_def: dict, current_time: datetime):
        """Evaluate bias at a checkpoint."""
        self._checkpoints_done.add(cp_name)
        self._current_cp = cp_name

        slots_done = cp_def["slots_done"]
        cum_return = sum(self._slot_returns[s]["return"] for s in slots_done)
        directions = [self._slot_returns[s]["direction"] for s in slots_done]
        all_same = all(d == directions[0] for d in directions) and directions[0] != 0
        n_up = sum(1 for d in directions if d > 0)
        n_dn = sum(1 for d in directions if d < 0)

        old_bias = self._bias
        old_strength = self._strength

        # ── Evaluate based on checkpoint ──
        if cp_name == "CP1":
            # Opening read: modest signal
            if abs(cum_return) > self.strong_move_pts * 0.5:
                self._bias = 1 if cum_return > 0 else -1
                self._strength = 0.3
            else:
                self._bias = 0
                self._strength = 0.0

        elif cp_name == "CP2":
            # STRONGEST checkpoint: S1+S2 strong move → rest of day WR 69%
            if abs(cum_return) > self.strong_move_pts:
                self._bias = 1 if cum_return > 0 else -1
                self._strength = 0.7  # high confidence
            elif abs(cum_return) > self.strong_move_pts * 0.5:
                self._bias = 1 if cum_return > 0 else -1
                self._strength = 0.4
            else:
                # Weak move, reduce any existing bias
                self._strength = max(0.0, self._strength - 0.2)
                if self._strength == 0.0:
                    self._bias = 0

        elif cp_name == "CP3":
            # AM trend established: all same dir → WR 65-70%
            if all_same and abs(cum_return) > self.strong_move_pts:
                self._bias = directions[0]
                self._strength = 0.8  # highest confidence
            elif all_same:
                self._bias = directions[0]
                self._strength = 0.6
            elif abs(cum_return) > self.strong_move_pts:
                self._bias = 1 if cum_return > 0 else -1
                self._strength = 0.5
            else:
                # Choppy AM → reduce bias
                self._strength = max(0.0, self._strength - 0.3)
                if self._strength < 0.2:
                    self._bias = 0
                    self._strength = 0.0

            # Store AM return for PM prediction
            self._am_return = cum_return

        elif cp_name == "CP4":
            # PM open: AM result → PM prediction
            # Research: AM→PM is ~50/50, but strong AM moves have slight edge
            # Gap DN days: PM tends to REVERSE (58.8%)
            am_ret = self._am_return
            if abs(am_ret) > self.strong_move_pts * 2:
                # Very strong AM → slight continuation into PM open
                self._bias = 1 if am_ret > 0 else -1
                self._strength = 0.3  # low — AM→PM weak
            else:
                # Normal AM → PM is coin flip
                self._bias = 0
                self._strength = 0.0

        elif cp_name == "CP5":
            # PM direction established
            s5_ret = self._slot_returns["S5"]["return"]
            if abs(s5_ret) > self.strong_move_pts * 0.7:
                self._bias = 1 if s5_ret > 0 else -1
                self._strength = 0.5
            elif self._strength > 0:
                # Maintain any existing bias weakly
                self._strength = max(0.0, self._strength - 0.2)
            else:
                self._bias = 0
                self._strength = 0.0

        # ── Alert if bias changed ──
        bias_changed = (old_bias != self._bias) or (abs(old_strength - self._strength) > 0.15)

        result = {
            "checkpoint": cp_name,
            "label": cp_def["label"],
            "cum_return": cum_return,
            "bias": self._bias,
            "strength": self._strength,
            "all_same_dir": all_same,
            "n_up": n_up,
            "n_dn": n_dn,
            "slot_details": {s: self._slot_returns[s]["return"] for s in slots_done},
        }
        self._history.append(result)

        # Print and notify
        bias_str = {-1: "SHORT", 0: "NEUTRAL", 1: "LONG"}.get(self._bias, "?")
        bias_icon = {-1: "\U0001f534", 0: "\u26aa", 1: "\U0001f7e2"}.get(self._bias, "?")
        strength_bar = "\u2588" * int(self._strength * 5) + "\u2591" * (5 - int(self._strength * 5))

        slot_line = " | ".join(
            f"{s}:{self._slot_returns[s]['return']:+.1f}" for s in slots_done
        )
        print(f"  [INTRADAY] {cp_def['label']}: {bias_icon} {bias_str} "
              f"[{strength_bar}] cum={cum_return:+.1f}pts | {slot_line}")

        if self.notifier and bias_changed:
            edge_note = self._get_edge_note(cp_name, cum_return, all_same)
            msg = (
                f"{bias_icon} <b>[Intraday Bias] {cp_def['label']}</b>\n"
                f"Bias: <b>{bias_str}</b> | Strength: {self._strength:.0%}\n"
                f"Cumulative: {cum_return:+.1f} pts\n"
                f"Slots: {slot_line}\n"
            )
            if edge_note:
                msg += f"\n{edge_note}"
            if self.shadow_mode:
                msg += "\n<i>Shadow mode</i>"
            self.notifier.send(msg)

        self._log_checkpoint(current_time, result)

    def _get_edge_note(self, cp_name: str, cum_return: float, all_same: bool) -> str:
        """Return research-backed edge note for the checkpoint."""
        if cp_name == "CP2" and abs(cum_return) > self.strong_move_pts:
            dir_str = "UP" if cum_return > 0 else "DN"
            return f"\u26a1 S1+S2 strong {dir_str} ({cum_return:+.1f}pts) \u2192 rest of day WR ~69%"
        if cp_name == "CP3" and all_same and abs(cum_return) > self.strong_move_pts:
            dir_str = "UP" if cum_return > 0 else "DN"
            return f"\u26a1 S1+S2+S3 ALL {dir_str} \u2192 S4+PM continuation WR ~65-70%"
        if cp_name == "CP3" and all_same:
            dir_str = "UP" if cum_return > 0 else "DN"
            return f"\U0001f4ca S1+S2+S3 consistent {dir_str} \u2192 moderate continuation edge"
        if cp_name == "CP4":
            return "\U0001f4ca AM\u2192PM: ~50/50 historically, bias weakened"
        return ""

    def should_filter(self, direction_int: int) -> bool:
        """Return True if entry should be blocked based on intraday bias.

        Only blocks entries opposing strong bias (strength >= 0.5).
        """
        if not self.enabled or self._strength < 0.5:
            return False
        if self._bias != 0 and direction_int != self._bias:
            return True
        return False

    def _log_checkpoint(self, current_time: datetime, result: dict):
        """Log checkpoint to JSONL."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": current_time.isoformat(),
                **result,
            }
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  [INTRADAY] Log error: {e}")
