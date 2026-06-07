import pandas as pd
import pandas_ta as ta

from combos.base import BaseCombo


class NR4Combo(BaseCombo):
    """NR4 SHORT — Narrowest Range in 4 bars, SHORT-only breakout.

    Signal: current bar is the smallest candle in last N, range < threshold×ATR.
    Direction: SHORT only (scanner queues SELL pending, ignores BUY).
    All params loaded from strategy_config.yaml combos.NR4 section.
    """

    name = "NR4"
    timeframe = "5m"

    def __init__(self):
        super().__init__()
        dir_str = self._cfg.get("direction", "SHORT")
        self.DIRECTION = -1 if dir_str == "SHORT" else (1 if dir_str == "LONG" else 0)
        ef = self._cfg.get("entry_filter", {})
        tw = self._cfg.get("time_windows", {})

        self._nr4_lookback = self._cfg.get("nr4_lookback", 4)
        self._threshold = self._cfg.get("compression_threshold", 0.7)
        self._atr_min = ef.get("atr_min", 2.5)
        self._atr_max = ef.get("atr_max", 4.5)

        def parse_time(s, default):
            try:
                parts = str(s).split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except Exception:
                return default

        self._time_windows = {
            "AM": (parse_time(tw.get("am_start", "09:15"), 555),
                   parse_time(tw.get("am_end", "10:45"), 645)),
            "PM": (parse_time(tw.get("pm_start", "13:15"), 795),
                   parse_time(tw.get("pm_end", "14:15"), 855)),
        }

    @property
    def nr4_lookback(self) -> int:
        return self._nr4_lookback

    def detect(self, df: pd.DataFrame) -> dict | None:
        """Detect NR4 compression on last complete bar."""
        if len(df) < 20:
            return None

        atr_s = ta.atr(df["high"], df["low"], df["close"], length=14)
        if atr_s is None:
            return None

        last_idx = len(df) - 1
        atr_v = float(atr_s.iloc[last_idx])
        if pd.isna(atr_v) or not (self._atr_min <= atr_v <= self._atr_max):
            return None

        # Time filter
        try:
            t = pd.to_datetime(df["time"].iloc[-1]) if "time" in df.columns else df.index[-1]
            mins = t.hour * 60 + t.minute
        except Exception:
            return None

        am_start, am_end = self._time_windows["AM"]
        pm_start, pm_end = self._time_windows["PM"]
        if not (am_start <= mins <= am_end or pm_start <= mins <= pm_end):
            return None

        # NR4: current bar range is smallest in last N
        if last_idx < self._nr4_lookback:
            return None

        ranges = (df["high"] - df["low"]).values
        cur_range = float(ranges[last_idx])
        prev_ranges = ranges[last_idx - self._nr4_lookback + 1: last_idx]
        if cur_range >= float(min(prev_ranges)):
            return None

        # Range must be < threshold × ATR
        if cur_range >= self._threshold * atr_v:
            return None

        session = "AM" if mins < 12 * 60 else "PM"
        time_str = str(df["time"].iloc[-1]) if "time" in df.columns else str(df.index[-1])

        return {
            "price": float(df["close"].iloc[-1]),
            "high": float(df["high"].iloc[-1]),
            "low": float(df["low"].iloc[-1]),
            "atr": atr_v,
            "session": session,
            "time": time_str,
            "interval": self.timeframe,
            "confidence": 1,
        }
