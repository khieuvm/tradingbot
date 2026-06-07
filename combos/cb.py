import pandas as pd
import pandas_ta as ta

from combos.base import BaseCombo


class CBCombo(BaseCombo):
    """CB — Compression Breakout on 5m.

    Signal: N-bar squeeze (max range < threshold×ATR) → breakout direction.
    All params loaded from strategy_config.yaml combos.CB section.
    """

    name = "CB"
    timeframe = "5m"

    def __init__(self):
        super().__init__()
        ef = self._cfg.get("entry_filter", {})
        comp = self._cfg.get("compression", {})
        tw = self._cfg.get("time_windows", {})

        self._atr_min = ef.get("atr_min", 2.5)
        self._atr_max = ef.get("atr_max", 4.5)
        self._atr_min_pm = ef.get("atr_min_pm", self._atr_min)
        self._atr_max_pm = ef.get("atr_max_pm", self._atr_max)
        self._rsi_period = ef.get("rsi_period", 14)
        self._rsi_max = ef.get("rsi_max", 70)

        self._n_bars = comp.get("n_bars", 3)
        self._threshold = comp.get("threshold", 0.7)

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

        self._dead_zones = []
        for slot in self._cfg.get("dead_zones", []):
            parts = str(slot).split("-")
            if len(parts) == 2:
                self._dead_zones.append((parse_time(parts[0], 0), parse_time(parts[1], 0)))

    def detect(self, df: pd.DataFrame) -> dict | None:
        """Detect CB compression on last complete 5m bar."""
        if len(df) < 20:
            return None

        atr_s = ta.atr(df["high"], df["low"], df["close"], length=14)
        rsi_s = ta.rsi(df["close"], length=self._rsi_period)
        if atr_s is None or rsi_s is None:
            return None

        last_idx = len(df) - 1
        atr_v = float(atr_s.iloc[last_idx])
        rsi_v = float(rsi_s.iloc[last_idx])

        if pd.isna(atr_v) or pd.isna(rsi_v):
            return None

        # Time filter
        try:
            t = pd.to_datetime(df["time"].iloc[-1]) if "time" in df.columns else df.index[-1]
            mins = t.hour * 60 + t.minute
        except Exception:
            return None

        am_start, am_end = self._time_windows["AM"]
        pm_start, pm_end = self._time_windows["PM"]
        in_am = am_start <= mins <= am_end
        in_pm = pm_start <= mins <= pm_end
        if not (in_am or in_pm):
            return None

        # Exclude weak AM dead slots
        if in_am:
            for slot_start, slot_end in self._dead_zones:
                if slot_start <= mins < slot_end:
                    return None

        # ATR filter — wider range for PM
        if in_pm:
            if not (self._atr_min_pm <= atr_v <= self._atr_max_pm):
                return None
        else:
            if not (self._atr_min <= atr_v <= self._atr_max):
                return None

        if rsi_v >= self._rsi_max:
            return None

        # N-bar compression
        if last_idx < self._n_bars:
            return None
        ranges = (df["high"] - df["low"]).values
        if max(ranges[last_idx - self._n_bars: last_idx]) >= self._threshold * atr_v:
            return None

        session = "AM" if mins < 12 * 60 else "PM"
        time_str = str(df["time"].iloc[-1]) if "time" in df.columns else str(df.index[-1])

        return {
            "price": float(df["close"].iloc[-1]),
            "high": float(df["high"].iloc[-1]),
            "low": float(df["low"].iloc[-1]),
            "atr": atr_v,
            "rsi": rsi_v,
            "session": session,
            "time": time_str,
            "interval": self.timeframe,
            "confidence": 1,
        }
