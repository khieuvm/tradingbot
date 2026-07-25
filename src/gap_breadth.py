"""
VN30 Gap Breadth Signal — Opening gap of top VN30 stocks → VN30F1M bias
========================================================================
Detect at 9:05: how many VN30 stocks gap down (or up) vs yesterday's close.
≥5/10 stocks gap DN <-0.5% → SHORT bias for VN30F1M (WR 72.7% ở 6M gần nhất).

Research:
  - 6M: ≥5 gap DN → WR 72.7%, PF 3.01, +10.17 pts after cost (n=11)
  - 1Y: ≥5 gap DN → WR 67.7%, +8.60 after cost (n=31)
  - Gap UP signals degraded → not used
  - Volume NOT needed (counterintuitive — high vol makes it worse)

Usage in scanner.py:
    gap_breadth = GapBreadthSignal(notifier=notifier)
    # At 9:05 every trading day:
    gap_breadth.check(fetcher)
    bias = gap_breadth.get_bias()  # 0, 1, or -1
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import pandas as pd

from src.strategy_config import get_config

VN_TZ = timezone(timedelta(hours=7))

# Top 10 VN30 stocks by market cap + correlation with VN30F1M
BASKET_STOCKS = ["VIC", "VHM", "VNM", "HPG", "MSN", "VCB", "BID", "CTG", "TCB", "FPT"]

# Gap threshold (pct)
DEFAULT_GAP_THRESHOLD = -0.5   # gap DN < -0.5%
DEFAULT_MIN_STOCKS_DN = 5      # ≥5/10 stocks gap DN → SHORT bias


class GapBreadthSignal:
    """Detect VN30 stocks gap opening → generate bias for VN30F1M."""

    def __init__(self, notifier=None):
        cfg = get_config().get("gap_breadth", {})
        self.enabled = cfg.get("enabled", False)
        self.shadow_mode = cfg.get("shadow_mode", True)
        self.gap_threshold_dn = cfg.get("gap_threshold_dn", DEFAULT_GAP_THRESHOLD)
        self.min_stocks_dn = cfg.get("min_stocks_dn", DEFAULT_MIN_STOCKS_DN)
        self.stocks = cfg.get("stocks", BASKET_STOCKS)

        self.notifier = notifier
        self._log_path = Path("logs/gap_breadth_shadow.jsonl")

        self._reset_state()

    def _reset_state(self):
        self._today_date = ""
        self._checked = False
        self._bias = 0          # 0=neutral, -1=SHORT, 1=LONG
        self._gap_details = {}  # {stock: gap_pct}
        self._n_gap_dn = 0
        self._n_gap_up = 0
        self._avg_gap = 0.0

    def reset(self):
        self._reset_state()

    @property
    def bias(self) -> int:
        return self._bias

    @property
    def is_checked(self) -> bool:
        return self._checked

    def get_bias(self) -> int:
        """Return current bias: -1=SHORT, 0=neutral, 1=LONG."""
        return self._bias

    def get_details(self) -> dict:
        """Return full gap breadth details for logging/display."""
        return {
            "date": self._today_date,
            "bias": self._bias,
            "n_gap_dn": self._n_gap_dn,
            "n_gap_up": self._n_gap_up,
            "avg_gap": self._avg_gap,
            "gaps": self._gap_details,
        }

    def check(self, fetcher, current_time: datetime = None):
        """Check gap breadth at market open (~9:05).

        Fetches yesterday's daily close + today's first 1m bar open for each stock.
        Should be called once per day, around 9:05.
        """
        if not self.enabled:
            return

        if current_time is None:
            current_time = datetime.now(VN_TZ)

        today_str = current_time.strftime("%Y-%m-%d")

        # Reset if new day
        if self._today_date != today_str:
            self._reset_state()
            self._today_date = today_str

        # Only check once per day
        if self._checked:
            return

        # Only check between 9:03 and 9:30
        hhmm = current_time.hour * 60 + current_time.minute
        if hhmm < 9 * 60 + 3 or hhmm > 9 * 60 + 30:
            return

        print(f"  [GAP_BREADTH] Checking {len(self.stocks)} VN30 stocks...")

        gaps = {}
        # Fetch window: 10 days back for daily (cover weekends/holidays)
        start_daily = (current_time - timedelta(days=10)).strftime("%Y-%m-%d")
        # For intraday open: fetch today's 1m data
        start_1m = (current_time - timedelta(days=1)).strftime("%Y-%m-%d")

        for stock in self.stocks:
            try:
                # 1) Get yesterday's close from daily data
                df_daily = fetcher.get_historical_ohlcv(
                    stock, start_daily, today_str, interval="1D"
                )
                if df_daily is None or len(df_daily) < 1:
                    print(f"    {stock}: no daily data")
                    continue

                df_daily["time"] = pd.to_datetime(df_daily["time"])
                df_daily = df_daily.sort_values("time").reset_index(drop=True)

                # Last complete daily bar = yesterday's close
                # If last bar is today (incomplete), use second-to-last
                last_date_str = df_daily.iloc[-1]["time"].strftime("%Y-%m-%d")
                if last_date_str == today_str and len(df_daily) >= 2:
                    prev_close = float(df_daily.iloc[-2]["close"])
                    today_open = float(df_daily.iloc[-1]["open"])
                elif last_date_str == today_str:
                    print(f"    {stock}: only today's bar, no prev close")
                    continue
                else:
                    # Today's daily bar not available yet → get open from 1m
                    prev_close = float(df_daily.iloc[-1]["close"])
                    today_open = self._get_today_open_1m(fetcher, stock, today_str, start_1m)
                    if today_open is None:
                        print(f"    {stock}: no intraday open available")
                        continue

                gap_pct = (today_open / prev_close - 1) * 100
                gaps[stock] = round(gap_pct, 3)

            except Exception as e:
                print(f"    {stock}: error {e}")
                continue

        if len(gaps) < 5:
            print(f"  [GAP_BREADTH] Only {len(gaps)} stocks fetched — skipping")
            return

        self._checked = True
        self._gap_details = gaps

        # Count gap down / gap up
        self._n_gap_dn = sum(1 for g in gaps.values() if g < self.gap_threshold_dn)
        self._n_gap_up = sum(1 for g in gaps.values() if g > abs(self.gap_threshold_dn))
        self._avg_gap = sum(gaps.values()) / len(gaps)

        # Determine bias
        if self._n_gap_dn >= self.min_stocks_dn:
            self._bias = -1  # SHORT
        else:
            self._bias = 0   # Neutral (gap UP signals degraded, not used)

        # Sort gaps for display
        sorted_gaps = sorted(gaps.items(), key=lambda x: x[1])
        gap_lines = []
        for stock, gap in sorted_gaps:
            icon = "\U0001f534" if gap < self.gap_threshold_dn else (
                "\U0001f7e2" if gap > abs(self.gap_threshold_dn) else "\u26aa"
            )
            gap_lines.append(f"  {icon} {stock}: {gap:+.2f}%")

        bias_str = {-1: "SHORT \U0001f534", 0: "NEUTRAL \u26aa", 1: "LONG \U0001f7e2"}[self._bias]
        print(f"  [GAP_BREADTH] {self._n_gap_dn} DN / {self._n_gap_up} UP / "
              f"{len(gaps) - self._n_gap_dn - self._n_gap_up} flat → Bias: {bias_str}")

        # Telegram alert
        if self.notifier:
            msg_lines = [
                f"\U0001f4ca <b>[Gap Breadth] {bias_str}</b>",
                f"Gap DN (<{self.gap_threshold_dn}%): <b>{self._n_gap_dn}/{len(gaps)}</b> stocks",
                f"Avg gap: {self._avg_gap:+.2f}%",
                "",
            ]
            msg_lines.extend(gap_lines)
            if self._bias == -1:
                msg_lines.extend([
                    "",
                    f"\u26a1 <b>\u2265{self.min_stocks_dn} stocks gap DN \u2192 SHORT bias</b>",
                    f"Edge: WR 72.7% (6M), +10.17 pts after cost",
                ])
                if self.shadow_mode:
                    msg_lines.append("<i>Shadow mode \u2014 bias filter only</i>")
            self.notifier.send("\n".join(msg_lines))

        # Log
        self._log_signal(current_time)

    @staticmethod
    def _get_today_open_1m(fetcher, stock: str, today_str: str, start_1m: str) -> float | None:
        """Get today's opening price from 1m intraday data."""
        try:
            df_1m = fetcher.get_historical_ohlcv(stock, start_1m, today_str, interval="1m")
            if df_1m is None or df_1m.empty:
                return None
            df_1m["time"] = pd.to_datetime(df_1m["time"])
            today_bars = df_1m[df_1m["time"].dt.strftime("%Y-%m-%d") == today_str]
            if today_bars.empty:
                return None
            return float(today_bars.iloc[0]["open"])
        except Exception:
            return None

    def should_filter(self, direction_int: int) -> bool:
        """Return True if entry should be blocked based on gap breadth bias.

        Only blocks entries that OPPOSE the bias.
        - bias=-1 (SHORT): block BUY entries (direction_int=1)
        - bias=0: no blocking
        """
        if not self.enabled or not self._checked:
            return False
        if self._bias == -1 and direction_int == 1:
            return True
        return False

    def _log_signal(self, current_time: datetime):
        """Log gap breadth signal to JSONL."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": current_time.isoformat(),
                "date": self._today_date,
                "bias": self._bias,
                "n_gap_dn": self._n_gap_dn,
                "n_gap_up": self._n_gap_up,
                "avg_gap": self._avg_gap,
                "gaps": self._gap_details,
            }
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  [GAP_BREADTH] Log error: {e}")
