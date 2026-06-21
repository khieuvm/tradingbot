"""
DNSE WebSocket Stream — Real-time OHLC bars and trade ticks for VN30F1M.

Provides:
- Real-time 1m/5m OHLC bars via WebSocket (replaces vnstock polling)
- Live trade ticks for instant price updates
- Rolling DataFrame buffer compatible with existing scanner interface
- Automatic reconnection and error handling

Usage:
    stream = DnseStream(api_key, api_secret, symbol="41I1G6000")
    stream.start()  # non-blocking, runs in daemon thread

    df_5m = stream.get_ohlcv("5m")   # pandas DataFrame, same format as vnstock
    df_1m = stream.get_ohlcv("1m")
    price = stream.get_latest_price()  # float or None
"""
import asyncio
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import Config

VN_TZ = timezone(timedelta(hours=7))
MAX_BARS_1M = 500
MAX_BARS_5M = 300


class DnseStream:
    """WebSocket stream manager for VN30F1M real-time data."""

    def __init__(self, api_key: str = None, api_secret: str = None,
                 symbol: str = "41I1G6000"):
        self._api_key = api_key or Config.DNSE_API_KEY
        self._api_secret = api_secret or Config.DNSE_API_SECRET
        self._symbol = symbol
        self._thread: threading.Thread | None = None
        self._stream = None
        self._running = False

        self._bars_1m: deque = deque(maxlen=MAX_BARS_1M)
        self._bars_5m: deque = deque(maxlen=MAX_BARS_5M)
        self._last_trade_price: float | None = None
        self._last_trade_time: float = 0
        self._lock = threading.Lock()
        self._connected_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def is_connected(self) -> bool:
        return self._running and self._connected_event.is_set()

    def start(self, timeout: float = 10.0) -> bool:
        """Start WebSocket stream in background thread.

        Returns True if connected within timeout, False otherwise.
        """
        if self.is_running:
            return True

        if not self._api_key or not self._api_secret:
            print("[STREAM] No API key/secret configured. Stream disabled.")
            return False

        self._connected_event.clear()
        self._thread = threading.Thread(target=self._run_stream, daemon=True)
        self._thread.start()
        self._connected_event.wait(timeout=timeout)
        return self._running

    def stop(self):
        """Stop the WebSocket stream."""
        if self._stream:
            self._stream.stop()
        self._running = False

    def get_latest_price(self) -> float | None:
        """Get latest trade price (thread-safe)."""
        with self._lock:
            if time.time() - self._last_trade_time > 300:
                return None
            return self._last_trade_price

    def get_ohlcv(self, tf: str = "5m") -> pd.DataFrame | None:
        """Get OHLCV DataFrame from stream buffer.

        Returns DataFrame with columns: [time, open, high, low, close, volume]
        matching the format expected by scanner.py and data_fetcher.
        Returns None if insufficient data (< 20 bars).
        """
        with self._lock:
            bars = list(self._bars_5m if tf == "5m" else self._bars_1m)

        if len(bars) < 20:
            return None

        df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        # DNSE WebSocket timestamps are integers — detect seconds vs milliseconds
        if df['time'].dtype in ('int64', 'float64') or (len(df) > 0 and isinstance(df['time'].iloc[0], (int, float))):
            sample = int(df['time'].iloc[0])
            unit = 'ms' if sample > 1e12 else 's'
            df['time'] = pd.to_datetime(df['time'], unit=unit, utc=True).dt.tz_convert(VN_TZ).dt.tz_localize(None)
        else:
            df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        df = df.drop_duplicates(subset='time', keep='last').reset_index(drop=True)
        return df

    def bar_count(self, tf: str = "5m") -> int:
        """Number of bars buffered for given timeframe."""
        with self._lock:
            return len(self._bars_5m if tf == "5m" else self._bars_1m)

    def _run_stream(self):
        """Internal: run the async stream (called in daemon thread)."""
        try:
            from dnse.stream import DnseMarketStream

            self._stream = DnseMarketStream(
                api_key=self._api_key,
                api_secret=self._api_secret,
            )

            async def on_ohlc_1m(bar):
                if bar.symbol != self._symbol:
                    return
                with self._lock:
                    self._bars_1m.append((
                        bar.timestamp,
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                    ))

            async def on_ohlc_5m(bar):
                if bar.symbol != self._symbol:
                    return
                with self._lock:
                    self._bars_5m.append((
                        bar.timestamp,
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                    ))

            async def on_trade(trade):
                if trade.symbol != self._symbol:
                    return
                with self._lock:
                    self._last_trade_price = trade.price
                    self._last_trade_time = time.time()

            self._stream.subscribe_ohlc(self._symbol, on_ohlc_1m, timeframe="1m")
            self._stream.subscribe_ohlc(self._symbol, on_ohlc_5m, timeframe="5m")
            self._stream.subscribe_trades(self._symbol, on_trade)

            self._running = True
            self._connected_event.set()
            print(f"[STREAM] Connected to DNSE WebSocket ({self._symbol})")
            self._stream.run()
            # run() returned — stream disconnected
            self._running = False
            print("[STREAM] Disconnected. Falling back to vnstock polling.")

        except ImportError as e:
            print(f"[STREAM] dnse.stream not available: {e}")
            self._running = False
        except Exception as e:
            if self._running:
                print(f"[STREAM] Error: {e}")
            self._running = False
        finally:
            self._connected_event.set()
