"""
Multi-Timeframe Signal Scanner
================================
Strategy:
  1. Scan 5m + 15m for signal direction (BUY/SELL)
  2. Use 1m to find optimal Limit Order entry via ATR pullback
  3. Runs every 60s (one 1m candle)

Usage:
    py scanner.py
    py scanner.py --once
    py scanner.py --interval 60 --combo "K: Smart Mean Reversion"
"""

import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import Config
from src.data_fetcher import DataFetcher
from src.signals import (
    COMBO_PRESETS, COND_LABELS, ALL_COND_KEYS,
    generate_combined_signals,
)
from src.notifier import TelegramNotifier

# ─── CONFIG ───────────────────────────────────────────────
SYMBOL = "VN30F1M"
SIGNAL_TIMEFRAMES = ["5m", "15m"]  # Signal detection
ENTRY_TIMEFRAME = "1m"             # Entry timing
COMBO = "D: Trend Confirmation (safest)"
SCAN_INTERVAL = 60  # seconds (= 1 candle on 1m)

# Vietnam timezone (UTC+7)
VN_TZ = timezone(timedelta(hours=7))

# Trading hours (Vietnam market: Mon-Fri 9:00-14:30)
MARKET_OPEN = (9, 0)
MARKET_CLOSE = (14, 30)

# ATR-based Limit Order parameters
ENTRY_ATR_PULLBACK = 0.5   # Entry = price ± 0.5*ATR (pullback from current)
SL_ATR_MULT = 1.5          # Stop Loss = entry ± 1.5*ATR
TP_ATR_MULT = 3.0          # Take Profit = entry ± 3.0*ATR

# Signal parameters
PARAMS = {
    "fast_ma": 10,
    "slow_ma": 20,
    "rsi_period": 7,
    "oversold": 35,
    "overbought": 70,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "vol_mult": 1.5,
}


def vn_now() -> datetime:
    """Get current time in Vietnam timezone."""
    return datetime.now(VN_TZ)


def is_trading_hours() -> bool:
    """Check if current time is within VN30F trading hours (Mon-Fri 9:00-14:30)."""
    now = vn_now()
    if now.weekday() >= 5:
        return False
    current = (now.hour, now.minute)
    return MARKET_OPEN <= current <= MARKET_CLOSE


def get_enabled_from_combo(combo_name: str) -> dict:
    """Build enabled dict from a combo preset."""
    preset = COMBO_PRESETS.get(combo_name, {})
    enabled = {}
    for cond in preset.get("primary", []) + preset.get("confirm", []) + preset.get("gate", []):
        enabled[cond] = True
    return enabled


def scan_timeframe(fetcher: DataFetcher, symbol: str, interval: str,
                   enabled: dict, combo_name: str) -> dict | None:
    """Scan a single timeframe for signal. Returns signal info or None."""
    now = vn_now()
    if interval in ("1m", "5m"):
        days_back = 5
    elif interval == "15m":
        days_back = 10
    else:
        days_back = 30

    start = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")

    try:
        df = fetcher.get_futures_ohlcv(symbol, start, end, interval=interval)
    except Exception as e:
        print(f"  [{interval}] Error: {e}")
        return None

    if df is None or df.empty or len(df) < 30:
        print(f"  [{interval}] Insufficient data ({len(df) if df is not None else 0} bars)")
        return None

    sig_df = generate_combined_signals(
        df, **PARAMS, enabled=enabled, combo_mode=combo_name,
    )

    last = sig_df.iloc[-1]
    signal = int(last.get("signal", 0))

    if signal == 0:
        return None

    prefix = "_b_" if signal == 1 else "_s_"
    fired = [COND_LABELS.get(k, k) for k in ALL_COND_KEYS
             if last.get(f"{prefix}{k}", 0) == 1]

    return {
        "interval": interval,
        "signal": "BUY" if signal == 1 else "SELL",
        "price": float(last["close"]),
        "atr": float(last.get("atr", 0)),
        "confidence": int(last.get("signal_confidence", 0)),
        "conditions": fired,
        "rsi": float(last.get("rsi", 0)),
        "ema_slope": float(last.get("ema_slope", 0)),
        "adx": float(last.get("adx", 0)),
        "time": str(last.get("time", last.name)),
    }


def get_1m_entry(fetcher: DataFetcher, symbol: str, direction: str) -> dict | None:
    """Use 1m chart to find optimal Limit Order entry based on ATR pullback.

    For BUY: Limit = current_price - pullback (buy the dip)
    For SELL: Limit = current_price + pullback (sell the rally)
    """
    now = vn_now()
    start = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")

    try:
        df = fetcher.get_futures_ohlcv(symbol, start, end, interval="1m")
    except Exception as e:
        print(f"  [1m entry] Error: {e}")
        return None

    if df is None or df.empty or len(df) < 30:
        return None

    # Calculate ATR on 1m
    import pandas_ta as ta
    df["atr_1m"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    last = df.iloc[-1]
    price = float(last["close"])
    atr_1m = float(last["atr_1m"]) if pd.notna(last.get("atr_1m")) else 0

    if atr_1m <= 0:
        return None

    # Recent support/resistance from last 20 bars
    recent = df.tail(20)
    recent_low = float(recent["low"].min())
    recent_high = float(recent["high"].max())

    pullback = atr_1m * ENTRY_ATR_PULLBACK

    if direction == "BUY":
        # Limit buy below current price (at pullback level)
        limit_price = price - pullback
        # Don't place below recent support (too aggressive)
        limit_price = max(limit_price, recent_low)
        sl = limit_price - SL_ATR_MULT * atr_1m
        tp = limit_price + TP_ATR_MULT * atr_1m
    else:  # SELL
        # Limit sell above current price (at rally level)
        limit_price = price + pullback
        # Don't place above recent resistance
        limit_price = min(limit_price, recent_high)
        sl = limit_price + SL_ATR_MULT * atr_1m
        tp = limit_price - TP_ATR_MULT * atr_1m

    rr_ratio = abs(tp - limit_price) / abs(sl - limit_price) if abs(sl - limit_price) > 0 else 0

    return {
        "current_price": price,
        "limit_price": limit_price,
        "sl": sl,
        "tp": tp,
        "atr_1m": atr_1m,
        "rr_ratio": rr_ratio,
        "recent_low": recent_low,
        "recent_high": recent_high,
        "distance_pct": abs(limit_price - price) / price * 100,
    }


def run_scan(fetcher: DataFetcher, notifier: TelegramNotifier,
             enabled: dict, combo_name: str, sent_alerts: dict):
    """Run one scan cycle: 5m/15m signal → 1m entry."""
    print(f"\n[{vn_now().strftime('%H:%M:%S')}] Scanning {SYMBOL}...")

    # Step 1: Scan 5m and 15m for signals
    signals = []
    for tf in SIGNAL_TIMEFRAMES:
        result = scan_timeframe(fetcher, SYMBOL, tf, enabled, combo_name)
        if result:
            signals.append(result)
            print(f"  [{tf}] {result['signal']} @ {result['price']:,.1f} "
                  f"(conf={result['confidence']}, conditions={len(result['conditions'])})")
        else:
            print(f"  [{tf}] No signal")

    if not signals:
        print("  No signal on 5m/15m. Skipping entry scan.")
        return

    # Determine direction: prefer 15m, or 5m if aligned
    # If both agree → strong; if only one → use it
    directions = set(s["signal"] for s in signals)
    if len(directions) > 1:
        print("  5m and 15m disagree. Skipping.")
        return

    direction = signals[0]["signal"]
    best_signal = max(signals, key=lambda x: x["confidence"])
    tfs_with_signal = ", ".join(s["interval"] for s in signals)

    # Step 2: Use 1m to find optimal Limit entry
    print(f"  -> {direction} confirmed on [{tfs_with_signal}]. Finding 1m entry...")
    entry = get_1m_entry(fetcher, SYMBOL, direction)

    if not entry:
        print("  [1m] Could not calculate entry.")
        return

    print(f"  [1m] Limit {direction}: {entry['limit_price']:,.1f} "
          f"(current={entry['current_price']:,.1f}, pullback={entry['distance_pct']:.2f}%)")
    print(f"       SL={entry['sl']:,.1f} | TP={entry['tp']:,.1f} | R:R={entry['rr_ratio']:.1f}")

    # Dedup: unique key based on signal TF + direction + signal bar time
    alert_key = f"{SYMBOL}_{direction}_{best_signal['time']}"
    if alert_key in sent_alerts:
        print("  (Already alerted for this signal)")
        return

    # Step 3: Send Telegram alert
    msg = (
        f"{'━' * 25}\n"
        f"<b>{'🟢 BUY' if direction == 'BUY' else '🔴 SELL'} SIGNAL</b>\n"
        f"{'━' * 25}\n"
        f"Symbol: <code>{SYMBOL}</code>\n"
        f"Signal from: <b>{tfs_with_signal}</b>\n"
        f"Confidence: {'⭐' * best_signal['confidence']} ({best_signal['confidence']}/3)\n"
        f"Conditions: {', '.join(best_signal['conditions']) or 'Score-based'}\n"
        f"\n"
        f"<b>📋 LIMIT ORDER:</b>\n"
        f"  Entry: <code>{entry['limit_price']:,.1f}</code>\n"
        f"  Current: <code>{entry['current_price']:,.1f}</code>\n"
        f"  Distance: {entry['distance_pct']:.2f}%\n"
        f"\n"
        f"  SL: <code>{entry['sl']:,.1f}</code>\n"
        f"  TP: <code>{entry['tp']:,.1f}</code>\n"
        f"  R:R = <b>{entry['rr_ratio']:.1f}</b>\n"
        f"\n"
        f"<i>ATR(1m)={entry['atr_1m']:.1f} | "
        f"RSI={best_signal['rsi']:.0f} | "
        f"ADX={best_signal['adx']:.0f}</i>\n"
        f"<i>Range: {entry['recent_low']:,.1f} - {entry['recent_high']:,.1f}</i>"
    )
    notifier.send(msg)
    sent_alerts[alert_key] = time.time()
    print(f"  -> ALERT SENT!")

    # Cleanup old alerts (older than 1 hour)
    cutoff = time.time() - 3600
    expired = [k for k, t in sent_alerts.items() if t < cutoff]
    for k in expired:
        del sent_alerts[k]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VN30F1M Multi-TF Signal Scanner")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    parser.add_argument("--interval", type=int, default=SCAN_INTERVAL, help="Scan interval (seconds)")
    parser.add_argument("--combo", type=str, default=COMBO, help="Combo preset name")
    args = parser.parse_args()

    combo_name = args.combo
    if combo_name not in COMBO_PRESETS:
        matches = [k for k in COMBO_PRESETS if combo_name.lower() in k.lower()]
        if matches:
            combo_name = matches[0]
        else:
            print(f"Unknown combo: {combo_name}")
            print(f"Available: {list(COMBO_PRESETS.keys())}")
            sys.exit(1)

    enabled = get_enabled_from_combo(combo_name)
    if not enabled:
        print(f"Warning: No conditions enabled for '{combo_name}'. Using all conditions.")
        enabled = {k: True for k in ALL_COND_KEYS}

    fetcher = DataFetcher()
    notifier = TelegramNotifier()

    if not notifier.is_configured():
        print("WARNING: Telegram not configured. Alerts will only print to console.")
    else:
        print(f"Telegram configured. Chat ID: {notifier.chat_id}")

    print(f"Scanner started: {SYMBOL}")
    print(f"Signal TFs: {', '.join(SIGNAL_TIMEFRAMES)}")
    print(f"Entry TF: {ENTRY_TIMEFRAME}")
    print(f"Combo: {combo_name}")
    print(f"Conditions: {[COND_LABELS.get(k, k) for k in enabled]}")
    print(f"Interval: {args.interval}s")
    print(f"Entry: Limit @ {ENTRY_ATR_PULLBACK}*ATR pullback | SL {SL_ATR_MULT}*ATR | TP {TP_ATR_MULT}*ATR")
    print("=" * 40)

    sent_alerts = {}

    if args.once:
        run_scan(fetcher, notifier, enabled, combo_name, sent_alerts)
        return

    # Send startup notification
    notifier.send(
        f"<b>Scanner Started</b>\n"
        f"Symbol: <code>{SYMBOL}</code>\n"
        f"Signal: {', '.join(SIGNAL_TIMEFRAMES)} → Entry: {ENTRY_TIMEFRAME}\n"
        f"Strategy: {combo_name}\n"
        f"Interval: {args.interval}s\n"
        f"Limit: {ENTRY_ATR_PULLBACK}×ATR pullback | R:R {TP_ATR_MULT/SL_ATR_MULT:.0f}:1"
    )

    while True:
        try:
            if not is_trading_hours():
                now = vn_now()
                print(f"\r[{now.strftime('%H:%M:%S')}] Outside trading hours (9:00-14:30 Mon-Fri). Waiting...", end="")
                time.sleep(60)
                continue

            run_scan(fetcher, notifier, enabled, combo_name, sent_alerts)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            time.sleep(args.interval)
            continue

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
