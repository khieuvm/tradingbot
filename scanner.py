"""
Multi-Timeframe Signal Scanner
================================
Runs every 30 seconds. Scans VN30F1M on 1m, 5m, 15m timeframes.
Sends Telegram alert when signal fires on any timeframe.

Usage:
    py scanner.py
    py scanner.py --once        (run once, no loop)
    py scanner.py --interval 60 (custom interval in seconds)
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
TIMEFRAMES = ["1m", "5m", "15m"]
COMBO = "D: Trend Confirmation (safest)"  # default combo preset
SCAN_INTERVAL = 30  # seconds

# Vietnam timezone (UTC+7)
VN_TZ = timezone(timedelta(hours=7))

# Trading hours (Vietnam market: Mon-Fri 9:00-14:30, lunch break 11:30-13:00)
MARKET_OPEN = (9, 0)      # 09:00
LUNCH_START = (11, 30)    # 11:30
LUNCH_END = (13, 0)       # 13:00
MARKET_CLOSE = (14, 30)   # 14:30


def vn_now() -> datetime:
    """Get current time in Vietnam timezone."""
    return datetime.now(VN_TZ)


def is_trading_hours() -> bool:
    """Check if current time is within VN30F trading hours (Mon-Fri 9:00-14:30, skip lunch 11:30-13:00)."""
    now = vn_now()
    # Weekend check (Saturday=5, Sunday=6)
    if now.weekday() >= 5:
        return False
    current = (now.hour, now.minute)
    # Outside market hours
    if current < MARKET_OPEN or current > MARKET_CLOSE:
        return False
    # Lunch break (futures still trades but less liquid — optional skip)
    # Uncomment next line to skip lunch break:
    # if LUNCH_START <= current < LUNCH_END:
    #     return False
    return True

# Signal parameters (same as web.py defaults)
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


def get_enabled_from_combo(combo_name: str) -> dict:
    """Build enabled dict from a combo preset."""
    preset = COMBO_PRESETS.get(combo_name, {})
    enabled = {}
    for cond in preset.get("primary", []) + preset.get("confirm", []) + preset.get("gate", []):
        enabled[cond] = True
    return enabled


def scan_timeframe(fetcher: DataFetcher, symbol: str, interval: str,
                   enabled: dict, combo_name: str) -> dict | None:
    """Scan a single timeframe. Returns signal info dict or None."""
    # Calculate date range based on interval
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
        print(f"  [{interval}] Error fetching data: {e}")
        return None

    if df is None or df.empty or len(df) < 30:
        print(f"  [{interval}] Insufficient data ({len(df) if df is not None else 0} bars)")
        return None

    # Generate signals
    sig_df = generate_combined_signals(
        df, **PARAMS, enabled=enabled, combo_mode=combo_name,
    )

    # Check last bar
    last = sig_df.iloc[-1]
    signal = int(last.get("signal", 0))

    if signal == 0:
        return None

    # Collect fired conditions
    prefix = "_b_" if signal == 1 else "_s_"
    fired = [COND_LABELS.get(k, k) for k in ALL_COND_KEYS
             if last.get(f"{prefix}{k}", 0) == 1]

    price = float(last["close"])
    atr = float(last.get("atr", 0))
    confidence = int(last.get("signal_confidence", 0))

    if signal == 1:
        sl = price - 1.5 * atr
        tp = price + 3.0 * atr
    else:
        sl = price + 1.5 * atr
        tp = price - 3.0 * atr

    return {
        "interval": interval,
        "signal": "BUY" if signal == 1 else "SELL",
        "price": price,
        "sl": sl,
        "tp": tp,
        "atr": atr,
        "confidence": confidence,
        "conditions": fired,
        "rsi": float(last.get("rsi", 0)),
        "ema_slope": float(last.get("ema_slope", 0)),
        "adx": float(last.get("adx", 0)),
        "time": str(last.get("time", last.name)),
    }


def run_scan(fetcher: DataFetcher, notifier: TelegramNotifier,
             enabled: dict, combo_name: str, sent_alerts: dict):
    """Run one scan cycle across all timeframes."""
    print(f"\n[{vn_now().strftime('%H:%M:%S')}] Scanning {SYMBOL}...")
    results = []

    for tf in TIMEFRAMES:
        result = scan_timeframe(fetcher, SYMBOL, tf, enabled, combo_name)
        if result:
            results.append(result)
            print(f"  [{tf}] {result['signal']} @ {result['price']:,.1f} "
                  f"(conf={result['confidence']}, conditions={len(result['conditions'])})")
        else:
            print(f"  [{tf}] No signal")

    if not results:
        return

    # Send alerts for new signals (avoid duplicates)
    for r in results:
        # Unique key: symbol + timeframe + signal direction + bar time
        alert_key = f"{SYMBOL}_{r['interval']}_{r['signal']}_{r['time']}"
        if alert_key in sent_alerts:
            continue

        # Build multi-TF context
        other_tfs = [x for x in results if x["interval"] != r["interval"]]
        mtf_note = ""
        if other_tfs:
            aligned = [x["interval"] for x in other_tfs if x["signal"] == r["signal"]]
            if aligned:
                mtf_note = f"MTF Aligned: {', '.join(aligned)}"

        extra = {
            "Timeframe": r["interval"],
            "RSI": f"{r['rsi']:.1f}",
            "EMA Slope": f"{r['ema_slope']:.3f}",
            "ADX": f"{r['adx']:.1f}",
            "ATR": f"{r['atr']:.2f}",
            "Bar Time": r["time"],
        }
        if mtf_note:
            extra["MTF"] = mtf_note

        notifier.send_signal_alert(
            symbol=SYMBOL,
            signal=r["signal"],
            price=r["price"],
            conditions_fired=r["conditions"],
            confidence=r["confidence"],
            combo_name=f"{combo_name} [{r['interval']}]",
            sl=r["sl"],
            tp=r["tp"],
            extra=extra,
        )
        sent_alerts[alert_key] = time.time()
        print(f"  -> Alert sent: {r['signal']} on {r['interval']}")

    # Bonus: if ALL timeframes agree, send a strong confluence alert
    if len(results) >= 2:
        directions = set(r["signal"] for r in results)
        if len(directions) == 1:
            direction = directions.pop()
            tfs = ", ".join(r["interval"] for r in results)
            confluence_key = f"{SYMBOL}_MTF_{direction}_{results[0]['time']}"
            if confluence_key not in sent_alerts:
                best = max(results, key=lambda x: x["confidence"])
                msg = (
                    f"{'=' * 20}\n"
                    f"<b>MULTI-TF CONFLUENCE</b>\n"
                    f"{'=' * 20}\n"
                    f"<b>{direction}</b> on <b>{tfs}</b>\n"
                    f"Symbol: <code>{SYMBOL}</code>\n"
                    f"Price: <code>{best['price']:,.1f}</code>\n"
                    f"Best confidence: {best['confidence']}/3\n"
                    f"SL: <code>{best['sl']:,.1f}</code> | TP: <code>{best['tp']:,.1f}</code>\n"
                )
                notifier.send(msg)
                sent_alerts[confluence_key] = time.time()
                print(f"  -> MTF CONFLUENCE ALERT: {direction} on {tfs}")

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
        # Try partial match
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
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Combo: {combo_name}")
    print(f"Conditions: {[COND_LABELS.get(k, k) for k in enabled]}")
    print(f"Interval: {args.interval}s")
    print("=" * 40)

    sent_alerts = {}

    if args.once:
        run_scan(fetcher, notifier, enabled, combo_name, sent_alerts)
        return

    # Send startup notification
    notifier.send(
        f"<b>Scanner Started</b>\n"
        f"Symbol: <code>{SYMBOL}</code>\n"
        f"TFs: {', '.join(TIMEFRAMES)}\n"
        f"Strategy: {combo_name}\n"
        f"Interval: {args.interval}s"
    )

    while True:
        try:
            if not is_trading_hours():
                now = vn_now()
                print(f"\r[{now.strftime('%H:%M:%S')}] Outside trading hours (9:00-14:30 Mon-Fri). Waiting...", end="")
                time.sleep(60)  # check every minute outside hours
                continue

            run_scan(fetcher, notifier, enabled, combo_name, sent_alerts)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            # Don't spam on repeated errors
            time.sleep(args.interval)
            continue

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
