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
from pathlib import Path

# Force UTF-8 on Windows console to handle emoji/Unicode from vnstock banner
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import yaml

from config import Config
from src.data_fetcher import DataFetcher
from src.signals import (
    COMBO_PRESETS, COND_LABELS, ALL_COND_KEYS,
    generate_combined_signals,
    compute_volume_profile,
)
from src.notifier import TelegramNotifier

# ─── LOAD CONFIG FROM YAML ────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "strategy_config.yaml"


def load_strategy_config() -> dict:
    """Load strategy configuration from YAML file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_config(cfg: dict):
    """Apply YAML config to module-level variables."""
    global SYMBOL, SIGNAL_TIMEFRAMES, ENTRY_TIMEFRAME, COMBO, SCAN_INTERVAL
    global MARKET_OPEN, MARKET_CLOSE
    global ENTRY_ATR_PULLBACK, SL_ATR_MULT, TP_ATR_MULT, PARAMS

    SYMBOL = cfg.get("symbol", "VN30F1M")
    SIGNAL_TIMEFRAMES = cfg.get("signal_timeframes", ["5m", "15m"])
    ENTRY_TIMEFRAME = cfg.get("entry_timeframe", "1m")
    COMBO = cfg.get("active_combo", "D: Trend Confirmation (safest)")
    SCAN_INTERVAL = cfg.get("scan_interval", 60)

    # Trading hours
    open_str = cfg.get("market_open", "09:00")
    close_str = cfg.get("market_close", "14:30")
    MARKET_OPEN = tuple(int(x) for x in open_str.split(":"))
    MARKET_CLOSE = tuple(int(x) for x in close_str.split(":"))

    # Entry parameters
    entry = cfg.get("entry", {})
    ENTRY_ATR_PULLBACK = entry.get("atr_pullback", 0.5)
    SL_ATR_MULT = entry.get("sl_atr_mult", 1.5)
    TP_ATR_MULT = entry.get("tp_atr_mult", 3.0)

    # Indicator parameters
    ind = cfg.get("indicators", {})
    PARAMS = {
        "fast_ma": ind.get("fast_ma", 10),
        "slow_ma": ind.get("slow_ma", 20),
        "rsi_period": ind.get("rsi_period", 7),
        "oversold": ind.get("oversold", 35),
        "overbought": ind.get("overbought", 70),
        "macd_fast": ind.get("macd_fast", 12),
        "macd_slow": ind.get("macd_slow", 26),
        "macd_signal": ind.get("macd_signal", 9),
        "vol_mult": ind.get("vol_mult", 1.5),
    }

    # Sync combo presets from YAML back to signals module
    combos = cfg.get("combos", {})
    if combos:
        for key, combo_cfg in combos.items():
            name = combo_cfg.get("name", key)
            COMBO_PRESETS[name] = {
                "desc": combo_cfg.get("desc", ""),
                "primary": combo_cfg.get("primary", []),
                "confirm": combo_cfg.get("confirm", []),
                "gate": combo_cfg.get("gate", []),
            }


# Initialize with defaults, then override from YAML
SYMBOL = "VN30F1M"
SIGNAL_TIMEFRAMES = ["5m", "15m"]
ENTRY_TIMEFRAME = "1m"
COMBO = "D: Trend Confirmation (safest)"
SCAN_INTERVAL = 60
MARKET_OPEN = (9, 0)
MARKET_CLOSE = (14, 30)
ENTRY_ATR_PULLBACK = 0.5
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
PARAMS = {
    "fast_ma": 10, "slow_ma": 20, "rsi_period": 7,
    "oversold": 35, "overbought": 70,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "vol_mult": 1.5,
}

# Load from YAML if exists
if CONFIG_PATH.exists():
    _cfg = load_strategy_config()
    apply_config(_cfg)

# Vietnam timezone (UTC+7)
VN_TZ = timezone(timedelta(hours=7))


def vn_now() -> datetime:
    """Get current time in Vietnam timezone."""
    return datetime.now(VN_TZ)


def is_trading_hours() -> bool:
    """Check if current time is within VN30F trading hours (Mon-Fri 9:00-11:30, 13:00-14:30)."""
    now = vn_now()
    if now.weekday() >= 5:
        return False
    current = (now.hour, now.minute)
    session_1 = (9, 0) <= current <= (11, 30)
    session_2 = (13, 0) <= current <= (14, 30)
    return session_1 or session_2


def get_enabled_from_combo(combo_name: str) -> dict:
    """Build enabled dict from a combo preset."""
    preset = COMBO_PRESETS.get(combo_name, {})
    enabled = {}
    for cond in preset.get("primary", []) + preset.get("confirm", []) + preset.get("gate", []):
        enabled[cond] = True
    return enabled


def scan_timeframe(fetcher: DataFetcher, symbol: str, interval: str,
                   enabled: dict, combo_name: str,
                   df: pd.DataFrame = None) -> tuple[dict | None, pd.DataFrame | None]:
    """Scan a single timeframe for signal. Returns (signal_info, sig_df) or (None, sig_df).

    If `df` is provided, skip fetching and use it directly (cache-friendly).
    """
    if df is None:
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
            return None, None

    if df is None or df.empty or len(df) < 30:
        print(f"  [{interval}] Insufficient data ({len(df) if df is not None else 0} bars)")
        return None, None

    sig_df = generate_combined_signals(
        df, **PARAMS, enabled=enabled, combo_mode=combo_name,
    )

    last = sig_df.iloc[-1]
    signal = int(last.get("signal", 0))

    if signal == 0:
        return None, sig_df

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
    }, sig_df


def scan_patterns(fetcher: DataFetcher, symbol: str, interval: str,
                  enabled: dict, combo_name: str,
                  sig_df: pd.DataFrame = None) -> list[dict]:
    """Scan a timeframe for candlestick patterns (independent of combo signals).

    Returns a list of detected patterns on the last bar, each with volume confirm status.
    If sig_df is provided, reuse it instead of fetching/computing again.
    """
    if sig_df is None:
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
            print(f"  [{interval}] Pattern scan error: {e}")
            return []

        if df is None or df.empty or len(df) < 30:
            return []

        sig_df = generate_combined_signals(
            df, **PARAMS, enabled=enabled, combo_mode=combo_name,
        )

    last = sig_df.iloc[-1]
    vol_confirm = bool(last.get("pat_volume_confirm", 0))
    price = float(last["close"])
    rsi = float(last.get("rsi", 0))
    atr = float(last.get("atr", 0))
    bar_time = str(last.get("time", last.name))

    detected = []

    # Bullish patterns
    if last.get("pat_morning_star", 0) == 1:
        detected.append({"name": "Morning Star", "direction": "BUY",
                         "vol_confirm": vol_confirm})
    if last.get("pat_bull_engulfing", 0) == 1:
        detected.append({"name": "Bullish Engulfing", "direction": "BUY",
                         "vol_confirm": vol_confirm})
    if last.get("pat_head_shoulders_bottom", 0) == 1:
        detected.append({"name": "Inv. Head & Shoulders", "direction": "BUY",
                         "vol_confirm": vol_confirm})

    # Bearish patterns
    if last.get("pat_evening_star", 0) == 1:
        detected.append({"name": "Evening Star", "direction": "SELL",
                         "vol_confirm": vol_confirm})
    if last.get("pat_bear_engulfing", 0) == 1:
        detected.append({"name": "Bearish Engulfing", "direction": "SELL",
                         "vol_confirm": vol_confirm})
    if last.get("pat_head_shoulders_top", 0) == 1:
        detected.append({"name": "Head & Shoulders", "direction": "SELL",
                         "vol_confirm": vol_confirm})

    # Attach common info
    for p in detected:
        p["interval"] = interval
        p["price"] = price
        p["rsi"] = rsi
        p["atr"] = atr
        p["time"] = bar_time

    return detected


def get_1m_entry(fetcher: DataFetcher, symbol: str, direction: str) -> dict | None:
    """Use 1m chart to find optimal Limit Order entry.

    Implements tiered TP/SL:
    - TP1=immediate (+1xATR), TP2=volume target (VP VAH/VAL or VWAP), TP3=extended
    - Stepped SL: initial -> breakeven after TP1 hit
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

    import pandas_ta as ta
    df["atr_1m"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # VWAP on 1m data
    try:
        _df_v = df.copy()
        if "time" in _df_v.columns and not isinstance(_df_v.index, pd.DatetimeIndex):
            _df_v.index = pd.to_datetime(_df_v["time"])
        _vwap = ta.vwap(_df_v["high"], _df_v["low"], _df_v["close"],
                        _df_v["volume"].astype(float))
        df["vwap_1m"] = _vwap.values if _vwap is not None else df["close"].values
    except Exception:
        df["vwap_1m"] = df["close"]

    last = df.iloc[-1]
    price = float(last["close"])
    atr_1m = float(last["atr_1m"]) if pd.notna(last.get("atr_1m")) else 0
    vwap_price = float(last.get("vwap_1m", price))

    if atr_1m <= 0:
        return None

    recent = df.tail(20)
    recent_low = float(recent["low"].min())
    recent_high = float(recent["high"].max())

    # Volume Profile: last 100 bars
    vp = compute_volume_profile(df, period=100, vol_pct=0.70)
    vp_poc = vp.get("poc")
    vp_vah = vp.get("vah")
    vp_val = vp.get("val")

    pullback = atr_1m * ENTRY_ATR_PULLBACK

    if direction == "BUY":
        limit_price = max(price - pullback, recent_low)
        sl = max(limit_price - SL_ATR_MULT * atr_1m, recent_low - atr_1m)
        # Tiered TP:
        tp1 = limit_price + 1.0 * atr_1m                          # immediate: +1R
        tp3 = limit_price + TP_ATR_MULT * atr_1m                  # extended: +nR
        if vp_vah is not None and vp_vah > limit_price + 0.5 * atr_1m:
            tp2 = min(vp_vah, tp3)                                 # VP VAH target
        elif vwap_price > limit_price + 0.5 * atr_1m:
            tp2 = min(vwap_price, tp3)                             # VWAP target
        else:
            tp2 = limit_price + 1.5 * atr_1m                      # mid fallback
    else:  # SELL
        limit_price = min(price + pullback, recent_high)
        sl = min(limit_price + SL_ATR_MULT * atr_1m, recent_high + atr_1m)
        tp1 = limit_price - 1.0 * atr_1m
        tp3 = limit_price - TP_ATR_MULT * atr_1m
        if vp_val is not None and vp_val < limit_price - 0.5 * atr_1m:
            tp2 = max(vp_val, tp3)                                 # VP VAL target
        elif vwap_price < limit_price - 0.5 * atr_1m:
            tp2 = max(vwap_price, tp3)                             # VWAP target
        else:
            tp2 = limit_price - 1.5 * atr_1m

    # Stepped SL: after TP1 hit -> move SL to breakeven
    sl2_breakeven = limit_price

    risk = abs(limit_price - sl)
    rr_tp2 = abs(tp2 - limit_price) / risk if risk > 0 else 0
    rr_tp3 = abs(tp3 - limit_price) / risk if risk > 0 else 0

    return {
        "current_price": price,
        "limit_price": limit_price,
        "sl": sl,
        "sl2_breakeven": sl2_breakeven,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "atr_1m": atr_1m,
        "rr_tp2": rr_tp2,
        "rr_tp3": rr_tp3,
        "rr_ratio": rr_tp2,
        "recent_low": recent_low,
        "recent_high": recent_high,
        "distance_pct": abs(limit_price - price) / price * 100,
        "vp_poc": vp_poc,
        "vp_vah": vp_vah,
        "vp_val": vp_val,
        "vwap": vwap_price,
    }


def _tp2_label(entry: dict, direction: str) -> str:
    """Return a short label describing the TP2 source (VP VAH/VAL, VWAP, or fallback)."""
    atr = entry.get("atr_1m", 1) or 1
    tp2 = entry.get("tp2", 0)
    vah = entry.get("vp_vah")
    val = entry.get("vp_val")
    vwap = entry.get("vwap")
    if direction == "BUY" and vah is not None and abs(tp2 - vah) < atr * 0.5:
        return "VP VAH"
    if direction == "SELL" and val is not None and abs(tp2 - val) < atr * 0.5:
        return "VP VAL"
    if vwap is not None and abs(tp2 - vwap) < atr * 0.5:
        return "VWAP"
    return "+1.5xATR"


def run_scan(fetcher: DataFetcher, notifier: TelegramNotifier, sent_alerts: dict):
    """Run one scan cycle: ALL combos + independent pattern detection."""
    print(f"\n[{vn_now().strftime('%H:%M:%S')}] Scanning {SYMBOL}...")

    # All combos that have primary conditions defined
    active_combos = [
        name for name, preset in COMBO_PRESETS.items()
        if preset.get("primary")
    ]

    # Pre-fetch OHLCV once per TF to stay within API rate limits (20 req/min)
    _now = vn_now()
    raw_data: dict = {}
    for _tf in SIGNAL_TIMEFRAMES:
        _days = 5 if _tf in ("1m", "5m") else 10
        _start = (_now - timedelta(days=_days)).strftime("%Y-%m-%d")
        _end   = _now.strftime("%Y-%m-%d")
        try:
            raw_data[_tf] = fetcher.get_futures_ohlcv(SYMBOL, _start, _end, interval=_tf)
        except Exception as _e:
            print(f"  [{_tf}] Fetch error: {_e}")
            raw_data[_tf] = None

    # PART A: Independent Pattern Detection (reuse cached data)
    pat_enabled = get_enabled_from_combo(active_combos[0])
    for tf in SIGNAL_TIMEFRAMES:
        _, sig_df = scan_timeframe(fetcher, SYMBOL, tf, pat_enabled, active_combos[0],
                                   df=raw_data.get(tf))
        patterns = scan_patterns(fetcher, SYMBOL, tf, pat_enabled, active_combos[0], sig_df=sig_df)
        if patterns:
            for pat in patterns:
                pat_key = f"PAT_{SYMBOL}_{pat['name']}_{pat['time']}"
                if pat_key in sent_alerts:
                    continue
                vol_str = "Vol OK" if pat['vol_confirm'] else "No Vol"
                msg = (
                    f"----------\n"
                    f"<b>{pat['direction']} - {pat['name']}</b>\n"
                    f"----------\n"
                    f"Symbol: <code>{SYMBOL}</code>\n"
                    f"Timeframe: <b>{pat['interval']}</b>\n"
                    f"Direction: <b>{pat['direction']}</b>\n"
                    f"Price: <code>{pat['price']:,.1f}</code>\n"
                    f"Volume: <b>{vol_str}</b>\n"
                    f"\n"
                    f"<i>RSI={pat['rsi']:.0f} | ATR={pat['atr']:.1f}</i>"
                )
                notifier.send(msg)
                sent_alerts[pat_key] = time.time()
                print(f"  [{tf}] Pattern: {pat['name']} ({pat['direction']}) "
                      f"vol={'OK' if pat['vol_confirm'] else 'no'}")
        else:
            print(f"  [{tf}] No pattern")

    # PART B: ALL Combo Signals (reuse cached data per TF)
    any_signal = False
    for combo_name in active_combos:
        combo_enabled = get_enabled_from_combo(combo_name)
        signals = []
        for tf in SIGNAL_TIMEFRAMES:
            result, _ = scan_timeframe(fetcher, SYMBOL, tf, combo_enabled, combo_name,
                                       df=raw_data.get(tf))
            if result:
                signals.append(result)

        if not signals:
            continue

        # Check direction agreement across TFs
        directions = set(s["signal"] for s in signals)
        if len(directions) > 1:
            continue

        direction = signals[0]["signal"]
        best_signal = max(signals, key=lambda x: x["confidence"])
        tfs_with_signal = ", ".join(s["interval"] for s in signals)

        # Dedup: unique per combo + direction + signal bar time
        alert_key = f"{SYMBOL}_{combo_name}_{direction}_{best_signal['time']}"
        if alert_key in sent_alerts:
            continue

        any_signal = True
        # Short combo label (e.g. "A" from "A: Trend Pullback (~65% WR)")
        combo_short = combo_name.split(":")[0].strip()

        print(f"  [{tfs_with_signal}] Combo {combo_short}: {direction} "
              f"(conf={best_signal['confidence']}, conds={len(best_signal['conditions'])})")

        # Get 1m entry
        entry = get_1m_entry(fetcher, SYMBOL, direction)
        if not entry:
            # Send without entry info
            msg = (
                f"----------\n"
                f"<b>{'BUY' if direction == 'BUY' else 'SELL'} Combo {combo_short}</b>\n"
                f"----------\n"
                f"Symbol: <code>{SYMBOL}</code>\n"
                f"Signal from: <b>{tfs_with_signal}</b>\n"
                f"Confidence: {best_signal['confidence']}/3\n"
                f"Conditions: {', '.join(best_signal['conditions']) or 'Score-based'}\n"
                f"Price: <code>{best_signal['price']:,.1f}</code>\n"
                f"\n"
                f"<i>RSI={best_signal['rsi']:.0f} | ADX={best_signal['adx']:.0f}</i>"
            )
        else:
            msg = (
                f"----------\n"
                f"<b>{'BUY' if direction == 'BUY' else 'SELL'} Combo {combo_short}</b>\n"
                f"----------\n"
                f"Symbol: <code>{SYMBOL}</code>\n"
                f"Signal from: <b>{tfs_with_signal}</b>\n"
                f"Confidence: {best_signal['confidence']}/3\n"
                f"Conditions: {', '.join(best_signal['conditions']) or 'Score-based'}\n"
                f"\n"
                f"<b>LIMIT ORDER</b>\n"
                f"  Entry: <code>{entry['limit_price']:,.1f}</code>\n"
                f"  Current: <code>{entry['current_price']:,.1f}</code>\n"
                f"  Distance: {entry['distance_pct']:.2f}%\n"
                f"\n"
                f"  SL1:  <code>{entry['sl']:,.1f}</code>  (initial -{SL_ATR_MULT}xATR)\n"
                f"  SL2:  <code>{entry['sl2_breakeven']:,.1f}</code>  (breakeven after TP1)\n"
                f"\n"
                f"  TP1:  <code>{entry['tp1']:,.1f}</code>  (+1xATR) [then SL->BE]\n"
                f"  TP2:  <code>{entry['tp2']:,.1f}</code>  ({_tp2_label(entry, direction)}) [main target]\n"
                f"  TP3:  <code>{entry['tp3']:,.1f}</code>  (+{TP_ATR_MULT:.0f}xATR) [extended]\n"
                f"\n"
                f"  R:R (TP2) = <b>{entry['rr_tp2']:.1f}:1</b>  |  TP3 = {entry['rr_tp3']:.1f}:1\n"
                f"\n"
                f"<i>PoC={'N/A' if entry.get('vp_poc') is None else '{:,.1f}'.format(entry['vp_poc'])}"
                f" | VWAP={entry['vwap']:,.1f} | ATR={entry['atr_1m']:.1f}</i>\n"
                f"<i>RSI={best_signal['rsi']:.0f} | ADX={best_signal['adx']:.0f} | "
                f"Range: {entry['recent_low']:,.1f}-{entry['recent_high']:,.1f}</i>"
            )

        notifier.send(msg)
        sent_alerts[alert_key] = time.time()
        print(f"  -> ALERT SENT! (Combo {combo_short})")

    if not any_signal:
        print("  No combo signal from any strategy.")

    # Cleanup old alerts (older than 1 hour)
    cutoff = time.time() - 3600
    expired = [k for k, t in sent_alerts.items() if t < cutoff]
    for k in expired:
        del sent_alerts[k]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VN30F1M Multi-TF Signal Scanner")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    args = parser.parse_args()

    # All combos with primary conditions
    active_combos = [name for name, p in COMBO_PRESETS.items() if p.get("primary")]

    fetcher = DataFetcher()
    notifier = TelegramNotifier()

    if not notifier.is_configured():
        print("WARNING: Telegram not configured. Alerts will only print to console.")
    else:
        print(f"Telegram configured. Chat ID: {notifier.chat_id}")

    print(f"Scanner started: {SYMBOL}")
    print(f"Signal TFs: {', '.join(SIGNAL_TIMEFRAMES)}")
    print(f"Entry TF: {ENTRY_TIMEFRAME}")
    print(f"Combos: {[n.split(':')[0].strip() for n in active_combos]}")
    print(f"Patterns: Independent (all TFs)")
    print(f"Interval: {SCAN_INTERVAL}s")
    print(f"Entry: Limit @ {ENTRY_ATR_PULLBACK}*ATR pullback | SL {SL_ATR_MULT}*ATR | TP {TP_ATR_MULT}*ATR")
    print("=" * 40)

    sent_alerts = {}

    if args.once:
        run_scan(fetcher, notifier, sent_alerts)
        return

    # Send startup notification
    combo_labels = ", ".join(n.split(':')[0].strip() for n in active_combos)
    notifier.send(
        f"<b>Scanner Started</b>\n"
        f"Symbol: <code>{SYMBOL}</code>\n"
        f"Signal: {', '.join(SIGNAL_TIMEFRAMES)} - Entry: {ENTRY_TIMEFRAME}\n"
        f"Combos: {combo_labels} + Patterns\n"
        f"Interval: {SCAN_INTERVAL}s\n"
        f"Limit: {ENTRY_ATR_PULLBACK}xATR pullback | R:R {TP_ATR_MULT/SL_ATR_MULT:.0f}:1"
    )

    while True:
        try:
            if not is_trading_hours():
                now = vn_now()
                print(f"\r[{now.strftime('%H:%M:%S')}] Outside trading hours (9:00-11:30 / 13:00-14:30). Waiting...", end="")
                time.sleep(60)
                continue

            run_scan(fetcher, notifier, sent_alerts)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            time.sleep(SCAN_INTERVAL)
            continue

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
