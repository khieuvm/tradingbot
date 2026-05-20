"""
Multi-Timeframe Signal Scanner v2
===================================
Strategy:
  1. Scan 1m + 5m + 15m for ALL combos independently
  2. Rate signal strength by TF agreement:
     - 1 TF  = NORMAL
     - 2 TFs = STRONG
     - 3 TFs = SUPER STRONG (all timeframes agree)
  3. Entry via ATR pullback on lowest TF with signal
  4. Notification includes: R:R, conditions fired, reasoning

Usage:
    py scanner.py
    py scanner.py --once
"""

import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Force UTF-8 on Windows console to handle emoji/Unicode from vnstock banner
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import yaml

# Suppress noisy third-party warnings from pandas_ta / pandas internals
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="Downcasting object dtype arrays", category=FutureWarning)

from config import Config
from src.data_fetcher import DataFetcher
from src.signals import (
    COMBO_PRESETS, COND_LABELS, ALL_COND_KEYS,
    generate_combined_signals,
    compute_volume_profile,
)
from src.notifier import TelegramNotifier
from src.portfolio_manager import PortfolioManager
from src.position_manager import PositionManager
from src.trade_logger import TradeLogger

# --- LOAD CONFIG FROM YAML ----------------------------------------
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
    global COMBO_TF_MAP

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

    # Combo-TF effectiveness map (from YAML or use default)
    tf_map = cfg.get("combo_tf_map", {})
    if tf_map:
        COMBO_TF_MAP.update(tf_map)

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
SIGNAL_TIMEFRAMES = ["1m", "5m", "15m"]
ENTRY_TIMEFRAME = "1m"
COMBO = "all"
SCAN_INTERVAL = 35
MARKET_OPEN = (9, 0)
MARKET_CLOSE = (14, 30)
ENTRY_ATR_PULLBACK = 0.5
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
MIN_COMBOS_ENTRY = 2       # Minimum combos agreeing to simulate a trade entry
PARAMS = {
    "fast_ma": 10, "slow_ma": 20, "rsi_period": 7,
    "oversold": 35, "overbought": 70,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "vol_mult": 1.5,
}

# Combo -> Timeframe effectiveness map (44-day portfolio test, updated May 21 2026)
# Result: 66 trades, WR 49%, Net +15.1M, MaxDD -3.4M, Sharpe 4.14
COMBO_TF_MAP = {
    "G": ["5m", "15m"],      # Main driver: +13.7M, PF 1.55
    "F": ["15m"],            # +1.1M in portfolio context
    "M": ["5m"],             # +1.4M, 60% WR
    "K": ["15m"],            # +1.8M, very selective
    "G+": ["5m", "15m"],     # +6.8M standalone, PF 2.20-2.46
    # DISABLED: F+(PF 0.98, -1.1M portfolio), L(reversal conflicts),
    #           H(too noisy), I(negative), B(negative), C(low PF)
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

    # Drop the last (incomplete/forming) candle during market hours
    # to avoid false signals from partial data
    if is_trading_hours() and len(df) > 30:
        df = df.iloc[:-1]

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


def run_scan(fetcher: DataFetcher, notifier: TelegramNotifier, sent_alerts: dict,
             position_manager: PositionManager | None = None,
             portfolio_mgr: PortfolioManager | None = None):
    """Run one scan cycle: ALL combos x ALL timeframes, then consolidated alert per direction."""
    print(f"\n[{vn_now().strftime('%H:%M:%S')}] Scanning {SYMBOL}...")

    # All combos that have primary conditions defined
    active_combos = [
        name for name, preset in COMBO_PRESETS.items()
        if preset.get("primary")
    ]

    # --- Portfolio price update (check SL/TP for open positions) ---
    if portfolio_mgr and portfolio_mgr.n_open > 0:
        try:
            _now_pm = vn_now()
            _start_pm = (_now_pm - timedelta(days=1)).strftime("%Y-%m-%d")
            _end_pm = _now_pm.strftime("%Y-%m-%d")
            _df_pm = fetcher.get_futures_ohlcv(SYMBOL, _start_pm, _end_pm, interval="1m")
            if _df_pm is not None and len(_df_pm) > 1:
                _last = _df_pm.iloc[-1]
                _atr = float((_df_pm["high"] - _df_pm["low"]).rolling(14).mean().iloc[-1])
                portfolio_mgr.update_prices(
                    SYMBOL,
                    high=float(_last["high"]),
                    low=float(_last["low"]),
                    close=float(_last["close"]),
                    atr=_atr,
                )
            portfolio_mgr.tick()  # decrease cooldown
        except Exception as _e:
            print(f"  [PORTFOLIO UPDATE ERROR] {_e}")
    elif portfolio_mgr:
        portfolio_mgr.tick()

    # Pre-fetch OHLCV once per TF to stay within API rate limits
    _now = vn_now()
    raw_data: dict = {}
    for _tf in SIGNAL_TIMEFRAMES:
        _days = 3 if _tf == "1m" else (5 if _tf == "5m" else 10)
        _start = (_now - timedelta(days=_days)).strftime("%Y-%m-%d")
        _end = _now.strftime("%Y-%m-%d")
        try:
            raw_data[_tf] = fetcher.get_futures_ohlcv(SYMBOL, _start, _end, interval=_tf)
        except Exception as _e:
            print(f"  [{_tf}] Fetch error: {_e}")
            raw_data[_tf] = None

    # --- Pass 1: Scan ALL combos, collect results (no alert yet) ---
    collected: list[dict] = []

    for combo_name in active_combos:
        combo_enabled = get_enabled_from_combo(combo_name)
        combo_short = combo_name.split(":")[0].strip()

        # Filter TFs based on backtest effectiveness
        allowed_tfs = COMBO_TF_MAP.get(combo_short, SIGNAL_TIMEFRAMES)

        # Collect signals per TF
        tf_signals = {}
        for tf in SIGNAL_TIMEFRAMES:
            if tf not in allowed_tfs or raw_data.get(tf) is None:
                continue
            result, _ = scan_timeframe(fetcher, SYMBOL, tf, combo_enabled, combo_name,
                                       df=raw_data.get(tf))
            if result:
                tf_signals[tf] = result

        if not tf_signals:
            continue

        buy_tfs  = [tf for tf, s in tf_signals.items() if s["signal"] == "BUY"]
        sell_tfs = [tf for tf, s in tf_signals.items() if s["signal"] == "SELL"]

        if len(buy_tfs) >= len(sell_tfs) and buy_tfs:
            direction, aligned_tfs = "BUY", buy_tfs
        elif sell_tfs:
            direction, aligned_tfs = "SELL", sell_tfs
        else:
            continue

        n_agree  = len(aligned_tfs)
        max_tfs  = len(allowed_tfs)
        best_tf  = max(aligned_tfs, key=lambda tf: tf_signals[tf]["confidence"])
        best_sig = tf_signals[best_tf]

        all_conds: set[str] = set()
        for tf in aligned_tfs:
            all_conds.update(tf_signals[tf]["conditions"])

        highest_tf = aligned_tfs[-1]
        alert_key  = f"{SYMBOL}_{combo_short}_{direction}_{tf_signals[highest_tf]['time']}"

        collected.append({
            "direction":   direction,
            "combo_name":  combo_name,
            "combo_short": combo_short,
            "aligned_tfs": aligned_tfs,
            "n_agree":     n_agree,
            "max_tfs":     max_tfs,
            "best_sig":    best_sig,
            "all_conds":   all_conds,
            "alert_key":   alert_key,
        })

    # --- Pass 2: Group by direction, send ONE consolidated alert per direction ---
    buy_items  = [c for c in collected if c["direction"] == "BUY"]
    sell_items = [c for c in collected if c["direction"] == "SELL"]

    any_signal = False
    for direction, items in [("BUY", buy_items), ("SELL", sell_items)]:
        if not items:
            continue

        # Only send if at least one combo-key is new (dedup)
        new_items = [i for i in items if i["alert_key"] not in sent_alerts]
        if not new_items:
            continue

        any_signal = True

        # --- Overall strength score ---
        # Score per combo = n_agree * confidence (max 3*3=9)
        total_score  = sum(i["n_agree"] * i["best_sig"]["confidence"] for i in items)
        max_agree    = max(i["n_agree"] for i in items)
        n_combos     = len(items)
        n_all_combos = len(active_combos)

        if max_agree >= 3 or (max_agree == 2 and n_combos >= 2):
            overall_strength = "SUPER STRONG"
            dir_icon = "\U0001f7e2\U0001f7e2\U0001f7e2" if direction == "BUY" else "\U0001f534\U0001f534\U0001f534"
        elif max_agree == 2 or n_combos >= 2:
            overall_strength = "STRONG"
            dir_icon = "\U0001f7e2\U0001f7e2" if direction == "BUY" else "\U0001f534\U0001f534"
        else:
            overall_strength = "NORMAL"
            dir_icon = "\U0001f7e2" if direction == "BUY" else "\U0001f534"

        # --- Per-combo summary lines ---
        combo_lines = []
        for i in sorted(items, key=lambda x: x["n_agree"] * x["best_sig"]["confidence"], reverse=True):
            stars = "\u2605" * i["best_sig"]["confidence"] + "\u2606" * (3 - i["best_sig"]["confidence"])
            tfs_str = ",".join(i["aligned_tfs"])
            combo_lines.append(
                f"  <b>{i['combo_short']}</b>: {i['n_agree']}/{i['max_tfs']} TF [{tfs_str}] {stars}"
            )

        # Best combo overall (highest score) — used for conditions display & position
        best_item = max(items, key=lambda x: x["n_agree"] * x["best_sig"]["confidence"])
        ref_sig   = best_item["best_sig"]
        all_conds_merged: set[str] = set()
        for i in items:
            all_conds_merged.update(i["all_conds"])

        print(f"  [{overall_strength}] {direction}: {n_combos} combos "
              f"(score={total_score}) -> ALERT")

        # --- Entry / risk info ---
        entry = get_1m_entry(fetcher, SYMBOL, direction)

        if entry and entry.get("atr_1m", 0) > 0:
            risk_pts   = abs(entry["limit_price"] - entry["sl"])
            tp2_src    = _tp2_label(entry, direction)
            rr_str     = f"{entry['rr_tp2']:.1f}:1"
            entry_block = (
                f"\n<b>ENTRY:</b>\n"
                f"  Limit: <code>{entry['limit_price']:,.1f}</code>"
                f"  (now: <code>{entry['current_price']:,.1f}</code>)\n"
                f"\n<b>RISK / REWARD:</b>\n"
                f"  SL:  <code>{entry['sl']:,.1f}</code>  (-{risk_pts:.1f} pts)\n"
                f"  TP1: <code>{entry['tp1']:,.1f}</code>  (+1xATR) \u2192 SL\u2192BE\n"
                f"  TP2: <code>{entry['tp2']:,.1f}</code>  ({tp2_src})\n"
                f"  TP3: <code>{entry['tp3']:,.1f}</code>  (+{TP_ATR_MULT:.0f}xATR)\n"
                f"  <b>R:R = {rr_str}</b>"
            )
            indicator_line = (
                f"<i>RSI={ref_sig['rsi']:.0f} | ADX={ref_sig['adx']:.0f} | "
                f"ATR={entry['atr_1m']:.1f} | VWAP={entry['vwap']:,.1f}</i>"
            )
        else:
            risk_pts  = ref_sig["atr"] * SL_ATR_MULT
            reward_pts = ref_sig["atr"] * TP_ATR_MULT
            rr_str    = f"{TP_ATR_MULT/SL_ATR_MULT:.1f}:1"
            entry_block = (
                f"\n<b>RISK / REWARD (est.):</b>\n"
                f"  Price: <code>{ref_sig['price']:,.1f}</code>\n"
                f"  SL: ~{risk_pts:.1f} pts | TP: ~{reward_pts:.1f} pts\n"
                f"  <b>R:R = {rr_str}</b>"
            )
            indicator_line = (
                f"<i>RSI={ref_sig['rsi']:.0f} | ADX={ref_sig['adx']:.0f} | "
                f"ATR={ref_sig['atr']:.1f}</i>"
            )

        conds_str = ", ".join(sorted(all_conds_merged)) if all_conds_merged else "Score-based"

        # --- Build consolidated message ---
        sep = "\u2500" * 22
        msg = (
            f"{dir_icon} <b>{direction} \u2014 {SYMBOL}</b>\n"
            f"{sep}\n"
            f"<b>Combos ({n_combos}/{n_all_combos} agree):</b>\n"
            + "\n".join(combo_lines) + "\n"
            f"\n"
            f"<b>Overall:</b> {overall_strength} | Score: {total_score}\n"
            f"<b>Conditions:</b> {conds_str}\n"
            + entry_block + "\n"
            f"\n"
            + indicator_line
        )

        notifier.send(msg)

        # Mark all keys for this direction as sent
        for i in items:
            sent_alerts[i["alert_key"]] = time.time()

        # --- Portfolio position management ---
        if portfolio_mgr and entry and n_combos >= MIN_COMBOS_ENTRY:
            direction_int = 1 if direction == "BUY" else -1
            best_tf = best_item["aligned_tfs"][-1]  # highest TF
            best_conf = best_item["best_sig"]["confidence"]

            # Check if we need to flip direction
            if portfolio_mgr.should_flip(direction_int, best_tf, best_conf):
                portfolio_mgr.execute_flip(entry["current_price"])

            # Try to open position (respects direction lock + capacity)
            if portfolio_mgr.can_open(direction_int):
                portfolio_mgr.open_position(
                    symbol=SYMBOL,
                    direction=direction_int,
                    entry_price=entry["limit_price"],
                    atr=entry["atr_1m"],
                    combo=best_item["combo_short"],
                    timeframe=best_tf,
                    confidence=best_conf,
                )
            elif portfolio_mgr.in_cooldown:
                print(f"  [COOLDOWN] Signal rejected (cooldown={portfolio_mgr.cooldown_remaining})")

    if not any_signal:
        print(f"  No signal. Portfolio: {portfolio_mgr.status_str() if portfolio_mgr else 'N/A'}")

    # Cleanup old alerts (older than 30 min)
    cutoff = time.time() - 1800
    for k in [k for k, t in sent_alerts.items() if t < cutoff]:
        del sent_alerts[k]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VN30F1M Multi-TF Signal Scanner v2")
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

    print(f"Scanner v2 started: {SYMBOL}")
    print(f"Timeframes: {', '.join(SIGNAL_TIMEFRAMES)} (multi-TF agreement)")
    print(f"Combos (TF-filtered by backtest):")
    for c in active_combos:
        cs = c.split(':')[0].strip()
        tfs = COMBO_TF_MAP.get(cs, SIGNAL_TIMEFRAMES)
        print(f"  {cs}: {', '.join(tfs)}")
    print(f"Interval: {SCAN_INTERVAL}s")
    print(f"Strength: 1TF=Normal, 2TF=Strong, 3TF=Super Strong")
    print(f"Risk: SL {SL_ATR_MULT}*ATR | TP {TP_ATR_MULT}*ATR | R:R {TP_ATR_MULT/SL_ATR_MULT:.1f}:1")
    print("=" * 50)

    trade_logger = TradeLogger(log_dir="logs")
    portfolio_manager = PortfolioManager(
        notifier=notifier,
        logger=trade_logger,
        max_contracts=3,
        flip_cooldown=3,
    )
    # Legacy single-position manager (kept for compatibility)
    position_manager = PositionManager(
        notifier=notifier,
        logger=trade_logger,
        sl_atr_mult=SL_ATR_MULT,
        tp_atr_mult=TP_ATR_MULT,
    )

    sent_alerts = {}

    if args.once:
        run_scan(fetcher, notifier, sent_alerts, position_manager,
                 portfolio_mgr=portfolio_manager)
        return

    # Send startup notification
    combo_labels = ", ".join(n.split(':')[0].strip() for n in active_combos)
    notifier.send(
        f"<b>Scanner v3 Started</b>\n"
        f"Symbol: <code>{SYMBOL}</code>\n"
        f"TFs: {', '.join(SIGNAL_TIMEFRAMES)} (multi-TF agreement)\n"
        f"Combos: {combo_labels}\n"
        f"Scan: every {SCAN_INTERVAL}s\n"
        f"Portfolio: max {portfolio_manager.max_contracts} contracts, "
        f"cooldown {portfolio_manager.flip_cooldown} bars\n"
        f"Status: {portfolio_manager.status_str()}"
    )

    _eod_sent_date: str = ""  # track which date we already sent EOD summary

    while True:
        try:
            now = vn_now()

            # --- End-of-day: close all positions + send summary ---
            today_str = now.strftime("%Y-%m-%d")
            after_close = (now.hour, now.minute) >= (14, 31)
            if after_close and now.weekday() < 5 and _eod_sent_date != today_str:
                _eod_sent_date = today_str
                # Close any remaining positions
                if portfolio_manager.n_open > 0:
                    try:
                        _df_eod = fetcher.get_futures_ohlcv(
                            SYMBOL, today_str, today_str, interval="1m")
                        if _df_eod is not None and len(_df_eod) > 0:
                            last_price = float(_df_eod.iloc[-1]["close"])
                            portfolio_manager.close_all(last_price, reason="EOD")
                    except Exception as _e:
                        print(f"  [EOD CLOSE ERROR] {_e}")
                # Send daily summary
                summary = trade_logger.daily_summary(today_str)
                notifier.send(summary)
                print(f"[EOD] Daily summary sent for {today_str}")

            if not is_trading_hours():
                print(f"\r[{now.strftime('%H:%M:%S')}] Outside trading hours. Waiting...", end="")
                time.sleep(60)
                continue

            run_scan(fetcher, notifier, sent_alerts, position_manager,
                     portfolio_mgr=portfolio_manager)
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
