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
    global COMBO_TF_MAP, COMBO_RISK

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

    # Combo-TF effectiveness map (from YAML overrides hardcoded defaults)
    tf_map = cfg.get("combo_tf_map", {})
    if tf_map:
        COMBO_TF_MAP.clear()
        COMBO_TF_MAP.update(tf_map)

    # Combo risk params (per-combo/TF SL/TP + hyperopt params)
    COMBO_RISK = cfg.get("combo_risk", {})

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
COMBO_RISK = {}            # Per-combo/TF risk params from YAML
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


def get_1m_entry(fetcher: DataFetcher, symbol: str, direction: str,
                 sl_mult: float = None, tp_mult: float = None,
                 pullback_atr: float = None) -> dict | None:
    """Use 1m chart to find optimal Limit Order entry.

    Implements tiered TP/SL:
    - TP1=immediate (+1xATR), TP2=volume target (VP VAH/VAL or VWAP), TP3=extended
    - Stepped SL: initial -> breakeven after TP1 hit
    """
    _sl = sl_mult if sl_mult is not None else SL_ATR_MULT
    _tp = tp_mult if tp_mult is not None else TP_ATR_MULT
    _pullback_mult = pullback_atr if pullback_atr is not None else ENTRY_ATR_PULLBACK
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

    pullback = atr_1m * _pullback_mult

    if direction == "BUY":
        limit_price = max(price - pullback, recent_low)
        sl = max(limit_price - _sl * atr_1m, recent_low - atr_1m)
        # Tiered TP:
        tp1 = limit_price + 1.0 * atr_1m                          # immediate: +1R
        tp3 = limit_price + _tp * atr_1m                           # extended: +nR
        if vp_vah is not None and vp_vah > limit_price + 0.5 * atr_1m:
            tp2 = min(vp_vah, tp3)                                 # VP VAH target
        elif vwap_price > limit_price + 0.5 * atr_1m:
            tp2 = min(vwap_price, tp3)                             # VWAP target
        else:
            tp2 = limit_price + 1.5 * atr_1m                      # mid fallback
    else:  # SELL
        limit_price = min(price + pullback, recent_high)
        sl = min(limit_price + _sl * atr_1m, recent_high + atr_1m)
        tp1 = limit_price - 1.0 * atr_1m
        tp3 = limit_price - _tp * atr_1m
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

    # Skip if TP distance < 3 points
    tp_distance = abs(tp3 - limit_price)
    if tp_distance < 3.0:
        return None

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
             portfolio_mgr: PortfolioManager | None = None,
             pending_signals: dict | None = None):
    """Run one scan cycle: ALL combos x ALL timeframes, then consolidated alert per direction.

    pending_signals: dict persisting across cycles for next-bar confirmation.
    Keys are combo_short, values are dicts with trigger info.
    """
    if pending_signals is None:
        pending_signals = {}
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

        # Skip combos not in COMBO_TF_MAP (disabled)
        if combo_short not in COMBO_TF_MAP:
            continue

        # Filter TFs based on backtest effectiveness
        allowed_tfs = COMBO_TF_MAP[combo_short]

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

    # --- Pass 2: Next-bar confirmation + entry ---
    # Step A: Check if any PENDING signals from previous cycle are now confirmed
    any_signal = False
    confirmed_items = []

    for pkey in list(pending_signals.keys()):
        ps = pending_signals[pkey]
        direction = ps["direction"]
        trigger = ps["trigger_price"]
        combo_short = ps["combo_short"]
        best_tf = ps["best_tf"]

        # Get current price from raw_data
        tf_data = raw_data.get(best_tf)
        if tf_data is None or tf_data.empty:
            # Can't verify, expire pending
            del pending_signals[pkey]
            continue

        last_bar = tf_data.iloc[-1]
        current_high = float(last_bar["high"])
        current_low = float(last_bar["low"])

        # Check confirmation: current bar broke trigger?
        confirmed = False
        if direction == "BUY" and current_high > trigger:
            confirmed = True
        elif direction == "SELL" and current_low < trigger:
            confirmed = True

        if confirmed:
            confirmed_items.append(ps)
            del pending_signals[pkey]
            print(f"  [{combo_short}/{best_tf}] {direction} CONFIRMED (trigger {trigger:.1f})")
        else:
            # Expire if too old (1 bar window = expired on next cycle after detection)
            ps["age"] = ps.get("age", 0) + 1
            if ps["age"] >= 2:
                del pending_signals[pkey]
                print(f"  [{combo_short}/{best_tf}] {direction} EXPIRED (no confirmation)")

    # Step B: Process confirmed signals (alert + entry)
    for ps in confirmed_items:
        direction = ps["direction"]
        combo_name = ps["combo_name"]
        combo_short = ps["combo_short"]
        alert_key = ps["alert_key"]
        aligned_tfs = ps["aligned_tfs"]
        n_agree = ps["n_agree"]
        max_tfs = ps["max_tfs"]
        best_sig = ps["best_sig"]
        all_conds = ps["all_conds"]

        # Dedup
        if alert_key in sent_alerts:
            continue

        any_signal = True

        # --- Strength rating ---
        conf = best_sig["confidence"]
        if n_agree >= 3 or (n_agree == 2 and conf >= 3):
            strength = "SUPER STRONG"
            stars = "\u2b50\u2b50\u2b50"
        elif n_agree >= 2 or conf >= 3:
            strength = "STRONG"
            stars = "\u2b50\u2b50"
        else:
            strength = "NORMAL"
            stars = "\u2b50"

        dir_icon = "\U0001f7e2" if direction == "BUY" else "\U0001f534"

        # --- Per-combo risk params ---
        best_tf = ps["best_tf"]
        risk_key = f"{combo_short}/{best_tf}"
        combo_risk_cfg = COMBO_RISK.get(risk_key, {})
        _sl_mult = combo_risk_cfg.get("sl_atr_mult", SL_ATR_MULT)
        _tp_mult = combo_risk_cfg.get("tp_atr_mult", TP_ATR_MULT)
        _max_hold = combo_risk_cfg.get("max_hold", 30)

        # --- Dynamic limit offset based on signal bar strength ---
        dyn_offset = ps["dynamic_offset"]

        # --- Entry calculation with dynamic offset ---
        entry = get_1m_entry(fetcher, SYMBOL, direction,
                             sl_mult=_sl_mult, tp_mult=_tp_mult,
                             pullback_atr=dyn_offset)

        # --- Build REASON block ---
        preset = COMBO_PRESETS.get(combo_name, {})
        combo_desc = preset.get("desc", "")
        primary_conds = [COND_LABELS.get(c, c) for c in preset.get("primary", [])]
        confirm_conds = [COND_LABELS.get(c, c) for c in preset.get("confirm", [])]
        gate_conds = [COND_LABELS.get(c, c) for c in preset.get("gate", [])]
        fired_list = sorted(all_conds)

        reason_block = (
            f"<b>LÝ DO VÀO LỆNH:</b>\n"
            f"  {combo_desc}\n"
            f"  \u2714 Đã kích hoạt: {', '.join(fired_list)}\n"
            f"  Tín hiệu chính: {', '.join(primary_conds)}\n"
            f"  Xác nhận: {', '.join(confirm_conds)}\n"
            f"  Điều kiện gate: {', '.join(gate_conds)}"
        )

        # --- CONFIDENCE block ---
        tfs_str = ", ".join(aligned_tfs)
        conf_block = (
            f"\n<b>ĐỘ TIN CẬY:</b> {stars} ({strength})\n"
            f"  TF đồng thuận: {n_agree}/{max_tfs} [{tfs_str}]\n"
            f"  Điểm signal: {conf}/3\n"
            f"  RSI={best_sig['rsi']:.0f} | ADX={best_sig['adx']:.0f}\n"
            f"  Entry offset: {dyn_offset:.2f} ATR (dynamic)"
        )

        # --- RISK block ---
        if entry and entry.get("atr_1m", 0) > 0:
            risk_pts = abs(entry["limit_price"] - entry["sl"])
            reward_pts = abs(entry["tp2"] - entry["limit_price"])
            rr = reward_pts / risk_pts if risk_pts > 0 else 0
            tp2_src = _tp2_label(entry, direction)
            loss_vnd = risk_pts * 100_000
            gain_vnd = reward_pts * 100_000
            hold_min = _max_hold * (3 if "3m" in best_tf else 5 if "5m" in best_tf else 15)

            risk_block = (
                f"\n<b>VÀO LỆNH & RỦI RO:</b>\n"
                f"  Entry: <code>{entry['limit_price']:,.1f}</code> (hiện tại: {entry['current_price']:,.1f})\n"
                f"  SL: <code>{entry['sl']:,.1f}</code> (-{risk_pts:.1f}pts = -{loss_vnd:,.0f}\u20ab)\n"
                f"  TP: <code>{entry['tp2']:,.1f}</code> (+{reward_pts:.1f}pts = +{gain_vnd:,.0f}\u20ab) [{tp2_src}]\n"
                f"  R:R = <b>{rr:.1f}:1</b> | Giữ tối đa: ~{hold_min} phút\n"
            )

            if direction == "BUY":
                risk_scenarios = (
                    f"\n<b>RỦI RO CÓ THỂ XẢY RA:</b>\n"
                    f"  \u26a0 SL hit ({risk_pts:.1f}pts): Giá break support, trend tiếp tục giảm\n"
                    f"  \u26a0 Timeout: Sideway không TP, close \u00b10 sau {hold_min}p\n"
                    f"  \u26a0 Trap: Spike xuống quét SL rồi quay lên"
                )
            else:
                risk_scenarios = (
                    f"\n<b>RỦI RO CÓ THỂ XẢY RA:</b>\n"
                    f"  \u26a0 SL hit ({risk_pts:.1f}pts): Giá break resistance, trend tiếp tục tăng\n"
                    f"  \u26a0 Timeout: Sideway không TP, close \u00b10 sau {hold_min}p\n"
                    f"  \u26a0 Trap: Spike lên quét SL rồi quay xuống"
                )
        else:
            risk_pts = best_sig["atr"] * _sl_mult
            reward_pts = best_sig["atr"] * _tp_mult
            rr = _tp_mult / _sl_mult
            loss_vnd = risk_pts * 100_000
            gain_vnd = reward_pts * 100_000

            risk_block = (
                f"\n<b>VÀO LỆNH & RỦI RO (ước tính):</b>\n"
                f"  Giá: <code>{best_sig['price']:,.1f}</code>\n"
                f"  SL: ~{risk_pts:.1f}pts (-{loss_vnd:,.0f}\u20ab) | TP: ~{reward_pts:.1f}pts (+{gain_vnd:,.0f}\u20ab)\n"
                f"  R:R = <b>{rr:.1f}:1</b>"
            )
            risk_scenarios = (
                f"\n<b>RỦI RO:</b>\n"
                f"  \u26a0 Max loss nếu SL: -{loss_vnd:,.0f}\u20ab ({risk_pts:.1f}pts)"
            )

        bt_block = ""

        # --- Final message ---
        sep = "\u2500" * 24
        msg = (
            f"{dir_icon} <b>{direction} {combo_short} [{best_tf}] \u2014 {SYMBOL}</b>\n"
            f"{sep}\n"
            f"{reason_block}\n"
            f"{conf_block}\n"
            f"{risk_block}"
            f"{risk_scenarios}"
            f"{bt_block}"
        )

        print(f"  [{combo_short}/{best_tf}] {direction} {strength} conf={conf} -> ALERT")
        notifier.send(msg)
        sent_alerts[alert_key] = time.time()

        # --- Portfolio position management ---
        if portfolio_mgr and entry and len(confirmed_items) >= 1:
            direction_int = 1 if direction == "BUY" else -1

            if portfolio_mgr.should_flip(direction_int, best_tf, conf):
                portfolio_mgr.execute_flip(entry["current_price"])

            if portfolio_mgr.can_open(direction_int):
                portfolio_mgr.open_position(
                    symbol=SYMBOL,
                    direction=direction_int,
                    entry_price=entry["limit_price"],
                    atr=entry["atr_1m"],
                    combo=combo_short,
                    timeframe=best_tf,
                    confidence=conf,
                )
            elif portfolio_mgr.in_cooldown:
                print(f"  [COOLDOWN] Signal rejected (cooldown={portfolio_mgr.cooldown_remaining})")

    # Step C: Store NEW signals as pending (wait for next-bar confirmation)
    for item in collected:
        direction = item["direction"]
        combo_name = item["combo_name"]
        combo_short = item["combo_short"]
        alert_key = item["alert_key"]

        # Skip if already sent or already pending
        if alert_key in sent_alerts:
            continue
        pkey = f"{combo_short}_{direction}"
        if pkey in pending_signals:
            continue

        best_sig = item["best_sig"]
        aligned_tfs = item["aligned_tfs"]
        best_tf = aligned_tfs[-1]
        sig_price = best_sig["price"]
        sig_atr = best_sig["atr"]

        # Exhaustion filter: check recent signals from raw data
        tf_data = raw_data.get(best_tf)
        skip_exhaustion = False
        if tf_data is not None and len(tf_data) > 10:
            # Re-generate signals to check for consecutive same-direction
            combo_enabled = get_enabled_from_combo(combo_name)
            sig_check = generate_combined_signals(
                tf_data.iloc[-15:].copy(), **PARAMS,
                enabled=combo_enabled, combo_mode=combo_name,
            )
            direction_int = 1 if direction == "BUY" else -1
            recent_sigs = sig_check["signal"].iloc[-6:-1]  # last 5 bars before current
            same_count = 0
            for s in reversed(recent_sigs.values):
                if int(s) == direction_int:
                    same_count += 1
                else:
                    break
            if same_count >= 2:
                skip_exhaustion = True
                print(f"  [{combo_short}/{best_tf}] {direction} SKIP (exhaustion: {same_count} consecutive)")

        if skip_exhaustion:
            continue

        # Determine trigger price (signal bar high/low)
        if tf_data is not None and not tf_data.empty:
            last_bar = tf_data.iloc[-1]
            if direction == "BUY":
                trigger = float(last_bar["high"]) + 0.1
            else:
                trigger = float(last_bar["low"]) - 0.1
        else:
            # Fallback: use price + small offset
            trigger = sig_price + 0.1 if direction == "BUY" else sig_price - 0.1

        # Compute dynamic offset from signal bar characteristics
        if tf_data is not None and not tf_data.empty:
            last_bar = tf_data.iloc[-1]
            bar_open = float(last_bar["open"])
            bar_close = float(last_bar["close"])
            bar_high = float(last_bar["high"])
            bar_low = float(last_bar["low"])
            body = abs(bar_close - bar_open)
            bar_range = bar_high - bar_low
            body_ratio = body / max(bar_range, 0.01)

            # Volume ratio
            if len(tf_data) >= 20:
                vol_avg = tf_data.iloc[-21:-1]["volume"].astype(float).mean()
            else:
                vol_avg = float(last_bar["volume"])
            vol_ratio = float(last_bar["volume"]) / max(vol_avg, 1)

            # Dynamic offset formula:
            # Strong (body>0.7 or vol>3x) → 0.3 ATR (enter quick)
            # Weak (body<0.5) → 0.8 ATR (wait for deeper pullback)
            # Normal → 0.5 ATR
            if body_ratio >= 0.7 or vol_ratio >= 3.0:
                dyn_offset = 0.3 * sig_atr
            elif body_ratio < 0.5:
                dyn_offset = 0.8 * sig_atr
            else:
                dyn_offset = 0.5 * sig_atr
            dyn_offset = max(0.3, min(dyn_offset, 3.0))
        else:
            dyn_offset = 0.5 * sig_atr

        pending_signals[pkey] = {
            "direction": direction,
            "combo_name": combo_name,
            "combo_short": combo_short,
            "alert_key": alert_key,
            "aligned_tfs": aligned_tfs,
            "n_agree": item["n_agree"],
            "max_tfs": item["max_tfs"],
            "best_sig": best_sig,
            "best_tf": best_tf,
            "all_conds": item["all_conds"],
            "trigger_price": trigger,
            "dynamic_offset": dyn_offset,
            "age": 0,
        }
        print(f"  [{combo_short}/{best_tf}] {direction} PENDING (trigger={trigger:.1f}, offset={dyn_offset:.2f})")

    if not any_signal and not pending_signals:
        print(f"  No signal. Portfolio: {portfolio_mgr.status_str() if portfolio_mgr else 'N/A'}")

    # Cleanup old alerts (older than 30 min)
    cutoff = time.time() - 1800
    for k in [k for k, t in sent_alerts.items() if t < cutoff]:
        del sent_alerts[k]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VN30F1M Multi-TF Signal Scanner v3")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    args = parser.parse_args()

    # Only combos in COMBO_TF_MAP are active
    active_combos = [
        name for name, p in COMBO_PRESETS.items()
        if p.get("primary") and name.split(":")[0].strip() in COMBO_TF_MAP
    ]

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
    pending_signals = {}  # Persists across scan cycles for next-bar confirmation

    if args.once:
        run_scan(fetcher, notifier, sent_alerts, position_manager,
                 portfolio_mgr=portfolio_manager, pending_signals=pending_signals)
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
                     portfolio_mgr=portfolio_manager,
                     pending_signals=pending_signals)
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
