"""
Feature computation for ML meta-labeling.

Computes 25 features from a 5m OHLCV DataFrame at a given signal bar index.
Features are chosen for their predictive power in classifying CB trade outcomes.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta


META_LABEL_FEATURES = [
    "atr_14",
    "atr_pct",
    "bb_width",
    "atr_ratio_5_14",
    "rsi_14",
    "rsi_3",
    "macd_hist",
    "adx",
    "ema50_dist",
    "ema_spread",
    "ret_3",
    "ret_5",
    "ret_13",
    "vol_ratio",
    "compression_depth",
    "pre_move_ratio",
    "session_minutes",
    "is_pm",
    "body_pct",
    "range_pct",
    "upper_wick_ratio",
    "bb_pos",
    "ret_1",
    "stoch_k",
    "time_to_session_end",
]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute all indicators needed for feature extraction.

    Args:
        df: OHLCV DataFrame with columns: time, open, high, low, close, volume
    Returns:
        DataFrame with indicator columns appended.
    """
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df.get("volume")

    df["atr_14"] = ta.atr(h, l, c, length=14)
    df["atr_5"] = ta.atr(h, l, c, length=5)
    df["rsi_14"] = ta.rsi(c, length=14)
    df["rsi_3"] = ta.rsi(c, length=3)
    df["adx"] = ta.adx(h, l, c, length=14)["ADX_14"] if len(df) >= 14 else np.nan

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd_hist"] = macd.iloc[:, 1]
    else:
        df["macd_hist"] = np.nan

    df["ema8"] = ta.ema(c, length=8)
    df["ema21"] = ta.ema(c, length=21)
    df["ema50"] = ta.ema(c, length=50)

    sma20 = ta.sma(c, length=20)
    std20 = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (4 * std20 / sma20).where(sma20 > 0, np.nan)
    df["bb_pos"] = ((c - sma20) / (2 * std20)).where(std20 > 0, np.nan)

    stoch = ta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        df["stoch_k"] = stoch.iloc[:, 0]
    else:
        df["stoch_k"] = np.nan

    if v is not None and not v.isna().all():
        vol_ema20 = ta.ema(v.astype(float), length=20)
        df["vol_ratio"] = (v / vol_ema20).where(vol_ema20 > 0, np.nan)
    else:
        df["vol_ratio"] = 1.0

    return df


def extract_features_at(df: pd.DataFrame, idx: int, direction: int = 0) -> dict | None:
    """Extract feature vector at a specific bar index.

    Args:
        df: DataFrame with indicators already computed (via compute_indicators).
        idx: Integer position (iloc-based) of the signal bar.
        direction: Trade direction (1=BUY, -1=SELL, 0=unknown).
    Returns:
        Dict of feature_name → value, or None if insufficient data.
    """
    if idx < 50 or idx >= len(df):
        return None

    row = df.iloc[idx]
    c = float(row["close"])
    h = float(row["high"])
    l = float(row["low"])
    o = float(row["open"])

    if c <= 0:
        return None

    atr_14 = float(row["atr_14"]) if not pd.isna(row["atr_14"]) else None
    if atr_14 is None or atr_14 <= 0:
        return None

    atr_5 = float(row["atr_5"]) if not pd.isna(row.get("atr_5", np.nan)) else atr_14

    # Parse time
    t = pd.to_datetime(row["time"]) if "time" in df.columns else df.index[idx]
    mins = t.hour * 60 + t.minute
    is_pm = 1 if mins >= 13 * 60 else 0
    session_minutes = mins - (13 * 60 if is_pm else 9 * 60)
    if is_pm:
        time_to_session_end = max(0, 14 * 60 + 30 - mins)
    else:
        time_to_session_end = max(0, 11 * 60 + 30 - mins)

    # Returns
    closes = df["close"].iloc[max(0, idx - 13):idx + 1].values
    ret_1 = (c - closes[-2]) / closes[-2] if len(closes) >= 2 and closes[-2] > 0 else 0
    ret_3 = (c - closes[-4]) / closes[-4] if len(closes) >= 4 and closes[-4] > 0 else 0
    ret_5 = (c - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] > 0 else 0
    ret_13 = (c - closes[0]) / closes[0] if len(closes) >= 14 and closes[0] > 0 else 0

    # Compression depth (CB-specific: how tight the squeeze is)
    ranges = (df["high"] - df["low"]).iloc[max(0, idx - 3):idx].values
    compression_depth = float(max(ranges) / atr_14) if len(ranges) >= 3 and atr_14 > 0 else 0.7

    # Pre-move ratio
    if idx >= 4 and direction != 0:
        pre_close = float(df["close"].iloc[idx - 1])
        pre_open = float(df["open"].iloc[idx - 3])
        pre_move_ratio = (pre_close - pre_open) * direction / atr_14
    else:
        pre_move_ratio = 0.0

    # Candle shape
    bar_range = h - l
    body_pct = (c - o) / c if c > 0 else 0
    range_pct = bar_range / c if c > 0 else 0
    upper_wick_ratio = (h - max(c, o)) / bar_range if bar_range > 0 else 0

    features = {
        "atr_14": atr_14,
        "atr_pct": atr_14 / c,
        "bb_width": _safe_float(row, "bb_width", 0),
        "atr_ratio_5_14": atr_5 / atr_14 if atr_14 > 0 else 1.0,
        "rsi_14": _safe_float(row, "rsi_14", 50),
        "rsi_3": _safe_float(row, "rsi_3", 50),
        "macd_hist": _safe_float(row, "macd_hist", 0) / c,
        "adx": _safe_float(row, "adx", 20),
        "ema50_dist": (c - _safe_float(row, "ema50", c)) / c,
        "ema_spread": ((c - _safe_float(row, "ema8", c)) / c
                       - (c - _safe_float(row, "ema50", c)) / c),
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_5": ret_5,
        "ret_13": ret_13,
        "vol_ratio": _safe_float(row, "vol_ratio", 1.0),
        "compression_depth": compression_depth,
        "pre_move_ratio": pre_move_ratio,
        "session_minutes": session_minutes,
        "is_pm": is_pm,
        "body_pct": body_pct,
        "range_pct": range_pct,
        "upper_wick_ratio": upper_wick_ratio,
        "bb_pos": _safe_float(row, "bb_pos", 0),
        "stoch_k": _safe_float(row, "stoch_k", 50),
        "time_to_session_end": time_to_session_end,
    }
    return features


def _safe_float(row, col, default):
    v = row.get(col, default) if isinstance(row, dict) else row[col] if col in row.index else default
    return float(v) if not pd.isna(v) else default
