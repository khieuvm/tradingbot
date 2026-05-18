"""
Signal Conditions & Strategy Formulas
======================================
Tất cả công thức tín hiệu và chiến lược được định nghĩa tại đây.
File này tách riêng để dễ quản lý, xem lại và chỉnh sửa công thức.

Nguồn tham khảo:
- Investopedia: https://www.investopedia.com
- ATR: https://www.investopedia.com/terms/a/atr.asp
- MACD: https://www.investopedia.com/terms/m/macd.asp
- Bollinger Bands: https://www.investopedia.com/terms/b/bollingerbands.asp
- Stochastic: https://www.investopedia.com/terms/s/stochasticoscillator.asp
- RSI: https://www.investopedia.com/terms/r/rsi.asp
"""

import pandas as pd
import numpy as np
import pandas_ta as ta


# ===================== DYNAMIC S/R CALCULATOR =====================
def calculate_dynamic_sr(df: pd.DataFrame) -> pd.DataFrame:
    """Tính Support/Resistance động kết hợp nhiều phương pháp.

    Phương pháp kết hợp:
    ─────────────────────
    1. Swing Points (Fractal):  Tìm đỉnh/đáy cục bộ thực sự (N=5 bars mỗi bên)
    2. EMA Confluence:          EMA5/12/21 là S/R động — nếu swing point gần EMA → mạnh hơn
    3. ATR Zone:                S/R là VÙNG chứ không phải 1 điểm (± 0.5×ATR)
    4. Volume Weight:           Swing point có volume cao → S/R mạnh hơn
    5. Recency Weight:          Swing point gần đây → quan trọng hơn
    6. Cluster Scoring:         Nhiều swing points cùng vùng → S/R rất mạnh

    Output columns:
    ───────────────
    - dynamic_resistance: mức kháng cự gần nhất (weighted by strength)
    - dynamic_support:    mức hỗ trợ gần nhất (weighted by strength)
    - sr_strength:        độ mạnh của S/R gần nhất (1-5 scale)
    """
    n = len(df)
    swing_window = 5  # bars mỗi bên để xác định swing point

    # --- Step 1: Tìm Swing Highs & Swing Lows (Fractal method) ---
    # Swing High: bar có high >= high của N bars trước VÀ N bars sau
    # Swing Low: bar có low <= low của N bars trước VÀ N bars sau
    swing_highs = pd.Series(np.nan, index=df.index, dtype=float)
    swing_lows = pd.Series(np.nan, index=df.index, dtype=float)
    swing_high_vol = pd.Series(np.nan, index=df.index, dtype=float)
    swing_low_vol = pd.Series(np.nan, index=df.index, dtype=float)

    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values.astype(float)

    for i in range(swing_window, n - swing_window):
        # Check swing high
        is_swing_high = True
        for j in range(1, swing_window + 1):
            if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.iloc[i] = highs[i]
            swing_high_vol.iloc[i] = volumes[i]

        # Check swing low
        is_swing_low = True
        for j in range(1, swing_window + 1):
            if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.iloc[i] = lows[i]
            swing_low_vol.iloc[i] = volumes[i]

    # --- Step 2: Cho mỗi bar, tìm S/R động gần nhất ---
    # Sử dụng rolling window 50 bars gần nhất để tìm swing points
    atr_vals = df["atr"].values
    ema5_vals = df["ema5"].values
    ema12_vals = df["ema12"].values
    ema21_vals = df["ema21"].values
    close_vals = df["close"].values

    dynamic_res = pd.Series(np.nan, index=df.index, dtype=float)
    dynamic_sup = pd.Series(np.nan, index=df.index, dtype=float)
    sr_strength_col = pd.Series(0.0, index=df.index, dtype=float)

    lookback = 50  # bars nhìn lại để tìm swing points

    for i in range(swing_window + 1, n):
        current_close = close_vals[i]
        current_atr = atr_vals[i] if not np.isnan(atr_vals[i]) else (highs[i] - lows[i])
        if current_atr == 0:
            current_atr = current_close * 0.02

        zone_radius = 0.5 * current_atr  # ATR zone: ± 0.5×ATR

        # Collect swing highs above current price (resistance candidates)
        # Collect swing lows below current price (support candidates)
        start_idx = max(0, i - lookback)

        res_candidates = []  # (price, strength_score)
        sup_candidates = []

        # --- Swing point candidates ---
        for j in range(start_idx, i):
            recency_weight = 1.0 + 0.5 * (j - start_idx) / max(1, i - start_idx)  # newer = higher weight

            # Resistance: swing highs above close
            sh = swing_highs.iloc[j]
            if not np.isnan(sh) and sh > current_close:
                vol_weight = 1.0
                sv = swing_high_vol.iloc[j]
                if not np.isnan(sv) and not np.isnan(df["vol_sma"].iloc[j]) and df["vol_sma"].iloc[j] > 0:
                    vol_weight = min(2.0, sv / df["vol_sma"].iloc[j])  # cap 2x

                # EMA confluence: if swing point is near an EMA → bonus
                ema_bonus = 0.0
                for ema_val in [ema5_vals[j], ema12_vals[j], ema21_vals[j]]:
                    if not np.isnan(ema_val) and abs(sh - ema_val) <= zone_radius:
                        ema_bonus += 0.3

                score = recency_weight * vol_weight + ema_bonus
                res_candidates.append((sh, score))

            # Support: swing lows below close
            sl = swing_lows.iloc[j]
            if not np.isnan(sl) and sl < current_close:
                vol_weight = 1.0
                sv = swing_low_vol.iloc[j]
                if not np.isnan(sv) and not np.isnan(df["vol_sma"].iloc[j]) and df["vol_sma"].iloc[j] > 0:
                    vol_weight = min(2.0, sv / df["vol_sma"].iloc[j])

                ema_bonus = 0.0
                for ema_val in [ema5_vals[j], ema12_vals[j], ema21_vals[j]]:
                    if not np.isnan(ema_val) and abs(sl - ema_val) <= zone_radius:
                        ema_bonus += 0.3

                score = recency_weight * vol_weight + ema_bonus
                sup_candidates.append((sl, score))

        # --- Step 3: Cluster nearby swing points into zones ---
        # Group candidates within zone_radius of each other, sum scores
        best_res = np.nan
        best_res_score = 0.0
        if res_candidates:
            # Sort by price (ascending) to cluster
            res_candidates.sort(key=lambda x: x[0])
            clusters = []
            current_cluster = [res_candidates[0]]
            for k in range(1, len(res_candidates)):
                if res_candidates[k][0] - current_cluster[0][0] <= zone_radius * 2:
                    current_cluster.append(res_candidates[k])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [res_candidates[k]]
            clusters.append(current_cluster)

            # Find cluster with highest total score, closest to price
            for cluster in clusters:
                total_score = sum(c[1] for c in cluster)
                # Bonus for multi-touch (cluster size)
                total_score *= (1 + 0.2 * (len(cluster) - 1))
                avg_price = sum(c[0] for c in cluster) / len(cluster)
                # Prefer closer resistance
                dist_penalty = (avg_price - current_close) / current_close
                adjusted_score = total_score / (1 + dist_penalty * 5)

                if adjusted_score > best_res_score:
                    best_res_score = adjusted_score
                    best_res = avg_price

        best_sup = np.nan
        best_sup_score = 0.0
        if sup_candidates:
            sup_candidates.sort(key=lambda x: x[0])
            clusters = []
            current_cluster = [sup_candidates[0]]
            for k in range(1, len(sup_candidates)):
                if sup_candidates[k][0] - current_cluster[0][0] <= zone_radius * 2:
                    current_cluster.append(sup_candidates[k])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [sup_candidates[k]]
            clusters.append(current_cluster)

            for cluster in clusters:
                total_score = sum(c[1] for c in cluster)
                total_score *= (1 + 0.2 * (len(cluster) - 1))
                avg_price = sum(c[0] for c in cluster) / len(cluster)
                dist_penalty = (current_close - avg_price) / current_close
                adjusted_score = total_score / (1 + dist_penalty * 5)

                if adjusted_score > best_sup_score:
                    best_sup_score = adjusted_score
                    best_sup = avg_price

        # --- Step 4: EMA as fallback dynamic S/R ---
        # If no swing points found, use nearest EMA above/below as S/R
        if np.isnan(best_res):
            ema_above = [v for v in [ema5_vals[i], ema12_vals[i], ema21_vals[i]]
                         if not np.isnan(v) and v > current_close]
            if ema_above:
                best_res = min(ema_above)
                best_res_score = 1.0

        if np.isnan(best_sup):
            ema_below = [v for v in [ema5_vals[i], ema12_vals[i], ema21_vals[i]]
                         if not np.isnan(v) and v < current_close]
            if ema_below:
                best_sup = max(ema_below)
                best_sup_score = 1.0

        dynamic_res.iloc[i] = best_res
        dynamic_sup.iloc[i] = best_sup
        sr_strength_col.iloc[i] = min(5.0, max(best_res_score, best_sup_score))

    df["dynamic_resistance"] = dynamic_res
    df["dynamic_support"] = dynamic_sup
    df["sr_strength"] = sr_strength_col.clip(0, 5)

    return df


# ===================== COMBO PRESETS =====================
# Each preset defines:
#   primary: conditions that trigger entry (need >= 1 to fire)
#   confirm: conditions that add confidence (HIGH / MED / LOW)
#   gate: conditions that MUST pass or signal is blocked (filter)
COMBO_PRESETS = {
    "Custom": {"primary": [], "confirm": []},
    "A: Trend Pullback (~65% WR)": {
        "desc": "EMA Ribbon aligned + price pullback to EMA21. MACD + ADX confirm strong trend.",
        "primary": ["ema_ribbon", "ema_pullback"],
        "confirm": ["adx_di", "macd_cross", "macd_hist_rev"],
    },
    "B: Momentum Breakout (R:R 3:1)": {
        "desc": "BB Squeeze breakout + 20-day high/low break. ADX + Volume confirm momentum.",
        "primary": ["bb_squeeze", "sr_breakout"],
        "confirm": ["adx_di", "macd_hist_rev", "inside_bar"],
    },
    "C: Mean Reversion (~60% WR)": {
        "desc": "Price touches BB + Stochastic extreme -> mean reversion. RSI Div + Hammer confirm reversal.",
        "primary": ["bb_bounce", "stoch_cross"],
        "confirm": ["rsi_div", "hammer_star", "engulfing"],
    },
    "D: Trend Confirmation (safest)": {
        "desc": "SMA Cross + MACD Cross agree -> new trend. EMA Ribbon + ADX confirm strong trend.",
        "primary": ["sma_cross", "macd_cross"],
        "confirm": ["ema_ribbon", "adx_di", "macd_hist_rev"],
    },
    "K: Smart Mean Reversion": {
        "desc": "Base from C + 2 gates: (1) no BUY when MACD below signal, no SELL when MACD above signal. "
                "(2) no BUY on red volume, no SELL on green volume.",
        "primary": ["bb_bounce", "stoch_cross"],
        "confirm": ["rsi_div", "hammer_star", "engulfing"],
        "gate": ["macd_filter", "vol_color_filter"],
    },
}

# Condition labels for display
COND_LABELS = {
    "sma_cross": "SMA Cross", "macd_cross": "MACD Cross",
    "ema_pullback": "EMA Pullback", "bb_squeeze": "BB Squeeze",
    "rsi_div": "RSI Divergence", "macd_hist_rev": "MACD Hist Rev",
    "stoch_cross": "Stoch Cross", "bb_bounce": "BB Bounce",
    "engulfing": "Engulfing",
    "ema_ribbon": "EMA Ribbon", "inside_bar": "Inside Bar",
    "hammer_star": "Hammer/Star", "adx_di": "ADX+DI",
    "sr_breakout": "S/R Breakout", "sr_atr": "S/R ± ATR",
    "macd_filter": "MACD Filter", "vol_color_filter": "Vol Color Filter",
}

# All condition keys
ALL_COND_KEYS = [
    "sma_cross", "macd_cross", "ema_pullback", "bb_squeeze", "rsi_div",
    "macd_hist_rev", "stoch_cross", "bb_bounce", "engulfing",
    "ema_ribbon", "inside_bar", "hammer_star", "adx_di", "sr_breakout",
    "sr_atr", "macd_filter", "vol_color_filter",
]


# ===================== SIGNAL ANALYSIS HELPER =====================
def analyze_signal_performance(sig_df, atr_sl_mult=1.5, atr_tp_mult=3.0, max_hold=30):
    """Analyze each signal: compute SL/TP, forward-simulate, return stats dict + detail rows.

    Parameters:
    -----------
    sig_df : DataFrame with 'signal', 'close', 'high', 'low', 'atr' columns
    atr_sl_mult : Stop Loss = entry ± (mult × ATR)
    atr_tp_mult : Take Profit = entry ± (mult × ATR)
    max_hold : Maximum bars to hold before forced exit

    Returns: dict with 'rows', 'total', 'wins', 'losses', 'win_rate', etc. or None
    """
    sig_rows = sig_df[sig_df["signal"] != 0].copy()
    if sig_rows.empty:
        return None

    analysis_rows = []
    for idx, row in sig_rows.iterrows():
        entry = float(row["close"])
        atr_val = float(row["atr"]) if pd.notna(row.get("atr")) else entry * 0.02
        sig_type = int(row["signal"])
        confidence = int(row.get("signal_confidence", 0))
        time_str = str(row["time"]) if "time" in row.index else str(idx)

        if sig_type == 1:
            sl = entry - atr_sl_mult * atr_val
            tp = entry + atr_tp_mult * atr_val
        else:
            sl = entry + atr_sl_mult * atr_val
            tp = entry - atr_tp_mult * atr_val

        pos = sig_df.index.get_loc(idx)
        future = sig_df.iloc[pos + 1: pos + 1 + max_hold]
        outcome = "Pending"
        exit_price = None
        pnl_pct = 0.0

        for fi, frow in future.iterrows():
            if sig_type == 1:
                if float(frow["high"]) >= tp:
                    outcome = "WIN (TP)"; exit_price = tp; break
                if float(frow["low"]) <= sl:
                    outcome = "LOSS (SL)"; exit_price = sl; break
            else:
                if float(frow["low"]) <= tp:
                    outcome = "WIN (TP)"; exit_price = tp; break
                if float(frow["high"]) >= sl:
                    outcome = "LOSS (SL)"; exit_price = sl; break

        if exit_price is None and not future.empty:
            exit_price = float(future.iloc[-1]["close"])
            if sig_type == 1:
                outcome = "Timeout" if exit_price >= entry else "Timeout (loss)"
            else:
                outcome = "Timeout" if exit_price <= entry else "Timeout (loss)"

        if exit_price is not None:
            pnl_pct = ((exit_price - entry) / entry * 100) if sig_type == 1 else ((entry - exit_price) / entry * 100)

        conf_str = "*" * confidence if confidence > 0 else "-"
        analysis_rows.append({
            "Time": time_str[:10] if len(time_str) > 10 else time_str,
            "Type": "BUY" if sig_type == 1 else "SELL",
            "Entry": f"{entry:,.1f}",
            "Stop Loss": f"{sl:,.1f}",
            "Take Profit": f"{tp:,.1f}",
            "R:R": f"1:{atr_tp_mult / atr_sl_mult:.1f}",
            "Confidence": conf_str,
            "Result": outcome,
            "P&L %": f"{pnl_pct:+.2f}%",
            "_pnl": pnl_pct,
            "_win": 1 if "WIN" in outcome else 0,
            "_loss": 1 if "LOSS" in outcome else 0,
        })

    if not analysis_rows:
        return None

    adf = pd.DataFrame(analysis_rows)
    total = len(adf)
    wins = int(adf["_win"].sum())
    losses = int(adf["_loss"].sum())
    win_rate = wins / max(1, wins + losses) * 100
    avg_win = adf.loc[adf["_win"] == 1, "_pnl"].mean() if wins > 0 else 0
    avg_loss = abs(adf.loc[adf["_loss"] == 1, "_pnl"].mean()) if losses > 0 else 0
    profit_factor = (avg_win * wins) / max(0.01, avg_loss * losses) if losses > 0 else float("inf")
    expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)
    total_pnl = adf["_pnl"].sum()

    return {
        "rows": adf,
        "total": total, "wins": wins, "losses": losses,
        "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": profit_factor, "expectancy": expectancy, "total_pnl": total_pnl,
    }


# ===================== MAIN SIGNAL GENERATOR =====================
def generate_combined_signals(data: pd.DataFrame, fast_ma=10, slow_ma=20,
                              rsi_period=7, oversold=35, overbought=70,
                              macd_fast=12, macd_slow=26, macd_signal=9,
                              vol_mult=1.5, bb_period=20, bb_std=2.0,
                              stoch_k=14, stoch_d=3,
                              enabled=None, combo_mode=None) -> pd.DataFrame:
    """Combined multi-condition signal system with individually toggleable conditions.

    Each condition can be enabled/disabled via the `enabled` dict.
    Signal fires based on primary/confirm/gate logic in combo mode,
    or flat score in custom mode.
    """
    if enabled is None:
        enabled = {}

    df = data.copy()

    # ==================== CORE INDICATORS ====================
    # ALL computed upfront (supports 30s refresh without re-selecting conditions)

    # --- Moving Averages ---
    df["sma_f"] = ta.sma(df["close"], length=fast_ma)
    df["sma_s"] = ta.sma(df["close"], length=slow_ma)
    df["ema5"] = ta.ema(df["close"], length=5)
    df["ema12"] = ta.ema(df["close"], length=12)
    df["ema8"] = ta.ema(df["close"], length=8)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["ema55"] = ta.ema(df["close"], length=55)

    # --- EMA Slope (rate of change of EMA12 over 3 bars, normalized by ATR) ---
    # Positive = bullish acceleration, Negative = bearish
    df["ema12_slope_raw"] = df["ema12"] - df["ema12"].shift(3)

    # --- RSI ---
    df["rsi"] = ta.rsi(df["close"], length=rsi_period)

    # --- MACD ---
    macd_result = ta.macd(df["close"], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    if macd_result is not None:
        df["macd_line"] = macd_result.iloc[:, 0]
        df["macd_sig"] = macd_result.iloc[:, 2]
        df["macd_hist"] = macd_result.iloc[:, 1]
    else:
        df["macd_line"] = 0
        df["macd_sig"] = 0
        df["macd_hist"] = 0

    # --- Bollinger Bands ---
    bb = ta.bbands(df["close"], length=bb_period, std=bb_std)
    if bb is not None:
        df["bb_upper"] = bb.iloc[:, 2]
        df["bb_mid"] = bb.iloc[:, 1]
        df["bb_lower"] = bb.iloc[:, 0]
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    else:
        df["bb_upper"] = df["bb_mid"] = df["bb_lower"] = df["close"]
        df["bb_width"] = 0

    # BB Width normalized (% of mid) — dùng để detect squeeze
    df["bb_width_pct"] = df["bb_width"] / df["bb_mid"].replace(0, np.nan)
    df["bb_squeeze_flag"] = df["bb_width_pct"] <= df["bb_width_pct"].rolling(20).quantile(0.2)

    # --- Stochastic ---
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=stoch_k, d=stoch_d)
    if stoch is not None:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]
    else:
        df["stoch_k"] = 50
        df["stoch_d"] = 50

    # --- Volume ---
    df["vol_sma"] = ta.sma(df["volume"].astype(float), length=20)
    df["vol_ok"] = df["volume"] > (vol_mult * df["vol_sma"])

    # --- ATR (14-day, Investopedia standard) ---
    atr_result = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr"] = atr_result if atr_result is not None else (df["high"] - df["low"])

    # EMA Slope normalized by ATR (dimensionless: >0.5 = strong trend)
    df["ema_slope"] = df["ema12_slope_raw"] / df["atr"].replace(0, np.nan)

    # --- Z-Score (how far price is from EMA21, in std units) ---
    # Z > 2 = overextended up, Z < -2 = overextended down
    _std20 = df["close"].rolling(20).std()
    df["zscore"] = (df["close"] - df["ema21"]) / _std20.replace(0, np.nan)

    # --- Support / Resistance levels ---
    # Simple: 20-day rolling (for sr_breakout)
    df["resistance_20"] = df["high"].rolling(20).max().shift(1)
    df["support_20"] = df["low"].rolling(20).min().shift(1)

    # Dynamic: Multi-method S/R (Swing + EMA Confluence + ATR Zone + Volume + Cluster)
    df = calculate_dynamic_sr(df)

    # --- Trend direction ---
    df["trend_bull"] = df["sma_f"] > df["sma_s"]
    df["trend_bear"] = df["sma_f"] < df["sma_s"]

    # --- ADX + Directional Indicators ---
    adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_result is not None:
        df["adx"] = adx_result.iloc[:, 0]
        df["plus_di"] = adx_result.iloc[:, 1]
        df["minus_di"] = adx_result.iloc[:, 2]
    else:
        df["adx"] = 0
        df["plus_di"] = 0
        df["minus_di"] = 0

    # ==================== SCORE SYSTEM ====================
    df["buy_score"] = 0
    df["sell_score"] = 0

    # Per-condition tracking columns
    for _k in ALL_COND_KEYS:
        df[f"_b_{_k}"] = 0
        df[f"_s_{_k}"] = 0

    # ==================== CONDITIONS ====================

    # --- 1. SMA Cross (Golden Cross / Death Cross) ---
    # Nguồn: Investopedia - Golden Cross is lagging → confirm with other indicators
    # Logic: SMA fast crosses SMA slow + RSI 30-70 filter (avoid extreme zones)
    if enabled.get("sma_cross", False):
        sma_golden = (df["sma_f"] > df["sma_s"]) & (df["sma_f"].shift(1) <= df["sma_s"].shift(1))
        sma_death = (df["sma_f"] < df["sma_s"]) & (df["sma_f"].shift(1) >= df["sma_s"].shift(1))
        _buy = sma_golden & (df["rsi"] < 70) & (df["rsi"] > 30)
        _sell = sma_death & (df["rsi"] > 30) & (df["rsi"] < 70)
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_sma_cross"] = 1
        df.loc[_sell, "_s_sma_cross"] = 1

    # --- 2. MACD Cross (Signal Line Crossover) ---
    # Nguồn: Investopedia - "Crossovers more reliable when conform to prevailing trend"
    # Logic: MACD line crosses signal line. Accept: trend-confirming OR reversal near zero
    if enabled.get("macd_cross", False):
        macd_buy = (df["macd_line"] > df["macd_sig"]) & (df["macd_line"].shift(1) <= df["macd_sig"].shift(1))
        macd_sell = (df["macd_line"] < df["macd_sig"]) & (df["macd_line"].shift(1) >= df["macd_sig"].shift(1))
        buy_valid = df["trend_bull"] | (df["macd_line"] <= 0)
        sell_valid = df["trend_bear"] | (df["macd_line"] >= 0)
        _buy = macd_buy & buy_valid
        _sell = macd_sell & sell_valid
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_macd_cross"] = 1
        df.loc[_sell, "_s_macd_cross"] = 1

    # --- 3. EMA Pullback (EMA21 Bounce) ---
    # Logic: In trend, price pulls back to EMA21 then bounces with strength
    # Buy: uptrend + low touches EMA21 + close in upper half of candle
    # Sell: downtrend + high touches EMA21 + close in lower half
    if enabled.get("ema_pullback", False):
        touched_ema21 = df["low"] <= df["ema21"] * 1.005
        closed_above_ema21 = df["close"] > df["ema21"]
        bounce_strength_buy = (df["close"] - df["low"]) > (df["high"] - df["low"]) * 0.5
        _buy = df["trend_bull"] & touched_ema21 & closed_above_ema21 & bounce_strength_buy
        df.loc[_buy, "buy_score"] += 1
        df.loc[_buy, "_b_ema_pullback"] = 1
        touched_ema21_hi = df["high"] >= df["ema21"] * 0.995
        closed_below_ema21 = df["close"] < df["ema21"]
        bounce_strength_sell = (df["high"] - df["close"]) > (df["high"] - df["low"]) * 0.5
        _sell = df["trend_bear"] & touched_ema21_hi & closed_below_ema21 & bounce_strength_sell
        df.loc[_sell, "sell_score"] += 1
        df.loc[_sell, "_s_ema_pullback"] = 1

    # --- 4. Bollinger Squeeze Breakout ---
    # Logic: BB width at 20-period minimum (squeeze) → breakout khi giá vượt BB
    # Squeeze = low volatility → expansion expected
    if enabled.get("bb_squeeze", False):
        bb_width_min = df["bb_width"].rolling(window=20).min()
        squeeze = df["bb_width"].shift(1) <= (bb_width_min.shift(1) * 1.05)
        breakout_up = df["close"] > df["bb_upper"]
        breakout_down = df["close"] < df["bb_lower"]
        _buy = squeeze & breakout_up
        _sell = squeeze & breakout_down
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_bb_squeeze"] = 1
        df.loc[_sell, "_s_bb_squeeze"] = 1

    # --- 5. RSI Divergence ---
    # Logic: Price makes lower low but RSI makes higher low (bullish div) → momentum weakening
    # Lookback 10 bars, RSI < 45 for buy / > 55 for sell
    if enabled.get("rsi_div", False):
        lookback = 10
        price_ll = df["close"] < df["close"].rolling(lookback).min().shift(1)
        rsi_hl = df["rsi"] > df["rsi"].rolling(lookback).min().shift(1)
        _buy = price_ll & rsi_hl & (df["rsi"] < 45)
        df.loc[_buy, "buy_score"] += 1
        df.loc[_buy, "_b_rsi_div"] = 1
        price_hh = df["close"] > df["close"].rolling(lookback).max().shift(1)
        rsi_lh = df["rsi"] < df["rsi"].rolling(lookback).max().shift(1)
        _sell = price_hh & rsi_lh & (df["rsi"] > 55)
        df.loc[_sell, "sell_score"] += 1
        df.loc[_sell, "_s_rsi_div"] = 1

    # --- 6. MACD Histogram Reversal ---
    # Logic: Histogram 3-bar momentum shift (turns from negative to less negative)
    # Detects early momentum change before full crossover
    if enabled.get("macd_hist_rev", False):
        hist_turn_up = (df["macd_hist"] > df["macd_hist"].shift(1)) & (df["macd_hist"].shift(1) < df["macd_hist"].shift(2)) & (df["macd_hist"] < 0)
        hist_turn_down = (df["macd_hist"] < df["macd_hist"].shift(1)) & (df["macd_hist"].shift(1) > df["macd_hist"].shift(2)) & (df["macd_hist"] > 0)
        df.loc[hist_turn_up, "buy_score"] += 1
        df.loc[hist_turn_down, "sell_score"] += 1
        df.loc[hist_turn_up, "_b_macd_hist_rev"] = 1
        df.loc[hist_turn_down, "_s_macd_hist_rev"] = 1

    # --- 7. Stochastic Extreme Cross ---
    # Nguồn: Investopedia - Overbought >80, Oversold <20
    # Logic: K crosses D when LEAVING extreme zone (not entering)
    if enabled.get("stoch_cross", False):
        stoch_buy = (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) & (df["stoch_k"].shift(1) < 20)
        stoch_sell = (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) & (df["stoch_k"].shift(1) > 80)
        df.loc[stoch_buy, "buy_score"] += 1
        df.loc[stoch_sell, "sell_score"] += 1
        df.loc[stoch_buy, "_b_stoch_cross"] = 1
        df.loc[stoch_sell, "_s_stoch_cross"] = 1

    # --- 8. Bollinger Bounce ---
    # Logic: Price touches lower BB in uptrend + RSI < 35 → oversold bounce
    # RSI threshold tightened for stronger confirmation
    if enabled.get("bb_bounce", False):
        touch_lower = df["low"] <= df["bb_lower"]
        touch_upper = df["high"] >= df["bb_upper"]
        _buy = touch_lower & df["trend_bull"] & (df["rsi"] < 35)
        _sell = touch_upper & df["trend_bear"] & (df["rsi"] > 65)
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_bb_bounce"] = 1
        df.loc[_sell, "_s_bb_bounce"] = 1

    # --- 9. Engulfing Candle ---
    # Logic: Current candle completely engulfs previous candle body
    # Bullish engulfing + RSI < 50 → reversal from oversold
    if enabled.get("engulfing", False):
        body = df["close"] - df["open"]
        prev_body = body.shift(1)
        bull_engulf = (body > 0) & (prev_body < 0) & (df["open"] <= df["close"].shift(1)) & (df["close"] >= df["open"].shift(1))
        bear_engulf = (body < 0) & (prev_body > 0) & (df["open"] >= df["close"].shift(1)) & (df["close"] <= df["open"].shift(1))
        _buy = bull_engulf & (df["rsi"] < 50)
        _sell = bear_engulf & (df["rsi"] > 50)
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_engulfing"] = 1
        df.loc[_sell, "_s_engulfing"] = 1

    # --- 10. EMA Ribbon Alignment (Triple EMA 8/21/55) ---
    # Logic: All 3 EMAs aligned (8>21>55 bull / 8<21<55 bear) + price pulls to EMA21
    # Win rate ~60-65% in trending markets
    if enabled.get("ema_ribbon", False):
        ribbon_bull = (df["ema8"] > df["ema21"]) & (df["ema21"] > df["ema55"])
        ribbon_bear = (df["ema8"] < df["ema21"]) & (df["ema21"] < df["ema55"])
        pull_to_21_buy = (df["low"] <= df["ema21"] * 1.005) & (df["close"] > df["ema21"])
        pull_to_21_sell = (df["high"] >= df["ema21"] * 0.995) & (df["close"] < df["ema21"])
        _buy = ribbon_bull & pull_to_21_buy
        _sell = ribbon_bear & pull_to_21_sell
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_ema_ribbon"] = 1
        df.loc[_sell, "_s_ema_ribbon"] = 1

    # --- 11. Inside Bar Breakout ---
    # Logic: Consolidation bar (range inside previous bar) → breakout in trend direction
    # Win rate ~55-60%, R:R ~3:1
    if enabled.get("inside_bar", False):
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        inside_bar = (df["high"].shift(1) < df["high"].shift(2)) & (df["low"].shift(1) > df["low"].shift(2))
        break_up = inside_bar & (df["close"] > prev_high)
        break_down = inside_bar & (df["close"] < prev_low)
        _buy = break_up & df["trend_bull"]
        _sell = break_down & df["trend_bear"]
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_inside_bar"] = 1
        df.loc[_sell, "_s_inside_bar"] = 1

    # --- 12. Hammer / Shooting Star ---
    # Logic: Reversal candle at key support/resistance level
    # Hammer: long lower shadow ≥2x body, near BB lower or EMA21 + RSI < 45
    # Star: long upper shadow ≥2x body, near BB upper or EMA21 + RSI > 55
    if enabled.get("hammer_star", False):
        body_size = (df["close"] - df["open"]).abs()
        candle_range = df["high"] - df["low"]
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
        upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
        hammer = (lower_shadow >= 2 * body_size) & (upper_shadow < body_size) & (candle_range > 0)
        star = (upper_shadow >= 2 * body_size) & (lower_shadow < body_size) & (candle_range > 0)
        near_support = (df["low"] <= df["bb_lower"] * 1.01) | (df["low"] <= df["ema21"] * 1.005)
        near_resistance = (df["high"] >= df["bb_upper"] * 0.99) | (df["high"] >= df["ema21"] * 0.995)
        _buy = hammer & near_support & (df["rsi"] < 45)
        _sell = star & near_resistance & (df["rsi"] > 55)
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_hammer_star"] = 1
        df.loc[_sell, "_s_hammer_star"] = 1

    # --- 13. ADX Trend Strength + DI Cross ---
    # Nguồn: Investopedia - ADX > 25 = trending market
    # Logic: +DI crosses -DI when ADX > 25 → trend confirmed
    if enabled.get("adx_di", False):
        strong_trend = df["adx"] > 25
        di_cross_up = (df["plus_di"] > df["minus_di"]) & (df["plus_di"].shift(1) <= df["minus_di"].shift(1))
        di_cross_down = (df["minus_di"] > df["plus_di"]) & (df["minus_di"].shift(1) <= df["plus_di"].shift(1))
        _buy = di_cross_up & strong_trend
        _sell = di_cross_down & strong_trend
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_adx_di"] = 1
        df.loc[_sell, "_s_adx_di"] = 1

    # --- 14. Support/Resistance Breakout (20-day High/Low) ---
    # Logic: Price breaks 20-day high/low with RSI confirmation (50-75 for buy, 25-50 for sell)
    if enabled.get("sr_breakout", False):
        _buy = (df["close"] > df["resistance_20"]) & (df["rsi"] > 50) & (df["rsi"] < 75)
        _sell = (df["close"] < df["support_20"]) & (df["rsi"] < 50) & (df["rsi"] > 25)
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_sr_breakout"] = 1
        df.loc[_sell, "_s_sr_breakout"] = 1

    # --- 15. Smart S/R ± ATR (Multi-Factor Breakout) ---
    # Phiên bản thông minh: dùng Dynamic S/R (swing + EMA + volume + cluster)
    #
    # CÔNG THỨC:
    #   Resistance = dynamic_resistance (multi-method: swing fractal + EMA confluence
    #                + volume weight + recency + cluster scoring)
    #   Support    = dynamic_support (tương tự)
    #
    #   BUY khi TẤT CẢ thỏa:
    #     1. Price > Dynamic Resistance + ATR×buffer  (breakout qua vùng S/R thật)
    #     2. BB Squeeze trước đó                     (tích lũy → nổ)
    #     3. Z-score < 2.5                            (chưa quá xa mean → còn room)
    #     4. EMA Slope > 0                            (xu hướng đang tăng tốc)
    #     5. EMA5 > EMA12                             (short-term momentum bullish)
    #     6. BB Width expanding                       (volatility đang nở = breakout thật)
    #
    #   SELL khi TẤT CẢ thỏa (ngược lại)
    #
    # S/R được tính bởi calculate_dynamic_sr():
    #   - Swing Points (fractal 5-bar): đỉnh/đáy cục bộ thực sự
    #   - EMA Confluence: swing gần EMA5/12/21 → bonus điểm
    #   - ATR Zone: cluster swing points trong ±0.5×ATR thành 1 vùng
    #   - Volume Weight: swing point volume cao → S/R mạnh hơn (cap 2x)
    #   - Recency: swing gần đây quan trọng hơn (linear decay 50 bars)
    #   - Cluster: nhiều swing cùng vùng → score × (1 + 0.2 × count)
    if enabled.get("sr_atr", False):
        # Factor 1: Price breaks Dynamic S/R + ATR buffer
        has_res = df["dynamic_resistance"].notna()
        has_sup = df["dynamic_support"].notna()
        break_above = has_res & (df["close"] > (df["dynamic_resistance"] + df["atr"]))
        break_below = has_sup & (df["close"] < (df["dynamic_support"] - df["atr"]))

        # Factor 2: BB was in squeeze recently (within last 3 bars)
        squeeze_recent = (
            df["bb_squeeze_flag"] |
            df["bb_squeeze_flag"].shift(1).fillna(False) |
            df["bb_squeeze_flag"].shift(2).fillna(False)
        )

        # Factor 3: Z-score not overextended
        zscore_ok_buy = df["zscore"] < 2.5
        zscore_ok_sell = df["zscore"] > -2.5

        # Factor 4: EMA Slope confirms direction
        slope_bull = df["ema_slope"] > 0
        slope_bear = df["ema_slope"] < 0

        # Factor 5: Short-term EMA alignment
        ema_short_bull = df["ema5"] > df["ema12"]
        ema_short_bear = df["ema5"] < df["ema12"]

        # Factor 6: BB Width expanding (current > previous bar)
        bb_expanding = df["bb_width"] > df["bb_width"].shift(1)

        # Combine all factors
        _buy = break_above & squeeze_recent & zscore_ok_buy & slope_bull & ema_short_bull & bb_expanding
        _sell = break_below & squeeze_recent & zscore_ok_sell & slope_bear & ema_short_bear & bb_expanding

        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_sr_atr"] = 1
        df.loc[_sell, "_s_sr_atr"] = 1

    # --- 16. MACD Momentum Filter (GATE) ---
    # Logic: Không BUY khi MACD line (xanh) < Signal line (cam) → momentum bearish
    #        Không SELL khi MACD line (xanh) > Signal line (cam) → momentum bullish
    if enabled.get("macd_filter", False):
        buy_ok = df["macd_line"] >= df["macd_sig"]
        sell_ok = df["macd_line"] <= df["macd_sig"]
        df.loc[buy_ok, "_b_macd_filter"] = 1
        df.loc[sell_ok, "_s_macd_filter"] = 1

    # --- 17. Volume Color Filter (GATE) ---
    # Logic: Không BUY khi nến đỏ (close < open) → selling pressure
    #        Không SELL khi nến xanh (close >= open) → buying pressure
    if enabled.get("vol_color_filter", False):
        green_candle = df["close"] >= df["open"]
        red_candle = df["close"] < df["open"]
        df.loc[green_candle, "_b_vol_color_filter"] = 1
        df.loc[red_candle, "_s_vol_color_filter"] = 1

    # ==================== PATTERN DETECTION (informational) ====================
    # These are reported in alerts for context, not used for signal scoring.

    body = df["close"] - df["open"]
    body_abs = body.abs()
    prev_body = body.shift(1)
    prev2_body = body.shift(2)

    # --- Morning Star (bullish reversal) ---
    # 3-candle pattern: big red → small body (indecision) → big green
    big_red_2 = (prev2_body < 0) & (prev2_body.abs() > body_abs.rolling(20).mean())
    small_body_1 = body_abs.shift(1) < body_abs.rolling(20).mean() * 0.5
    big_green_0 = (body > 0) & (body_abs > body_abs.rolling(20).mean())
    # Close of 3rd candle must recover > 50% of 1st candle body
    recover = df["close"] > (df["open"].shift(2) + df["close"].shift(2)) / 2
    df["pat_morning_star"] = (big_red_2 & small_body_1 & big_green_0 & recover).astype(int)

    # --- Evening Star (bearish reversal) ---
    # 3-candle pattern: big green → small body → big red
    big_green_2 = (prev2_body > 0) & (prev2_body.abs() > body_abs.rolling(20).mean())
    small_body_1_ev = body_abs.shift(1) < body_abs.rolling(20).mean() * 0.5
    big_red_0 = (body < 0) & (body_abs > body_abs.rolling(20).mean())
    drop = df["close"] < (df["open"].shift(2) + df["close"].shift(2)) / 2
    df["pat_evening_star"] = (big_green_2 & small_body_1_ev & big_red_0 & drop).astype(int)

    # --- Engulfing (already a condition, but track separately for pattern alert) ---
    bull_engulf = (body > 0) & (prev_body < 0) & (df["open"] <= df["close"].shift(1)) & (df["close"] >= df["open"].shift(1))
    bear_engulf = (body < 0) & (prev_body > 0) & (df["open"] >= df["close"].shift(1)) & (df["close"] <= df["open"].shift(1))
    df["pat_bull_engulfing"] = bull_engulf.astype(int)
    df["pat_bear_engulfing"] = bear_engulf.astype(int)

    # --- Head and Shoulders (simplified: 5-bar swing detection) ---
    # Detect: left shoulder (high) < head (higher high) > right shoulder (high ≈ left)
    h = df["high"]
    l = df["low"]
    # Bearish H&S (top reversal)
    left_sh = h.shift(4)
    head = h.shift(2)
    right_sh = h.shift(0)
    neckline = df[["low"]].shift(1).rolling(3).min().shift(0)
    head_higher = (head > left_sh) & (head > right_sh)
    shoulders_similar = (right_sh >= left_sh * 0.97) & (right_sh <= left_sh * 1.03)
    break_neck_bear = df["close"] < neckline["low"]
    df["pat_head_shoulders_top"] = (head_higher & shoulders_similar & break_neck_bear).astype(int)

    # Bullish inverse H&S (bottom reversal)
    left_sh_inv = l.shift(4)
    head_inv = l.shift(2)
    right_sh_inv = l.shift(0)
    neckline_inv = df[["high"]].shift(1).rolling(3).max().shift(0)
    head_lower = (head_inv < left_sh_inv) & (head_inv < right_sh_inv)
    shoulders_similar_inv = (right_sh_inv >= left_sh_inv * 0.97) & (right_sh_inv <= left_sh_inv * 1.03)
    break_neck_bull = df["close"] > neckline_inv["high"]
    df["pat_head_shoulders_bottom"] = (head_lower & shoulders_similar_inv & break_neck_bull).astype(int)

    # --- Volume Confirmation ---
    # Volume > 1.5x 20-bar average on the signal bar
    vol_avg = df["volume"].rolling(20).mean()
    df["pat_volume_confirm"] = (df["volume"] > vol_avg * 1.5).astype(int)

    # ==================== VOLUME GATE ====================
    use_vol = enabled.get("vol_filter", True)

    # ==================== FIRE SIGNALS ====================
    df["signal"] = 0
    df["signal_confidence"] = 0

    preset = COMBO_PRESETS.get(combo_mode) if combo_mode else None

    if preset and preset.get("primary"):
        # --- COMBO MODE ---
        primary_keys = set(preset["primary"])
        confirm_keys = set(preset["confirm"])
        all_cond_keys = [k for k in enabled if k != "vol_filter" and enabled.get(k)]

        df["primary_buy"] = 0
        df["primary_sell"] = 0
        df["confirm_buy"] = 0
        df["confirm_sell"] = 0

        for cond_key in all_cond_keys:
            b_col = f"_b_{cond_key}"
            s_col = f"_s_{cond_key}"
            if b_col not in df.columns:
                continue
            if cond_key in primary_keys:
                df["primary_buy"] += df[b_col]
                df["primary_sell"] += df[s_col]
            elif cond_key in confirm_keys:
                df["confirm_buy"] += df[b_col]
                df["confirm_sell"] += df[s_col]

        buy_triggered = df["primary_buy"] >= 1
        sell_triggered = df["primary_sell"] >= 1

        # --- GATE: must ALL pass ---
        gate_keys = set(preset.get("gate", []))
        for gate_key in gate_keys:
            b_col = f"_b_{gate_key}"
            s_col = f"_s_{gate_key}"
            if b_col in df.columns:
                buy_triggered = buy_triggered & (df[b_col] == 1)
            if s_col in df.columns:
                sell_triggered = sell_triggered & (df[s_col] == 1)

        buy_confidence = df["confirm_buy"].clip(0, 3).astype(int) + 1
        sell_confidence = df["confirm_sell"].clip(0, 3).astype(int) + 1

        if use_vol:
            df.loc[buy_triggered & df["vol_ok"], "signal"] = 1
            df.loc[buy_triggered & df["vol_ok"], "signal_confidence"] = buy_confidence
            df.loc[sell_triggered & df["vol_ok"], "signal"] = -1
            df.loc[sell_triggered & df["vol_ok"], "signal_confidence"] = sell_confidence
        else:
            df.loc[buy_triggered, "signal"] = 1
            df.loc[buy_triggered, "signal_confidence"] = buy_confidence
            df.loc[sell_triggered, "signal"] = -1
            df.loc[sell_triggered, "signal_confidence"] = sell_confidence

        df["signal_confidence"] = df["signal_confidence"].clip(0, 3)
        df.loc[df["signal"] == 0, "signal_confidence"] = 0

        # Drop intermediate scoring columns but keep _b_/_s_ condition flags
        df.drop(columns=["primary_buy", "primary_sell", "confirm_buy", "confirm_sell"],
                inplace=True, errors="ignore")
    else:
        # --- CUSTOM MODE ---
        n_enabled = sum(1 for k, v in enabled.items() if v and k != "vol_filter")
        min_score = max(1, n_enabled // 3)

        if use_vol:
            df.loc[(df["buy_score"] >= min_score) & df["vol_ok"], "signal"] = 1
            df.loc[(df["sell_score"] >= min_score) & df["vol_ok"], "signal"] = -1
        else:
            df.loc[df["buy_score"] >= min_score, "signal"] = 1
            df.loc[df["sell_score"] >= min_score, "signal"] = -1

        df.loc[df["signal"] == 1, "signal_confidence"] = (df["buy_score"] / max(1, n_enabled) * 3).clip(1, 3).astype(int)
        df.loc[df["signal"] == -1, "signal_confidence"] = (df["sell_score"] / max(1, n_enabled) * 3).clip(1, 3).astype(int)

    return df
