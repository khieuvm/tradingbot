"""
Signal Conditions & Strategy Formulas
======================================
All signal formulas and strategy logic defined here.
Separated for easy management and review.

References:
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
    """Calculate Dynamic Support/Resistance using multiple methods.

    Methods combined:
    ---------------------
    1. Swing Points (Fractal):  Find true local highs/lows (N=5 bars each side)
    2. EMA Confluence:          EMA5/12/21 as dynamic S/R -- swing near EMA = stronger
    3. ATR Zone:                S/R is a ZONE not a point (+/- 0.5xATR)
    4. Volume Weight:           High volume swing point = stronger S/R
    5. Recency Weight:          Recent swing points = more important
    6. Cluster Scoring:         Multiple swings in same zone = very strong S/R

    Output columns:
    ---------------
    - dynamic_resistance: nearest resistance level (weighted by strength)
    - dynamic_support:    nearest support level (weighted by strength)
    - sr_strength:        S/R strength score (1-5 scale)
    """
    n = len(df)
    swing_window = 5  # bars each side to confirm swing point

    # --- Step 1: Find Swing Highs & Swing Lows (Fractal method) ---
    # Swing High: bar with high >= high of N bars before AND N bars after
    # Swing Low: bar with low <= low of N bars before AND N bars after
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

    # --- Step 2: For each bar, find nearest dynamic S/R ---
    # Use rolling window of 50 nearest bars to find swing points
    atr_vals = df["atr"].values
    ema5_vals = df["ema5"].values
    ema12_vals = df["ema12"].values
    ema21_vals = df["ema21"].values
    close_vals = df["close"].values

    dynamic_res = pd.Series(np.nan, index=df.index, dtype=float)
    dynamic_sup = pd.Series(np.nan, index=df.index, dtype=float)
    sr_strength_col = pd.Series(0.0, index=df.index, dtype=float)

    lookback = 50  # bars to look back for swing points

    for i in range(swing_window + 1, n):
        current_close = close_vals[i]
        current_atr = atr_vals[i] if not np.isnan(atr_vals[i]) else (highs[i] - lows[i])
        if current_atr == 0:
            current_atr = current_close * 0.02

        zone_radius = 0.5 * current_atr  # ATR zone: +/- 0.5xATR

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

                # EMA confluence: if swing point is near an EMA -> bonus
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


# ===================== VOLUME PROFILE CALCULATOR =====================
def compute_volume_profile(df: pd.DataFrame, period: int = 100, vol_pct: float = 0.70) -> dict:
    """Compute Volume Profile (PoC / VAH / VAL) from last N OHLCV candles.

    Expand from PoC in both directions, adding the higher-volume side first,
    until vol_pct% of total volume is captured.

    Returns:
        dict with keys 'poc', 'vah', 'val' (all float or None on failure)
    """
    subset = df.tail(period).copy()
    if len(subset) < 10:
        return {"poc": None, "vah": None, "val": None}

    price_range = float(subset["high"].max() - subset["low"].min())
    if price_range <= 0:
        return {"poc": None, "vah": None, "val": None}

    # Adaptive bucket step: aim for ~20 price buckets
    step = max(1.0, round(price_range / 20, 0))

    # Typical price (H+L+C)/3 as representative price for each bar
    typical = (subset["high"] + subset["low"] + subset["close"]) / 3
    subset = subset.assign(bucket=(typical / step).round(0) * step)
    vp = subset.groupby("bucket")["volume"].sum().reset_index()
    vp.columns = ["price", "volume"]
    vp = vp.sort_values("price").reset_index(drop=True)

    if len(vp) == 0:
        return {"poc": None, "vah": None, "val": None}

    # PoC = price level with highest volume
    poc_idx = int(vp["volume"].idxmax())
    poc = float(vp.loc[poc_idx, "price"])
    total_vol = float(vp["volume"].sum())
    threshold = total_vol * vol_pct

    # Expand from PoC outward; pick the higher-volume side at each step
    above = vp[vp["price"] > poc].sort_values("price").reset_index(drop=True)
    below = vp[vp["price"] < poc].sort_values("price", ascending=False).reset_index(drop=True)

    cum_vol = float(vp.loc[poc_idx, "volume"])
    vah = poc
    val = poc
    ia, ib = 0, 0

    while cum_vol < threshold and (ia < len(above) or ib < len(below)):
        a_vol = float(above.loc[ia, "volume"]) if ia < len(above) else 0.0
        b_vol = float(below.loc[ib, "volume"]) if ib < len(below) else 0.0

        if a_vol == 0.0 and b_vol == 0.0:
            break

        if a_vol >= b_vol:
            vah = float(above.loc[ia, "price"])
            cum_vol += a_vol
            ia += 1
        else:
            val = float(below.loc[ib, "price"])
            cum_vol += b_vol
            ib += 1

    return {"poc": poc, "vah": float(vah), "val": float(val)}


# ===================== COMBO PRESETS =====================
# Each preset defines:
#   primary: conditions that trigger entry (need >= 1 to fire)
#   confirm: conditions that add confidence (HIGH / MED / LOW)
#   gate: conditions that MUST pass or signal is blocked (filter)
COMBO_PRESETS = {
    "Custom": {"primary": [], "confirm": []},
    "A: Trend Pullback (~65% WR)": {
        "desc": "EMA Ribbon aligned + price pullback to EMA21. Gated by MACD momentum + OBV + TEMA.",
        "primary": ["ema_ribbon", "ema_pullback"],
        "confirm": ["adx_di", "macd_cross", "macd_hist_rev", "obv_confirm"],
        "gate": ["macd_filter", "tema_guard", "volume_ratio_gate"],
    },
    "B: Momentum Breakout (R:R 3:1)": {
        "desc": "BB Squeeze breakout + 20-day high/low break. Gated by MACD + OBV + volume ratio.",
        "primary": ["bb_squeeze", "sr_breakout"],
        "confirm": ["adx_di", "macd_hist_rev", "inside_bar", "obv_confirm"],
        "gate": ["macd_filter", "volume_ratio_gate"],
    },
    "C: Mean Reversion (~60% WR)": {
        "desc": "Price touches BB + Stoch extreme + MFI oversold. BB percent filter ensures entry in zone.",
        "primary": ["bb_bounce", "stoch_cross", "mfi_confirm"],
        "confirm": ["rsi_div", "hammer_star", "engulfing"],
        "gate": ["bb_percent_filter", "ranging_filter"],
    },
    "K: Smart Mean Reversion": {
        "desc": "Base from C + MACD gate + TEMA guard for timing precision.",
        "primary": ["bb_bounce", "stoch_cross", "mfi_confirm"],
        "confirm": ["rsi_div", "hammer_star", "engulfing", "obv_confirm"],
        "gate": ["macd_filter", "tema_guard", "bb_percent_filter"],
    },
    "F: Supertrend Momentum": {
        "desc": "Supertrend direction flip + ADX strong trend. Gated by OBV + TEMA + volume ratio.",
        "primary": ["supertrend_flip"],
        "confirm": ["adx_di", "ema_ribbon", "macd_cross", "obv_confirm"],
        "gate": ["macd_filter", "volume_ratio_gate", "tema_guard"],
    },
    "G: Multi-Oscillator Reversal": {
        "desc": "CCI + Williams %R both extreme + MFI confirmation. Volume ratio gate.",
        "primary": ["cci_extreme", "williams_extreme", "mfi_confirm"],
        "confirm": ["stoch_cross", "bb_bounce", "rsi_div", "obv_confirm"],
        "gate": ["volume_ratio_gate"],
    },
    "H: Alpha Momentum": {
        "desc": "Midpoint momentum + Vol-price divergence + OBV. Momentum ratio gate.",
        "primary": ["midpoint_momentum", "vol_price_divergence"],
        "confirm": ["obv_confirm", "macd_hist_rev", "adx_di"],
        "gate": ["momentum_ratio_gate", "volume_ratio_gate"],
    },
    "I: Alpha Reversal": {
        "desc": "Deviation acceleration + BB bounce + MFI. Vol-return correlation gate.",
        "primary": ["deviation_accel", "bb_bounce", "mfi_confirm"],
        "confirm": ["vol_price_divergence", "stoch_cross", "rsi_div"],
        "gate": ["vol_return_corr_gate", "bb_percent_filter"],
    },
    "G+: Enhanced Multi-Oscillator": {
        "desc": "G combo + momentum ratio gate + vol-return correlation. "
                "+EMA200 bias + EMA slope: ensures oscillator extremes align with dominant trend.",
        "primary": ["cci_extreme", "williams_extreme", "mfi_confirm"],
        "confirm": ["stoch_cross", "bb_bounce", "rsi_div", "obv_confirm", "vol_price_divergence"],
        "gate": ["volume_ratio_gate", "momentum_ratio_gate", "ema200_bias_gate", "ema_slope_gate"],
    },
    "G+T: Trend-Oscillator": {
        "desc": "G+ with HTF trend alignment gate. BUY only in uptrend (EMA9>EMA21), SELL only in downtrend. "
                "Eliminates counter-trend losses: in bull market SELL signals are blocked. "
                "Best on 5m where EMA9/EMA21 cross is meaningful.",
        "primary": ["cci_extreme", "williams_extreme", "mfi_confirm"],
        "confirm": ["stoch_cross", "bb_bounce", "rsi_div", "obv_confirm", "vol_price_divergence"],
        "gate": ["volume_ratio_gate", "momentum_ratio_gate", "htf_trend_filter"],
    },
    "F+: Enhanced Supertrend": {
        "desc": "F combo + vol-return correlation gate.",
        "primary": ["supertrend_flip"],
        "confirm": ["adx_di", "ema_ribbon", "macd_cross", "obv_confirm", "midpoint_momentum"],
        "gate": ["macd_filter", "volume_ratio_gate", "tema_guard", "vol_return_corr_gate"],
    },
    "L: Adaptive Reversal": {
        "desc": "Normalized + Adaptive RSI reversal with CoG leading. KB squeeze gate ensures low-vol entry.",
        "primary": ["norm_rsi", "adaptive_rsi", "cog_reversal"],
        "confirm": ["stoch_cross", "bb_bounce", "mfi_confirm", "obv_confirm"],
        "gate": ["kb_squeeze", "volume_ratio_gate"],
    },
    "M: Trend Breakout": {
        "desc": "Trend composite + KB squeeze release + Supertrend. Trend strength gate ensures quality trend.",
        "primary": ["trend_composite", "supertrend_flip", "sr_breakout"],
        "confirm": ["adx_di", "ema_ribbon", "macd_cross", "obv_confirm"],
        "gate": ["trend_strength_gate", "kb_squeeze", "volume_ratio_gate"],
    },
    "N: Regime Adaptive": {
        "desc": "Crypto-inspired: ADX detects regime. Trending=EMA cross+MACD+DI. Ranging=RSI+BB+OBV. EMA200 bias gate.",
        "primary": ["regime_ema_cross", "regime_range_bounce"],
        "confirm": ["supertrend_flip", "obv_confirm", "macd_hist_rev", "adx_di"],
        "gate": ["ema200_bias"],
    },
    # ===== 1M SCALP COMBOS =====
    "P: Fast EMA Scalp": {
        "desc": "1m scalp: EMA3 x EMA8 crossover + strong directional candle. HTF trend gate filters noise.",
        "primary": ["fast_ema_cross", "candle_momentum"],
        "confirm": ["obv_confirm", "macd_hist_rev", "adx_di"],
        "gate": ["htf_trend_filter", "volume_ratio_gate"],
    },
    "Q: ORB Scalp": {
        "desc": "1m scalp: Open Range Breakout of first 15-min high/low. HTF + volume + VWAP confirm.",
        "primary": ["orb_breakout"],
        "confirm": ["adx_di", "obv_confirm", "vwap_dev", "candle_momentum"],
        "gate": ["htf_trend_filter", "volume_ratio_gate"],
    },
    "R: VWAP Bounce Scalp": {
        "desc": "1m mean reversion: price deviates from VWAP + BB bounce + Stoch extreme. Ranging gate only (no HTF — counter-trend by design).",
        "primary": ["vwap_dev", "bb_bounce"],
        "confirm": ["stoch_cross", "obv_confirm", "mfi_confirm", "candle_momentum"],
        "gate": ["volume_ratio_gate", "ranging_filter"],
    },
    "S: HTF-Filtered VWAP Scalp": {
        "desc": "1m premium: VWAP pullback touch + micro-momentum + HTF 5m trend alignment. "
                "Most selective 1m combo (GitHub pattern: TrendRider multi-TF + VWAP bounce).",
        "primary": ["vwap_pullback"],
        "confirm": ["candle_momentum", "obv_confirm", "adx_di", "macd_hist_rev"],
        "gate": ["htf_trend_filter", "volume_ratio_gate"],
    },
    "T: Micro Momentum Burst": {
        "desc": "1m breakout: 5-bar high/low break with volume surge + HTF trend. "
                "Captures momentum continuation after liquidity sweep (common in VN30F futures).",
        "primary": ["momentum_burst"],
        "confirm": ["candle_momentum", "adx_di", "obv_confirm"],
        "gate": ["htf_trend_filter", "volume_ratio_gate"],
    },
    # ===== BIG-MOVE CATCHER =====
    "W: EMA21 Bounce": {
        "desc": "Data-driven (216d, 827 swings): EMA21 acts as dynamic S/R. "
                "BUY when price dips within 1.2xATR of EMA21 + Supertrend bullish + ADX>20 + prior pullback. "
                "Lift 1.34x vs base avg. SELL symmetric. "
                "+EMA200 bias + prev_bar gate: ensures trend alignment + momentum confirmation. "
                "Oscillator confirms add quality filter. SL=1.2xATR, TP=4.0xATR.",
        "primary": ["ema21_pullback"],
        "confirm": ["cci_extreme", "williams_extreme", "mfi_confirm", "stoch_cross",
                    "bb_bounce", "adx_di", "obv_confirm"],
        "gate":    ["session_open_filter", "w_regime_gate", "ema21_trend_gate",
                    "ema200_bias_gate", "prev_bar_gate"],
    },
    "O: ORB Momentum": {
        "desc": "ORB breakout in trending+volatile regime (ADX>=22 + ATR>=3.0 + EMA aligned). "
                "BUY when 15-min opening range high breaks in bull EMA context. "
                "SELL when 15-min opening range low breaks in bear EMA context. "
                "Max 2 signals per session (AM + PM ORB). SL=1.5xATR, TP=3.5xATR.",
        "primary": ["orb_breakout"],
        "confirm": ["adx_di", "obv_confirm", "candle_momentum"],
        "gate":    ["session_open_filter", "b_regime_gate"],
    },
    "V2: Bounce Confirm": {
        "desc": "Data-driven V2: wait for FIRST reversal candle after osc extreme + pullback. "
                "Fixes V's 'falling knife' problem by requiring confirmed bounce before entry. "
                "Entry = recent CCI/WR extreme + prior pullback + first bullish/bearish candle. "
                "+EMA200 bias + strong candle: WR boost by trend alignment + quality filter. "
                "Best on 5m. SL=1.5xATR, TP=3.5xATR.",
        "primary": ["bounce_confirm"],
        "confirm": ["adx_di", "obv_confirm", "adx_trend_filter"],
        "gate":    ["session_open_filter", "ema200_bias_gate", "strong_candle_gate"],
    },
    "V: VWAP Dip Bounce": {
        "desc": "Data-driven: 74% of 5+pt moves start with 3-bar pullback below daily VWAP + osc extreme. "
                "HTF 5m trend filter prevents counter-trend knives. "
                "+EMA200 bias + strong candle: V/5m SC+EMA200 WR 49→75% (4T). "
                "Best on 3m/5m. SL=1.5xATR, TP=3.5xATR.",
        "primary": ["vwap_side_bounce"],
        "confirm": ["cci_extreme", "williams_extreme", "bb_bounce",
                     "adx_di", "obv_confirm", "mfi_confirm", "adx_trend_filter"],
        "gate": ["session_open_filter", "ema200_bias_gate", "strong_candle_gate"],
    },
    # ===== BIG MOVE CATCHER (7+ pts analysis-driven) =====
    "X: Big Move Catcher": {
        "desc": "Data-driven (153 moves >=7pts, 120d): 59% preceded by STOP HUNT (liquidity sweep). "
                "88% have at least 1 trap. Entry after trap detection + momentum confirmation. "
                "Strong candle filter (body>55% range) improves WR from 34% to 48-56%. "
                "Primary: liquidity_sweep + pullback_reversal. "
                "Confirm: supertrend aligned + ADX/DMI + volume surge + MACD momentum. "
                "Best params: SL=1.5, TP=2.5, MH=30. WR=56.5%, PF=1.51, +22.5pts/120d.",
        "primary": ["liquidity_sweep", "pullback_reversal"],
        "confirm": ["supertrend_flip", "adx_di", "obv_confirm", "macd_hist_rev",
                    "cci_extreme", "williams_extreme"],
        "gate":    ["volume_ratio_gate", "trap_confirmation_gate", "strong_candle_gate",
                    "ema_slope_gate"],
    },
    # ===== SWING ANALYSIS — H2/H3/H5 DIRECTIONS =====
    "D: AM Mid Trend": {
        "desc": "H2: 9:30-10:15 AM trend entry after opening noise clears. "
                "EMA21 slope confirmed + MACD hist direction + ADX>=18. "
                "25% of all swings start in this window (avg 12.3pts). "
                "Volume gates: vol_surge + OBV trend + MFI direction for quality filter. "
                "+EMA200 bias gate: WR 65→75% (20T, PF=2.62). "
                "Gate: session filter + adx_trend_filter + vol_surge_gate + obv_trend_gate + mfi_direction_gate + ema200_bias_gate.",
        "primary": ["am_mid_entry"],
        "confirm": ["adx_di", "obv_confirm", "candle_momentum"],
        "gate":    ["session_open_filter", "adx_trend_filter",
                    "vol_surge_gate", "obv_trend_gate", "mfi_direction_gate",
                    "ema200_bias_gate"],
    },
    "E: Strong Momentum": {
        "desc": "H3: Strong trending regime entry via supertrend + DI± direction + EMA alignment. "
                "ADX>=25 + ATR>=4.0 regime gate. Targets 45%+ big moves in trending months. "
                "Best: Mar-26 ADX=28 avg 15+pts, ADX>30+ATR>5 swings avg 18pts. "
                "Volume gates: vol_surge + OBV trend filter to block fake recoveries in Apr-type markets. "
                "SL=1.5xATR, TP=3.5xATR in high-ADX regime.",
        "primary": ["strong_trend_entry"],
        "confirm": ["adx_di", "obv_confirm", "candle_momentum"],
        "gate":    ["session_open_filter", "strong_trend_gate",
                    "vol_surge_gate", "obv_trend_gate"],
    },
    "J: RSI Momentum": {
        "desc": "H5: Counter-intuitive momentum continuation. "
                "RSI<30 in downtrend = SELL (momentum continuation in crash), "
                "RSI>70 in uptrend = BUY (momentum in strong rally). "
                "Data: SHORT@RSI<30 avg 16.3pts (best class from 199-swing analysis). "
                "Volume gates: vol_surge + OBV trend blocks choppy RSI extremes without real conviction. "
                "+EMA200 bias + slope gate: boost WR by filtering counter-trend noise. "
                "Gate: session_open_filter + b_regime_gate + vol_surge_gate + obv_trend_gate + ema200_bias_gate + ema_slope_gate.",
        "primary": ["rsi_momentum_entry"],
        "confirm": ["adx_di", "obv_confirm", "supertrend_flip"],
        "gate":    ["session_open_filter", "b_regime_gate",
                    "vol_surge_gate", "obv_trend_gate",
                    "ema200_bias_gate", "ema_slope_gate"],
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
    "sr_breakout": "S/R Breakout", "sr_atr": "S/R +/- ATR",
    "macd_filter": "MACD Filter", "vol_color_filter": "Vol Color Filter",
    # New indicators
    "ichimoku_cross": "Ichimoku Cross", "supertrend_flip": "Supertrend Flip",
    "vwap_dev": "VWAP Deviation", "donchian_break": "Donchian Break",
    "cci_extreme": "CCI Extreme", "psar_flip": "PSAR Flip",
    "williams_extreme": "Williams %R",
    "pvt_confirm": "PVT Confirm",
    # Gate filters
    "ribbon_trend_filter": "Ribbon Trend", "session_open_filter": "Session Filter",
    "adx_trend_filter": "ADX Trend", "market_regime_filter": "Market Regime",
    "ranging_filter": "Ranging Filter",
    # Freqtrade-inspired new indicators
    "mfi_confirm": "MFI Confirm", "obv_confirm": "OBV Confirm",
    "volume_ratio_gate": "Vol Ratio Gate", "tema_guard": "TEMA Guard",
    "bb_percent_filter": "BB% Filter",
    # Alpha Zoo inspired (Vibe-Trading)
    "momentum_ratio_gate": "Momentum Ratio", "vol_price_divergence": "Vol-Price Div",
    "midpoint_momentum": "Midpoint Mom", "deviation_accel": "Deviation Accel",
    "vol_return_corr_gate": "Vol-Ret Corr",
    # Superalgos-inspired
    "kb_squeeze": "KB Squeeze", "norm_rsi": "Norm RSI Z",
    "adaptive_rsi": "Adaptive RSI", "cog_reversal": "CoG Reversal",
    "trend_composite": "Trend Composite", "trend_strength_gate": "Trend Gate",
    # Crypto-inspired (ComboH regime adaptive)
    "regime_ema_cross": "Regime EMA Cross", "regime_range_bounce": "Regime Range Bounce",
    "ema200_bias": "EMA200 Bias",
    # 1m Scalp conditions
    "fast_ema_cross": "Fast EMA Cross (3x8)",
    "candle_momentum": "Candle Momentum",
    "orb_breakout": "ORB Breakout",
    # HTF + 1m premium conditions
    "htf_trend_filter": "HTF 5m Trend Filter",
    "vwap_pullback": "VWAP Pullback Touch",
    "momentum_burst": "Momentum Burst (5-bar)",
    # Big-move catcher
    "vwap_side_bounce": "VWAP Dip/Rally Bounce",
    "bounce_confirm":   "Bounce Confirm (V2)",
    "ema21_pullback":         "EMA21 Dip Bounce (W)",
    # Regime gates
    "w_regime_gate":          "W Regime Gate (ATR≥3.5 + EMA55 gentle)",
    "g_regime_gate":          "G+ Regime Gate (ATR 3.5-7 + ADX≥18)",
    "b_regime_gate":          "B Regime Gate (ADX≥22 + ATR≥3.0)",
    # Trend breakout (B)
    "trend_momentum_entry":   "Trend Momentum Re-entry (B)",
    # H2/H3/H5 new conditions
    "am_mid_entry":           "AM Mid Entry (9:30-10:15)",
    "strong_trend_gate":      "Strong Trend Gate (ADX≥25 + ATR≥4.0)",
    "strong_trend_entry":     "Strong Trend Entry (H3: DI+ST+EMA aligned)",
    "rsi_momentum_entry":     "RSI Momentum Continuation (H5)",
    # Volume & flow gates (D/E/J)
    "vol_surge_gate":         "Vol Surge Gate (vol_ratio≥1.2)",
    "obv_trend_gate":         "OBV Trend Gate (OBV_EMA slope 5-bar)",
    "mfi_direction_gate":     "MFI Direction Gate (MFI>50 BUY / <50 SELL)",
    # EMA21 trend clarity gate (W)
    "ema21_trend_gate":       "EMA21 Trend Gate (slope>0 BUY / <0 SELL + dist≥0.5×ATR)",
    # Big Move Catcher (X)
    "liquidity_sweep":        "Liquidity Sweep (break recent H/L then reverse)",
    "pullback_reversal":      "Pullback Reversal (mom3 opposite then candle flip)",
    "trap_confirmation_gate": "Trap Confirm Gate (sweep + vol + momentum)",
    "strong_candle_gate":     "Strong Candle Gate (body/range > 55%)",
    # Universal WR boost gates
    "ema200_bias_gate":       "EMA200 Bias Gate (trade with dominant trend)",
    "ema_slope_gate":         "EMA Slope Gate (EMA20 slope alignment)",
    "prev_bar_gate":          "Prev Bar Gate (prev candle confirms direction)",
}

# All condition keys
ALL_COND_KEYS = [
    "sma_cross", "macd_cross", "ema_pullback", "bb_squeeze", "rsi_div",
    "macd_hist_rev", "stoch_cross", "bb_bounce", "engulfing",
    "ema_ribbon", "inside_bar", "hammer_star", "adx_di", "sr_breakout",
    "sr_atr", "macd_filter", "vol_color_filter",
    # New
    "ichimoku_cross", "supertrend_flip", "vwap_dev", "donchian_break",
    "cci_extreme", "psar_flip", "williams_extreme",
    "pvt_confirm",
    # Gate filters
    "ribbon_trend_filter", "session_open_filter", "adx_trend_filter",
    "market_regime_filter", "ranging_filter",
    # Freqtrade-inspired
    "mfi_confirm", "obv_confirm", "volume_ratio_gate", "tema_guard",
    "bb_percent_filter",
    # Alpha Zoo inspired (Vibe-Trading)
    "momentum_ratio_gate", "vol_price_divergence", "midpoint_momentum",
    "deviation_accel", "vol_return_corr_gate",
    # Superalgos-inspired
    "kb_squeeze", "norm_rsi", "adaptive_rsi", "cog_reversal",
    "trend_composite", "trend_strength_gate",
    # Crypto-inspired (regime adaptive)
    "regime_ema_cross", "regime_range_bounce", "ema200_bias",
    # 1m Scalp conditions
    "fast_ema_cross", "candle_momentum", "orb_breakout",
    # HTF + 1m premium
    "htf_trend_filter", "vwap_pullback", "momentum_burst",
    # Big-move catcher
    "vwap_side_bounce",
    "bounce_confirm",
    "ema21_pullback",
    # Regime gates
    "w_regime_gate", "g_regime_gate", "b_regime_gate",
    # Trend breakout (B)
    "trend_momentum_entry",
    # H2/H3/H5 new conditions
    "am_mid_entry", "strong_trend_gate", "strong_trend_entry", "rsi_momentum_entry",
    # Volume & flow gates
    "vol_surge_gate", "obv_trend_gate", "mfi_direction_gate",
    # EMA21 trend clarity gate
    "ema21_trend_gate",
    # Big Move Catcher (X)
    "liquidity_sweep", "pullback_reversal", "trap_confirmation_gate",
    "strong_candle_gate",
    # Universal WR boost gates
    "ema200_bias_gate", "ema_slope_gate", "prev_bar_gate",
]


# ===================== SIGNAL ANALYSIS HELPER =====================
def analyze_signal_performance(sig_df, atr_sl_mult=1.5, atr_tp_mult=3.0, max_hold=30):
    """Analyze each signal: compute SL/TP, forward-simulate, return stats dict + detail rows.

    Parameters:
    -----------
    sig_df : DataFrame with 'signal', 'close', 'high', 'low', 'atr' columns
    atr_sl_mult : Stop Loss = entry +/- (mult x ATR)
    atr_tp_mult : Take Profit = entry +/- (mult x ATR)
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
    df["ema3"]  = ta.ema(df["close"], length=3)   # Ultra-fast (1m scalp)
    df["ema5"]  = ta.ema(df["close"], length=5)
    df["ema8"]  = ta.ema(df["close"], length=8)
    df["ema12"] = ta.ema(df["close"], length=12)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["ema55"] = ta.ema(df["close"], length=55)
    df["ema200"] = ta.ema(df["close"], length=200)

    # --- Open Range Breakout (ORB, 1m scalp) ---
    # ORB = high/low of first 15 bars of morning session (09:00-09:14 VN time)
    if "time" in df.columns:
        _t = pd.to_datetime(df["time"])
        _h, _m = _t.dt.hour, _t.dt.minute
        _date = _t.dt.date
        _is_am_orb = (_h == 9) & (_m < 15)           # 09:00-09:14
        _is_pm_orb = (_h == 13) & (_m < 10)          # 13:00-13:09
        _df_orb = df.copy()
        _df_orb["_date"] = _date
        _am_high = _df_orb.loc[_is_am_orb].groupby("_date")["high"].max()
        _am_low  = _df_orb.loc[_is_am_orb].groupby("_date")["low"].min()
        _pm_high = _df_orb.loc[_is_pm_orb].groupby("_date")["high"].max()
        _pm_low  = _df_orb.loc[_is_pm_orb].groupby("_date")["low"].min()
        df["orb_high"] = _df_orb["_date"].map(_am_high).values
        df["orb_low"]  = _df_orb["_date"].map(_am_low).values
        # PM ORB (fallback to AM if no PM data)
        df["orb_pm_high"] = _df_orb["_date"].map(_pm_high).fillna(df["orb_high"]).values
        df["orb_pm_low"]  = _df_orb["_date"].map(_pm_low).fillna(df["orb_low"]).values
        # Combined: best ORB reference at each bar
        _in_am = (_h >= 9) & ((_h < 11) | ((_h == 11) & (_m <= 30)))
        _in_pm = (_h >= 13) & ((_h < 14) | ((_h == 14) & (_m <= 45)))
        df["orb_ref_high"] = np.where(_in_pm, df["orb_pm_high"], df["orb_high"])
        df["orb_ref_low"]  = np.where(_in_pm, df["orb_pm_low"],  df["orb_low"])
        # ORB window: signal valid only AFTER ORB formation
        df["_in_trading_window"] = _in_am | _in_pm
        df["_after_orb"] = (
            ((_h == 9) & (_m >= 15)) | ((_h > 9) & (_h < 12)) |
            ((_h == 13) & (_m >= 10)) | ((_h > 13) & (_h < 15))
        ).values
    else:
        df["orb_high"] = df["orb_low"] = df["orb_ref_high"] = df["orb_ref_low"] = np.nan
        df["_in_trading_window"] = True
        df["_after_orb"] = True

    # --- HTF 5m Trend Filter (resample 1m→5m, inspired by TrendRider/multi-TF strategies) ---
    # Only take longs when 5m EMA9 > EMA21 (uptrend), shorts when EMA9 < EMA21 (downtrend).
    # This is the #1 WR improvement for 1m scalping from GitHub community strategies.
    if "time" in df.columns:
        try:
            _t_idx = pd.to_datetime(df["time"])
            _t5 = _t_idx.dt.floor("5min")
            # 5m close = last 1m close in each 5m window
            _5m_close = df.groupby(_t5)["close"].last()
            _5m_ema9  = ta.ema(_5m_close, length=9)
            _5m_ema21 = ta.ema(_5m_close, length=21)
            df["htf_ema9"]  = _t5.map(_5m_ema9).ffill().values
            df["htf_ema21"] = _t5.map(_5m_ema21).ffill().values
        except Exception:
            df["htf_ema9"]  = df["ema21"]
            df["htf_ema21"] = df["ema55"]
    else:
        # Fallback when no time column: use EMA21 vs EMA55 as proxy
        df["htf_ema9"]  = df["ema21"]
        df["htf_ema21"] = df["ema55"]
    df["htf_bull"] = (df["htf_ema9"] > df["htf_ema21"]).astype(int)
    df["htf_bear"] = (df["htf_ema9"] < df["htf_ema21"]).astype(int)

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

    # BB Width normalized (% of mid) -- used to detect squeeze
    df["bb_width_pct"] = df["bb_width"] / df["bb_mid"].replace(0, np.nan)
    df["bb_squeeze_flag"] = df["bb_width_pct"] <= df["bb_width_pct"].rolling(20).quantile(0.2)

    # BB Percent (0-1 normalized position within BB)
    # 0 = at lower band, 1 = at upper band, <0.3 = oversold zone, >0.7 = overbought zone
    df["bb_percent"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # --- Stochastic ---
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=stoch_k, d=stoch_d)
    if stoch is not None:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]
    else:
        df["stoch_k"] = 50
        df["stoch_d"] = 50

    # --- TEMA (Triple Exponential Moving Average, period 9) ---
    # More responsive than EMA, less lag -> better entry timing
    df["tema9"] = ta.tema(df["close"], length=9)
    df["tema_rising"] = df["tema9"] > df["tema9"].shift(1)
    df["tema_falling"] = df["tema9"] < df["tema9"].shift(1)

    # --- Volume ---
    df["vol_sma"] = ta.sma(df["volume"].astype(float), length=20)
    df["vol_ok"] = df["volume"] > (vol_mult * df["vol_sma"])

    # --- Volume Ratio (continuous, from TrendRider) ---
    # vol_ratio > 1.0 = above average, > 1.5 = strong volume surge
    df["vol_ema20"] = ta.ema(df["volume"].astype(float), length=20)
    df["volume_ratio"] = df["volume"].astype(float) / (df["vol_ema20"] + 1e-10)

    # --- MFI (Money Flow Index, 14-period) ---
    # Volume-weighted RSI - better reversal confirmation than RSI alone
    try:
        mfi_result = ta.mfi(df["high"], df["low"], df["close"], df["volume"].astype(float), length=14)
        df["mfi"] = mfi_result if mfi_result is not None else 50.0
    except Exception:
        df["mfi"] = 50.0

    # --- OBV (On-Balance Volume) + OBV EMA ---
    # OBV > OBV_EMA = accumulation (bullish), < = distribution (bearish)
    try:
        obv_result = ta.obv(df["close"], df["volume"].astype(float))
        df["obv"] = obv_result if obv_result is not None else 0.0
    except Exception:
        df["obv"] = 0.0
    df["obv_ema"] = ta.ema(df["obv"], length=20)
    df["obv_rising"] = df["obv"] > df["obv_ema"]

    # --- ATR (14-day, Investopedia standard) ---
    # Skip if caller pre-computed ATR (e.g. scaled for non-5m timeframes)
    if "atr" not in df.columns or df["atr"].isna().all():
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

    # --- Ichimoku Cloud (Tenkan 9 / Kijun 26 / Senkou 52) ---
    try:
        ichi = ta.ichimoku(df["high"], df["low"], df["close"], tenkan=9, kijun=26, senkou=52)
        if ichi is not None and len(ichi) == 2:
            ichi_df = ichi[0]
            cols = ichi_df.columns.tolist()
            tenkan_col = next((c for c in cols if c.startswith("ITS")), None)
            kijun_col  = next((c for c in cols if c.startswith("IKS")), None)
            span_a_col = next((c for c in cols if c.startswith("ISA")), None)
            span_b_col = next((c for c in cols if c.startswith("ISB")), None)
            df["ichi_tenkan"] = ichi_df[tenkan_col] if tenkan_col else np.nan
            df["ichi_kijun"]  = ichi_df[kijun_col]  if kijun_col  else np.nan
            df["ichi_span_a"] = ichi_df[span_a_col] if span_a_col else np.nan
            df["ichi_span_b"] = ichi_df[span_b_col] if span_b_col else np.nan
        else:
            df["ichi_tenkan"] = df["ichi_kijun"] = df["ichi_span_a"] = df["ichi_span_b"] = np.nan
    except Exception:
        df["ichi_tenkan"] = df["ichi_kijun"] = df["ichi_span_a"] = df["ichi_span_b"] = np.nan

    # --- Supertrend (length=7, multiplier=3.0) ---
    try:
        st = ta.supertrend(df["high"], df["low"], df["close"], length=7, multiplier=3.0)
        if st is not None:
            st_dir_col = next((c for c in st.columns if "SUPERTd" in c), None)
            df["supertrend_dir"] = st[st_dir_col].fillna(0) if st_dir_col else 0
        else:
            df["supertrend_dir"] = 0
    except Exception:
        df["supertrend_dir"] = 0

    # --- VWAP (requires DatetimeIndex) ---
    try:
        _df_vwap = df.copy()
        if "time" in _df_vwap.columns and not isinstance(_df_vwap.index, pd.DatetimeIndex):
            _df_vwap.index = pd.to_datetime(_df_vwap["time"])
        vwap_result = ta.vwap(_df_vwap["high"], _df_vwap["low"], _df_vwap["close"], _df_vwap["volume"].astype(float))
        df["vwap"] = vwap_result.values if vwap_result is not None else df["close"].values
    except Exception:
        df["vwap"] = df["close"]
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["atr"].replace(0, np.nan)

    # --- Daily-Anchored VWAP (resets at 09:00 each VN trading day) ---
    # This is the correct VWAP for intraday scalping — price should oscillate near it.
    # Formula: cumsum(typical_price × volume) / cumsum(volume), reset per calendar day.
    if "time" in df.columns:
        try:
            _t_vd = pd.to_datetime(df["time"])
            _date_vd = _t_vd.dt.date.astype(str)
            _tp = (df["high"] + df["low"] + df["close"]) / 3.0
            _tp_vol = _tp * df["volume"].astype(float)
            _cumvol    = df.groupby(_date_vd)["volume"].cumsum().astype(float)
            _cumtpvol  = _tp_vol.groupby(_date_vd).cumsum()
            df["vwap_d"] = (_cumtpvol / _cumvol.replace(0, float("nan"))).values
        except Exception:
            df["vwap_d"] = df["close"]
    else:
        df["vwap_d"] = df["bb_mid"]   # fallback: BB midline as price mean
    df["vwap_d_dist"] = (df["close"] - df["vwap_d"]) / df["atr"].replace(0, float("nan"))

    # --- Donchian Channels (shifted 1 bar to avoid look-ahead) ---
    try:
        dc = ta.donchian(df["high"], df["low"], lower_length=20, upper_length=20)
        if dc is not None:
            dcu = next((c for c in dc.columns if "DCU" in c), None)
            dcl = next((c for c in dc.columns if "DCL" in c), None)
            df["dc_upper"] = dc[dcu].shift(1) if dcu else df["resistance_20"]
            df["dc_lower"] = dc[dcl].shift(1) if dcl else df["support_20"]
        else:
            df["dc_upper"] = df["resistance_20"]
            df["dc_lower"] = df["support_20"]
    except Exception:
        df["dc_upper"] = df["resistance_20"]
        df["dc_lower"] = df["support_20"]

    # --- CCI (Commodity Channel Index, 14-period) ---
    try:
        cci_result = ta.cci(df["high"], df["low"], df["close"], length=14)
        df["cci"] = cci_result if cci_result is not None else 0
    except Exception:
        df["cci"] = 0

    # --- Parabolic SAR ---
    try:
        psar = ta.psar(df["high"], df["low"], df["close"])
        if psar is not None:
            psar_l = next((c for c in psar.columns if "PSARl" in c), None)
            # PSARl has values (not NaN) only during bullish phase
            df["psar_bull"] = psar[psar_l].notna() if psar_l else True
        else:
            df["psar_bull"] = True
    except Exception:
        df["psar_bull"] = True
    df["psar_bull_prev"] = df["psar_bull"].shift(1).fillna(True)

    # --- Williams %R (14-period) ---
    try:
        willr_result = ta.willr(df["high"], df["low"], df["close"], length=14)
        df["willr"] = willr_result if willr_result is not None else -50
    except Exception:
        df["willr"] = -50

    # --- PVT (Price Volume Trend) ---
    # Rising PVT => volume confirms bullish price move
    # Falling PVT => volume confirms bearish price move
    try:
        _pvt_chg = df["volume"].astype(float) * (
            (df["close"] - df["close"].shift(1)) /
            df["close"].shift(1).replace(0, np.nan)
        )
        df["pvt"] = _pvt_chg.fillna(0).cumsum()
        df["pvt_bull"] = df["pvt"] > df["pvt"].shift(3)  # rising over 3 bars
    except Exception:
        df["pvt"] = 0.0
        df["pvt_bull"] = True

    # --- Keltner Channels (for KB Squeeze detection) ---
    # Keltner = EMA(20) +/- 1.5 * ATR(10)
    _kc_ema = ta.ema(df["close"], length=20)
    _kc_atr = ta.atr(df["high"], df["low"], df["close"], length=10)
    if _kc_atr is None:
        _kc_atr = df["atr"]
    df["kc_upper"] = _kc_ema + 1.5 * _kc_atr
    df["kc_lower"] = _kc_ema - 1.5 * _kc_atr
    # Squeeze: BB is INSIDE Keltner Channel (volatility compressed)
    df["kb_squeeze_on"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
    # Squeeze release: was squeezing, now released
    df["kb_squeeze_off"] = (~df["kb_squeeze_on"]) & (df["kb_squeeze_on"].shift(1).fillna(False))

    # --- Normalized RSI (Z-score dynamic bands, from Superalgos Normalized_Momentum) ---
    # Z = (RSI - SMA(RSI,10)) / StdDev(RSI,10)
    # Dynamic bands = SMA(Z,20) +/- 2*StdDev(Z,20)
    _rsi_sma = df["rsi"].rolling(10, min_periods=10).mean()
    _rsi_std = df["rsi"].rolling(10, min_periods=10).std()
    df["norm_rsi_z"] = (df["rsi"] - _rsi_sma) / _rsi_std.replace(0, np.nan)
    _z_sma = df["norm_rsi_z"].rolling(20, min_periods=10).mean()
    _z_std = df["norm_rsi_z"].rolling(20, min_periods=10).std()
    df["norm_rsi_upper"] = _z_sma + 2.0 * _z_std
    df["norm_rsi_lower"] = _z_sma - 2.0 * _z_std

    # --- Auto-Adaptive RSI (DFT dominant cycle detection, from Superalgos) ---
    # Use FFT to find dominant cycle period (8-50 range), then use as RSI period
    # Much faster than rolling autocorrelation
    _close_arr = df["close"].values.astype(float)
    _adaptive_period = np.full(len(df), 14.0)  # default 14
    _min_period, _max_period = 8, 50
    _dft_lookback = 60  # bars to analyze

    for i in range(_dft_lookback, len(df)):
        _segment = _close_arr[i - _dft_lookback:i]
        _segment = _segment - np.mean(_segment)  # detrend
        _seg_std = np.std(_segment)
        if _seg_std < 1e-10:
            continue
        # FFT-based dominant cycle detection
        _fft = np.fft.rfft(_segment)
        _power = np.abs(_fft) ** 2
        # Convert FFT bins to periods: period = lookback / frequency_index
        # Skip bin 0 (DC) and bins with period outside our range
        _best_period = 14
        _best_power = 0.0
        for _bin in range(1, len(_power)):
            _period = _dft_lookback / _bin
            if _min_period <= _period <= _max_period:
                if _power[_bin] > _best_power:
                    _best_power = _power[_bin]
                    _best_period = int(round(_period))
        _adaptive_period[i] = _best_period

    df["adaptive_rsi_period"] = _adaptive_period
    # Compute adaptive RSI using rolling periods (batch by common periods for speed)
    _unique_periods = np.unique(_adaptive_period[_dft_lookback:].astype(int))
    df["adaptive_rsi"] = df["rsi"].copy()  # fallback
    for _per in _unique_periods:
        _mask = df["adaptive_rsi_period"].astype(int) == _per
        if _mask.sum() > 0:
            _rsi_p = ta.rsi(df["close"], length=int(_per))
            if _rsi_p is not None:
                df.loc[_mask, "adaptive_rsi"] = _rsi_p[_mask]

    # Normalize adaptive RSI with Z-score bands
    _arsi_sma = df["adaptive_rsi"].rolling(10, min_periods=5).mean()
    _arsi_std = df["adaptive_rsi"].rolling(10, min_periods=5).std()
    df["adaptive_rsi_z"] = (df["adaptive_rsi"] - _arsi_sma) / _arsi_std.replace(0, np.nan)

    # --- Center of Gravity Oscillator (Ehlers, from Superalgos Polus mine) ---
    # CoG = -Sum(price[i] * (i+1), i=0..N-1) / Sum(price[i], i=0..N-1)
    # Leading indicator - turns BEFORE price
    _cog_period = 10
    _weights = np.arange(1, _cog_period + 1, dtype=float)
    # Vectorized rolling weighted sum
    _price_series = df["close"].values.astype(float)
    _cog_values = np.full(len(df), np.nan)
    # Use rolling window approach via stride tricks for speed
    if len(df) >= _cog_period:
        from numpy.lib.stride_tricks import sliding_window_view
        _windows = sliding_window_view(_price_series, _cog_period)
        _sum_pw = np.sum(_windows * _weights, axis=1)
        _sum_p = np.sum(_windows, axis=1)
        _sum_p[np.abs(_sum_p) < 1e-10] = np.nan
        _cog_calc = -_sum_pw / _sum_p
        _cog_values[_cog_period - 1:] = _cog_calc
    df["cog"] = _cog_values
    # CoG signal line (3-period EMA of CoG)
    df["cog_signal"] = pd.Series(_cog_values).ewm(span=3, adjust=False).mean().values

    # --- Trend Composite (Linear Regression + R² + Aroon) ---
    # LinReg slope: direction and steepness of trend
    # R²: quality of trend (1.0 = perfect trend, 0.0 = no trend)
    # Aroon: trend maturity (how many bars since high/low)
    _linreg_period = 20
    _linreg_slope = np.full(len(df), 0.0)
    _linreg_r2 = np.full(len(df), 0.0)

    _x = np.arange(_linreg_period, dtype=float)
    _x_mean = _x.mean()
    _x_var = np.sum((_x - _x_mean) ** 2)

    # Vectorized using sliding_window_view
    if len(df) >= _linreg_period:
        from numpy.lib.stride_tricks import sliding_window_view
        _y_windows = sliding_window_view(_close_arr, _linreg_period)
        _y_means = _y_windows.mean(axis=1)
        _y_centered = _y_windows - _y_means[:, np.newaxis]
        _x_centered = _x - _x_mean
        _cov = np.sum(_y_centered * _x_centered, axis=1)
        _slopes = _cov / _x_var
        _y_var = np.sum(_y_centered ** 2, axis=1)
        _r_sq = np.where(_y_var > 0, (_cov ** 2) / (_x_var * _y_var), 0.0)
        _linreg_slope[_linreg_period - 1:] = _slopes
        _linreg_r2[_linreg_period - 1:] = _r_sq

    df["linreg_slope"] = _linreg_slope
    # Normalize slope by ATR for comparability
    df["linreg_slope_norm"] = _linreg_slope / df["atr"].replace(0, np.nan).values
    df["linreg_r2"] = _linreg_r2

    # Aroon oscillator (periods since highest high / lowest low)
    _aroon_period = 25
    try:
        _aroon = ta.aroon(df["high"], df["low"], length=_aroon_period)
        if _aroon is not None:
            _aroon_up_col = next((c for c in _aroon.columns if "AROONU" in c), None)
            _aroon_dn_col = next((c for c in _aroon.columns if "AROOND" in c), None)
            df["aroon_up"] = _aroon[_aroon_up_col] if _aroon_up_col else 50.0
            df["aroon_down"] = _aroon[_aroon_dn_col] if _aroon_dn_col else 50.0
        else:
            df["aroon_up"] = 50.0
            df["aroon_down"] = 50.0
    except Exception:
        df["aroon_up"] = 50.0
        df["aroon_down"] = 50.0
    df["aroon_osc"] = df["aroon_up"] - df["aroon_down"]  # +100 = strong up, -100 = strong down

    # Trend Composite Score: combines slope direction + R² quality + Aroon maturity
    # Range roughly -3 to +3 (positive = bullish trend, negative = bearish)
    _slope_score = np.clip(df["linreg_slope_norm"].values, -1.5, 1.5)  # cap ±1.5
    _r2_score = df["linreg_r2"].values  # 0 to 1
    _aroon_score = df["aroon_osc"].values / 100.0  # -1 to +1
    df["trend_composite"] = _slope_score + _r2_score * np.sign(_slope_score) + _aroon_score

    # ==================== SCORE SYSTEM ====================
    df["buy_score"] = 0
    df["sell_score"] = 0

    # Per-condition tracking columns
    for _k in ALL_COND_KEYS:
        df[f"_b_{_k}"] = 0
        df[f"_s_{_k}"] = 0

    # ==================== CONDITIONS ====================

    # --- 1. SMA Cross (Golden Cross / Death Cross) ---
    # Ref: Investopedia - Golden Cross is lagging -> confirm with other indicators
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
    # Ref: Investopedia - "Crossovers more reliable when conform to prevailing trend"
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
    # Logic: BB width at 20-period minimum (squeeze) -> breakout when price exceeds BB
    # Squeeze = low volatility -> expansion expected
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
    # Logic: Price makes lower low but RSI makes higher low (bullish div) -> momentum weakening
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
    # Ref: Investopedia - Overbought >80, Oversold <20
    # Logic: K crosses D when LEAVING extreme zone (not entering)
    if enabled.get("stoch_cross", False):
        stoch_buy = (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) & (df["stoch_k"].shift(1) < 20)
        stoch_sell = (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) & (df["stoch_k"].shift(1) > 80)
        df.loc[stoch_buy, "buy_score"] += 1
        df.loc[stoch_sell, "sell_score"] += 1
        df.loc[stoch_buy, "_b_stoch_cross"] = 1
        df.loc[stoch_sell, "_s_stoch_cross"] = 1

    # --- 8. Bollinger Bounce ---
    # Logic: Price touches lower BB in uptrend + RSI < 35 -> oversold bounce
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
    # Bullish engulfing + RSI < 50 -> reversal from oversold
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
    # Logic: Consolidation bar (range inside previous bar) -> breakout in trend direction
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
    # Hammer: long lower shadow >=2x body, near BB lower or EMA21 + RSI < 45
    # Star: long upper shadow >=2x body, near BB upper or EMA21 + RSI > 55
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
    # Ref: Investopedia - ADX > 25 = trending market
    # Logic: +DI crosses -DI when ADX > 25 -> trend confirmed
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

    # --- 15. Smart S/R +/- ATR (Multi-Factor Breakout) ---
    # Smart version: uses Dynamic S/R (swing + EMA + volume + cluster)
    #
    # FORMULA:
    #   Resistance = dynamic_resistance (multi-method: swing fractal + EMA confluence
    #                + volume weight + recency + cluster scoring)
    #   Support    = dynamic_support (same method)
    #
    #   BUY when ALL conditions met:
    #     1. Price > Dynamic Resistance + ATRxbuffer  (breakout past real S/R zone)
    #     2. BB Squeeze prior                        (consolidation -> explosion)
    #     3. Z-score < 2.5                            (not too far from mean -> room left)
    #     4. EMA Slope > 0                            (trend accelerating)
    #     5. EMA5 > EMA12                             (short-term momentum bullish)
    #     6. BB Width expanding                       (volatility expanding = real breakout)
    #
    #   SELL when ALL conditions met (inverse)
    #
    # S/R calculated by calculate_dynamic_sr():
    #   - Swing Points (fractal 5-bar): real local highs/lows
    #   - EMA Confluence: swing near EMA5/12/21 -> bonus score
    #   - ATR Zone: cluster swing points within +/-0.5xATR into 1 zone
    #   - Volume Weight: high volume swing point -> stronger S/R (cap 2x)
    #   - Recency: recent swings weighted more (linear decay 50 bars)
    #   - Cluster: multiple swings in same zone -> score x (1 + 0.2 x count)
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
    # Logic: Do NOT BUY when MACD line < Signal line -> momentum bearish
    #        Do NOT SELL when MACD line > Signal line -> momentum bullish
    if enabled.get("macd_filter", False):
        buy_ok = df["macd_line"] >= df["macd_sig"]
        sell_ok = df["macd_line"] <= df["macd_sig"]
        df.loc[buy_ok, "_b_macd_filter"] = 1
        df.loc[sell_ok, "_s_macd_filter"] = 1

    # --- 17. Volume Color Filter (GATE) ---
    # No BUY on red candle (close < open) = selling pressure
    # No SELL on green candle (close >= open) = buying pressure
    if enabled.get("vol_color_filter", False):
        green_candle = df["close"] >= df["open"]
        red_candle = df["close"] < df["open"]
        df.loc[green_candle, "_b_vol_color_filter"] = 1
        df.loc[red_candle, "_s_vol_color_filter"] = 1

    # --- 18. Ichimoku Cloud Cross (TK Cross + Cloud Position) ---
    # BUY: Tenkan crosses above Kijun AND price is above the cloud
    # SELL: Tenkan crosses below Kijun AND price is below the cloud
    if enabled.get("ichimoku_cross", False):
        tk_cross_up = (
            (df["ichi_tenkan"] > df["ichi_kijun"]) &
            (df["ichi_tenkan"].shift(1) <= df["ichi_kijun"].shift(1))
        )
        tk_cross_down = (
            (df["ichi_tenkan"] < df["ichi_kijun"]) &
            (df["ichi_tenkan"].shift(1) >= df["ichi_kijun"].shift(1))
        )
        cloud_top = df[["ichi_span_a", "ichi_span_b"]].max(axis=1)
        cloud_bot = df[["ichi_span_a", "ichi_span_b"]].min(axis=1)
        _buy  = tk_cross_up   & (df["close"] > cloud_top)
        _sell = tk_cross_down & (df["close"] < cloud_bot)
        df.loc[_buy,  "buy_score"]  += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy,  "_b_ichimoku_cross"] = 1
        df.loc[_sell, "_s_ichimoku_cross"] = 1

    # --- 19. Supertrend Direction Flip ---
    # BUY: Supertrend flips from -1 (bearish) to +1 (bullish)
    # SELL: Supertrend flips from +1 to -1
    if enabled.get("supertrend_flip", False):
        st_flip_bull = (df["supertrend_dir"] == 1)  & (df["supertrend_dir"].shift(1) == -1)
        st_flip_bear = (df["supertrend_dir"] == -1) & (df["supertrend_dir"].shift(1) == 1)
        df.loc[st_flip_bull, "buy_score"]  += 1
        df.loc[st_flip_bear, "sell_score"] += 1
        df.loc[st_flip_bull, "_b_supertrend_flip"] = 1
        df.loc[st_flip_bear, "_s_supertrend_flip"] = 1

    # --- 20. VWAP Deviation Mean Reversion ---
    # BUY: Price >2 ATR below VWAP and starting to return
    # SELL: Price >2 ATR above VWAP and starting to return
    if enabled.get("vwap_dev", False):
        _buy  = (df["vwap_dist"] < -2.0) & (df["close"] > df["close"].shift(1))
        _sell = (df["vwap_dist"] >  2.0) & (df["close"] < df["close"].shift(1))
        df.loc[_buy,  "buy_score"]  += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy,  "_b_vwap_dev"] = 1
        df.loc[_sell, "_s_vwap_dev"] = 1

    # --- 21. Donchian Channel Breakout ---
    # BUY: close breaks above 20-period upper channel (fresh breakout)
    # SELL: close breaks below 20-period lower channel
    if enabled.get("donchian_break", False):
        _buy  = (df["close"] > df["dc_upper"]) & (df["close"].shift(1) <= df["dc_upper"].shift(1))
        _sell = (df["close"] < df["dc_lower"]) & (df["close"].shift(1) >= df["dc_lower"].shift(1))
        df.loc[_buy,  "buy_score"]  += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy,  "_b_donchian_break"] = 1
        df.loc[_sell, "_s_donchian_break"] = 1

    # --- 22. CCI Extreme Zone Exit ---
    # BUY: CCI crosses from below -100 back above -100 (exits oversold)
    # SELL: CCI crosses from above +100 back below +100 (exits overbought)
    if enabled.get("cci_extreme", False):
        _buy  = (df["cci"] > -100) & (df["cci"].shift(1) <= -100)
        _sell = (df["cci"] <  100) & (df["cci"].shift(1) >=  100)
        df.loc[_buy,  "buy_score"]  += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy,  "_b_cci_extreme"] = 1
        df.loc[_sell, "_s_cci_extreme"] = 1

    # --- 23. Parabolic SAR Flip ---
    # BUY: PSAR flips from bearish (dots above) to bullish (dots below price)
    # SELL: PSAR flips from bullish to bearish
    if enabled.get("psar_flip", False):
        _buy  = df["psar_bull"] & ~df["psar_bull_prev"]
        _sell = ~df["psar_bull"] & df["psar_bull_prev"]
        df.loc[_buy,  "buy_score"]  += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy,  "_b_psar_flip"] = 1
        df.loc[_sell, "_s_psar_flip"] = 1

    # --- 24. Williams %R Extreme Zone Exit ---
    # BUY: Williams %R crosses from oversold (<-80) back above -80
    # SELL: Williams %R crosses from overbought (>-20) back below -20
    if enabled.get("williams_extreme", False):
        _buy  = (df["willr"] > -80) & (df["willr"].shift(1) <= -80)
        _sell = (df["willr"] < -20) & (df["willr"].shift(1) >= -20)
        df.loc[_buy,  "buy_score"]  += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy,  "_b_williams_extreme"] = 1
        df.loc[_sell, "_s_williams_extreme"] = 1

    # --- 25. PVT (Price Volume Trend) Confirmation ---
    # BUY only when PVT is rising (volume confirms bullish price move)
    # SELL only when PVT is falling (volume confirms bearish price move)
    if enabled.get("pvt_confirm", False):
        _buy  = df["pvt_bull"].fillna(True)
        _sell = ~df["pvt_bull"].fillna(True)
        df.loc[_buy,  "buy_score"]  += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy,  "_b_pvt_confirm"] = 1
        df.loc[_sell, "_s_pvt_confirm"] = 1

    # --- 26. Ribbon Trend Filter (GATE) ---
    # Only BUY when EMA8 > EMA21 > EMA55 (full bullish ribbon alignment)
    # Only SELL when EMA8 < EMA21 < EMA55 (full bearish ribbon alignment)
    if enabled.get("ribbon_trend_filter", False):
        ribbon_bull_gate = (df["ema8"] > df["ema21"]) & (df["ema21"] > df["ema55"])
        ribbon_bear_gate = (df["ema8"] < df["ema21"]) & (df["ema21"] < df["ema55"])
        df.loc[ribbon_bull_gate, "_b_ribbon_trend_filter"] = 1
        df.loc[ribbon_bear_gate, "_s_ribbon_trend_filter"] = 1

    # --- 27. Session Open Filter (GATE) ---
    # Block signals in first 5 minutes after market open (9:00-9:05, 13:00-13:05)
    # These are volatile/noisy periods with unreliable signals
    if enabled.get("session_open_filter", False):
        if "time" in df.columns:
            _times = pd.to_datetime(df["time"])
            _h = _times.dt.hour
            _m = _times.dt.minute
            _session_settled = ~(
                ((_h == 9) & (_m < 5)) |
                ((_h == 13) & (_m < 5))
            )
        else:
            _session_settled = pd.Series(True, index=df.index)
        df.loc[_session_settled, "_b_session_open_filter"] = 1
        df.loc[_session_settled, "_s_session_open_filter"] = 1

    # --- 28. ADX Trend Filter (GATE) ---
    # Only allow entry when ADX > 20 (market is trending, not choppy)
    if enabled.get("adx_trend_filter", False):
        _adx_trending = df["adx"] > 20
        df.loc[_adx_trending, "_b_adx_trend_filter"] = 1
        df.loc[_adx_trending, "_s_adx_trend_filter"] = 1

    # --- 29. Market Regime Filter (GATE) ---
    # BUY only when price above EMA55 (bullish regime)
    # SELL only when price below EMA55 (bearish regime)
    if enabled.get("market_regime_filter", False):
        _above_ema55 = df["close"] > df["ema55"]
        _below_ema55 = df["close"] < df["ema55"]
        df.loc[_above_ema55, "_b_market_regime_filter"] = 1
        df.loc[_below_ema55, "_s_market_regime_filter"] = 1

    # --- 30. Ranging Filter (GATE) ---
    # Only allow entry when ADX < 25 (market is ranging, good for mean reversion)
    if enabled.get("ranging_filter", False):
        _adx_ranging = df["adx"] < 25
        df.loc[_adx_ranging, "_b_ranging_filter"] = 1
        df.loc[_adx_ranging, "_s_ranging_filter"] = 1

    # --- 31. MFI Confirmation (volume-weighted RSI) ---
    # BUY: MFI < 30 (oversold with volume confirmation)
    # SELL: MFI > 70 (overbought with volume confirmation)
    # Better than RSI alone because it weights by volume
    if enabled.get("mfi_confirm", False):
        _buy = (df["mfi"] < 30) & (df["mfi"] > df["mfi"].shift(1))  # oversold + turning up
        _sell = (df["mfi"] > 70) & (df["mfi"] < df["mfi"].shift(1))  # overbought + turning down
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_mfi_confirm"] = 1
        df.loc[_sell, "_s_mfi_confirm"] = 1

    # --- 32. OBV Trend Confirmation ---
    # BUY only when OBV > OBV_EMA (accumulation = smart money buying)
    # SELL only when OBV < OBV_EMA (distribution = smart money selling)
    if enabled.get("obv_confirm", False):
        _buy = df["obv_rising"]
        _sell = ~df["obv_rising"]
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_obv_confirm"] = 1
        df.loc[_sell, "_s_obv_confirm"] = 1

    # --- 33. Volume Ratio Gate (continuous, replaces binary vol_color_filter) ---
    # BUY/SELL only when volume_ratio >= 0.8 (minimum volume threshold)
    # Ensures we don't enter on dead/thin volume bars
    if enabled.get("volume_ratio_gate", False):
        _vol_ok = df["volume_ratio"] >= 0.8
        df.loc[_vol_ok, "_b_volume_ratio_gate"] = 1
        df.loc[_vol_ok, "_s_volume_ratio_gate"] = 1

    # --- 34. TEMA Guard (GATE) ---
    # BUY only when TEMA9 rising (short-term momentum confirming entry)
    # SELL only when TEMA9 falling
    if enabled.get("tema_guard", False):
        df.loc[df["tema_rising"], "_b_tema_guard"] = 1
        df.loc[df["tema_falling"], "_s_tema_guard"] = 1

    # --- 35. BB Percent Filter (GATE) ---
    # BUY only when bb_percent < 0.35 (price in lower zone of BB, room to rise)
    # SELL only when bb_percent > 0.65 (price in upper zone, room to fall)
    if enabled.get("bb_percent_filter", False):
        _buy_zone = df["bb_percent"] < 0.35
        _sell_zone = df["bb_percent"] > 0.65
        df.loc[_buy_zone, "_b_bb_percent_filter"] = 1
        df.loc[_sell_zone, "_s_bb_percent_filter"] = 1

    # ==================== ALPHA ZOO CONDITIONS (Vibe-Trading) ====================

    # --- 36. Momentum Ratio Gate (GTJA #53 adapted) ---
    # Up-bar ratio in last 12 bars: >55% = bullish momentum, <45% = bearish
    # Only allow BUY when majority of recent bars were up (momentum confirms)
    # Only allow SELL when majority of recent bars were down
    if enabled.get("momentum_ratio_gate", False):
        _up_bars = (df["close"] > df["close"].shift(1)).astype(float)
        _up_ratio = _up_bars.rolling(12, min_periods=12).mean()
        _bull_mom = _up_ratio > 0.55
        _bear_mom = _up_ratio < 0.45
        df.loc[_bull_mom, "_b_momentum_ratio_gate"] = 1
        df.loc[_bear_mom, "_s_momentum_ratio_gate"] = 1

    # --- 37. Volume-Price Divergence (Kakushadze #12 adapted) ---
    # sign(Δvolume) × (-1 × Δclose): positive = vol up + price down (bullish reversal)
    # BUY: vol increasing + price falling (divergence = reversal signal)
    # SELL: vol increasing + price rising (overextension)
    if enabled.get("vol_price_divergence", False):
        _dvol = df["volume"].astype(float).diff(1)
        _dclose = df["close"].diff(1)
        _vpd = np.sign(_dvol) * (-1.0 * _dclose)
        # Normalize by ATR to make threshold-free
        _vpd_norm = _vpd / df["atr"].replace(0, np.nan)
        _buy = _vpd_norm > 0.3  # vol up + price down → bullish divergence
        _sell = _vpd_norm < -0.3  # vol up + price up (or vol down + price down) → bearish
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_vol_price_divergence"] = 1
        df.loc[_sell, "_s_vol_price_divergence"] = 1

    # --- 38. Midpoint Momentum (GTJA #9 adapted) ---
    # EWM of (midpoint_change × range / volume): quality momentum
    # High positive = strong bullish with range expansion & volume
    # Uses ewm(alpha=2/7) ≈ 7-bar half-life
    if enabled.get("midpoint_momentum", False):
        _mid = (df["high"] + df["low"]) / 2.0
        _pmid = (df["high"].shift(1) + df["low"].shift(1)) / 2.0
        _range = df["high"] - df["low"]
        _vol_safe = df["volume"].astype(float).replace(0, np.nan)
        _raw = (_mid - _pmid) * _range / _vol_safe
        _midmom = _raw.ewm(alpha=2.0 / 7.0, adjust=False).mean()
        # Normalize: compare to rolling std for adaptive threshold
        _midmom_std = _midmom.rolling(20, min_periods=8).std()
        _midmom_z = _midmom / _midmom_std.replace(0, np.nan)
        _buy = _midmom_z > 1.0  # strong bullish midpoint momentum
        _sell = _midmom_z < -1.0  # strong bearish midpoint momentum
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_midpoint_momentum"] = 1
        df.loc[_sell, "_s_midpoint_momentum"] = 1

    # --- 39. Deviation Acceleration (GTJA #22 adapted) ---
    # EWM of 3-bar-change in (close - MA6)/MA6: how fast price deviates from mean
    # Positive acceleration from below = reversal starting (good for mean-reversion entry)
    # Used as confirmation for C/K combos
    if enabled.get("deviation_accel", False):
        _ma6 = df["close"].rolling(6, min_periods=6).mean()
        _z_dev = (df["close"] - _ma6) / _ma6.replace(0, np.nan)
        _z_diff = _z_dev - _z_dev.shift(3)
        _accel = _z_diff.ewm(alpha=1.0 / 12.0, adjust=False).mean()
        # BUY: acceleration turning positive (from below, mean reverting up)
        # SELL: acceleration turning negative (from above, mean reverting down)
        _buy = (_accel > 0) & (_accel.shift(1) <= 0)  # cross above zero
        _sell = (_accel < 0) & (_accel.shift(1) >= 0)  # cross below zero
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_deviation_accel"] = 1
        df.loc[_sell, "_s_deviation_accel"] = 1

    # --- 40. Volume-Return Correlation Gate (GTJA #1 adapted) ---
    # rolling_corr(Δlog(volume), intraday_return, 6): when >0, moves are vol-confirmed
    # GATE: only trade when recent volume changes correlate with price moves
    if enabled.get("vol_return_corr_gate", False):
        _log_vol = np.log(df["volume"].astype(float).replace(0, np.nan))
        _dvol_log = _log_vol.diff(1)
        _intra_ret = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
        _vr_corr = _dvol_log.rolling(6, min_periods=6).corr(_intra_ret)
        # When correlation > 0: volume confirms price moves (reliable signal env)
        _confirmed = _vr_corr > 0
        df.loc[_confirmed, "_b_vol_return_corr_gate"] = 1
        df.loc[_confirmed, "_s_vol_return_corr_gate"] = 1

    # ==================== SUPERALGOS-INSPIRED CONDITIONS ====================

    # --- 41. Keltner-Bollinger Squeeze Gate (from Superalgos Keltner_Bollinger_Strategy) ---
    # GATE: Only allow entry when squeeze is ON or just released (volatility expansion imminent)
    # Squeeze ON = BB inside Keltner = compressed volatility → breakout coming
    # Squeeze OFF (release) = the actual breakout moment
    if enabled.get("kb_squeeze", False):
        _squeeze_active = df["kb_squeeze_on"] | df["kb_squeeze_off"]
        df.loc[_squeeze_active, "_b_kb_squeeze"] = 1
        df.loc[_squeeze_active, "_s_kb_squeeze"] = 1

    # --- 42. Normalized RSI (Z-score dynamic bands, from Superalgos Normalized_Momentum) ---
    # BUY: Norm RSI Z crosses back ABOVE lower band (leaving oversold zone = reversal)
    # SELL: Norm RSI Z crosses back BELOW upper band (leaving overbought zone = reversal)
    # Better than fixed 70/30 because thresholds adapt to current regime
    if enabled.get("norm_rsi", False):
        _was_os = df["norm_rsi_z"].shift(1) < df["norm_rsi_lower"].shift(1)
        _now_above = df["norm_rsi_z"] >= df["norm_rsi_lower"]
        _was_ob = df["norm_rsi_z"].shift(1) > df["norm_rsi_upper"].shift(1)
        _now_below = df["norm_rsi_z"] <= df["norm_rsi_upper"]
        _buy = _was_os & _now_above  # leaving oversold
        _sell = _was_ob & _now_below  # leaving overbought
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_norm_rsi"] = 1
        df.loc[_sell, "_s_norm_rsi"] = 1

    # --- 43. Adaptive RSI (DFT dominant cycle, from Superalgos Auto-Adaptive RSI) ---
    # BUY: Adaptive RSI Z < -1.5 (deep oversold relative to its own adaptive mean)
    # SELL: Adaptive RSI Z > 1.5 (deep overbought relative to its own adaptive mean)
    # RSI period auto-adjusts to market rhythm → fewer false signals
    if enabled.get("adaptive_rsi", False):
        _buy = (df["adaptive_rsi_z"] < -1.5) & (df["adaptive_rsi_z"].shift(1) >= -1.5)
        _sell = (df["adaptive_rsi_z"] > 1.5) & (df["adaptive_rsi_z"].shift(1) <= 1.5)
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_adaptive_rsi"] = 1
        df.loc[_sell, "_s_adaptive_rsi"] = 1

    # --- 44. Center of Gravity Reversal (Ehlers, from Superalgos Polus mine) ---
    # CoG is a LEADING indicator - crosses signal line before price turns
    # BUY: CoG crosses above signal line (momentum shifting up)
    # SELL: CoG crosses below signal line (momentum shifting down)
    if enabled.get("cog_reversal", False):
        _cog_cross_up = (df["cog"] > df["cog_signal"]) & (df["cog"].shift(1) <= df["cog_signal"].shift(1))
        _cog_cross_dn = (df["cog"] < df["cog_signal"]) & (df["cog"].shift(1) >= df["cog_signal"].shift(1))
        df.loc[_cog_cross_up, "buy_score"] += 1
        df.loc[_cog_cross_dn, "sell_score"] += 1
        df.loc[_cog_cross_up, "_b_cog_reversal"] = 1
        df.loc[_cog_cross_dn, "_s_cog_reversal"] = 1

    # --- 45. Trend Composite (LinReg + R² + Aroon, from Superalgos Masters/Zeus) ---
    # BUY: Trend composite > 1.0 (clear bullish trend with high quality)
    # SELL: Trend composite < -1.0 (clear bearish trend with high quality)
    # Combines direction (slope), quality (R²), and maturity (Aroon)
    if enabled.get("trend_composite", False):
        _buy = (df["trend_composite"] > 1.0) & (df["trend_composite"].shift(1) <= 1.0)
        _sell = (df["trend_composite"] < -1.0) & (df["trend_composite"].shift(1) >= -1.0)
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_trend_composite"] = 1
        df.loc[_sell, "_s_trend_composite"] = 1

    # --- 46. Trend Strength Gate (GATE, from Superalgos trend systems) ---
    # Only BUY in confirmed uptrend: trend_composite > 0.5 AND R² > 0.3
    # Only SELL in confirmed downtrend: trend_composite < -0.5 AND R² > 0.3
    # R² threshold ensures trend is clean (not choppy)
    if enabled.get("trend_strength_gate", False):
        _strong_up = (df["trend_composite"] > 0.5) & (df["linreg_r2"] > 0.3)
        _strong_dn = (df["trend_composite"] < -0.5) & (df["linreg_r2"] > 0.3)
        df.loc[_strong_up, "_b_trend_strength_gate"] = 1
        df.loc[_strong_dn, "_s_trend_strength_gate"] = 1

    # --- 47. Regime EMA Cross (PRIMARY, from ComboH crypto) ---
    # TRENDING mode (ADX > 20): EMA13 crosses EMA34 + either MACD or DI confirms
    # Relaxed from crypto (was ADX>25 + both MACD + DI)
    if enabled.get("regime_ema_cross", False):
        _adx_trend = df["adx"] > 20
        _ema13 = ta.ema(df["close"], length=13)
        _ema34 = ta.ema(df["close"], length=34)
        _cross_up = (_ema13 > _ema34) & (_ema13.shift(1) <= _ema34.shift(1))
        _cross_dn = (_ema13 < _ema34) & (_ema13.shift(1) >= _ema34.shift(1))
        # Allow cross within 2 bars (momentum still valid)
        _recent_cross_up = _cross_up | _cross_up.shift(1).fillna(False) | _cross_up.shift(2).fillna(False)
        _recent_cross_dn = _cross_dn | _cross_dn.shift(1).fillna(False) | _cross_dn.shift(2).fillna(False)
        _macd_up = df["macd_hist"] > 0
        _macd_dn = df["macd_hist"] < 0
        _di_up = df["plus_di"] > df["minus_di"]
        _di_dn = df["minus_di"] > df["plus_di"]
        _buy = _adx_trend & _recent_cross_up & (_macd_up | _di_up)
        _sell = _adx_trend & _recent_cross_dn & (_macd_dn | _di_dn)
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_regime_ema_cross"] = 1
        df.loc[_sell, "_s_regime_ema_cross"] = 1

    # --- 48. Regime Range Bounce (PRIMARY, from ComboH crypto) ---
    # RANGING mode (ADX <= 25): RSI turning from extreme + near BB band + OBV confirm
    # Relaxed: removed candle color requirement, widened BB/RSI thresholds
    if enabled.get("regime_range_bounce", False):
        _adx_range = df["adx"] <= 25
        _rsi_os = (df["rsi"] < 40) & (df["rsi"] > df["rsi"].shift(1))
        _rsi_ob = (df["rsi"] > 60) & (df["rsi"] < df["rsi"].shift(1))
        _bb_low = df["close"] < df["bb_lower"] * 1.01
        _bb_high = df["close"] > df["bb_upper"] * 0.99
        _obv_up = df["obv"] > df["obv_ema"]
        _obv_dn = df["obv"] < df["obv_ema"]
        _buy = _adx_range & _rsi_os & _bb_low & _obv_up
        _sell = _adx_range & _rsi_ob & _bb_high & _obv_dn
        df.loc[_buy, "buy_score"] += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy, "_b_regime_range_bounce"] = 1
        df.loc[_sell, "_s_regime_range_bounce"] = 1

    # --- 49. EMA200 Bias Gate (GATE, from ComboH crypto) ---
    # Only BUY when price > EMA200 (long-term bullish)
    # Only SELL when price < EMA200 (long-term bearish)
    if enabled.get("ema200_bias", False):
        if "ema200" in df.columns and df["ema200"].notna().any():
            _above = df["close"] > df["ema200"]
            _below = df["close"] < df["ema200"]
            df.loc[_above, "_b_ema200_bias"] = 1
            df.loc[_below, "_s_ema200_bias"] = 1
        else:
            # If not enough data for EMA200, allow all
            df["_b_ema200_bias"] = 1
            df["_s_ema200_bias"] = 1

    # --- 50. Fast EMA Cross (PRIMARY, 1m scalp) ---
    # EMA3 crosses EMA8 — ultra-fast trend signal for 1m scalping.
    # Allow cross within 2 bars (1-2 min lag tolerance).
    # Requires MACD hist or DI alignment to filter noise.
    if enabled.get("fast_ema_cross", False):
        _ema3 = df["ema3"] if "ema3" in df.columns else ta.ema(df["close"], length=3)
        _ema8 = df["ema8"] if "ema8" in df.columns else ta.ema(df["close"], length=8)
        _cross_up = (_ema3 > _ema8) & (_ema3.shift(1) <= _ema8.shift(1))
        _cross_dn = (_ema3 < _ema8) & (_ema3.shift(1) >= _ema8.shift(1))
        _recent_up = _cross_up | _cross_up.shift(1).fillna(False)
        _recent_dn = _cross_dn | _cross_dn.shift(1).fillna(False)
        # Require momentum confirmation (MACD hist OR DI direction)
        _mom_up = (df["macd_hist"] > 0) | (df["plus_di"] > df["minus_di"])
        _mom_dn = (df["macd_hist"] < 0) | (df["minus_di"] > df["plus_di"])
        _buy  = _recent_up & _mom_up
        _sell = _recent_dn & _mom_dn
        df.loc[_buy,  "buy_score"]  += 1
        df.loc[_sell, "sell_score"] += 1
        df.loc[_buy,  "_b_fast_ema_cross"] = 1
        df.loc[_sell, "_s_fast_ema_cross"] = 1

    # --- 51. Candle Momentum (PRIMARY, 1m scalp) ---
    # Strong directional candle: body >= 60% of range AND close in top/bottom 30%.
    # Catches momentum surges (market orders dominating).
    if enabled.get("candle_momentum", False):
        _body_cm  = df["close"] - df["open"]
        _range_cm = (df["high"] - df["low"]).replace(0, np.nan)
        _body_ratio = _body_cm.abs() / _range_cm
        # Bull: green body >= 60% range, close in top 30%
        _bull_cm = (
            (_body_cm > 0) &
            (_body_ratio >= 0.60) &
            ((df["close"] - df["low"]) / _range_cm >= 0.70)
        )
        # Bear: red body >= 60% range, close in bottom 30%
        _bear_cm = (
            (_body_cm < 0) &
            (_body_ratio >= 0.60) &
            ((df["high"] - df["close"]) / _range_cm >= 0.70)
        )
        df.loc[_bull_cm, "buy_score"]  += 1
        df.loc[_bear_cm, "sell_score"] += 1
        df.loc[_bull_cm, "_b_candle_momentum"] = 1
        df.loc[_bear_cm, "_s_candle_momentum"] = 1

    # --- 52. ORB Breakout (PRIMARY, 1m scalp) ---
    # Open Range Breakout: price breaks the first 15-min session high/low.
    # Morning ORB: 09:00-09:14 VN time. Afternoon ORB: 13:00-13:09.
    # Signal: first close above/below ORB after formation.
    if enabled.get("orb_breakout", False):
        if "orb_ref_high" in df.columns and df["orb_ref_high"].notna().any():
            _orb_h = df["orb_ref_high"]
            _orb_l = df["orb_ref_low"]
            _after = df["_after_orb"].astype(bool)
            # First bar to break ORB (not just above it from before)
            _break_up = _after & (df["close"] > _orb_h) & (df["close"].shift(1) <= _orb_h)
            # Allow 1-bar lag
            _break_up = _break_up | (_after & _break_up.shift(1).fillna(False) & (df["close"] > _orb_h))
            _break_dn = _after & (df["close"] < _orb_l) & (df["close"].shift(1) >= _orb_l)
            _break_dn = _break_dn | (_after & _break_dn.shift(1).fillna(False) & (df["close"] < _orb_l))
            df.loc[_break_up,  "buy_score"]  += 1
            df.loc[_break_dn, "sell_score"] += 1
            df.loc[_break_up,  "_b_orb_breakout"] = 1
            df.loc[_break_dn, "_s_orb_breakout"] = 1
        # If no time column: skip silently (ORB not available)

    # --- 53. HTF Trend Filter (GATE, 1m scalp) ---
    # Only allow longs when 5m EMA9 > EMA21, shorts when 5m EMA9 < EMA21.
    # #1 pattern from GitHub community strategies (TrendRider, MoniGoMani):
    # higher timeframe alignment is the single biggest WR improvement for 1m.
    if enabled.get("htf_trend_filter", False):
        _htf_bull = df["htf_bull"].astype(bool)
        _htf_bear = df["htf_bear"].astype(bool)
        df.loc[_htf_bull, "buy_score"]  += 1
        df.loc[_htf_bear, "sell_score"] += 1
        df.loc[_htf_bull, "_b_htf_trend_filter"] = 1
        df.loc[_htf_bear, "_s_htf_trend_filter"] = 1

    # --- 54. VWAP Pullback (PRIMARY, 1m scalp) ---
    # Price came from extended VWAP deviation, pulled back to VWAP touch zone,
    # and now bouncing. Far cleaner than raw vwap_dev.
    # Inspired by: VWAP-based strategies widely used in 1m equity/futures scalping.
    # Long:  was > 0.5*ATR above daily VWAP, returns within 0.4*ATR, bounce candle
    # Short: was < -0.5*ATR below daily VWAP, returns within 0.4*ATR, bounce candle
    # Uses vwap_d (daily-anchored VWAP) for intraday distance accuracy.
    if enabled.get("vwap_pullback", False):
        _vwap_col = "vwap_d" if "vwap_d" in df.columns else "vwap"
        _vwap  = df[_vwap_col]
        _atr   = df["atr"]
        _dist  = df["close"] - _vwap          # signed distance
        # "At VWAP": tight threshold for precision
        _at_vwap = _dist.abs() <= 0.4 * _atr
        # Was extended 3 bars ago: rolling mean > 0.5 ATR from VWAP
        _was_above = _dist.rolling(3).mean().shift(1) > 0.5 * _atr
        _was_below = _dist.rolling(3).mean().shift(1) < -0.5 * _atr
        # Bounce candle direction
        _bull_candle = df["close"] > df["open"]
        _bear_candle = df["close"] < df["open"]
        _vpb_long  = _at_vwap & _was_above & _bull_candle
        _vpb_short = _at_vwap & _was_below & _bear_candle
        df.loc[_vpb_long,  "buy_score"]  += 2   # Primary: +2
        df.loc[_vpb_short, "sell_score"] += 2
        df.loc[_vpb_long,  "_b_vwap_pullback"] = 1
        df.loc[_vpb_short, "_s_vwap_pullback"] = 1

    # --- 55. Momentum Burst (PRIMARY, 1m scalp) ---
    # Micro-structure breakout: close breaks the 5-bar high/low with volume surge.
    # Based on: Opening momentum / liquidity sweep patterns common in 1m futures.
    # Long:  close > max(high[-5:-1]) with volume > 1.4x average
    # Short: close < min(low[-5:-1]) with volume > 1.4x average
    if enabled.get("momentum_burst", False):
        _5bar_high = df["high"].shift(1).rolling(5).max()
        _5bar_low  = df["low"].shift(1).rolling(5).min()
        _vol_surge = df["volume_ratio"] > 1.4
        _mb_long   = (df["close"] > _5bar_high) & _vol_surge & (df["close"] > df["open"])
        _mb_short  = (df["close"] < _5bar_low)  & _vol_surge & (df["close"] < df["open"])
        df.loc[_mb_long,  "buy_score"]  += 2
        df.loc[_mb_short, "sell_score"] += 2
        df.loc[_mb_long,  "_b_momentum_burst"] = 1
        df.loc[_mb_short, "_s_momentum_burst"] = 1

    # --- 56. VWAP Side Bounce (PRIMARY, Big-Move Catcher) ---
    # Data-driven condition based on 120-day analysis of VN30F1M 3m data (2553 bars):
    # 74% of UP big-moves (5+pts) start when:
    #   (a) price is BELOW daily VWAP  (the cheap side relative to today's mean)
    #   (b) 3-bar momentum is negative (mini pullback happened)
    #   (c) current candle is bullish  (bounce starting)
    #   (d) at least one oscillator in extreme zone (CCI<-80 OR WR<-80 OR BB%<0.35)
    #       → adds precision; CCI/WR were #1 correlators with move size in analysis
    # Symmetric inverse for DOWN big-moves.
    # Uses vwap_d (daily-anchored VWAP) for accuracy.
    if enabled.get("vwap_side_bounce", False):
        _vwap_col = "vwap_d" if "vwap_d" in df.columns else "vwap"
        _vwap  = df[_vwap_col]
        _mom3  = df["close"].pct_change(3) * 100          # 3-bar return (%)
        _bull_candle = df["close"] > df["open"]
        _bear_candle = df["close"] < df["open"]
        _c_rising    = df["close"] > df["close"].shift(1)
        _c_falling   = df["close"] < df["close"].shift(1)

        # Oscillator extreme detection (re-use pre-computed indicator columns if available)
        _cci_col  = "cci"   if "cci"   in df.columns else None
        _willr_col= "willr" if "willr" in df.columns else None
        _bb_pct_c = "bb_percent" if "bb_percent" in df.columns else None

        _osc_bull = pd.Series(False, index=df.index)
        _osc_bear = pd.Series(False, index=df.index)
        if _cci_col:
            _osc_bull |= (df[_cci_col] < -80)
            _osc_bear |= (df[_cci_col] >  80)
        if _willr_col:
            _osc_bull |= (df[_willr_col] < -80)
            _osc_bear |= (df[_willr_col] > -20)
        if _bb_pct_c:
            _osc_bull |= (df[_bb_pct_c] < 0.35)
            _osc_bear |= (df[_bb_pct_c] > 0.65)
        # Fallback if none computed yet: use RSI
        if not _cci_col and not _willr_col and not _bb_pct_c:
            _osc_bull |= (df["rsi"] < 40)
            _osc_bear |= (df["rsi"] > 60)

        # Long: below VWAP + 3-bar pullback + bullish bounce + at least 1 oscillator extreme
        _vsb_long  = (
            (df["close"] < _vwap) &
            (_mom3 < -0.05) &
            _bull_candle &
            _c_rising &
            _osc_bull
        )
        # Short: above VWAP + 3-bar bounce + bearish drop + at least 1 oscillator extreme
        _vsb_short = (
            (df["close"] > _vwap) &
            (_mom3 > 0.05) &
            _bear_candle &
            _c_falling &
            _osc_bear
        )
        df.loc[_vsb_long,  "buy_score"]  += 2   # Primary: +2
        df.loc[_vsb_short, "sell_score"] += 2
        df.loc[_vsb_long,  "_b_vwap_side_bounce"] = 1
        df.loc[_vsb_short, "_s_vwap_side_bounce"] = 1

    # --- 57. Bounce Confirm (V2 — Reversal Confirmation Entry) ---
    # Fix cho V: thay vì vào TRONG KHI giá đang rơi, chờ NẾN XÁC NHẬN đảo chiều đầu tiên.
    # Data-driven (từ analyze_big_moves.py):
    #   - 74% UP moves: prior pullback (mom5<0) + oscillator extreme trong 5 nến gần nhất
    #   - 66% UP moves: dưới VWAP ngày
    #   - ADX>20: 69% của UP moves
    # Entry khi:
    #   [Context]  Rolling 5-bar min CCI<-80 OR WR<-80 OR BB%<0.2 (extreme XẢY RA gần đây)
    #   [Context]  mom5 < -0.10% (giá đã kéo xuống đủ)
    #   [Context]  ADX > 20 (trending market)
    #   [Confirm]  close > open AND close > prev_close (nến bullish + đang tăng lại)
    # → Điểm vào chậm hơn V 1-2 nến nhưng reversal đã được xác nhận.
    if enabled.get("bounce_confirm", False):
        _vwap_bc = df["vwap_d"] if "vwap_d" in df.columns else df.get("vwap", df["close"])
        _mom5_bc = df["close"].pct_change(5) * 100

        # Reversal candle check
        _bc_bull = (df["close"] > df["open"]) & (df["close"] > df["close"].shift(1))
        _bc_bear = (df["close"] < df["open"]) & (df["close"] < df["close"].shift(1))

        # ADX filter
        _bc_adx = df["adx"] > 20 if "adx" in df.columns else pd.Series(True, index=df.index)

        # Rolling 5-bar oscillator extreme (extreme happened recently, not necessarily now)
        _null = pd.Series(False, index=df.index)
        _cci_lo5  = (df["cci"].rolling(5, min_periods=1).min() < -80)   if "cci"        in df.columns else _null
        _wr_lo5   = (df["willr"].rolling(5, min_periods=1).min() < -80) if "willr"      in df.columns else _null
        _bb_lo5   = (df["bb_percent"].rolling(5, min_periods=1).min() < 0.2) if "bb_percent" in df.columns else _null
        _cci_hi5  = (df["cci"].rolling(5, min_periods=1).max() > 80)    if "cci"        in df.columns else _null
        _wr_hi5   = (df["willr"].rolling(5, min_periods=1).max() > -20) if "willr"      in df.columns else _null
        _bb_hi5   = (df["bb_percent"].rolling(5, min_periods=1).max() > 0.8) if "bb_percent" in df.columns else _null

        _recent_bull_osc = _cci_lo5 | _wr_lo5 | _bb_lo5
        _recent_bear_osc = _cci_hi5 | _wr_hi5 | _bb_hi5

        _bc_long = (
            (df["close"] < _vwap_bc) &
            (_mom5_bc < -0.10) &
            _bc_adx &
            _recent_bull_osc &
            _bc_bull
        )
        _bc_short = (
            (df["close"] > _vwap_bc) &
            (_mom5_bc >  0.10) &
            _bc_adx &
            _recent_bear_osc &
            _bc_bear
        )
        df.loc[_bc_long,  "buy_score"]  += 2
        df.loc[_bc_short, "sell_score"] += 2
        df.loc[_bc_long,  "_b_bounce_confirm"] = 1
        df.loc[_bc_short, "_s_bounce_confirm"] = 1

    # --- 59. W Regime Gate (GATE for W: EMA21 Bounce) ---
    # Allow W signals only when ATR >= 3.5 (sufficient room for SL=1xATR).
    # KEY insight: W fails in thin-bar months (Apr-26 ATR=2.95, May-26 ATR=3.02, Feb-26 ATR=2.82)
    #   because SL=1.0*ATR is too tight — noise clips it before the trade works.
    # ATR filter: saves Apr(-22.7), May(-1.2), Feb(-5.2). Oct+Dec+Jan+Mar remain.
    if enabled.get("w_regime_gate", False):
        _atr_ok_w = df["atr"] >= 3.5
        df.loc[_atr_ok_w, "_b_w_regime_gate"] = 1
        df.loc[_atr_ok_w, "_s_w_regime_gate"] = 1

    # --- 60. G+ Regime Gate (GATE for G+: Enhanced Multi-Oscillator) ---
    # Allow G+ signals only when ATR in [3.5, 7.0] AND ADX >= 18.
    # Rationale: TP=5xATR needs ATR>=3.5 to be reachable (TP~18pts) within session.
    #   ATR>7 = chaotic/news-driven → filter out. ADX<18 = choppy, no momentum → skip.
    # Analysis: G+ wins Mar-26 (ATR=4.62, ADX=25.2) and Nov-25 (ATR=3.88, ADX=19.6).
    #   Fails Oct-25 (ADX=15.2), Apr/May-26 (ATR<3.5).
    if enabled.get("g_regime_gate", False):
        _atr_ok_g = (df["atr"] >= 3.5) & (df["atr"] <= 7.0)
        _adx_ok_g = df["adx"] >= 18
        _g_regime = _atr_ok_g & _adx_ok_g
        df.loc[_g_regime, "_b_g_regime_gate"] = 1
        df.loc[_g_regime, "_s_g_regime_gate"] = 1

    # --- 61. Trend Breakout Regime Gate (GATE for B: Trend Breakout) ---
    # BUY gate: ADX >= 22 + ATR >= 3.0 + EMA21 > EMA55 (medium-term bullish context)
    # SELL gate: ADX >= 22 + ATR >= 3.0 + EMA21 < EMA55 (medium-term bearish context)
    # EMA alignment prevents trading Supertrend COUNTER-TREND flips (false signals).
    # Directional separation: only ride flips that agree with medium-term EMA direction.
    if enabled.get("b_regime_gate", False):
        _adx_ok_b  = df["adx"] >= 22
        _atr_ok_b  = df["atr"] >= 3.0
        _base_b    = _adx_ok_b & _atr_ok_b
        _ema_bull  = df["ema21"] > df["ema55"]   # medium-term bull
        _ema_bear  = df["ema21"] < df["ema55"]   # medium-term bear
        df.loc[_base_b & _ema_bull, "_b_b_regime_gate"] = 1  # BUY only in bull EMA context
        df.loc[_base_b & _ema_bear, "_s_b_regime_gate"] = 1  # SELL only in bear EMA context

    # --- 62. Trend Momentum Re-entry (PRIMARY for B: Trend Breakout) ---
    # Data-driven: catches trend RESUMPTION after pullback.
    # In a bull regime (EMA21>EMA55 + ST=1 + ADX>22):
    #   price dips to/below EMA21 → then CROSSES BACK ABOVE → trend continuation confirmed.
    # In a bear regime (EMA21<EMA55 + ST=-1 + ADX>22):
    #   price rallies to/above EMA21 → then CROSSES BACK BELOW → bear continuation confirmed.
    # Stronger than ema21_pullback: requires confirmed cross, not just touch.
    # Analysis target: catch Dec-25 UP (+28.4 ADX), Mar-26 DOWN (-25.2 ADX), Apr-26 UP (+23.2).
    if enabled.get("trend_momentum_entry", False):
        _ema21_b = df["ema21"]
        _ema55_b = df["ema55"]
        _st_b    = df["supertrend_dir"]
        _adx_b   = df["adx"]
        # Bull regime conditions
        _bull_r = (_ema21_b > _ema55_b) & (_st_b == 1) & (_adx_b > 22)
        # Bear regime conditions
        _bear_r = (_ema21_b < _ema55_b) & (_st_b == -1) & (_adx_b > 22)
        # Cross above EMA21 (price was below, now above = pullback ends)
        _cross_up_b = (df["close"] > _ema21_b) & (df["close"].shift(1) <= _ema21_b.shift(1))
        # Cross below EMA21 (price was above, now below = rally ends)
        _cross_dn_b = (df["close"] < _ema21_b) & (df["close"].shift(1) >= _ema21_b.shift(1))
        # Confirm there was a genuine pullback (price was below EMA21 in last 5 bars)
        _was_below_b = (df["close"].rolling(5, min_periods=3).min() < _ema21_b.shift(1))
        # Confirm there was a genuine rally (price was above EMA21 in last 5 bars)
        _was_above_b = (df["close"].rolling(5, min_periods=3).max() > _ema21_b.shift(1))
        _tme_long  = _bull_r & _cross_up_b & _was_below_b
        _tme_short = _bear_r & _cross_dn_b & _was_above_b
        df.loc[_tme_long,  "buy_score"]  += 2
        df.loc[_tme_short, "sell_score"] += 2
        df.loc[_tme_long,  "_b_trend_momentum_entry"] = 1
        df.loc[_tme_short, "_s_trend_momentum_entry"] = 1

    # --- 58. EMA21 Pullback Bounce (PRIMARY) ---
    # Data-driven: analysis on 827 swing moves >=5pts (5m, 216d):
    #   "touch_ema21 + ADX>20 + Supertrend_bull" has LIFT=1.34x (19.3pts avg vs 14.4 base)
    #   EMA21 acts as dynamic support: price dips into EMA21 zone in an uptrend = bounce likely
    # BUY: abs(close-EMA21) <= 1.2*ATR + Supertrend bullish + ADX>20 + prior 3-bar pullback
    # SELL: abs(close-EMA21) <= 1.2*ATR + Supertrend bearish + ADX>20 + prior 3-bar rally
    # Optimal ATR params from analysis: SL=1.2x ATR, TP=4.0x ATR (p75 = 4.3x ATR move)
    if enabled.get("ema21_pullback", False):
        _atr_w = df["atr"]
        _e21_dist = (df["close"] - df["ema21"]).abs() / _atr_w.replace(0, np.nan)
        _touch_e21   = _e21_dist <= 1.2
        _st_w_bull   = df["supertrend_dir"] == 1
        _st_w_bear   = df["supertrend_dir"] == -1
        _adx_w_ok    = df["adx"] > 20
        _mom3_w      = df["close"].pct_change(3)
        _pullback_w  = _mom3_w < -0.001   # prior 3-bar decline
        _rally_w     = _mom3_w > 0.001    # prior 3-bar rise
        _e21_long    = _touch_e21 & _st_w_bull & _adx_w_ok & _pullback_w
        _e21_short   = _touch_e21 & _st_w_bear & _adx_w_ok & _rally_w
        df.loc[_e21_long,  "buy_score"]  += 2
        df.loc[_e21_short, "sell_score"] += 2
        df.loc[_e21_long,  "_b_ema21_pullback"] = 1
        df.loc[_e21_short, "_s_ema21_pullback"] = 1

    # --- 63. AM Mid Entry (PRIMARY for P: AM Mid Trend) ---
    # H2: After opening noise (9:00-9:30), the intraday trend direction becomes clear.
    # Window 9:30-10:15 VN time: 25% of all 5pt+ intraday swings start here.
    # BUY: EMA21 rising slope + close > EMA21 + MACD hist > 0 and trending up + ADX >= 18
    # SELL: EMA21 falling slope + close < EMA21 + MACD hist < 0 and trending down + ADX >= 18
    if enabled.get("am_mid_entry", False):
        if "time" in df.columns:
            _t_am = pd.to_datetime(df["time"])
            _hh_am = _t_am.dt.hour
            _mm_am = _t_am.dt.minute
            _min_of_day = _hh_am * 60 + _mm_am
            _am_mid_window = (_min_of_day >= 570) & (_min_of_day <= 615)  # 09:30-10:15
        else:
            _am_mid_window = pd.Series(True, index=df.index)
        _ema21_rising   = df["ema21"] > df["ema21"].shift(4)
        _ema21_falling  = df["ema21"] < df["ema21"].shift(4)
        _macd_bull_am   = df["macd_hist"] > 0
        _macd_bear_am   = df["macd_hist"] < 0
        _macd_rising_am = df["macd_hist"] > df["macd_hist"].shift(2)
        _macd_falling_am= df["macd_hist"] < df["macd_hist"].shift(2)
        _adx_am_ok      = df["adx"] >= 18
        _above_e21_am   = df["close"] > df["ema21"]
        _below_e21_am   = df["close"] < df["ema21"]
        _am_long  = (_am_mid_window & _ema21_rising  & _macd_bull_am
                     & _adx_am_ok & _above_e21_am)
        _am_short = (_am_mid_window & _ema21_falling & _macd_bear_am
                     & _adx_am_ok & _below_e21_am)
        df.loc[_am_long,  "buy_score"]  += 2
        df.loc[_am_short, "sell_score"] += 2
        df.loc[_am_long,  "_b_am_mid_entry"] = 1
        df.loc[_am_short, "_s_am_mid_entry"] = 1

    # --- 64. Strong Trend Gate (GATE for Q: Strong Momentum) ---
    # H3: ADX >= 25 AND ATR >= 4.0 — strong trending + volatile session.
    # Targets 45%+ of big moves that occur in high-momentum months (Mar-26, Nov-25).
    # Directional: allows both LONG and SHORT when regime is strong.
    if enabled.get("strong_trend_gate", False):
        _sq_adx = df["adx"] >= 25
        _sq_atr = df["atr"] >= 4.0
        _sq_ok  = _sq_adx & _sq_atr
        df.loc[_sq_ok, "_b_strong_trend_gate"] = 1
        df.loc[_sq_ok, "_s_strong_trend_gate"] = 1

    # --- 65. Strong Trend Entry (PRIMARY for Q: Strong Momentum) ---
    # H3: Supertrend aligned + DI direction + EMA55 medium-term bias.
    # BUY: supertrend=1 + DI+ > DI- + EMA21 > EMA55 + MACD hist > 0 + ADX>=22 + ATR>=3.5
    # SELL: supertrend=-1 + DI- > DI+ + EMA21 < EMA55 + MACD hist < 0 + ADX>=22 + ATR>=3.5
    # Catches big trending moves: ADX>30+ATR>5 avg 18pts/swing (from zigzag analysis).
    if enabled.get("strong_trend_entry", False):
        _ste_adx     = df["adx"] >= 22
        _ste_atr     = df["atr"] >= 3.5
        _ste_base    = _ste_adx & _ste_atr
        _ste_st_bull = df["supertrend_dir"] == 1
        _ste_st_bear = df["supertrend_dir"] == -1
        _ste_ema_bull= df["ema21"] > df["ema55"]
        _ste_ema_bear= df["ema21"] < df["ema55"]
        _ste_di_bull = df["plus_di"] > df["minus_di"]
        _ste_di_bear = df["minus_di"] > df["plus_di"]
        _ste_mh_bull = df["macd_hist"] > 0
        _ste_mh_bear = df["macd_hist"] < 0
        _ste_long  = _ste_base & _ste_st_bull & _ste_ema_bull & _ste_di_bull & _ste_mh_bull
        _ste_short = _ste_base & _ste_st_bear & _ste_ema_bear & _ste_di_bear & _ste_mh_bear
        df.loc[_ste_long,  "buy_score"]  += 2
        df.loc[_ste_short, "sell_score"] += 2
        df.loc[_ste_long,  "_b_strong_trend_entry"] = 1
        df.loc[_ste_short, "_s_strong_trend_entry"] = 1

    # --- 66. RSI Momentum Continuation (PRIMARY for R: RSI Momentum) ---
    # H5: Counter-intuitive — RSI extreme IN THE DIRECTION OF TREND = momentum continuation.
    # SHORT when RSI < 40 in downtrend (sellers are dominant, not oversold reversal).
    # LONG when RSI > 60 in uptrend (buyers are dominant, not overbought reversal).
    # Data: SHORT@RSI<30 in downtrend avg 16.3pts (best class from 199 swings analysis).
    # Gate: b_regime_gate handles regime (ADX>=22 + ATR>=3.0).
    if enabled.get("rsi_momentum_entry", False):
        _rsi_rm      = df["rsi"]
        _rm_ema_bull = df["ema21"] > df["ema55"]
        _rm_ema_bear = df["ema21"] < df["ema55"]
        _rm_st_bull  = df["supertrend_dir"] == 1
        _rm_st_bear  = df["supertrend_dir"] == -1
        _rm_adx      = df["adx"] >= 20
        _rm_atr      = df["atr"] >= 3.0
        _rm_base     = _rm_adx & _rm_atr
        # LONG: RSI > 70 confirming strong uptrend momentum (not overbought reversal)
        _rsi_mom_long  = (_rsi_rm > 70) & _rm_ema_bull & _rm_st_bull & _rm_base
        # SHORT: RSI < 30 confirming strong downtrend momentum (not oversold reversal)
        _rsi_mom_short = (_rsi_rm < 30) & _rm_ema_bear & _rm_st_bear & _rm_base
        df.loc[_rsi_mom_long,  "buy_score"]  += 2
        df.loc[_rsi_mom_short, "sell_score"] += 2
        df.loc[_rsi_mom_long,  "_b_rsi_momentum_entry"] = 1
        df.loc[_rsi_mom_short, "_s_rsi_momentum_entry"] = 1

    # --- 67. Volume Surge Gate (GATE) ---
    # volume_ratio >= 1.2 = meaningful above-average volume (stronger than volume_ratio_gate=0.8).
    # Filters out thin/noise entries: real momentum moves carry above-avg volume.
    # Applied to D/E/J to ensure signals occur during active, liquid candles.
    if enabled.get("vol_surge_gate", False):
        _vsg_ok = df["volume_ratio"] >= 1.2
        df.loc[_vsg_ok, "_b_vol_surge_gate"] = 1
        df.loc[_vsg_ok, "_s_vol_surge_gate"] = 1

    # --- 68. OBV Trend Gate (GATE, directional) ---
    # OBV EMA slope over 5 bars: captures smart-money accumulation vs distribution.
    # BUY gate: OBV EMA rising (smart money accumulating = buy side dominant).
    # SELL gate: OBV EMA falling (smart money distributing = sell side dominant).
    # More robust than obv_confirm (OBV vs OBV_EMA) because it checks the TREND of OBV.
    # Key use: In Apr-26, E fires LONGs but OBV_EMA was falling = filter out those LONGs.
    if enabled.get("obv_trend_gate", False):
        _obv_ema_up = df["obv_ema"] > df["obv_ema"].shift(5)
        _obv_ema_dn = df["obv_ema"] < df["obv_ema"].shift(5)
        df.loc[_obv_ema_up, "_b_obv_trend_gate"] = 1
        df.loc[_obv_ema_dn, "_s_obv_trend_gate"] = 1

    # --- 69. MFI Direction Gate (GATE, directional) ---
    # MFI (Money Flow Index) = volume-weighted RSI.
    # > 50: net buying pressure (buyers absorbing volume) → BUY side dominant.
    # < 50: net selling pressure → SELL side dominant.
    # Unlike mfi_confirm (requires extremes <30/>70), this checks dominant side only.
    # Applied to D/J: confirms that volume FLOW is in the direction of the trade.
    if enabled.get("mfi_direction_gate", False):
        _mfi_buy  = df["mfi"] > 50
        _mfi_sell = df["mfi"] < 50
        df.loc[_mfi_buy,  "_b_mfi_direction_gate"] = 1
        df.loc[_mfi_sell, "_s_mfi_direction_gate"] = 1

    # --- 70. EMA21 Trend Gate (GATE for W: EMA21 Bounce) ---
    # Filter: EMA21 must have clear slope direction AND price not hugging EMA21.
    # Insight (backtest_ema_clear.py): H1-W/5m: -24.1 → +2.4 (+26.5pts) with dist≥0.5+slope align.
    # LONG gate:  EMA21 3-bar slope > 0 (EMA rising = bull micro-trend)
    #             AND |close - EMA21| / ATR >= 0.5 (price moved away from EMA = momentum confirmed)
    # SHORT gate: EMA21 3-bar slope < 0 (EMA falling = bear micro-trend)
    #             AND |close - EMA21| / ATR >= 0.5
    # Combined with ema21_pullback (requires dist ≤ 1.2×ATR), this creates sweet zone: 0.5–1.2×ATR.
    # Do NOT apply to 3m (filter ineffective: 3m entries already have dist>0.5 naturally).
    if enabled.get("ema21_trend_gate", False):
        _etg_atr      = df["atr"].replace(0, np.nan)
        _etg_slope    = df["ema21"].diff(3)                          # 3-bar EMA21 change
        _etg_dist     = (df["close"] - df["ema21"]).abs() / _etg_atr # dist / ATR
        _etg_bull     = (_etg_slope > 0) & (_etg_dist >= 0.5)        # rising EMA + price moved up
        _etg_bear     = (_etg_slope < 0) & (_etg_dist >= 0.5)        # falling EMA + price moved down
        df.loc[_etg_bull, "_b_ema21_trend_gate"] = 1
        df.loc[_etg_bear, "_s_ema21_trend_gate"] = 1

    # --- 71. Liquidity Sweep (PRIMARY for X: Big Move Catcher) ---
    # Data-driven: 59% of 7+ pt moves are preceded by a STOP HUNT (liquidity sweep).
    # BUY: price dipped below 5-bar low (grabbed sell stops) then closed above it = sweep complete.
    # SELL: price spiked above 5-bar high (grabbed buy stops) then closed below it = sweep complete.
    # The key insight: smart money pushes price to grab stops, then reverses aggressively.
    if enabled.get("liquidity_sweep", False):
        _ls_low5 = df["low"].rolling(5).min().shift(1)   # 5-bar low BEFORE current bar
        _ls_high5 = df["high"].rolling(5).max().shift(1)  # 5-bar high BEFORE current bar
        # BUY sweep: current bar's low broke below prev 5-bar low, but close recovered above it
        _ls_buy = (df["low"] < _ls_low5) & (df["close"] > _ls_low5)
        # SELL sweep: current bar's high broke above prev 5-bar high, but close fell below it
        _ls_sell = (df["high"] > _ls_high5) & (df["close"] < _ls_high5)
        df.loc[_ls_buy,  "buy_score"]  += 2
        df.loc[_ls_sell, "sell_score"] += 2
        df.loc[_ls_buy,  "_b_liquidity_sweep"] = 1
        df.loc[_ls_sell, "_s_liquidity_sweep"] = 1

    # --- 72. Pullback Reversal (PRIMARY for X: Big Move Catcher) ---
    # Data-driven: 63% of 7+ pt moves have a prior pullback (mom3 opposite direction).
    # BUY: 3-bar momentum was negative (pullback) + current candle is bullish (reversal starts).
    # SELL: 3-bar momentum was positive (rally) + current candle is bearish (reversal starts).
    # Combined with liquidity_sweep, captures the "trap then reverse" pattern.
    if enabled.get("pullback_reversal", False):
        _pr_mom3 = df["close"].pct_change(3)
        _pr_bull_candle = df["close"] > df["open"]
        _pr_bear_candle = df["close"] < df["open"]
        # BUY: prior 3-bar move was down (pullback) + current bar is bullish
        _pr_buy = (_pr_mom3 < -0.001) & _pr_bull_candle
        # SELL: prior 3-bar move was up (rally) + current bar is bearish
        _pr_sell = (_pr_mom3 > 0.001) & _pr_bear_candle
        df.loc[_pr_buy,  "buy_score"]  += 2
        df.loc[_pr_sell, "sell_score"] += 2
        df.loc[_pr_buy,  "_b_pullback_reversal"] = 1
        df.loc[_pr_sell, "_s_pullback_reversal"] = 1

    # --- 73. Trap Confirmation Gate (GATE for X: Big Move Catcher) ---
    # Multi-factor trap detection: combines liquidity sweep presence + volume confirmation + 
    # momentum alignment. Requires sweep (low broke recent support/resistance) + vol >= 1.0x avg.
    # From analysis: moves with trap_score >= 3 have avg 11.9 pts (88% of all big moves have traps).
    if enabled.get("trap_confirmation_gate", False):
        _tcg_low5 = df["low"].rolling(5).min().shift(1)
        _tcg_high5 = df["high"].rolling(5).max().shift(1)
        _tcg_vol_ok = df["volume_ratio"] >= 1.0
        # BUY: sweep below detected (wick below 5-bar low) + vol ok + close recovered
        _tcg_sweep_buy = (df["low"] < _tcg_low5) & (df["close"] >= _tcg_low5) & _tcg_vol_ok
        # SELL: sweep above detected (wick above 5-bar high) + vol ok + close fell
        _tcg_sweep_sell = (df["high"] > _tcg_high5) & (df["close"] <= _tcg_high5) & _tcg_vol_ok
        # Also pass if supertrend just flipped (= strong reversal signal)
        _tcg_st_flip_buy = (df["supertrend_dir"] == 1) & (df["supertrend_dir"].shift(1) == -1)
        _tcg_st_flip_sell = (df["supertrend_dir"] == -1) & (df["supertrend_dir"].shift(1) == 1)
        _tcg_buy = _tcg_sweep_buy | _tcg_st_flip_buy
        _tcg_sell = _tcg_sweep_sell | _tcg_st_flip_sell
        df.loc[_tcg_buy,  "_b_trap_confirmation_gate"] = 1
        df.loc[_tcg_sell, "_s_trap_confirmation_gate"] = 1

    # --- 74. Strong Candle Gate (GATE for X: Big Move Catcher) ---
    # Filter: only allow signals where the signal candle has a strong body (body/range > 55%).
    # Backtest shows this filter improves X WR from 34% → 48-56% by filtering weak/doji entries.
    if enabled.get("strong_candle_gate", False):
        _scg_body = (df["close"] - df["open"]).abs()
        _scg_range = (df["high"] - df["low"]).replace(0, pd.NA)
        _scg_ratio = _scg_body / _scg_range
        _scg_strong = _scg_ratio > 0.55
        # Both BUY and SELL pass when candle is strong
        df.loc[_scg_strong,  "_b_strong_candle_gate"] = 1
        df.loc[_scg_strong,  "_s_strong_candle_gate"] = 1

    # --- 75. EMA200 Bias Gate (Universal WR booster) ---
    # Only allow BUY when price > EMA200, SELL when price < EMA200.
    # Backtest 120d: D +9.6% WR, V +26% WR, X +61% WR when applied.
    # Ensures trading with the dominant trend direction.
    if enabled.get("ema200_bias_gate", False):
        _ema200 = df["close"].ewm(span=200, min_periods=50).mean()
        df["ema200"] = _ema200
        _e200_buy = df["close"] > _ema200
        _e200_sell = df["close"] < _ema200
        df.loc[_e200_buy,  "_b_ema200_bias_gate"] = 1
        df.loc[_e200_sell, "_s_ema200_bias_gate"] = 1

    # --- 76. EMA Slope Gate (Universal WR booster) ---
    # Only allow BUY when EMA20 slope > 0 (rising), SELL when slope < 0 (falling).
    # 3-bar diff of EMA20 used as slope proxy. Ensures momentum alignment.
    # Backtest 120d: X/5m 39→71% WR, D 65→75% WR.
    if enabled.get("ema_slope_gate", False):
        _ema20_line = df["close"].ewm(span=20, min_periods=10).mean()
        _ema20_slope = _ema20_line.diff(3)
        _eslope_buy = _ema20_slope > 0
        _eslope_sell = _ema20_slope < 0
        df.loc[_eslope_buy,  "_b_ema_slope_gate"] = 1
        df.loc[_eslope_sell, "_s_ema_slope_gate"] = 1

    # --- 77. Prev Bar Gate (momentum confirmation) ---
    # Only allow BUY when previous bar was bullish, SELL when bearish.
    # Confirms that momentum is already moving in signal direction.
    if enabled.get("prev_bar_gate", False):
        _pb_bull = df["close"].shift(1) > df["open"].shift(1)
        _pb_bear = df["close"].shift(1) < df["open"].shift(1)
        df.loc[_pb_bull, "_b_prev_bar_gate"] = 1
        df.loc[_pb_bear, "_s_prev_bar_gate"] = 1

    # ==================== PATTERN DETECTION (informational) ====================
    # Improved implementation based on TA-Lib logic and swing-point analysis.
    # These are reported in alerts for context, not used for signal scoring.

    _open = df["open"]
    _close = df["close"]
    _high = df["high"]
    _low = df["low"]
    _body = _close - _open
    _body_abs = _body.abs()
    _candle_range = _high - _low

    # Adaptive body size thresholds (TA-Lib approach: rolling averages)
    _body_avg = _body_abs.rolling(10, min_periods=1).mean()
    _body_long = _body_abs > _body_avg * 1.5     # "Long" body
    _body_short = _body_abs < _body_avg * 0.5    # "Short" body (doji-like)

    # --- Morning Star (TA-Lib logic, adapted for intraday) ---
    # 3-candle bullish reversal at bottom
    # 1st (i-2): Long bearish (red)
    # 2nd (i-1): Short body + opens near/below 1st close (relaxed gap for intraday)
    # 3rd (i):   Bullish, body > short, close penetrates >=30% into 1st body
    _c1_long_bear = _body_long.shift(2) & (_body.shift(2) < 0)
    # Relaxed gap: 2nd candle opens at or below 1st candle's close (no strict gap needed intraday)
    _c2_upper = pd.concat([_open.shift(1), _close.shift(1)], axis=1).max(axis=1)
    _c1_lower = pd.concat([_open.shift(2), _close.shift(2)], axis=1).min(axis=1)
    _c1_close = _close.shift(2)
    _gap_down = _open.shift(1) <= _c1_close  # open of 2nd <= close of 1st (bearish continuation)
    _c2_short = _body_short.shift(1)
    # 3rd candle: bullish, not short, close penetrates 30% into 1st body
    _c3_bull = (_body > 0) & (~_body_short)
    _penetration = 0.3
    _c1_body_top = pd.concat([_open.shift(2), _close.shift(2)], axis=1).max(axis=1)
    _c1_body_size = _body_abs.shift(2)
    _c3_penetrate = _close > (_c1_body_top - _c1_body_size * (1 - _penetration))
    df["pat_morning_star"] = (_c1_long_bear & _c2_short & _gap_down & _c3_bull & _c3_penetrate).fillna(False).astype(int)

    # --- Evening Star (TA-Lib logic, adapted for intraday) ---
    # 3-candle bearish reversal at top
    # 1st (i-2): Long bullish (green)
    # 2nd (i-1): Short body + opens near/above 1st close
    # 3rd (i):   Bearish, body > short, close penetrates >=30% into 1st body
    _c1_long_bull = _body_long.shift(2) & (_body.shift(2) > 0)
    _c1_close_bull = _close.shift(2)
    _gap_up = _open.shift(1) >= _c1_close_bull  # open of 2nd >= close of 1st (bullish continuation)
    _c3_bear = (_body < 0) & (~_body_short)
    _c1_body_bottom = pd.concat([_open.shift(2), _close.shift(2)], axis=1).min(axis=1)
    _c3_penetrate_ev = _close < (_c1_body_bottom + _c1_body_size * (1 - _penetration))
    df["pat_evening_star"] = (_c1_long_bull & _c2_short & _gap_up & _c3_bear & _c3_penetrate_ev).fillna(False).astype(int)

    # --- Engulfing (TA-Lib logic) ---
    # Bullish: prev bearish + current bullish engulfs prev body entirely
    # Additional: current body is "long" (meaningful size)
    _prev_bear = _body.shift(1) < 0
    _prev_bull = _body.shift(1) > 0
    _engulf_buy = (
        (_body > 0) & _prev_bear &
        (_open <= _close.shift(1)) &  # open at or below prev close
        (_close >= _open.shift(1)) &   # close at or above prev open
        (_body_abs > _body_abs.shift(1))  # current body > prev body
    )
    _engulf_sell = (
        (_body < 0) & _prev_bull &
        (_open >= _close.shift(1)) &
        (_close <= _open.shift(1)) &
        (_body_abs > _body_abs.shift(1))
    )
    df["pat_bull_engulfing"] = _engulf_buy.fillna(False).astype(int)
    df["pat_bear_engulfing"] = _engulf_sell.fillna(False).astype(int)

    # --- Head and Shoulders (swing-point based, optimized) ---
    # Step 1: Find swing highs/lows using rolling window (order=3)
    _swing_order = 3  # bars on each side to confirm a swing (relaxed for intraday)
    _lookback = 50    # bars to search for the pattern

    # Vectorized swing detection: high == rolling max of 2*order+1 window centered on bar
    _win = 2 * _swing_order + 1
    _roll_max = _high.rolling(_win, center=True).max()
    _roll_min = _low.rolling(_win, center=True).min()
    _swing_high = (_high == _roll_max) & _roll_max.notna()
    _swing_low = (_low == _roll_min) & _roll_min.notna()

    df["pat_head_shoulders_top"] = 0
    df["pat_head_shoulders_bottom"] = 0

    # Step 2: Scan for H&S pattern in swing points
    _sh_indices = _swing_high[_swing_high].index.tolist()
    _sl_indices = _swing_low[_swing_low].index.tolist()

    # Bearish H&S (top): find 3 consecutive swing highs where middle is highest
    for i in range(2, len(_sh_indices)):
        ls_idx = _sh_indices[i - 2]  # left shoulder
        hd_idx = _sh_indices[i - 1]  # head
        rs_idx = _sh_indices[i]      # right shoulder

        ls_pos = df.index.get_loc(ls_idx)
        hd_pos = df.index.get_loc(hd_idx)
        rs_pos = df.index.get_loc(rs_idx)

        # Must be within lookback window and reasonably spaced
        if rs_pos - ls_pos > _lookback or rs_pos - ls_pos < 6:
            continue

        ls_val = _high.loc[ls_idx]
        hd_val = _high.loc[hd_idx]
        rs_val = _high.loc[rs_idx]

        # Head must be highest
        if hd_val <= ls_val or hd_val <= rs_val:
            continue

        # Shoulders roughly equal (within 3% of head height)
        shoulder_diff = abs(ls_val - rs_val)
        head_range = hd_val - min(ls_val, rs_val)
        if head_range == 0 or shoulder_diff / head_range > 0.3:
            continue

        # Neckline: lowest low between shoulders
        neck_slice = _low.iloc[ls_pos:rs_pos + 1]
        neckline = neck_slice.min()

        # Confirm: price breaks below neckline after right shoulder
        if rs_pos < len(df) - 1:
            post_close = _close.iloc[rs_pos:min(rs_pos + 5, len(df))]
            if (post_close < neckline).any():
                df.iloc[rs_pos, df.columns.get_loc("pat_head_shoulders_top")] = 1

    # Bullish inverse H&S (bottom): 3 consecutive swing lows where middle is lowest
    for i in range(2, len(_sl_indices)):
        ls_idx = _sl_indices[i - 2]
        hd_idx = _sl_indices[i - 1]
        rs_idx = _sl_indices[i]

        ls_pos = df.index.get_loc(ls_idx)
        hd_pos = df.index.get_loc(hd_idx)
        rs_pos = df.index.get_loc(rs_idx)

        if rs_pos - ls_pos > _lookback or rs_pos - ls_pos < 6:
            continue

        ls_val = _low.loc[ls_idx]
        hd_val = _low.loc[hd_idx]
        rs_val = _low.loc[rs_idx]

        # Head must be lowest
        if hd_val >= ls_val or hd_val >= rs_val:
            continue

        # Shoulders roughly equal
        shoulder_diff = abs(ls_val - rs_val)
        head_range = max(ls_val, rs_val) - hd_val
        if head_range == 0 or shoulder_diff / head_range > 0.3:
            continue

        # Neckline: highest high between shoulders
        neck_slice = _high.iloc[ls_pos:rs_pos + 1]
        neckline = neck_slice.max()

        # Confirm: price breaks above neckline after right shoulder
        if rs_pos < len(df) - 1:
            post_close = _close.iloc[rs_pos:min(rs_pos + 5, len(df))]
            if (post_close > neckline).any():
                df.iloc[rs_pos, df.columns.get_loc("pat_head_shoulders_bottom")] = 1

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
