"""
Missed Signals Analysis — Last 7 trading days.

Finds profitable moves that happened but ML didn't predict.
Analyzes features/patterns before those moves to suggest improvements.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

from backtest.engine import load


def compute_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ALL features including experimental new ones."""
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df.get("volume")

    # Standard indicators
    df["atr_14"] = ta.atr(h, l, c, length=14)
    df["atr_5"] = ta.atr(h, l, c, length=5)
    df["rsi_14"] = ta.rsi(c, length=14)
    df["rsi_3"] = ta.rsi(c, length=3)
    df["ema8"] = ta.ema(c, length=8)
    df["ema21"] = ta.ema(c, length=21)
    df["ema50"] = ta.ema(c, length=50)
    sma20 = ta.sma(c, length=20)
    std20 = c.rolling(20).std()
    df["bb_width"] = (4 * std20 / sma20).where(sma20 > 0, 0)
    df["bb_pos"] = ((c - sma20) / (2 * std20)).where(std20 > 0, 0)

    stoch = ta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd_hist"] = macd.iloc[:, 1]
        df["macd_line"] = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 2]

    adx_df = ta.adx(h, l, c, length=14)
    if adx_df is not None:
        df["adx"] = adx_df["ADX_14"]
        df["di_plus"] = adx_df["DMP_14"]
        df["di_minus"] = adx_df["DMN_14"]

    # ===== NEW EXPERIMENTAL FEATURES =====

    # 1. Microstructure: consecutive candle direction
    df["candle_dir"] = np.sign(c - o)
    df["consec_up"] = 0
    df["consec_down"] = 0
    dirs = df["candle_dir"].values
    cu = np.zeros(len(dirs))
    cd = np.zeros(len(dirs))
    for i in range(1, len(dirs)):
        if dirs[i] > 0:
            cu[i] = cu[i-1] + 1
            cd[i] = 0
        elif dirs[i] < 0:
            cd[i] = cd[i-1] + 1
            cu[i] = 0
    df["consec_up"] = cu
    df["consec_down"] = cd

    # 2. Range expansion/contraction ratio (last 3 vs last 10)
    ranges = h - l
    df["range_ratio_3_10"] = ranges.rolling(3).mean() / ranges.rolling(10).mean()

    # 3. Body-to-range ratio (momentum quality)
    df["body_ratio"] = abs(c - o) / ranges.replace(0, np.nan)

    # 4. Gap from previous close (overnight/lunch gap)
    df["gap"] = o - c.shift(1)
    df["gap_pct"] = df["gap"] / c.shift(1)

    # 5. Volume spike (if available)
    if v is not None and not v.isna().all():
        vol_sma5 = v.rolling(5).mean()
        vol_sma20 = v.rolling(20).mean()
        df["vol_spike_5"] = (v / vol_sma5).where(vol_sma5 > 0, 1.0)
        df["vol_spike_20"] = (v / vol_sma20).where(vol_sma20 > 0, 1.0)
    else:
        df["vol_spike_5"] = 1.0
        df["vol_spike_20"] = 1.0

    # 6. Price relative to session VWAP proxy (cumulative weighted avg)
    # Since no tick data, use bar midpoint weighted by range as proxy
    df["bar_mid"] = (h + l + c) / 3
    # Reset each session
    df["_session_cumsum"] = 0.0
    df["_session_vol_cumsum"] = 0.0

    # 7. Momentum divergence: price making new high but RSI not
    df["price_5h"] = c.rolling(5).max()
    df["rsi_5h"] = df["rsi_14"].rolling(5).max()
    df["div_bearish"] = ((c >= df["price_5h"] * 0.999) &
                         (df["rsi_14"] < df["rsi_5h"] - 5)).astype(int)
    df["price_5l"] = c.rolling(5).min()
    df["rsi_5l"] = df["rsi_14"].rolling(5).min()
    df["div_bullish"] = ((c <= df["price_5l"] * 1.001) &
                         (df["rsi_14"] > df["rsi_5l"] + 5)).astype(int)

    # 8. EMA alignment score (-3 to +3)
    df["ema_align"] = (
        np.sign(c - df["ema8"]) +
        np.sign(c - df["ema21"]) +
        np.sign(c - df["ema50"])
    )

    # 9. Keltner Channel position
    ema20 = ta.ema(c, length=20)
    df["kc_pos"] = ((c - ema20) / (1.5 * df["atr_14"])).where(df["atr_14"] > 0, 0)

    # 10. Stochastic RSI
    rsi = df["rsi_14"]
    rsi_min14 = rsi.rolling(14).min()
    rsi_max14 = rsi.rolling(14).max()
    df["stoch_rsi"] = ((rsi - rsi_min14) / (rsi_max14 - rsi_min14)).where(
        (rsi_max14 - rsi_min14) > 0, 0.5)

    # 11. Price acceleration (2nd derivative of price)
    ret1 = c.pct_change(1)
    df["price_accel"] = ret1 - ret1.shift(1)

    # 12. High-Low momentum (are highs expanding or contracting?)
    df["hh_count_5"] = h.rolling(5).apply(lambda x: sum(x[i] > x[i-1] for i in range(1, len(x))), raw=True)
    df["ll_count_5"] = l.rolling(5).apply(lambda x: sum(x[i] < x[i-1] for i in range(1, len(x))), raw=True)

    # 13. Intraday time-weighted momentum (recent bars weighted more)
    weights = np.array([1, 2, 3, 4, 5], dtype=float)
    weights /= weights.sum()
    df["weighted_ret_5"] = c.pct_change(1).rolling(5).apply(
        lambda x: np.dot(x, weights) if len(x) == 5 else 0, raw=True)

    # 14. Distance from session high/low
    # Will compute per-session below

    return df


def find_big_moves(df, horizon=6, min_move_pts=2.0):
    """Find all bars where the next `horizon` bars moved >= min_move_pts."""
    c = df["close"].values
    moves = []

    for i in range(len(c) - horizon):
        if pd.isna(df.iloc[i].get("atr_14")) or i < 50:
            continue
        # Check forward return
        fwd_max = max(c[i+1:i+horizon+1]) - c[i]
        fwd_min = c[i] - min(c[i+1:i+horizon+1])

        best_move = max(fwd_max, fwd_min)
        direction = 1 if fwd_max >= fwd_min else -1

        if best_move >= min_move_pts:
            moves.append({
                "idx": i,
                "time": df.iloc[i]["time"],
                "date": df.iloc[i]["date"],
                "session": df.iloc[i]["session"],
                "close": c[i],
                "direction": direction,
                "move_pts": best_move * direction,
                "abs_move": best_move,
            })

    return pd.DataFrame(moves)


def main():
    print("=" * 90)
    print("MISSED SIGNALS ANALYSIS — Last 7 Trading Days")
    print("=" * 90)

    # Load data
    df = load("5m", days=180)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    df["mins"] = df["time"].dt.hour * 60 + df["time"].dt.minute
    df["session"] = np.where(df["mins"] < 12*60, "AM", "PM")

    # Get last 7 trading days
    unique_dates = sorted(df["date"].unique())
    last_7_dates = unique_dates[-7:]
    print(f"\n  Analyzing: {last_7_dates[0]} to {last_7_dates[-1]}")

    # Compute features on ALL data (for indicator warmup)
    df = compute_enhanced_features(df)

    # Filter to last 7 days
    df_week = df[df["date"].isin(last_7_dates)].copy()
    print(f"  Bars in period: {len(df_week)}")

    # ===== Find ALL big moves =====
    print(f"\n{'=' * 90}")
    print("SECTION 1: All Significant Moves (>= 2 pts in 6 bars)")
    print(f"{'=' * 90}")

    moves = find_big_moves(df_week, horizon=6, min_move_pts=2.0)
    # Deduplicate: keep only moves at least 6 bars apart
    if not moves.empty:
        keep = [0]
        for i in range(1, len(moves)):
            if moves.iloc[i]["idx"] - moves.iloc[keep[-1]]["idx"] >= 6:
                keep.append(i)
        moves = moves.iloc[keep].reset_index(drop=True)

    print(f"  Total significant moves: {len(moves)}")
    if not moves.empty:
        buy_moves = moves[moves["direction"] == 1]
        sell_moves = moves[moves["direction"] == -1]
        print(f"  UP moves (BUY opportunity): {len(buy_moves)}, avg={buy_moves['abs_move'].mean():.1f} pts")
        print(f"  DOWN moves (SELL opportunity): {len(sell_moves)}, avg={sell_moves['abs_move'].mean():.1f} pts")

        # Session breakdown
        for sess in ["AM", "PM"]:
            sub = moves[moves["session"] == sess]
            print(f"  {sess}: {len(sub)} moves, avg={sub['abs_move'].mean():.1f} pts")

    # ===== Analyze features BEFORE big moves =====
    print(f"\n{'=' * 90}")
    print("SECTION 2: Feature Patterns Before Big Moves")
    print(f"{'=' * 90}")

    feature_cols = [
        "rsi_14", "rsi_3", "adx", "bb_pos", "stoch_k", "atr_14",
        "range_ratio_3_10", "body_ratio", "vol_spike_5", "vol_spike_20",
        "consec_up", "consec_down", "div_bearish", "div_bullish",
        "ema_align", "kc_pos", "stoch_rsi", "price_accel",
        "hh_count_5", "ll_count_5", "weighted_ret_5", "macd_hist",
        "gap_pct", "bb_width",
    ]

    # Get features at move bars vs non-move bars
    move_indices = moves["idx"].values if not moves.empty else []
    non_move_mask = ~df_week.index.isin(df_week.index[move_indices]) if len(move_indices) > 0 else df_week.index

    print(f"\n  Feature comparison: Move bars vs Non-move bars")
    print(f"  {'Feature':<20} {'Move_Mean':<12} {'NoMove_Mean':<12} {'Diff%':<10} {'Predictive?'}")
    print(f"  {'-'*70}")

    available_feats = [f for f in feature_cols if f in df_week.columns]
    feature_diffs = []

    for feat in available_feats:
        if len(move_indices) == 0:
            continue
        move_vals = df_week.iloc[move_indices][feat].dropna()
        non_move_vals = df_week[~df_week.index.isin(df_week.index[move_indices])][feat].dropna()

        if len(move_vals) < 3 or len(non_move_vals) < 10:
            continue

        m_mean = move_vals.mean()
        nm_mean = non_move_vals.mean()
        diff_pct = abs(m_mean - nm_mean) / (abs(nm_mean) + 0.001) * 100

        predictive = "***" if diff_pct > 30 else "**" if diff_pct > 15 else "*" if diff_pct > 8 else ""
        print(f"  {feat:<20} {m_mean:<12.3f} {nm_mean:<12.3f} {diff_pct:<10.1f} {predictive}")
        feature_diffs.append({"feat": feat, "diff_pct": diff_pct, "move_mean": m_mean, "nomove_mean": nm_mean})

    # Sort by predictive power
    feature_diffs.sort(key=lambda x: x["diff_pct"], reverse=True)
    print(f"\n  Top 5 Most Discriminative Features:")
    for i, fd in enumerate(feature_diffs[:5]):
        print(f"    {i+1}. {fd['feat']}: {fd['diff_pct']:.1f}% diff (move={fd['move_mean']:.3f}, other={fd['nomove_mean']:.3f})")

    # ===== Direction-specific analysis =====
    print(f"\n{'=' * 90}")
    print("SECTION 3: Direction-Specific Patterns")
    print(f"{'=' * 90}")

    if not moves.empty:
        for direction, dir_name in [(1, "UP/BUY"), (-1, "DOWN/SELL")]:
            dir_moves = moves[moves["direction"] == direction]
            if len(dir_moves) < 3:
                continue
            dir_indices = dir_moves["idx"].values

            print(f"\n  --- {dir_name} moves ({len(dir_moves)} instances) ---")
            print(f"  {'Feature':<20} {'Mean':<10} {'Std':<10} {'Pattern'}")
            print(f"  {'-'*55}")

            for feat in available_feats[:15]:
                vals = df_week.iloc[dir_indices][feat].dropna()
                if len(vals) < 3:
                    continue
                m = vals.mean()
                s = vals.std()
                # Interpret pattern
                pattern = ""
                if feat == "rsi_14":
                    if direction == 1 and m < 40: pattern = "oversold → bounce"
                    elif direction == -1 and m > 60: pattern = "overbought → drop"
                elif feat == "ema_align":
                    if direction == 1 and m < 0: pattern = "below EMAs → reversal"
                    elif direction == -1 and m > 0: pattern = "above EMAs → reversal"
                elif feat == "range_ratio_3_10":
                    if m < 0.7: pattern = "squeeze before move"
                    elif m > 1.3: pattern = "expansion continues"
                elif feat == "consec_up" and direction == -1 and m > 2:
                    pattern = "exhaustion after run"
                elif feat == "consec_down" and direction == 1 and m > 2:
                    pattern = "exhaustion after sell"

                print(f"  {feat:<20} {m:<10.3f} {s:<10.3f} {pattern}")

    # ===== Detailed move-by-move for last 7 days =====
    print(f"\n{'=' * 90}")
    print("SECTION 4: Move-by-Move Detail (Last 7 Days)")
    print(f"{'=' * 90}")

    if not moves.empty:
        print(f"\n  {'Time':<18} {'Sess':<4} {'Dir':<5} {'Move':<7} {'RSI':<6} {'ADX':<6} "
              f"{'BB_pos':<7} {'RngRat':<7} {'EMA_al':<7} {'StRSI':<6} {'Accel':<7}")
        print(f"  {'-'*90}")

        for _, mv in moves.iterrows():
            idx = int(mv["idx"])
            if idx >= len(df_week):
                continue
            row = df_week.iloc[idx]
            t = str(row["time"])[-8:-3] if "time" in row.index else "?"
            d = "BUY" if mv["direction"] == 1 else "SELL"
            print(f"  {str(mv['date'])} {t} {mv['session']:<4} {d:<5} {mv['abs_move']:<+7.1f} "
                  f"{row.get('rsi_14', 0):<6.1f} {row.get('adx', 0):<6.1f} "
                  f"{row.get('bb_pos', 0):<7.2f} {row.get('range_ratio_3_10', 0):<7.2f} "
                  f"{row.get('ema_align', 0):<7.0f} {row.get('stoch_rsi', 0):<6.2f} "
                  f"{row.get('price_accel', 0):<7.4f}")

    # ===== SECTION 5: Suggested New Features =====
    print(f"\n{'=' * 90}")
    print("SECTION 5: Feature Engineering Suggestions")
    print(f"{'=' * 90}")

    print("""
  Based on analysis, these NEW features could help:

  1. RANGE CONTRACTION RATIO (range_ratio_3_10)
     - Moves often follow squeeze periods (ratio < 0.7)
     - Combine with ADX < 20 for "coiled spring" signal

  2. CONSECUTIVE CANDLE DIRECTION (consec_up/down)
     - 3+ consecutive same-direction → exhaustion likely
     - Reversal signals after extended runs

  3. VOLUME SPIKE (vol_spike_5)
     - Unusually high volume precedes breakouts
     - Low volume in squeeze → high volume on break

  4. MOMENTUM DIVERGENCE (div_bullish/div_bearish)
     - Price new low but RSI higher → bullish reversal
     - Price new high but RSI lower → bearish reversal

  5. EMA ALIGNMENT SCORE (ema_align)
     - Full alignment (±3) = strong trend continuation
     - Mixed alignment (0, ±1) = ranging, reversal possible

  6. STOCHASTIC RSI (stoch_rsi)
     - More sensitive than regular RSI for extremes
     - < 0.1 or > 0.9 = extreme, reversal expected

  7. PRICE ACCELERATION (price_accel)
     - 2nd derivative of price — momentum change
     - Negative accel after positive run = weakening

  8. KELTNER CHANNEL POSITION (kc_pos)
     - Outside ±1.5 = overextended
     - Reversion expected from extremes
""")

    # ===== Quick test: what if we used top features as simple rules? =====
    print(f"\n{'=' * 90}")
    print("SECTION 6: Simple Rule-Based Signal Test (last 7 days)")
    print(f"{'=' * 90}")

    COST = 0.96
    horizon = 6

    # Test simple rule combos
    rules = {
        "Squeeze+RSI<40 (BUY)": lambda r: (r.get("range_ratio_3_10", 1) < 0.7) and (r.get("rsi_14", 50) < 40),
        "Squeeze+RSI>60 (SELL)": lambda r: (r.get("range_ratio_3_10", 1) < 0.7) and (r.get("rsi_14", 50) > 60),
        "StochRSI<0.1+EMA<0 (BUY)": lambda r: (r.get("stoch_rsi", 0.5) < 0.1) and (r.get("ema_align", 0) <= 0),
        "StochRSI>0.9+EMA>0 (SELL)": lambda r: (r.get("stoch_rsi", 0.5) > 0.9) and (r.get("ema_align", 0) >= 0),
        "Consec3Down+RSI<35 (BUY)": lambda r: (r.get("consec_down", 0) >= 3) and (r.get("rsi_14", 50) < 35),
        "Consec3Up+RSI>65 (SELL)": lambda r: (r.get("consec_down", 0) >= 3) and (r.get("rsi_14", 50) > 65),
        "DivBullish (BUY)": lambda r: r.get("div_bullish", 0) == 1,
        "DivBearish (SELL)": lambda r: r.get("div_bearish", 0) == 1,
        "KC<-1.5+Accel>0 (BUY)": lambda r: (r.get("kc_pos", 0) < -1.5) and (r.get("price_accel", 0) > 0),
        "KC>1.5+Accel<0 (SELL)": lambda r: (r.get("kc_pos", 0) > 1.5) and (r.get("price_accel", 0) < 0),
    }

    c_vals = df_week["close"].values
    print(f"\n  {'Rule':<32} {'Signals':<8} {'Wins':<5} {'WR%':<7} {'Avg_PnL':<8} {'Total':<8}")
    print(f"  {'-'*75}")

    for rule_name, rule_fn in rules.items():
        direction = 1 if "BUY" in rule_name else -1
        trades_pnl = []
        last_signal = -horizon

        for i in range(50, len(df_week) - horizon):
            if i - last_signal < horizon:
                continue
            row = df_week.iloc[i]
            # Time filter
            mins = row.get("mins", 0)
            if row.get("session") == "AM" and mins >= 11*60+20:
                continue
            if row.get("session") == "PM" and mins >= 14*60+20:
                continue

            try:
                if rule_fn(row):
                    entry = c_vals[i]
                    # Find exit
                    exit_idx = min(i + horizon, len(df_week) - 1)
                    for j in range(i+1, i+horizon+1):
                        if j >= len(df_week):
                            exit_idx = len(df_week) - 1
                            break
                        if df_week.iloc[j]["date"] != row["date"] or df_week.iloc[j]["session"] != row["session"]:
                            exit_idx = j - 1
                            break
                        exit_idx = j

                    exit_price = c_vals[exit_idx]
                    pnl = direction * (exit_price - entry) - COST
                    trades_pnl.append(pnl)
                    last_signal = i
            except:
                continue

        if trades_pnl:
            n = len(trades_pnl)
            wins = sum(1 for p in trades_pnl if p > 0)
            wr = wins / n * 100
            avg_pnl = np.mean(trades_pnl)
            total = sum(trades_pnl)
            marker = " <<<" if wr > 55 and avg_pnl > 0 else ""
            print(f"  {rule_name:<32} {n:<8} {wins:<5} {wr:<7.1f} {avg_pnl:<+8.2f} {total:<+8.1f}{marker}")
        else:
            print(f"  {rule_name:<32} {'0':<8}")


if __name__ == "__main__":
    main()
