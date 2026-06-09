"""
Standalone ML Signal Generator for VN30F1M.

Pure ML approach: predict price direction on every 5m bar.
No dependency on CB compression - generates its own entry/exit signals.

Methodology:
  - Fixed-horizon labeling: predict return N bars forward
  - Walk-forward validation: 60d train / 20d test / step 20d
  - Threshold-based signals: trade only when P(direction) > threshold
  - Cost: 0.96 pts/trade (same as CB)

Usage:
    python -m ml.standalone_signals
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from datetime import timedelta

from backtest.engine import load


COST = 0.96
REPORT_PATH = Path("ml/reports/standalone_ml_results.json")


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 40 features on every bar for standalone prediction."""
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df.get("volume")

    # Volatility
    df["atr_14"] = ta.atr(h, l, c, length=14)
    df["atr_5"] = ta.atr(h, l, c, length=5)
    df["atr_pct"] = df["atr_14"] / c
    df["atr_ratio"] = df["atr_5"] / df["atr_14"]
    sma20 = ta.sma(c, length=20)
    std20 = c.rolling(20).std()
    df["bb_width"] = (4 * std20 / sma20).where(sma20 > 0, 0)
    df["bb_pos"] = ((c - sma20) / (2 * std20)).where(std20 > 0, 0)

    # Momentum
    df["rsi_14"] = ta.rsi(c, length=14)
    df["rsi_3"] = ta.rsi(c, length=3)
    df["rsi_7"] = ta.rsi(c, length=7)
    df["rsi_delta"] = df["rsi_7"] - df["rsi_7"].shift(3)

    stoch = ta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]
    else:
        df["stoch_k"] = 50.0
        df["stoch_d"] = 50.0

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd_hist"] = macd.iloc[:, 1] / c
    else:
        df["macd_hist"] = 0.0

    adx_df = ta.adx(h, l, c, length=14)
    if adx_df is not None:
        df["adx"] = adx_df["ADX_14"]
        df["di_plus"] = adx_df["DMP_14"]
        df["di_minus"] = adx_df["DMN_14"]
    else:
        df["adx"] = 20.0
        df["di_plus"] = 20.0
        df["di_minus"] = 20.0

    # Trend
    df["ema8"] = ta.ema(c, length=8)
    df["ema21"] = ta.ema(c, length=21)
    df["ema50"] = ta.ema(c, length=50)
    df["ema8_dist"] = (c - df["ema8"]) / c
    df["ema21_dist"] = (c - df["ema21"]) / c
    df["ema50_dist"] = (c - df["ema50"]) / c
    df["ema_spread"] = df["ema8_dist"] - df["ema50_dist"]

    # Returns
    df["ret_1"] = c.pct_change(1)
    df["ret_2"] = c.pct_change(2)
    df["ret_3"] = c.pct_change(3)
    df["ret_5"] = c.pct_change(5)
    df["ret_8"] = c.pct_change(8)
    df["ret_13"] = c.pct_change(13)

    # Price action
    bar_range = h - l
    df["body_pct"] = (c - o) / c
    df["range_pct"] = bar_range / c
    df["upper_wick"] = (h - np.maximum(c, o)) / bar_range.replace(0, np.nan)
    df["lower_wick"] = (np.minimum(c, o) - l) / bar_range.replace(0, np.nan)
    df["close_vs_range"] = (c - l) / bar_range.replace(0, np.nan)

    # Volume
    if v is not None and not v.isna().all():
        vol_ema20 = ta.ema(v.astype(float), length=20)
        df["vol_ratio"] = (v / vol_ema20).where(vol_ema20 > 0, 1.0)
        vol_sma5 = v.rolling(5).mean()
        df["vol_spike_5"] = (v / vol_sma5).where(vol_sma5 > 0, 1.0)
    else:
        df["vol_ratio"] = 1.0
        df["vol_spike_5"] = 1.0

    # Temporal
    df["hour_sin"] = np.sin(2 * np.pi * (df["mins"] - 9*60) / (5.5*60))
    df["hour_cos"] = np.cos(2 * np.pi * (df["mins"] - 9*60) / (5.5*60))
    df["is_pm"] = (df["mins"] >= 13*60).astype(int)

    # === NEW V2 FEATURES (from missed_signals_analysis) ===

    # MACD raw (not normalized) — top discriminator
    macd2 = ta.macd(c, fast=12, slow=26, signal=9)
    if macd2 is not None:
        df["macd_raw"] = macd2.iloc[:, 0]
        df["macd_signal_line"] = macd2.iloc[:, 2]
        df["macd_hist_raw"] = macd2.iloc[:, 1]
    else:
        df["macd_raw"] = 0.0
        df["macd_signal_line"] = 0.0
        df["macd_hist_raw"] = 0.0

    # Range contraction ratio (3-bar vs 10-bar avg range)
    ranges = h - l
    range_ma3 = ranges.rolling(3).mean()
    range_ma10 = ranges.rolling(10).mean()
    df["range_ratio_3_10"] = (range_ma3 / range_ma10).where(range_ma10 > 0, 1.0)

    # Consecutive candle direction
    candle_dir = np.sign((c - o).values)
    consec_up = np.zeros(len(candle_dir))
    consec_down = np.zeros(len(candle_dir))
    for i in range(1, len(candle_dir)):
        if candle_dir[i] > 0:
            consec_up[i] = consec_up[i-1] + 1
        elif candle_dir[i] < 0:
            consec_down[i] = consec_down[i-1] + 1
    df["consec_up"] = consec_up
    df["consec_down"] = consec_down

    # Momentum divergence (RSI vs Price over 5 bars)
    price_5h = c.rolling(5).max()
    rsi_5h = df["rsi_14"].rolling(5).max()
    price_5l = c.rolling(5).min()
    rsi_5l = df["rsi_14"].rolling(5).min()
    df["div_bearish"] = ((c >= price_5h * 0.999) &
                         (df["rsi_14"] < rsi_5h - 5)).astype(int)
    df["div_bullish"] = ((c <= price_5l * 1.001) &
                         (df["rsi_14"] > rsi_5l + 5)).astype(int)

    # EMA alignment score (-3 to +3)
    df["ema_align"] = (
        np.sign(c - df["ema8"]).fillna(0) +
        np.sign(c - df["ema21"]).fillna(0) +
        np.sign(c - df["ema50"]).fillna(0)
    )

    # Keltner Channel position
    ema20 = ta.ema(c, length=20)
    df["kc_pos"] = ((c - ema20) / (1.5 * df["atr_14"])).where(df["atr_14"] > 0, 0)

    # Stochastic RSI
    rsi = df["rsi_14"]
    rsi_min14 = rsi.rolling(14).min()
    rsi_max14 = rsi.rolling(14).max()
    df["stoch_rsi"] = ((rsi - rsi_min14) / (rsi_max14 - rsi_min14)).where(
        (rsi_max14 - rsi_min14) > 0, 0.5)

    # Price acceleration (2nd derivative)
    ret1 = c.pct_change(1)
    df["price_accel"] = ret1 - ret1.shift(1)

    # Weighted recent momentum (last 5 bars, more weight to recent)
    weights = np.array([1, 2, 3, 4, 5], dtype=float)
    weights /= weights.sum()
    df["weighted_ret_5"] = ret1.rolling(5).apply(
        lambda x: np.dot(x, weights) if len(x) == 5 else 0, raw=True)

    # DI spread (directional indicator difference)
    df["di_spread"] = df["di_plus"] - df["di_minus"]

    return df


FEATURE_COLS = [
    # Original 35
    "atr_14", "atr_5", "atr_pct", "atr_ratio", "bb_width", "bb_pos",
    "rsi_14", "rsi_3", "rsi_7", "rsi_delta", "stoch_k", "stoch_d",
    "macd_hist", "adx", "di_plus", "di_minus",
    "ema8_dist", "ema21_dist", "ema50_dist", "ema_spread",
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_8", "ret_13",
    "body_pct", "range_pct", "upper_wick", "lower_wick", "close_vs_range",
    "vol_ratio", "hour_sin", "hour_cos", "is_pm",
    # New V2 features (from missed_signals_analysis)
    "macd_raw", "macd_signal_line", "macd_hist_raw",
    "range_ratio_3_10", "consec_up", "consec_down",
    "div_bearish", "div_bullish",
    "ema_align", "kc_pos", "stoch_rsi",
    "price_accel", "weighted_ret_5",
    "vol_spike_5", "di_spread",
]


def label_fixed_horizon(df: pd.DataFrame, horizon: int = 6) -> pd.Series:
    """Label bars based on forward return vs adaptive threshold.

    Args:
        df: DataFrame with close and atr_14 columns.
        horizon: bars ahead to check.

    Returns:
        Series: 1 (LONG), -1 (SHORT), 0 (NEUTRAL)
    """
    c = df["close"].values
    atr = df["atr_14"].values
    labels = np.zeros(len(c))

    for i in range(len(c) - horizon):
        if pd.isna(atr[i]) or atr[i] <= 0 or c[i] <= 0:
            continue
        fwd_ret = (c[i + horizon] - c[i]) / c[i]
        # Adaptive threshold based on volatility
        threshold = max(0.0004, 0.3 * atr[i] / c[i])
        if fwd_ret > threshold:
            labels[i] = 1
        elif fwd_ret < -threshold:
            labels[i] = -1

    return pd.Series(labels, index=df.index)


def simulate_ml_trades(df: pd.DataFrame, predictions: np.ndarray,
                       threshold: float, horizon: int,
                       max_trades_per_session: int = 3) -> pd.DataFrame:
    """Simulate trading based on ML predictions.

    Entry: when P(long) > threshold → BUY, P(long) < (1-threshold) → SELL
    Exit: fixed horizon (hold for `horizon` bars) OR session end
    Cost: 0.96 pts per trade
    """
    trades = []
    c = df["close"].values
    dates = df["date"].values
    sessions = df["session"].values
    mins_vals = df["mins"].values

    in_position = False
    session_trades = 0
    last_session_key = None

    for i in range(len(predictions)):
        session_key = (dates[i], sessions[i])
        if session_key != last_session_key:
            session_trades = 0
            last_session_key = session_key

        if in_position:
            continue

        if session_trades >= max_trades_per_session:
            continue

        # Time filter: don't enter too close to session end
        if sessions[i] == "AM" and mins_vals[i] >= 11*60+20:
            continue
        if sessions[i] == "PM" and mins_vals[i] >= 14*60+20:
            continue

        prob = predictions[i]
        if np.isnan(prob):
            continue

        direction = 0
        if prob > threshold:
            direction = 1
        elif prob < (1 - threshold):
            direction = -1

        if direction == 0:
            continue

        # Enter trade
        entry_price = c[i]
        exit_idx = None

        # Hold for horizon bars or until session ends
        for j in range(i + 1, min(i + horizon + 1, len(df))):
            if dates[j] != dates[i] or sessions[j] != sessions[i]:
                exit_idx = j - 1
                break
            if (sessions[j] == "AM" and mins_vals[j] >= 11*60+25):
                exit_idx = j
                break
            if (sessions[j] == "PM" and mins_vals[j] >= 14*60+25):
                exit_idx = j
                break
            if j == i + horizon:
                exit_idx = j
                break

        if exit_idx is None:
            exit_idx = min(i + horizon, len(df) - 1)

        exit_price = c[exit_idx]
        pnl = direction * (exit_price - entry_price) - COST
        mfe_prices = df["high"].iloc[i+1:exit_idx+1] if direction == 1 else df["low"].iloc[i+1:exit_idx+1]
        if direction == 1 and len(mfe_prices) > 0:
            mfe = float(mfe_prices.max()) - entry_price
        elif direction == -1 and len(mfe_prices) > 0:
            mfe = entry_price - float(mfe_prices.min())
        else:
            mfe = 0

        trades.append({
            "date": str(dates[i]),
            "session": sessions[i],
            "time": str(df["time"].iloc[i]),
            "direction": "BUY" if direction == 1 else "SELL",
            "entry": entry_price,
            "exit": exit_price,
            "pnl": pnl,
            "mfe": mfe,
            "bars_held": exit_idx - i,
            "prob": prob,
        })

        session_trades += 1
        # Block next `horizon` bars from new entries
        in_position = True
        # Simple cooldown
        for k in range(i + 1, min(i + horizon + 1, len(df))):
            if k == exit_idx:
                in_position = False
                break

        in_position = False

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def walk_forward_experiment(df: pd.DataFrame, horizon: int = 6,
                            train_days: int = 60, test_days: int = 20):
    """Walk-forward validation for standalone ML signals."""
    from sklearn.ensemble import ExtraTreesClassifier

    # Compute features and labels
    df = compute_all_features(df)
    df["label"] = label_fixed_horizon(df, horizon=horizon)

    # Filter to only LONG/SHORT labeled bars for training
    # (neutral bars = no trade, used in test for realistic simulation)
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    X_all = df[available_features].values
    y_all = df["label"].values
    dates_all = df["date"].values

    # Walk-forward
    unique_dates = sorted(df["date"].unique())
    results_by_threshold = {}

    all_test_indices = []
    all_test_predictions = []

    fold = 0
    train_start_idx = 0

    while True:
        train_end_idx = train_start_idx + train_days
        test_end_idx = train_end_idx + test_days

        if test_end_idx > len(unique_dates):
            break

        train_dates = set(unique_dates[train_start_idx:train_end_idx])
        test_dates = set(unique_dates[train_end_idx:test_end_idx])

        train_mask = np.array([d in train_dates for d in dates_all])
        test_mask = np.array([d in test_dates for d in dates_all])

        # For training: only use labeled bars (label != 0)
        train_labeled = train_mask & (y_all != 0)
        X_train = X_all[train_labeled]
        y_train = (y_all[train_labeled] == 1).astype(int)  # binary: 1=long, 0=short

        if len(X_train) < 100:
            train_start_idx += test_days
            continue

        # Train
        X_train_clean = np.nan_to_num(X_train, nan=0.0)
        model = ExtraTreesClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            max_features="sqrt", class_weight="balanced", n_jobs=-1,
        )
        model.fit(X_train_clean, y_train)

        # Predict on ALL test bars (including neutral)
        X_test = np.nan_to_num(X_all[test_mask], nan=0.0)
        predictions = model.predict_proba(X_test)[:, 1]

        test_indices = np.where(test_mask)[0]
        all_test_indices.extend(test_indices.tolist())
        all_test_predictions.extend(predictions.tolist())

        fold += 1
        train_start_idx += test_days

    if not all_test_predictions:
        print("  ERROR: No test predictions generated")
        return None

    # Reconstruct full prediction array
    full_predictions = np.full(len(df), np.nan)
    for idx, pred in zip(all_test_indices, all_test_predictions):
        full_predictions[idx] = pred

    # Simulate trades at various thresholds
    print(f"\n  Walk-forward: {fold} folds, {len(all_test_predictions)} test bars with predictions")
    print(f"  Label distribution (train): LONG={int((y_all==1).sum())}, SHORT={int((y_all==-1).sum())}, NEUTRAL={int((y_all==0).sum())}")

    test_df = df.iloc[all_test_indices].copy().reset_index(drop=True)
    test_preds = np.array(all_test_predictions)
    test_n_days = test_df["date"].nunique()

    print(f"  Test period: {test_n_days} days")
    print(f"\n  {'Threshold':<10} {'Trades':<8} {'Trades/d':<10} {'WR%':<8} {'PnL':<10} {'PnL/d':<8} {'Avg_PnL':<10} {'PF':<6}")
    print(f"  {'-'*75}")

    all_results = []

    for thr in [0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70]:
        trades_df = simulate_ml_trades(test_df, test_preds, thr, horizon)

        if trades_df.empty:
            continue

        n_trades = len(trades_df)
        trades_per_day = n_trades / test_n_days
        wr = (trades_df["pnl"] > 0).mean() * 100
        total_pnl = trades_df["pnl"].sum()
        pnl_per_day = total_pnl / test_n_days
        avg_pnl = trades_df["pnl"].mean()
        wins_pnl = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        loss_pnl = abs(trades_df[trades_df["pnl"] <= 0]["pnl"].sum())
        pf = wins_pnl / loss_pnl if loss_pnl > 0 else 999

        print(f"  {thr:<10.2f} {n_trades:<8} {trades_per_day:<10.1f} {wr:<8.1f} "
              f"{total_pnl:<+10.1f} {pnl_per_day:<+8.2f} {avg_pnl:<+10.2f} {pf:<6.2f}")

        all_results.append({
            "threshold": thr,
            "horizon": horizon,
            "n_trades": n_trades,
            "trades_per_day": round(trades_per_day, 2),
            "win_rate": round(wr, 1),
            "total_pnl": round(total_pnl, 2),
            "pnl_per_day": round(pnl_per_day, 2),
            "avg_pnl": round(avg_pnl, 3),
            "profit_factor": round(pf, 2),
            "test_days": test_n_days,
            "folds": fold,
        })

        # Session breakdown for best config
        if thr == 0.58:
            print(f"\n    --- Session breakdown (thr={thr}) ---")
            for sess in ["AM", "PM"]:
                sub = trades_df[trades_df["session"] == sess]
                if len(sub) > 0:
                    s_wr = (sub["pnl"] > 0).mean() * 100
                    s_pnl = sub["pnl"].sum()
                    print(f"    {sess}: {len(sub)} trades, WR={s_wr:.1f}%, PnL={s_pnl:+.1f}")
            print(f"    --- Direction breakdown ---")
            for dir_name in ["BUY", "SELL"]:
                sub = trades_df[trades_df["direction"] == dir_name]
                if len(sub) > 0:
                    s_wr = (sub["pnl"] > 0).mean() * 100
                    s_pnl = sub["pnl"].sum()
                    print(f"    {dir_name}: {len(sub)} trades, WR={s_wr:.1f}%, PnL={s_pnl:+.1f}")
            print()

    return all_results


def main():
    print("=" * 80)
    print("STANDALONE ML SIGNAL GENERATOR — VN30F1M 5m")
    print("=" * 80)
    print("Independent of CB strategy. Pure ML direction prediction.")
    print(f"Cost per trade: {COST} pts")

    print("\n[1] Loading 5m data...")
    df = load("5m", days=180)
    n_days = df["date"].nunique()
    print(f"  {len(df)} bars, {n_days} trading days")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    all_experiment_results = {}

    # Test multiple horizons
    for horizon in [3, 6, 10]:
        bars_min = horizon * 5
        print(f"\n{'=' * 80}")
        print(f"HORIZON = {horizon} bars ({bars_min} minutes)")
        print(f"{'=' * 80}")

        results = walk_forward_experiment(df, horizon=horizon, train_days=60, test_days=20)
        if results:
            all_experiment_results[f"horizon_{horizon}"] = results

    # Save all results
    with open(REPORT_PATH, "w") as f:
        json.dump(all_experiment_results, f, indent=2)
    print(f"\nResults saved: {REPORT_PATH}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY — Best configs across all horizons")
    print(f"{'=' * 80}")
    print(f"  {'Horizon':<10} {'Threshold':<10} {'Trades/d':<10} {'WR%':<8} {'PnL/d':<10} {'PF':<6}")
    print(f"  {'-'*55}")

    for horizon_key, results in all_experiment_results.items():
        if results:
            best = max(results, key=lambda r: r["pnl_per_day"])
            print(f"  {horizon_key:<10} {best['threshold']:<10.2f} {best['trades_per_day']:<10.1f} "
                  f"{best['win_rate']:<8.1f} {best['pnl_per_day']:<+10.2f} {best['profit_factor']:<6.2f}")


if __name__ == "__main__":
    main()
