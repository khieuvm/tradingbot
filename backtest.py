"""
100-Day Backtest Engine (v2 - Enhanced)
========================================
Enhanced with freqtrade-inspired techniques:
- Cascading time-based exits (cut losers early)
- Confidence score rejection (skip weak signals)
- Max 3 contracts, same direction only
- MFI + OBV + Volume Ratio + TEMA indicators
- Trailing SL with offset activation

Strategies tested:
1. All 6 active combos (A, B, C, F, G, K) with new indicators
2. Freqtrade-inspired: RSI + TEMA + BB + MFI + OBV
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.data_fetcher import DataFetcher
from src.signals import (
    COMBO_PRESETS,
    generate_combined_signals,
)


# ============== CONFIG ==============
SYMBOL = "VN30F1M"
BACKTEST_DAYS = 100
TIMEFRAMES = ["5m", "15m"]
INITIAL_CAPITAL = 100_000_000  # 100M VND
MAX_CONTRACTS = 3  # Maximum 3 contracts at any time
POINT_VALUE = 100_000  # 1 point = 100,000 VND for VN30F
COMMISSION = 0.47  # points per side (entry + exit)

# TP/SL variants to test
SL_ATR_MULT_OPTIONS = [1.0, 1.5, 2.0]
TP_ATR_MULT_OPTIONS = [2.0, 3.0, 4.0]
MAX_HOLD_BARS = 30  # force exit after N bars

# Trailing SL config
TRAILING_SL_OFFSET = 0.5  # activate trailing after 0.5*ATR profit
TRAILING_SL_STEP = 0.3    # trail by 0.3*ATR below/above peak

# Cascading time-based exit thresholds (bars, min_pnl_pct)
# Cut losers progressively earlier
CASCADE_EXITS = [
    (8, -0.015),   # After 8 bars: cut if PnL < -1.5%
    (15, 0.0),     # After 15 bars: cut if PnL < 0 (red)
    (22, 0.005),   # After 22 bars: cut if PnL < +0.5%
]

# Confidence threshold: reject signals below this confidence level
MIN_CONFIDENCE = 2  # minimum signal_confidence to take a trade (1-3 scale)

VN_TZ = timezone(timedelta(hours=7))


def fetch_data(fetcher: DataFetcher, tf: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV data for backtesting."""
    now = datetime.now(VN_TZ)
    end = now.strftime("%Y-%m-%d")
    start = (now - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"  Fetching {SYMBOL} {tf} from {start} to {end}...")
    df = fetcher.get_futures_ohlcv(SYMBOL, start, end, interval=tf)
    if df is not None and not df.empty:
        print(f"  Got {len(df)} bars")
    else:
        print(f"  ERROR: No data returned")
    return df


def get_enabled_from_combo(combo_name: str) -> dict:
    """Build enabled dict from a combo preset."""
    preset = COMBO_PRESETS.get(combo_name, {})
    enabled = {}
    for cond in preset.get("primary", []) + preset.get("confirm", []) + preset.get("gate", []):
        enabled[cond] = True
    return enabled


def simulate_trades_multi(sig_df: pd.DataFrame, sl_mult: float, tp_mult: float,
                          trailing: bool = False, max_hold: int = MAX_HOLD_BARS,
                          max_contracts: int = MAX_CONTRACTS,
                          use_cascade: bool = True,
                          min_confidence: int = MIN_CONFIDENCE) -> list[dict]:
    """Simulate trades with multi-contract support (same direction only).

    Rules:
    - Max N contracts open simultaneously
    - All open positions must be same direction (all BUY or all SELL)
    - New signal in opposite direction: close all, then open new
    - Cascading time-based exits: cut losers early
    - Confidence filtering: skip weak signals
    - Trailing SL with offset activation
    """
    trades = []
    open_positions = []  # list of dicts
    n = len(sig_df)

    for i in range(n):
        row = sig_df.iloc[i]
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_close = float(row["close"])

        # --- Update all open positions ---
        closed_indices = []
        for pos_idx, pos in enumerate(open_positions):
            pos["bars_held"] += 1
            direction = pos["direction"]
            atr = pos["atr"]

            # --- Check SL/TP hits ---
            if direction == 1:  # LONG
                if bar_low <= pos["trailing_sl"]:
                    pos["exit_price"] = pos["trailing_sl"]
                    pos["exit_reason"] = "SL"
                    closed_indices.append(pos_idx)
                    continue
                if bar_high >= pos["tp"]:
                    pos["exit_price"] = pos["tp"]
                    pos["exit_reason"] = "TP"
                    closed_indices.append(pos_idx)
                    continue
                # Trailing SL update
                if trailing:
                    unrealized = bar_high - pos["entry"]
                    if unrealized >= TRAILING_SL_OFFSET * atr:
                        new_trail = bar_high - TRAILING_SL_STEP * atr
                        if new_trail > pos["trailing_sl"]:
                            pos["trailing_sl"] = new_trail
            else:  # SHORT
                if bar_high >= pos["trailing_sl"]:
                    pos["exit_price"] = pos["trailing_sl"]
                    pos["exit_reason"] = "SL"
                    closed_indices.append(pos_idx)
                    continue
                if bar_low <= pos["tp"]:
                    pos["exit_price"] = pos["tp"]
                    pos["exit_reason"] = "TP"
                    closed_indices.append(pos_idx)
                    continue
                # Trailing SL update
                if trailing:
                    unrealized = pos["entry"] - bar_low
                    if unrealized >= TRAILING_SL_OFFSET * atr:
                        new_trail = bar_low + TRAILING_SL_STEP * atr
                        if new_trail < pos["trailing_sl"]:
                            pos["trailing_sl"] = new_trail

            # --- Cascading time-based exit ---
            if use_cascade and pos_idx not in closed_indices:
                pnl_pct = direction * (bar_close - pos["entry"]) / pos["entry"]
                for bar_threshold, min_pnl in CASCADE_EXITS:
                    if pos["bars_held"] >= bar_threshold and pnl_pct < min_pnl:
                        pos["exit_price"] = bar_close
                        pos["exit_reason"] = f"CASCADE_{bar_threshold}b"
                        closed_indices.append(pos_idx)
                        break

            # --- Max hold timeout ---
            if pos_idx not in closed_indices and pos["bars_held"] >= max_hold:
                pos["exit_price"] = bar_close
                pos["exit_reason"] = "TIMEOUT"
                closed_indices.append(pos_idx)

        # --- Close positions and record trades ---
        for pos_idx in sorted(set(closed_indices), reverse=True):
            pos = open_positions.pop(pos_idx)
            pnl_points = pos["direction"] * (pos["exit_price"] - pos["entry"]) - 2 * COMMISSION
            pnl_vnd = pnl_points * POINT_VALUE
            pnl_pct = pos["direction"] * (pos["exit_price"] - pos["entry"]) / pos["entry"] * 100

            trades.append({
                "entry_idx": pos["entry_idx"],
                "exit_idx": i,
                "direction": "BUY" if pos["direction"] == 1 else "SELL",
                "entry": pos["entry"],
                "exit": pos["exit_price"],
                "sl": pos["sl"],
                "tp": pos["tp"],
                "atr": pos["atr"],
                "bars_held": pos["bars_held"],
                "exit_reason": pos["exit_reason"],
                "pnl_points": pnl_points,
                "pnl_vnd": pnl_vnd,
                "pnl_pct": pnl_pct,
            })

        # --- Check for new signal ---
        signal = int(row.get("signal", 0))
        confidence = int(row.get("signal_confidence", 0))

        if signal == 0:
            continue

        # Confidence filter
        if confidence < min_confidence:
            continue

        atr = float(row.get("atr", 0))
        if atr <= 0:
            continue

        direction = signal  # 1=BUY, -1=SELL

        # Check direction conflict
        if open_positions:
            current_direction = open_positions[0]["direction"]
            if current_direction != direction:
                # Close all existing positions (direction reversal)
                for pos in open_positions:
                    pnl_points = pos["direction"] * (bar_close - pos["entry"]) - 2 * COMMISSION
                    pnl_vnd = pnl_points * POINT_VALUE
                    pnl_pct = pos["direction"] * (bar_close - pos["entry"]) / pos["entry"] * 100
                    trades.append({
                        "entry_idx": pos["entry_idx"],
                        "exit_idx": i,
                        "direction": "BUY" if pos["direction"] == 1 else "SELL",
                        "entry": pos["entry"],
                        "exit": bar_close,
                        "sl": pos["sl"],
                        "tp": pos["tp"],
                        "atr": pos["atr"],
                        "bars_held": pos["bars_held"],
                        "exit_reason": "REVERSAL",
                        "pnl_points": pnl_points,
                        "pnl_vnd": pnl_vnd,
                        "pnl_pct": pnl_pct,
                    })
                open_positions.clear()

        # Open new position if under max contracts
        if len(open_positions) < max_contracts:
            entry = bar_close
            sl = entry - direction * sl_mult * atr
            tp = entry + direction * tp_mult * atr

            open_positions.append({
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "trailing_sl": sl,
                "direction": direction,
                "entry_idx": i,
                "atr": atr,
                "bars_held": 0,
            })

    # --- Close remaining open positions at end ---
    if open_positions and n > 0:
        last_close = float(sig_df.iloc[-1]["close"])
        for pos in open_positions:
            pnl_points = pos["direction"] * (last_close - pos["entry"]) - 2 * COMMISSION
            pnl_vnd = pnl_points * POINT_VALUE
            pnl_pct = pos["direction"] * (last_close - pos["entry"]) / pos["entry"] * 100
            trades.append({
                "entry_idx": pos["entry_idx"],
                "exit_idx": n - 1,
                "direction": "BUY" if pos["direction"] == 1 else "SELL",
                "entry": pos["entry"],
                "exit": last_close,
                "sl": pos["sl"],
                "tp": pos["tp"],
                "atr": pos["atr"],
                "bars_held": pos["bars_held"],
                "exit_reason": "EOD",
                "pnl_points": pnl_points,
                "pnl_vnd": pnl_vnd,
                "pnl_pct": pnl_pct,
            })

    return trades


def compute_stats(trades: list[dict]) -> dict:
    """Compute performance statistics from trade list."""
    if not trades:
        return {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
            "expectancy": 0, "total_pnl": 0, "max_dd_pct": 0,
            "sharpe": 0, "avg_bars": 0, "total_vnd": 0,
        }

    df = pd.DataFrame(trades)
    total = len(df)
    wins = (df["pnl_points"] > 0).sum()
    losses = (df["pnl_points"] < 0).sum()
    win_rate = wins / max(1, total) * 100

    avg_win = df.loc[df["pnl_points"] > 0, "pnl_points"].mean() if wins > 0 else 0
    avg_loss = abs(df.loc[df["pnl_points"] < 0, "pnl_points"].mean()) if losses > 0 else 0.01

    profit_factor = (avg_win * wins) / max(0.01, avg_loss * losses)
    expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)
    total_pnl = df["pnl_points"].sum()
    total_vnd = df["pnl_vnd"].sum()

    # Max Drawdown
    equity = df["pnl_points"].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min()
    max_dd_pct = max_dd / max(1, peak.max()) * 100 if peak.max() > 0 else 0

    # Sharpe (daily-ish approximation)
    returns = df["pnl_pct"].values
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    avg_bars = df["bars_held"].mean()

    return {
        "total": total,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(win_rate, 1),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "profit_factor": round(float(profit_factor), 2),
        "expectancy": round(float(expectancy), 2),
        "total_pnl": round(float(total_pnl), 1),
        "total_vnd": round(float(total_vnd), 0),
        "max_dd_pct": round(float(max_dd_pct), 1),
        "sharpe": round(float(sharpe), 2),
        "avg_bars": round(float(avg_bars), 1),
    }


# ============== FREQTRADE-INSPIRED STRATEGY (v2) ==============
def freqtrade_strategy_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced Freqtrade strategy using:
    - RSI cross + TEMA9 position/slope
    - BB percent for zone detection
    - MFI for volume-weighted confirmation
    - OBV trend for accumulation/distribution
    - ADX for trend strength
    """
    import pandas_ta as ta

    df = df.copy()

    # Indicators
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["tema9"] = ta.tema(df["close"], length=9)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["adx_val"] = ta.adx(df["high"], df["low"], df["close"], length=14).iloc[:, 0]

    # MFI
    try:
        df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"].astype(float), length=14)
    except Exception:
        df["mfi"] = 50.0

    # OBV
    try:
        df["obv"] = ta.obv(df["close"], df["volume"].astype(float))
    except Exception:
        df["obv"] = 0.0
    df["obv_ema"] = ta.ema(df["obv"], length=20)

    # Bollinger Bands
    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is not None:
        bb_cols = bb.columns.tolist()
        upper_col = next((c for c in bb_cols if "BBU" in c), bb_cols[0])
        lower_col = next((c for c in bb_cols if "BBL" in c), bb_cols[2])
        mid_col = next((c for c in bb_cols if "BBM" in c), bb_cols[1])
        df["bb_upper"] = bb[upper_col]
        df["bb_lower"] = bb[lower_col]
        df["bb_mid"] = bb[mid_col]
    else:
        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_upper"] = df["bb_mid"] + 2 * df["close"].rolling(20).std()
        df["bb_lower"] = df["bb_mid"] - 2 * df["close"].rolling(20).std()

    df["bb_percent"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # Volume ratio
    vol_ema = ta.ema(df["volume"].astype(float), length=20)
    df["volume_ratio"] = df["volume"].astype(float) / (vol_ema + 1e-10)

    # Signals
    df["signal"] = 0
    df["signal_confidence"] = 0

    # TEMA slope
    tema_rising = df["tema9"] > df["tema9"].shift(1)
    tema_falling = df["tema9"] < df["tema9"].shift(1)

    # RSI cross from oversold/overbought
    rsi_cross_up = (df["rsi"] > 30) & (df["rsi"].shift(1) <= 30)
    rsi_cross_down = (df["rsi"] > 70) & (df["rsi"].shift(1) <= 70)

    # Guards
    tema_below_bb = df["tema9"] <= df["bb_mid"]
    tema_above_bb = df["tema9"] > df["bb_mid"]
    adx_strong = df["adx_val"] > 20
    mfi_oversold = df["mfi"] < 35
    mfi_overbought = df["mfi"] > 65
    obv_bullish = df["obv"] > df["obv_ema"]
    obv_bearish = df["obv"] < df["obv_ema"]
    vol_ok = df["volume_ratio"] > 0.8
    bb_buy_zone = df["bb_percent"] < 0.4
    bb_sell_zone = df["bb_percent"] > 0.6

    # Entry LONG: RSI cross up + TEMA below BB mid + TEMA rising + MFI/OBV
    buy_signal = (rsi_cross_up & tema_below_bb & tema_rising & adx_strong &
                  (mfi_oversold | obv_bullish) & vol_ok & bb_buy_zone)

    # Entry SHORT: RSI cross down + TEMA above BB mid + TEMA falling + MFI/OBV
    sell_signal = (rsi_cross_down & tema_above_bb & tema_falling & adx_strong &
                   (mfi_overbought | obv_bearish) & vol_ok & bb_sell_zone)

    # Confidence scoring (1-3)
    buy_conf = (adx_strong.astype(int) + mfi_oversold.astype(int) +
                obv_bullish.astype(int) + (df["volume_ratio"] > 1.2).astype(int))
    sell_conf = (adx_strong.astype(int) + mfi_overbought.astype(int) +
                 obv_bearish.astype(int) + (df["volume_ratio"] > 1.2).astype(int))

    df.loc[buy_signal, "signal"] = 1
    df.loc[sell_signal, "signal"] = -1
    df.loc[buy_signal, "signal_confidence"] = buy_conf[buy_signal].clip(1, 3)
    df.loc[sell_signal, "signal_confidence"] = sell_conf[sell_signal].clip(1, 3)

    return df


# ============== MAIN BACKTEST LOOP ==============
def run_backtest():
    """Run full backtest across all combos and strategies."""
    print("=" * 70)
    print("  100-DAY BACKTEST ENGINE v2 - VN30F1M")
    print("  Max 3 contracts | Same direction | Cascading exits")
    print("  New: MFI + OBV + Volume Ratio + TEMA + Confidence Score")
    print("=" * 70)

    fetcher = DataFetcher()

    # Fetch data for each timeframe
    data_cache = {}
    for tf in TIMEFRAMES:
        df = fetch_data(fetcher, tf, BACKTEST_DAYS)
        if df is not None and not df.empty:
            data_cache[tf] = df
        else:
            print(f"  SKIP {tf}: No data")

    if not data_cache:
        print("ERROR: No data available for backtesting!")
        return

    # Results accumulator
    all_results = []

    # ===================== TEST ALL EXISTING COMBOS =====================
    print("\n" + "=" * 70)
    print("  PHASE 1: Testing All COMBO Presets (with new indicators)")
    print("=" * 70)

    active_combos = [name for name, preset in COMBO_PRESETS.items() if preset.get("primary")]

    for combo_name in active_combos:
        enabled = get_enabled_from_combo(combo_name)
        print(f"\n--- {combo_name} ---")

        for tf in TIMEFRAMES:
            if tf not in data_cache:
                continue

            df = data_cache[tf].copy()
            sig_df = generate_combined_signals(
                df, fast_ma=10, slow_ma=20, rsi_period=7,
                oversold=35, overbought=70,
                macd_fast=12, macd_slow=26, macd_signal=9,
                vol_mult=1.5, enabled=enabled, combo_mode=combo_name,
            )

            # Test with different SL/TP combinations
            for sl_m in SL_ATR_MULT_OPTIONS:
                for tp_m in TP_ATR_MULT_OPTIONS:
                    # With trailing + cascade
                    trades = simulate_trades_multi(
                        sig_df, sl_m, tp_m,
                        trailing=True, use_cascade=True, min_confidence=2
                    )
                    stats = compute_stats(trades)
                    stats["combo"] = combo_name
                    stats["tf"] = tf
                    stats["sl_mult"] = sl_m
                    stats["tp_mult"] = tp_m
                    stats["trailing"] = True
                    stats["cascade"] = True
                    stats["strategy"] = "combo_v2"
                    all_results.append(stats)

                    # Without cascade (baseline comparison)
                    trades_nc = simulate_trades_multi(
                        sig_df, sl_m, tp_m,
                        trailing=True, use_cascade=False, min_confidence=1
                    )
                    stats_nc = compute_stats(trades_nc)
                    stats_nc["combo"] = combo_name
                    stats_nc["tf"] = tf
                    stats_nc["sl_mult"] = sl_m
                    stats_nc["tp_mult"] = tp_m
                    stats_nc["trailing"] = True
                    stats_nc["cascade"] = False
                    stats_nc["strategy"] = "combo_baseline"
                    all_results.append(stats_nc)

    # ===================== FREQTRADE STRATEGY v2 =====================
    print("\n" + "=" * 70)
    print("  PHASE 2: Freqtrade v2 (RSI+TEMA+BB+MFI+OBV)")
    print("=" * 70)

    for tf in TIMEFRAMES:
        if tf not in data_cache:
            continue
        df = data_cache[tf].copy()
        sig_df = freqtrade_strategy_v2(df)

        for sl_m in SL_ATR_MULT_OPTIONS:
            for tp_m in TP_ATR_MULT_OPTIONS:
                trades = simulate_trades_multi(
                    sig_df, sl_m, tp_m,
                    trailing=True, use_cascade=True, min_confidence=2
                )
                stats = compute_stats(trades)
                stats["combo"] = "Freqtrade_v2"
                stats["tf"] = tf
                stats["sl_mult"] = sl_m
                stats["tp_mult"] = tp_m
                stats["trailing"] = True
                stats["cascade"] = True
                stats["strategy"] = "freqtrade_v2"
                all_results.append(stats)

    # ===================== ANALYSIS =====================
    print("\n" + "=" * 70)
    print("  RESULTS ANALYSIS")
    print("=" * 70)

    results_df = pd.DataFrame(all_results)

    # Filter out zero-trade results
    results_df = results_df[results_df["total"] > 0]

    if results_df.empty:
        print("No trades generated in any strategy!")
        return

    # Save full results
    results_df.to_csv("backtest_100d_v2_results.csv", index=False)
    print(f"\n  Full results saved to backtest_100d_v2_results.csv ({len(results_df)} rows)")

    # ---- TOP 20 by composite score (min 5 trades) ----
    qualified = results_df[results_df["total"] >= 5].copy()

    if qualified.empty:
        print("  Not enough trades (min 5) to rank strategies.")
        qualified = results_df[results_df["total"] >= 2].copy()

    if not qualified.empty:
        # Rank by composite score: WR * PF * Expectancy / (1 + |MaxDD|)
        qualified["score"] = (
            qualified["win_rate"] / 100 *
            qualified["profit_factor"].clip(0, 10) *
            qualified["expectancy"].clip(-50, 50) /
            (1 + qualified["max_dd_pct"].abs() / 10)
        )

        top20 = qualified.nlargest(20, "score")

        print("\n" + "-" * 100)
        print(f"  {'#':<3}{'STRATEGY':<38}{'TF':<5}{'SL':<5}{'TP':<5}"
              f"{'TRADES':<7}{'WR%':<7}{'PF':<6}{'EXPECT':<8}{'PnL pts':<10}{'MDD%':<7}{'VND(M)':<10}")
        print("-" * 100)

        for rank, (_, row) in enumerate(top20.iterrows(), 1):
            combo_short = row["combo"][:35]
            vnd_m = row["total_vnd"] / 1_000_000
            print(f"  {rank:<3}{combo_short:<38}{row['tf']:<5}{row['sl_mult']:<5}"
                  f"{row['tp_mult']:<5}{row['total']:<7}"
                  f"{row['win_rate']:<7}{row['profit_factor']:<6}"
                  f"{row['expectancy']:<8}{row['total_pnl']:<10}"
                  f"{row['max_dd_pct']:<7}{vnd_m:<10.1f}")

        print("-" * 100)

        # Best overall
        best = top20.iloc[0]
        print(f"\n  *** BEST STRATEGY ***")
        print(f"  Strategy: {best['combo']}")
        print(f"  TF:       {best['tf']}")
        print(f"  SL/TP:    {best['sl_mult']}x / {best['tp_mult']}x ATR")
        print(f"  Trades:   {best['total']} (W:{best['wins']}/L:{best['losses']})")
        print(f"  Win Rate: {best['win_rate']}%")
        print(f"  PF:       {best['profit_factor']}")
        print(f"  Expect:   {best['expectancy']} pts/trade")
        print(f"  Total PnL:{best['total_pnl']:.1f} pts = {best['total_vnd']/1e6:.1f}M VND")
        print(f"  Max DD:   {best['max_dd_pct']:.1f}%")
        print(f"  Sharpe:   {best['sharpe']}")

        # Summary by strategy type
        print("\n\n  === SUMMARY BY STRATEGY TYPE (avg per config) ===")
        print("-" * 80)
        summary = qualified.groupby("strategy").agg({
            "total": "mean",
            "win_rate": "mean",
            "profit_factor": "mean",
            "expectancy": "mean",
            "total_pnl": "mean",
            "total_vnd": "mean",
            "sharpe": "mean",
        }).round(2)
        summary["total_vnd"] = (summary["total_vnd"] / 1e6).round(1)
        summary.columns = ["avg_trades", "avg_WR%", "avg_PF", "avg_expect", "avg_pnl_pts", "avg_VND(M)", "avg_sharpe"]
        print(summary.to_string())

        # Best config per combo
        print("\n\n  === BEST CONFIG PER COMBO ===")
        print("-" * 100)
        best_per_combo = qualified.loc[qualified.groupby("combo")["score"].idxmax()]
        best_per_combo = best_per_combo.sort_values("score", ascending=False)

        print(f"  {'COMBO':<38}{'TF':<5}{'SL':<5}{'TP':<5}{'STRAT':<15}"
              f"{'WR%':<7}{'PF':<6}{'EXPECT':<8}{'TRADES':<7}{'VND(M)':<10}{'SCORE':<8}")
        print("-" * 100)
        for _, row in best_per_combo.iterrows():
            combo_short = row["combo"][:35]
            vnd_m = row["total_vnd"] / 1_000_000
            print(f"  {combo_short:<38}{row['tf']:<5}{row['sl_mult']:<5}"
                  f"{row['tp_mult']:<5}{row['strategy']:<15}{row['win_rate']:<7}"
                  f"{row['profit_factor']:<6}{row['expectancy']:<8}"
                  f"{row['total']:<7}{vnd_m:<10.1f}{row['score']:<8.2f}")

        # Cascade vs No-Cascade comparison
        print("\n\n  === CASCADE EXIT vs BASELINE ===")
        print("-" * 70)
        cascade_df = qualified[qualified["strategy"] == "combo_v2"]
        baseline_df = qualified[qualified["strategy"] == "combo_baseline"]
        if not cascade_df.empty and not baseline_df.empty:
            print(f"  {'Metric':<20}{'Cascade (v2)':<20}{'Baseline':<20}{'Improvement':<15}")
            print(f"  {'Win Rate':<20}{cascade_df['win_rate'].mean():<20.1f}{baseline_df['win_rate'].mean():<20.1f}"
                  f"{cascade_df['win_rate'].mean() - baseline_df['win_rate'].mean():<+15.1f}")
            print(f"  {'Profit Factor':<20}{cascade_df['profit_factor'].mean():<20.2f}{baseline_df['profit_factor'].mean():<20.2f}"
                  f"{cascade_df['profit_factor'].mean() - baseline_df['profit_factor'].mean():<+15.2f}")
            print(f"  {'Expectancy':<20}{cascade_df['expectancy'].mean():<20.2f}{baseline_df['expectancy'].mean():<20.2f}"
                  f"{cascade_df['expectancy'].mean() - baseline_df['expectancy'].mean():<+15.2f}")
            print(f"  {'Max DD%':<20}{cascade_df['max_dd_pct'].mean():<20.1f}{baseline_df['max_dd_pct'].mean():<20.1f}"
                  f"{cascade_df['max_dd_pct'].mean() - baseline_df['max_dd_pct'].mean():<+15.1f}")

    print("\n" + "=" * 70)
    print("  BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest()
