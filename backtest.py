"""
90-Day Backtest Engine
======================
Compares all COMBO_PRESETS + optimized strategies derived from:
- Freqtrade approach: Trailing SL, time-based ROI, RSI guard filters
- Backtrader approach: Vectorized event-driven simulation
- Current project: Multi-condition combo scoring

Strategies tested:
1. All 11 existing combos (A, B, C, D, E, F, G, H, J, K, L)
2. Optimized combos with trailing SL
3. Freqtrade-inspired: RSI cross + TEMA + BB guard
4. Enhanced: ADX filter + ATR trailing + tiered exit

Output: Per-combo stats (WR, PF, Expectancy, Sharpe, MaxDD)
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Ensure project root is in path
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
BACKTEST_DAYS = 90
TIMEFRAMES = ["5m", "15m"]
INITIAL_CAPITAL = 100_000_000  # 100M VND
POSITION_SIZE = 1  # 1 contract VN30F
POINT_VALUE = 100_000  # 1 point = 100,000 VND for VN30F
COMMISSION = 0.47  # points per side (entry + exit)

# TP/SL variants to test
SL_ATR_MULT_OPTIONS = [1.0, 1.5, 2.0]
TP_ATR_MULT_OPTIONS = [2.0, 3.0, 4.0]
MAX_HOLD_BARS = 30  # force exit after N bars

# Trailing SL config (freqtrade-inspired)
TRAILING_SL_ENABLED = True
TRAILING_SL_OFFSET = 0.5  # activate trailing after 0.5*ATR profit
TRAILING_SL_STEP = 0.3    # trail by 0.3*ATR below/above peak

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


def simulate_trades(sig_df: pd.DataFrame, sl_mult: float, tp_mult: float,
                    trailing: bool = False, max_hold: int = MAX_HOLD_BARS) -> list[dict]:
    """Simulate trades on signal dataframe.

    For each signal bar:
    - Entry at close of signal bar
    - SL = entry -/+ sl_mult * ATR
    - TP = entry +/- tp_mult * ATR
    - Forward simulate bar by bar
    - Optional trailing SL
    - Force exit after max_hold bars

    Returns list of trade dicts.
    """
    trades = []
    i = 0
    n = len(sig_df)

    while i < n:
        row = sig_df.iloc[i]
        signal = int(row.get("signal", 0))

        if signal == 0:
            i += 1
            continue

        entry = float(row["close"])
        atr = float(row.get("atr", 0))
        if atr <= 0:
            i += 1
            continue

        direction = signal  # 1=BUY, -1=SELL
        sl = entry - direction * sl_mult * atr
        tp = entry + direction * tp_mult * atr
        trailing_sl = sl

        # Forward simulate
        exit_price = None
        exit_reason = ""
        bars_held = 0

        for j in range(i + 1, min(i + 1 + max_hold, n)):
            bars_held += 1
            bar = sig_df.iloc[j]
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            if direction == 1:  # LONG
                # Check SL hit (low touches SL)
                if bar_low <= trailing_sl:
                    exit_price = trailing_sl
                    exit_reason = "SL"
                    break
                # Check TP hit (high touches TP)
                if bar_high >= tp:
                    exit_price = tp
                    exit_reason = "TP"
                    break
                # Trailing SL update
                if trailing:
                    unrealized = bar_high - entry
                    if unrealized >= TRAILING_SL_OFFSET * atr:
                        new_trail = bar_high - TRAILING_SL_STEP * atr
                        if new_trail > trailing_sl:
                            trailing_sl = new_trail
            else:  # SHORT
                # Check SL hit (high touches SL)
                if bar_high >= trailing_sl:
                    exit_price = trailing_sl
                    exit_reason = "SL"
                    break
                # Check TP hit (low touches TP)
                if bar_low <= tp:
                    exit_price = tp
                    exit_reason = "TP"
                    break
                # Trailing SL update
                if trailing:
                    unrealized = entry - bar_low
                    if unrealized >= TRAILING_SL_OFFSET * atr:
                        new_trail = bar_low + TRAILING_SL_STEP * atr
                        if new_trail < trailing_sl:
                            trailing_sl = new_trail

        # Force exit at last bar close if no SL/TP hit
        if exit_price is None:
            if i + 1 < n:
                last_bar_idx = min(i + max_hold, n - 1)
                exit_price = float(sig_df.iloc[last_bar_idx]["close"])
                exit_reason = "TIMEOUT"
            else:
                i += 1
                continue

        # Calculate P&L
        pnl_points = direction * (exit_price - entry) - 2 * COMMISSION
        pnl_vnd = pnl_points * POINT_VALUE * POSITION_SIZE
        pnl_pct = direction * (exit_price - entry) / entry * 100

        trades.append({
            "entry_idx": i,
            "exit_idx": i + bars_held,
            "direction": "BUY" if direction == 1 else "SELL",
            "entry": entry,
            "exit": exit_price,
            "sl": sl,
            "tp": tp,
            "atr": atr,
            "bars_held": bars_held,
            "exit_reason": exit_reason,
            "pnl_points": pnl_points,
            "pnl_vnd": pnl_vnd,
            "pnl_pct": pnl_pct,
        })

        # Skip bars until trade exits (no overlapping trades)
        i = i + bars_held + 1

    return trades


def compute_stats(trades: list[dict]) -> dict:
    """Compute performance statistics from trade list."""
    if not trades:
        return {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
            "expectancy": 0, "total_pnl": 0, "max_dd_pct": 0,
            "sharpe": 0, "avg_bars": 0,
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
        "max_dd_pct": round(float(max_dd_pct), 1),
        "sharpe": round(float(sharpe), 2),
        "avg_bars": round(float(avg_bars), 1),
    }


# ============== FREQTRADE-INSPIRED STRATEGY ==============
def freqtrade_strategy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate signals using freqtrade SampleStrategy approach:
    - Entry LONG: RSI crosses above 30 + TEMA below BB mid + TEMA rising + Volume > 0
    - Entry SHORT: RSI crosses above 70 + TEMA above BB mid + TEMA falling + Volume > 0
    - Uses ADX > 25 as trend strength filter (from freqtrade best practices)
    """
    import pandas_ta as ta

    # Indicators
    df = df.copy()
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["tema"] = ta.tema(df["close"], length=9)
    df["adx_val"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is not None:
        bb_cols = bb.columns.tolist()
        mid_col = next((c for c in bb_cols if "BBM" in c), bb_cols[1])
        df["bb_mid"] = bb[mid_col]
    else:
        df["bb_mid"] = df["close"].rolling(20).mean()

    # Signals
    df["signal"] = 0
    df["signal_confidence"] = 0

    # RSI cross above 30 (from oversold)
    rsi_cross_up = (df["rsi"] > 30) & (df["rsi"].shift(1) <= 30)
    # RSI cross above 70 (overbought)
    rsi_cross_down = (df["rsi"] > 70) & (df["rsi"].shift(1) <= 70)

    # Guards
    tema_below_bb = df["tema"] <= df["bb_mid"]
    tema_above_bb = df["tema"] > df["bb_mid"]
    tema_rising = df["tema"] > df["tema"].shift(1)
    tema_falling = df["tema"] < df["tema"].shift(1)
    vol_ok = df["volume"] > 0
    adx_strong = df["adx_val"] > 25

    # Entry signals
    buy_signal = rsi_cross_up & tema_below_bb & tema_rising & vol_ok & adx_strong
    sell_signal = rsi_cross_down & tema_above_bb & tema_falling & vol_ok & adx_strong

    df.loc[buy_signal, "signal"] = 1
    df.loc[sell_signal, "signal"] = -1
    df.loc[buy_signal | sell_signal, "signal_confidence"] = 3

    return df


# ============== ENHANCED OPTIMAL STRATEGY ==============
def enhanced_strategy_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced strategy combining best practices from all repos:

    From Freqtrade:
    - Trailing SL with offset activation
    - Time-based ROI (exit after N bars if in profit)
    - ADX trend filter (only trade when ADX > 25)

    From current project (best combos):
    - EMA Ribbon alignment (trend direction)
    - BB Squeeze detection (volatility expansion)
    - Supertrend confirmation

    Entry LONG:
    - EMA5 > EMA12 > EMA21 (trend aligned)
    - Supertrend bullish
    - BB Width expanding (squeeze breakout)
    - ADX > 25 (strong trend)
    - RSI between 40-65 (not overbought)

    Entry SHORT: (inverse)
    """
    import pandas_ta as ta

    df = df.copy()

    # Core indicators
    df["ema5"] = ta.ema(df["close"], length=5)
    df["ema12"] = ta.ema(df["close"], length=12)
    df["ema21"] = ta.ema(df["close"], length=21)
    df["rsi"] = ta.rsi(df["close"], length=7)
    df["adx_val"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Bollinger
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

    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_width_min20"] = df["bb_width"].rolling(20).min()
    df["bb_squeeze"] = df["bb_width"] <= df["bb_width_min20"] * 1.1  # near squeeze
    df["bb_expanding"] = df["bb_width"] > df["bb_width"].shift(1)  # expanding

    # Supertrend
    try:
        st = ta.supertrend(df["high"], df["low"], df["close"], length=7, multiplier=3.0)
        if st is not None:
            st_dir_col = next((c for c in st.columns if "SUPERTd" in c), None)
            df["st_dir"] = st[st_dir_col].fillna(0) if st_dir_col else 0
        else:
            df["st_dir"] = 0
    except Exception:
        df["st_dir"] = 0

    # MACD for momentum confirmation
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        macd_cols = macd.columns.tolist()
        macd_line_col = next((c for c in macd_cols if "MACD_" in c and "s" not in c.lower() and "h" not in c.lower()), macd_cols[0])
        macd_sig_col = next((c for c in macd_cols if "MACDs" in c), macd_cols[1])
        df["macd_line"] = macd[macd_line_col]
        df["macd_sig"] = macd[macd_sig_col]
    else:
        df["macd_line"] = 0
        df["macd_sig"] = 0

    # Signal generation
    df["signal"] = 0
    df["signal_confidence"] = 0

    # EMA Ribbon aligned
    ema_bull = (df["ema5"] > df["ema12"]) & (df["ema12"] > df["ema21"])
    ema_bear = (df["ema5"] < df["ema12"]) & (df["ema12"] < df["ema21"])

    # Supertrend direction
    st_bull = df["st_dir"] == 1
    st_bear = df["st_dir"] == -1

    # BB expanding (breakout from squeeze)
    bb_breakout = df["bb_expanding"]

    # ADX strong
    adx_ok = df["adx_val"] > 25

    # RSI filter (not overbought/oversold)
    rsi_buy_ok = (df["rsi"] > 35) & (df["rsi"] < 65)
    rsi_sell_ok = (df["rsi"] > 35) & (df["rsi"] < 65)

    # MACD momentum
    macd_bull = df["macd_line"] > df["macd_sig"]
    macd_bear = df["macd_line"] < df["macd_sig"]

    # Combined entry
    buy_signal = ema_bull & st_bull & bb_breakout & adx_ok & rsi_buy_ok & macd_bull
    sell_signal = ema_bear & st_bear & bb_breakout & adx_ok & rsi_sell_ok & macd_bear

    # Confidence scoring
    buy_conf = (ema_bull.astype(int) + st_bull.astype(int) + adx_ok.astype(int) +
                macd_bull.astype(int) + bb_breakout.astype(int))
    sell_conf = (ema_bear.astype(int) + st_bear.astype(int) + adx_ok.astype(int) +
                 macd_bear.astype(int) + bb_breakout.astype(int))

    df.loc[buy_signal, "signal"] = 1
    df.loc[sell_signal, "signal"] = -1
    df.loc[buy_signal, "signal_confidence"] = buy_conf[buy_signal]
    df.loc[sell_signal, "signal_confidence"] = sell_conf[sell_signal]

    return df


# ============== MAIN BACKTEST LOOP ==============
def run_backtest():
    """Run full backtest across all combos and strategies."""
    print("=" * 70)
    print("  90-DAY BACKTEST ENGINE - VN30F1M")
    print("  Comparing: 11 Combos + Freqtrade Strategy + Enhanced Strategy")
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
    print("  PHASE 1: Testing All 11 COMBO Presets")
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

            # Test with different SL/TP
            for sl_m in SL_ATR_MULT_OPTIONS:
                for tp_m in TP_ATR_MULT_OPTIONS:
                    # Without trailing
                    trades = simulate_trades(sig_df, sl_m, tp_m, trailing=False)
                    stats = compute_stats(trades)
                    stats["combo"] = combo_name
                    stats["tf"] = tf
                    stats["sl_mult"] = sl_m
                    stats["tp_mult"] = tp_m
                    stats["trailing"] = False
                    stats["strategy"] = "combo"
                    all_results.append(stats)

                    # With trailing
                    trades_t = simulate_trades(sig_df, sl_m, tp_m, trailing=True)
                    stats_t = compute_stats(trades_t)
                    stats_t["combo"] = combo_name
                    stats_t["tf"] = tf
                    stats_t["sl_mult"] = sl_m
                    stats_t["tp_mult"] = tp_m
                    stats_t["trailing"] = True
                    stats_t["strategy"] = "combo+trail"
                    all_results.append(stats_t)

    # ===================== FREQTRADE STRATEGY =====================
    print("\n" + "=" * 70)
    print("  PHASE 2: Freqtrade-Inspired Strategy (RSI + TEMA + BB + ADX)")
    print("=" * 70)

    for tf in TIMEFRAMES:
        if tf not in data_cache:
            continue
        df = data_cache[tf].copy()
        sig_df = freqtrade_strategy_signals(df)

        for sl_m in SL_ATR_MULT_OPTIONS:
            for tp_m in TP_ATR_MULT_OPTIONS:
                trades = simulate_trades(sig_df, sl_m, tp_m, trailing=False)
                stats = compute_stats(trades)
                stats["combo"] = "Freqtrade RSI+TEMA"
                stats["tf"] = tf
                stats["sl_mult"] = sl_m
                stats["tp_mult"] = tp_m
                stats["trailing"] = False
                stats["strategy"] = "freqtrade"
                all_results.append(stats)

                trades_t = simulate_trades(sig_df, sl_m, tp_m, trailing=True)
                stats_t = compute_stats(trades_t)
                stats_t["combo"] = "Freqtrade RSI+TEMA"
                stats_t["tf"] = tf
                stats_t["sl_mult"] = sl_m
                stats_t["tp_mult"] = tp_m
                stats_t["trailing"] = True
                stats_t["strategy"] = "freqtrade+trail"
                all_results.append(stats_t)

    # ===================== ENHANCED STRATEGY =====================
    print("\n" + "=" * 70)
    print("  PHASE 3: Enhanced Optimal Strategy (EMA+ST+BB+ADX+MACD)")
    print("=" * 70)

    for tf in TIMEFRAMES:
        if tf not in data_cache:
            continue
        df = data_cache[tf].copy()
        sig_df = enhanced_strategy_signals(df)

        for sl_m in SL_ATR_MULT_OPTIONS:
            for tp_m in TP_ATR_MULT_OPTIONS:
                trades = simulate_trades(sig_df, sl_m, tp_m, trailing=False)
                stats = compute_stats(trades)
                stats["combo"] = "Enhanced (EMA+ST+BB+ADX)"
                stats["tf"] = tf
                stats["sl_mult"] = sl_m
                stats["tp_mult"] = tp_m
                stats["trailing"] = False
                stats["strategy"] = "enhanced"
                all_results.append(stats)

                trades_t = simulate_trades(sig_df, sl_m, tp_m, trailing=True)
                stats_t = compute_stats(trades_t)
                stats_t["combo"] = "Enhanced (EMA+ST+BB+ADX)"
                stats_t["tf"] = tf
                stats_t["sl_mult"] = sl_m
                stats_t["tp_mult"] = tp_m
                stats_t["trailing"] = True
                stats_t["strategy"] = "enhanced+trail"
                all_results.append(stats_t)

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
    results_df.to_csv("backtest_results_full.csv", index=False)
    print(f"\n  Full results saved to backtest_results_full.csv ({len(results_df)} rows)")

    # ---- TOP 20 by Win Rate (min 10 trades) ----
    qualified = results_df[results_df["total"] >= 10].copy()

    if qualified.empty:
        print("  Not enough trades (min 10) to rank strategies.")
        qualified = results_df[results_df["total"] >= 3].copy()

    if not qualified.empty:
        # Rank by composite score: WR * PF * Expectancy / (1 + |MaxDD|)
        qualified["score"] = (
            qualified["win_rate"] / 100 *
            qualified["profit_factor"].clip(0, 10) *
            qualified["expectancy"].clip(-50, 50) /
            (1 + qualified["max_dd_pct"].abs() / 10)
        )

        top20 = qualified.nlargest(20, "score")

        print("\n" + "-" * 90)
        print(f"  {'RANK':<5}{'STRATEGY':<35}{'TF':<5}{'SL':<5}{'TP':<5}{'TRAIL':<7}"
              f"{'TRADES':<7}{'WR%':<7}{'PF':<6}{'EXPECT':<8}{'P&L':<10}{'MDD%':<7}{'SHARPE':<7}")
        print("-" * 90)

        for rank, (_, row) in enumerate(top20.iterrows(), 1):
            trail_str = "YES" if row["trailing"] else "NO"
            combo_short = row["combo"][:32]
            print(f"  {rank:<5}{combo_short:<35}{row['tf']:<5}{row['sl_mult']:<5}"
                  f"{row['tp_mult']:<5}{trail_str:<7}{row['total']:<7}"
                  f"{row['win_rate']:<7}{row['profit_factor']:<6}"
                  f"{row['expectancy']:<8}{row['total_pnl']:<10}"
                  f"{row['max_dd_pct']:<7}{row['sharpe']:<7}")

        print("-" * 90)

        # Best overall
        best = top20.iloc[0]
        print(f"\n  *** BEST STRATEGY ***")
        print(f"  Combo:   {best['combo']}")
        print(f"  TF:      {best['tf']}")
        print(f"  SL/TP:   {best['sl_mult']}x / {best['tp_mult']}x ATR")
        print(f"  Trailing:{' YES' if best['trailing'] else ' NO'}")
        print(f"  Trades:  {best['total']}")
        print(f"  Win Rate:{best['win_rate']}%")
        print(f"  PF:      {best['profit_factor']}")
        print(f"  Expect:  {best['expectancy']} pts/trade")
        print(f"  Total PnL: {best['total_pnl']:.1f} pts")
        print(f"  Max DD:  {best['max_dd_pct']:.1f}%")
        print(f"  Sharpe:  {best['sharpe']}")

        # Summary by strategy type
        print("\n\n  === SUMMARY BY STRATEGY TYPE ===")
        print("-" * 70)
        summary = qualified.groupby("strategy").agg({
            "total": "sum",
            "win_rate": "mean",
            "profit_factor": "mean",
            "expectancy": "mean",
            "total_pnl": "sum",
            "sharpe": "mean",
        }).round(2)
        print(summary.to_string())

        # Summary by combo (best config per combo)
        print("\n\n  === BEST CONFIG PER COMBO ===")
        print("-" * 90)
        best_per_combo = qualified.loc[qualified.groupby("combo")["score"].idxmax()]
        best_per_combo = best_per_combo.sort_values("score", ascending=False)

        print(f"  {'COMBO':<35}{'TF':<5}{'SL':<5}{'TP':<5}{'TRAIL':<7}"
              f"{'WR%':<7}{'PF':<6}{'EXPECT':<8}{'TRADES':<7}{'SCORE':<8}")
        print("-" * 90)
        for _, row in best_per_combo.iterrows():
            trail_str = "YES" if row["trailing"] else "NO"
            combo_short = row["combo"][:32]
            print(f"  {combo_short:<35}{row['tf']:<5}{row['sl_mult']:<5}"
                  f"{row['tp_mult']:<5}{trail_str:<7}{row['win_rate']:<7}"
                  f"{row['profit_factor']:<6}{row['expectancy']:<8}"
                  f"{row['total']:<7}{row['score']:<8.2f}")

    print("\n" + "=" * 70)
    print("  BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_backtest()
