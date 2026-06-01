"""
120-Day Focused Backtest - Active Combos with Optimized Parameters
==================================================================
Tests: D/5m, J/5m, X/3m, X/5m, V/5m with gates + trailing + entry_limit_offset
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import yaml

from src.data_fetcher import DataFetcher
from src.signals import (
    COMBO_PRESETS,
    generate_combined_signals,
)

# ============== CONFIG ==============
SYMBOL = "VN30F1M"
BACKTEST_DAYS = 120
COMMISSION = 0.47  # points per side (entry + exit = 0.94 total)
POINT_VALUE = 100_000  # 1 point = 100,000 VND
VN_TZ = timezone(timedelta(hours=7))

# Load strategy config
with open("strategy_config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

COMBO_RISK = CFG.get("combo_risk", {})
COMBO_TF_MAP = CFG.get("combo_tf_map", {})

# Build mapping: short name -> full COMBO_PRESETS key
SHORT_TO_FULL = {}
for full_name in COMBO_PRESETS:
    short = full_name.split(":")[0].strip()
    SHORT_TO_FULL[short] = full_name

# Active combos from combo_tf_map
ACTIVE_PAIRS = []
for combo_short, tfs in COMBO_TF_MAP.items():
    for tf in tfs:
        ACTIVE_PAIRS.append((combo_short, tf))

print(f"Active pairs: {ACTIVE_PAIRS}")


def fetch_data(fetcher, tf, days):
    now = datetime.now(VN_TZ)
    end = now.strftime("%Y-%m-%d")
    start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    df = fetcher.get_futures_ohlcv(SYMBOL, start, end, interval=tf)
    if df is not None and not df.empty:
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")
    return df


def get_enabled_from_combo(combo_short):
    full_name = SHORT_TO_FULL.get(combo_short, combo_short)
    preset = COMBO_PRESETS.get(full_name, {})
    enabled = {}
    for cond in preset.get("primary", []) + preset.get("confirm", []) + preset.get("gate", []):
        enabled[cond] = True
    return enabled, full_name


def simulate_trades(sig_df, combo_key, entry_limit_offset=0.0):
    """Simulate trades for a single combo with its specific parameters.
    
    Uses per-combo SL/TP/trailing/max_hold from strategy_config.yaml.
    entry_limit_offset: ATR multiplier for limit order offset (0 = market order at close).
    
    Next-bar confirmation: signal fires on bar i, entry only if bar i+1
    breaks bar i's high (BUY) or low (SELL). This filters ~50% of traps.
    """
    risk = COMBO_RISK.get(combo_key, {})
    sl_mult = risk.get("sl_atr_mult", 1.5)
    tp_mult = risk.get("tp_atr_mult", 3.0)
    max_hold = risk.get("max_hold", 20)
    trail_offset = risk.get("trailing_offset", 0.5)
    trail_step = risk.get("trailing_step", 0.3)
    min_conf = risk.get("min_confidence", 1)

    trades = []
    open_pos = None
    pending_signal = None  # waiting for next-bar confirmation
    n = len(sig_df)

    for i in range(n):
        row = sig_df.iloc[i]
        bar_high = float(row["high"])
        bar_low = float(row["low"])
        bar_close = float(row["close"])
        bar_time = row.name if isinstance(row.name, (pd.Timestamp, datetime)) else sig_df.index[i]

        # --- Check pending signal confirmation ---
        if pending_signal is not None and open_pos is None:
            ps = pending_signal
            confirmed = False
            # Dynamic offset based on signal bar strength
            offset = ps["dynamic_offset"]

            if ps["direction"] == 1 and bar_high > ps["trigger_price"]:
                # BUY confirmed: enter with dynamic pullback from trigger
                entry_price = ps["trigger_price"] - offset
                if bar_low <= entry_price:
                    confirmed = True
                else:
                    # Pullback didn't reach, enter at trigger
                    entry_price = ps["trigger_price"]
                    confirmed = True
            elif ps["direction"] == -1 and bar_low < ps["trigger_price"]:
                # SELL confirmed: enter with dynamic pullback from trigger
                entry_price = ps["trigger_price"] + offset
                if bar_high >= entry_price:
                    confirmed = True
                else:
                    entry_price = ps["trigger_price"]
                    confirmed = True

            if confirmed:
                atr = ps["atr"]
                direction = ps["direction"]
                sl = entry_price - direction * sl_mult * atr
                tp = entry_price + direction * tp_mult * atr

                # Skip if TP distance < 3 points
                if abs(tp - entry_price) >= 3.0:
                    open_pos = {
                        "entry": entry_price,
                        "entry_time": bar_time,
                        "sl": sl,
                        "tp": tp,
                        "trailing_sl": sl,
                        "direction": direction,
                        "atr": atr,
                        "bars_held": 0,
                        "exit_price": None,
                        "exit_reason": None,
                    }
            # Pending expires after 1 bar (no confirmation = trap)
            pending_signal = None

        # --- Update open position ---
        if open_pos is not None:
            open_pos["bars_held"] += 1
            d = open_pos["direction"]
            atr = open_pos["atr"]

            # Check SL hit
            if d == 1:  # LONG
                if bar_low <= open_pos["trailing_sl"]:
                    open_pos["exit_price"] = open_pos["trailing_sl"]
                    open_pos["exit_reason"] = "SL"
                elif bar_high >= open_pos["tp"]:
                    open_pos["exit_price"] = open_pos["tp"]
                    open_pos["exit_reason"] = "TP"
                else:
                    # Trailing SL
                    unrealized = bar_high - open_pos["entry"]
                    if unrealized >= trail_offset * atr:
                        new_trail = bar_high - trail_step * atr
                        if new_trail > open_pos["trailing_sl"]:
                            open_pos["trailing_sl"] = new_trail
            else:  # SHORT
                if bar_high >= open_pos["trailing_sl"]:
                    open_pos["exit_price"] = open_pos["trailing_sl"]
                    open_pos["exit_reason"] = "SL"
                elif bar_low <= open_pos["tp"]:
                    open_pos["exit_price"] = open_pos["tp"]
                    open_pos["exit_reason"] = "TP"
                else:
                    unrealized = open_pos["entry"] - bar_low
                    if unrealized >= trail_offset * atr:
                        new_trail = bar_low + trail_step * atr
                        if new_trail < open_pos["trailing_sl"]:
                            open_pos["trailing_sl"] = new_trail

            # Max hold timeout
            if open_pos.get("exit_price") is None and open_pos["bars_held"] >= max_hold:
                open_pos["exit_price"] = bar_close
                open_pos["exit_reason"] = "TIMEOUT"

            # Close position if exit triggered
            if open_pos.get("exit_price") is not None:
                pnl = open_pos["direction"] * (open_pos["exit_price"] - open_pos["entry"]) - 2 * COMMISSION
                trades.append({
                    "entry_time": open_pos["entry_time"],
                    "exit_time": bar_time,
                    "direction": "BUY" if open_pos["direction"] == 1 else "SELL",
                    "entry": open_pos["entry"],
                    "exit": open_pos["exit_price"],
                    "atr": open_pos["atr"],
                    "bars_held": open_pos["bars_held"],
                    "exit_reason": open_pos["exit_reason"],
                    "pnl_points": round(pnl, 2),
                    "pnl_vnd": round(pnl * POINT_VALUE, 0),
                })
                open_pos = None

        # --- Check for new signal (creates pending, NOT immediate entry) ---
        if open_pos is not None or pending_signal is not None:
            continue  # already in a trade or waiting for confirmation

        signal = int(row.get("signal", 0))
        confidence = int(row.get("signal_confidence", 0))

        if signal == 0 or confidence < min_conf:
            continue

        atr = float(row.get("atr", 0))
        if atr <= 0:
            continue

        direction = signal  # 1=BUY, -1=SELL

        # Exhaustion filter: skip if 2+ consecutive same-dir signals
        if i >= 2:
            same_count = 0
            for j in range(i - 1, max(0, i - 6), -1):
                prev_sig = int(sig_df.iloc[j].get("signal", 0))
                if prev_sig == direction:
                    same_count += 1
                else:
                    break
            if same_count >= 2:
                continue

        # Set pending signal - wait for next bar confirmation
        # BUY: next bar must break current bar's high
        # SELL: next bar must break current bar's high/low
        trigger = bar_high + 0.1 if direction == 1 else bar_low - 0.1

        # Dynamic limit offset based on signal bar strength + volume
        bar_open = float(row["open"])
        body = abs(bar_close - bar_open)
        bar_range = bar_high - bar_low
        body_ratio = body / max(bar_range, 0.01)

        # Volume ratio vs 20-bar average
        if i >= 20:
            vol_avg = sig_df.iloc[i-20:i]["volume"].astype(float).mean()
        else:
            vol_avg = float(row["volume"])
        vol_ratio = float(row["volume"]) / max(vol_avg, 1)

        # Formula: strong signal = small offset (enter quick), weak = larger offset
        # Strong: body_ratio > 0.7 OR vol > 3x → offset = 0.3 ATR
        # Normal: offset = 0.5 ATR
        # Weak: body_ratio < 0.5 → offset = 0.8 ATR
        if body_ratio >= 0.7 or vol_ratio >= 3.0:
            dyn_offset = 0.3 * atr
        elif body_ratio < 0.5:
            dyn_offset = 0.8 * atr
        else:
            dyn_offset = 0.5 * atr

        # Cap offset to reasonable range [0.3, 3.0] pts
        dyn_offset = max(0.3, min(dyn_offset, 3.0))

        pending_signal = {
            "direction": direction,
            "trigger_price": trigger,
            "atr": atr,
            "signal_bar_idx": i,
            "dynamic_offset": dyn_offset,
        }

    # Close remaining position at end
    if open_pos is not None and n > 0:
        last_close = float(sig_df.iloc[-1]["close"])
        last_time = sig_df.index[-1]
        pnl = open_pos["direction"] * (last_close - open_pos["entry"]) - 2 * COMMISSION
        trades.append({
            "entry_time": open_pos["entry_time"],
            "exit_time": last_time,
            "direction": "BUY" if open_pos["direction"] == 1 else "SELL",
            "entry": open_pos["entry"],
            "exit": last_close,
            "atr": open_pos["atr"],
            "bars_held": open_pos["bars_held"],
            "exit_reason": "EOD",
            "pnl_points": round(pnl, 2),
            "pnl_vnd": round(pnl * POINT_VALUE, 0),
        })

    return trades


def compute_stats(trades):
    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl": 0, "profit_factor": 0, "avg_bars": 0}
    df = pd.DataFrame(trades)
    total = len(df)
    wins = (df["pnl_points"] > 0).sum()
    losses = (df["pnl_points"] <= 0).sum()
    wr = wins / total * 100

    gross_win = df.loc[df["pnl_points"] > 0, "pnl_points"].sum()
    gross_loss = abs(df.loc[df["pnl_points"] <= 0, "pnl_points"].sum())
    pf = gross_win / max(0.01, gross_loss)
    total_pnl = df["pnl_points"].sum()
    avg_bars = df["bars_held"].mean()

    return {
        "total": total,
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(wr, 1),
        "total_pnl": round(total_pnl, 1),
        "profit_factor": round(pf, 2),
        "avg_bars": round(avg_bars, 1),
    }


def main():
    print("=" * 70)
    print(f"  120-DAY BACKTEST - Active Combos (Optimized Params)")
    print(f"  Commission: {COMMISSION * 2} pts/round-trip")
    print("=" * 70)

    fetcher = DataFetcher()

    # Fetch data per timeframe
    tfs_needed = set(tf for _, tf in ACTIVE_PAIRS)
    data_cache = {}
    for tf in tfs_needed:
        print(f"\n  Fetching {SYMBOL} {tf} ({BACKTEST_DAYS} days)...")
        df = fetch_data(fetcher, tf, BACKTEST_DAYS)
        if df is not None and not df.empty:
            data_cache[tf] = df
            print(f"  -> {len(df)} bars, from {df.index[0]} to {df.index[-1]}")
        else:
            print(f"  -> NO DATA")

    # Run backtest per combo
    all_trades = {}  # key: "COMBO/TF" -> list of trades
    summary = []

    for combo, tf in ACTIVE_PAIRS:
        key = f"{combo}/{tf}"
        if tf not in data_cache:
            print(f"\n  SKIP {key}: no data")
            continue

        df = data_cache[tf].copy()
        enabled, full_name = get_enabled_from_combo(combo)
        sig_df = generate_combined_signals(
            df, fast_ma=10, slow_ma=20, rsi_period=7,
            oversold=35, overbought=70,
            macd_fast=12, macd_slow=26, macd_signal=9,
            vol_mult=1.5, enabled=enabled, combo_mode=full_name,
        )

        # Entry limit offset
        risk = COMBO_RISK.get(key, {})
        limit_offset = risk.get("entry_limit_offset", 0.0)

        trades = simulate_trades(sig_df, key, entry_limit_offset=limit_offset)
        all_trades[key] = trades
        stats = compute_stats(trades)
        stats["combo"] = key
        summary.append(stats)

        print(f"\n  {key}: {stats['total']}T, WR={stats['win_rate']}%, "
              f"P&L={stats['total_pnl']:+.1f}pts, PF={stats['profit_factor']}, "
              f"AvgBars={stats['avg_bars']}")

    # ============ OVERALL SUMMARY ============
    print("\n" + "=" * 70)
    print("  OVERALL 120-DAY SUMMARY")
    print("=" * 70)

    all_t = []
    for key, trades in all_trades.items():
        for t in trades:
            t["combo"] = key
        all_t.extend(trades)

    if not all_t:
        print("  No trades generated!")
        return

    df_all = pd.DataFrame(all_t)
    total = len(df_all)
    wins = (df_all["pnl_points"] > 0).sum()
    total_pnl = df_all["pnl_points"].sum()
    total_vnd = df_all["pnl_vnd"].sum()

    print(f"\n  Total Trades: {total}")
    print(f"  Win Rate: {wins/total*100:.1f}%")
    print(f"  Total P&L: {total_pnl:+.1f} pts ({total_vnd/1e6:+.1f}M VND)")
    print(f"  Avg P&L/trade: {total_pnl/total:+.2f} pts")

    # Per-combo breakdown
    print(f"\n  {'Combo':<10} {'Trades':>6} {'WR%':>6} {'P&L(pts)':>10} {'PF':>6} {'AvgBars':>8}")
    print(f"  {'-'*48}")
    for s in sorted(summary, key=lambda x: -x["total_pnl"]):
        print(f"  {s['combo']:<10} {s['total']:>6} {s['win_rate']:>5.1f}% "
              f"{s['total_pnl']:>+9.1f} {s['profit_factor']:>6.2f} {s['avg_bars']:>7.1f}")

    # Exit reason breakdown
    print(f"\n  Exit Reasons:")
    for reason, grp in df_all.groupby("exit_reason"):
        cnt = len(grp)
        pnl = grp["pnl_points"].sum()
        wr = (grp["pnl_points"] > 0).sum() / cnt * 100
        print(f"    {reason:<10}: {cnt:>3} trades, WR={wr:.0f}%, P&L={pnl:+.1f}")

    # ============ LAST 7 DAYS DETAIL ============
    print("\n" + "=" * 70)
    print("  LAST 7 DAYS - DETAILED TRADES")
    print("=" * 70)

    cutoff = datetime.now(VN_TZ) - timedelta(days=7)
    # Convert entry_time to comparable
    df_all["entry_time"] = pd.to_datetime(df_all["entry_time"])
    if df_all["entry_time"].dt.tz is None:
        df_all["entry_time"] = df_all["entry_time"].dt.tz_localize(VN_TZ)

    recent = df_all[df_all["entry_time"] >= cutoff].sort_values("entry_time")

    if recent.empty:
        print("  No trades in last 7 days.")
    else:
        print(f"\n  Found {len(recent)} trades in last 7 days:\n")
        print(f"  {'#':>2} {'Combo':<8} {'Dir':<5} {'Entry':>8} {'Exit':>8} "
              f"{'P&L':>7} {'Reason':<8} {'Bars':>4} {'Time'}")
        print(f"  {'-'*80}")

        running_pnl = 0
        for idx, (_, t) in enumerate(recent.iterrows(), 1):
            running_pnl += t["pnl_points"]
            entry_str = f"{t['entry']:.1f}"
            exit_str = f"{t['exit']:.1f}"
            time_str = t["entry_time"].strftime("%m/%d %H:%M")
            win_mark = "W" if t["pnl_points"] > 0 else "L"
            print(f"  {idx:>2} {t['combo']:<8} {t['direction']:<5} {entry_str:>8} {exit_str:>8} "
                  f"{t['pnl_points']:>+6.1f}{win_mark} {t['exit_reason']:<8} {t['bars_held']:>4} {time_str}")

        print(f"\n  7-Day Summary:")
        print(f"    Trades: {len(recent)}")
        print(f"    Wins: {(recent['pnl_points'] > 0).sum()}, Losses: {(recent['pnl_points'] <= 0).sum()}")
        print(f"    WR: {(recent['pnl_points'] > 0).sum()/len(recent)*100:.1f}%")
        print(f"    P&L: {recent['pnl_points'].sum():+.1f} pts ({recent['pnl_vnd'].sum()/1e6:+.1f}M VND)")
        print(f"    Best: {recent['pnl_points'].max():+.1f} pts")
        print(f"    Worst: {recent['pnl_points'].min():+.1f} pts")


if __name__ == "__main__":
    main()
