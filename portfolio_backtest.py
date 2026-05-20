"""
Portfolio Backtest Module
=========================
Simulates multi-combo portfolio trading with realistic constraints:
- Single direction at any time (all positions same side)
- Max N contracts simultaneously
- Flip cooldown after direction change
- 15m signal priority over 5m
- Minimum confidence filter

Usage:
    python portfolio_backtest.py [--days 44] [--no-f+] [--no-l]
"""

import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import yaml

from backtest import COMMISSION, POINT_VALUE, get_enabled_from_combo
from src.data_fetcher import DataFetcher
from src.signals import COMBO_PRESETS, generate_combined_signals

CONFIG_PATH = Path(__file__).parent / "strategy_config.yaml"


def load_combos_from_config():
    """Load active combos and their risk params from strategy_config.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    combo_tf_map = cfg.get("combo_tf_map", {})
    combo_risk = cfg.get("combo_risk", {})
    entry = cfg.get("entry", {})
    default_sl = entry.get("sl_atr_mult", 1.5)
    default_tp = entry.get("tp_atr_mult", 4.0)

    combos_config = {}
    for combo_name, tfs in combo_tf_map.items():
        risk = combo_risk.get(combo_name, {})
        combos_config[combo_name] = {
            "tfs": tfs,
            "sl": risk.get("sl_atr_mult", default_sl),
            "tp": risk.get("tp_atr_mult", default_tp),
        }
    return combos_config


def resolve_combo_names():
    """Map short combo names (G, F, etc.) to full COMBO_PRESETS keys."""
    name_map = {}
    for full_name in COMBO_PRESETS:
        short = full_name.split(":")[0].strip()
        name_map[short] = full_name
    return name_map


def generate_all_signals(combos_config, df5, df15, test_dates, min_confidence=2):
    """Generate filtered signals for all combos on their assigned timeframes."""
    name_map = resolve_combo_names()
    all_signals = []

    for short_name, cfg in combos_config.items():
        full_name = name_map.get(short_name)
        if not full_name:
            continue
        enabled = get_enabled_from_combo(full_name)
        for tf in cfg["tfs"]:
            df = df5.copy() if tf == "5m" else df15.copy()
            sig_df = generate_combined_signals(df, enabled=enabled, combo_mode=full_name)
            sig_df["time"] = pd.to_datetime(sig_df["time"])
            test_sigs = sig_df[
                (sig_df["time"].dt.date.isin(test_dates)) & (sig_df["signal"] != 0)
            ]
            for _, row in test_sigs.iterrows():
                conf = int(row.get("signal_confidence", 0))
                if conf < min_confidence:
                    continue
                all_signals.append(
                    {
                        "time": row["time"],
                        "combo": short_name,
                        "tf": tf,
                        "direction": int(row["signal"]),
                        "price": row["close"],
                        "atr": row["atr"],
                        "sl_mult": cfg["sl"],
                        "tp_mult": cfg["tp"],
                        "conf": conf,
                    }
                )

    all_signals.sort(key=lambda x: x["time"])
    return all_signals


def simulate_portfolio(
    all_signals,
    price_feed,
    max_contracts=3,
    flip_cooldown=3,
):
    """
    Run portfolio simulation with direction lock, flip cooldown, 15m priority.

    Returns list of closed trades.
    """
    open_positions = []
    closed_trades = []
    current_direction = 0
    cooldown_remaining = 0

    # Index signals by time for fast lookup
    signal_times = {}
    for sig in all_signals:
        t = sig["time"]
        if t not in signal_times:
            signal_times[t] = []
        signal_times[t].append(sig)

    for _, bar in price_feed.iterrows():
        bar_time = bar["time"]
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_close = bar["close"]

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # --- Check SL/TP exits ---
        to_close = []
        for i, pos in enumerate(open_positions):
            if pos["direction"] == 1:
                if bar_low <= pos["sl"]:
                    to_close.append((i, "SL", pos["sl"]))
                elif bar_high >= pos["tp"]:
                    to_close.append((i, "TP", pos["tp"]))
            else:
                if bar_high >= pos["sl"]:
                    to_close.append((i, "SL", pos["sl"]))
                elif bar_low <= pos["tp"]:
                    to_close.append((i, "TP", pos["tp"]))

        for i, reason, exit_p in sorted(to_close, reverse=True):
            pos = open_positions.pop(i)
            pnl = pos["direction"] * (exit_p - pos["entry"]) - 2 * COMMISSION
            closed_trades.append(
                {
                    "date": pos["entry_time"].strftime("%m/%d"),
                    "entry_time": pos["entry_time"],
                    "exit_time": bar_time,
                    "combo": pos["combo"],
                    "tf": pos["tf"],
                    "direction": pos["direction"],
                    "entry": pos["entry"],
                    "exit": exit_p,
                    "reason": reason,
                    "pnl_pts": pnl,
                    "pnl_vnd": pnl * POINT_VALUE,
                }
            )

        if not open_positions:
            current_direction = 0

        # --- Check signals at this bar ---
        signals_now = []
        for sig_time in list(signal_times.keys()):
            if bar_time <= sig_time < bar_time + pd.Timedelta(minutes=5):
                signals_now.extend(signal_times[sig_time])

        if not signals_now:
            continue

        # Priority: 15m > 5m, then by confidence
        sig_15m = [s for s in signals_now if s["tf"] == "15m"]
        sig_5m = [s for s in signals_now if s["tf"] == "5m"]
        if sig_15m:
            best_sig = max(sig_15m, key=lambda x: x["conf"])
        elif sig_5m:
            best_sig = max(sig_5m, key=lambda x: x["conf"])
        else:
            best_sig = signals_now[0]
        new_dir = best_sig["direction"]

        # --- Handle direction flip ---
        if current_direction != 0 and new_dir != current_direction:
            # Only flip if signal is 15m or conf >= 2
            if best_sig["tf"] != "15m" and best_sig["conf"] < 2:
                for sig_time in list(signal_times.keys()):
                    if bar_time <= sig_time < bar_time + pd.Timedelta(minutes=5):
                        del signal_times[sig_time]
                continue

            # Execute flip: close all positions at market
            for pos in open_positions:
                pnl = pos["direction"] * (bar_close - pos["entry"]) - 2 * COMMISSION
                closed_trades.append(
                    {
                        "date": pos["entry_time"].strftime("%m/%d"),
                        "entry_time": pos["entry_time"],
                        "exit_time": bar_time,
                        "combo": pos["combo"],
                        "tf": pos["tf"],
                        "direction": pos["direction"],
                        "entry": pos["entry"],
                        "exit": bar_close,
                        "reason": "FLIP",
                        "pnl_pts": pos["direction"] * (bar_close - pos["entry"])
                        - 2 * COMMISSION,
                        "pnl_vnd": pnl * POINT_VALUE,
                    }
                )
            open_positions.clear()
            current_direction = 0
            cooldown_remaining = flip_cooldown

        # --- Skip during cooldown ---
        if cooldown_remaining > 0:
            for sig_time in list(signal_times.keys()):
                if bar_time <= sig_time < bar_time + pd.Timedelta(minutes=5):
                    del signal_times[sig_time]
            continue

        # --- Open new position ---
        if len(open_positions) < max_contracts:
            entry = bar_close
            atr = best_sig["atr"]
            sl = entry - new_dir * best_sig["sl_mult"] * atr
            tp = entry + new_dir * best_sig["tp_mult"] * atr
            open_positions.append(
                {
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "direction": new_dir,
                    "combo": best_sig["combo"],
                    "tf": best_sig["tf"],
                    "entry_time": bar_time,
                    "atr": atr,
                }
            )
            current_direction = new_dir

        # Clean up processed signals
        for sig_time in list(signal_times.keys()):
            if bar_time <= sig_time < bar_time + pd.Timedelta(minutes=5):
                del signal_times[sig_time]

    # --- Close remaining positions at last bar (EOD) ---
    if open_positions:
        last_close = price_feed.iloc[-1]["close"]
        last_time = price_feed.iloc[-1]["time"]
        for pos in open_positions:
            pnl = pos["direction"] * (last_close - pos["entry"]) - 2 * COMMISSION
            closed_trades.append(
                {
                    "date": pos["entry_time"].strftime("%m/%d"),
                    "entry_time": pos["entry_time"],
                    "exit_time": last_time,
                    "combo": pos["combo"],
                    "tf": pos["tf"],
                    "direction": pos["direction"],
                    "entry": pos["entry"],
                    "exit": last_close,
                    "reason": "EOD",
                    "pnl_pts": pnl,
                    "pnl_vnd": pnl * POINT_VALUE,
                }
            )

    return closed_trades


def compute_portfolio_stats(closed_trades, n_days):
    """Compute summary statistics from closed trades."""
    if not closed_trades:
        return {"trades": 0}

    total_pnl = sum(t["pnl_vnd"] for t in closed_trades)
    wins = sum(1 for t in closed_trades if t["pnl_vnd"] > 0)
    losses = len(closed_trades) - wins

    daily_pnl = defaultdict(float)
    for t in closed_trades:
        daily_pnl[t["date"]] += t["pnl_vnd"]

    win_days = sum(1 for v in daily_pnl.values() if v > 0)
    loss_days = sum(1 for v in daily_pnl.values() if v <= 0)

    # Max drawdown
    cum_arr = []
    c = 0
    for d in sorted(daily_pnl.keys()):
        c += daily_pnl[d]
        cum_arr.append(c)
    peak = 0
    max_dd = 0
    for v in cum_arr:
        if v > peak:
            peak = v
        if v - peak < max_dd:
            max_dd = v - peak

    # Sharpe
    daily_vals = [daily_pnl[d] for d in sorted(daily_pnl.keys())]
    if len(daily_vals) > 1:
        sharpe = (np.mean(daily_vals) / np.std(daily_vals)) * np.sqrt(252)
    else:
        sharpe = 0

    # By combo
    combo_stats = {}
    for t in closed_trades:
        c = t["combo"]
        if c not in combo_stats:
            combo_stats[c] = {"trades": 0, "wins": 0, "pnl": 0}
        combo_stats[c]["trades"] += 1
        combo_stats[c]["pnl"] += t["pnl_vnd"]
        if t["pnl_vnd"] > 0:
            combo_stats[c]["wins"] += 1

    return {
        "trades": len(closed_trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / max(1, len(closed_trades)) * 100,
        "net_pnl": total_pnl,
        "avg_per_day": total_pnl / max(1, n_days),
        "avg_per_trade": total_pnl / max(1, len(closed_trades)),
        "max_drawdown": max_dd,
        "peak_equity": max(cum_arr) if cum_arr else 0,
        "sharpe": sharpe,
        "win_days": win_days,
        "loss_days": loss_days,
        "active_days": len(daily_pnl),
        "daily_pnl": dict(daily_pnl),
        "combo_stats": combo_stats,
    }


def print_report(stats, n_days, title="PORTFOLIO BACKTEST"):
    """Print formatted backtest report."""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)

    if stats["trades"] == 0:
        print("  No trades generated.")
        return

    # Weekly summary
    daily_pnl = stats["daily_pnl"]
    week_pnl = defaultdict(float)
    for d, pnl in daily_pnl.items():
        dt = pd.Timestamp(f"2026/{d}")
        wk = dt.isocalendar()[1]
        week_pnl[wk] += pnl

    print(f"\n  WEEKLY:")
    cum = 0
    for wk in sorted(week_pnl.keys()):
        cum += week_pnl[wk]
        flag = "W" if week_pnl[wk] > 0 else "L"
        print(f"    Wk{wk:2d}: {week_pnl[wk]:>+12,.0f}  (cum: {cum:>+13,.0f}) {flag}")

    print(f"\n  SUMMARY:")
    print(f"    Trades: {stats['trades']} | W:{stats['wins']} L:{stats['losses']} | WR: {stats['win_rate']:.1f}%")
    print(f"    Win Days: {stats['win_days']}/{stats['active_days']} ({stats['win_days']/max(1,stats['active_days'])*100:.0f}%)")
    print(f"    Net PnL: {stats['net_pnl']:+,.0f} VND")
    print(f"    Avg/Day: {stats['avg_per_day']:+,.0f} | Avg/Trade: {stats['avg_per_trade']:+,.0f}")
    print(f"    Max DD: {stats['max_drawdown']:,.0f} VND")
    print(f"    Peak: {stats['peak_equity']:+,.0f} VND")
    print(f"    Sharpe: {stats['sharpe']:.2f}")
    print(f"    Return on 100M: {stats['net_pnl']/100_000_000*100:.1f}%")

    print(f"\n  BY COMBO:")
    for c in sorted(stats["combo_stats"].keys(), key=lambda x: -stats["combo_stats"][x]["pnl"]):
        s = stats["combo_stats"][c]
        wr = s["wins"] / max(1, s["trades"]) * 100
        print(f"    {c:<6} {s['trades']:<4} trades  WR:{wr:<5.0f}%  {s['pnl']:>+13,.0f} VND")


def run_backtest(days=44, max_contracts=3, flip_cooldown=3, min_confidence=2):
    """Run full portfolio backtest."""
    fetcher = DataFetcher()
    combos_config = load_combos_from_config()

    print(f"Config: {list(combos_config.keys())}")
    print(f"Params: days={days}, max_contracts={max_contracts}, cooldown={flip_cooldown}, min_conf={min_confidence}")
    print("Fetching data...")

    df5 = fetcher.get_futures_ohlcv("VN30F1M", "2026-01-01", "2026-05-21", interval="5m")
    df15 = fetcher.get_futures_ohlcv("VN30F1M", "2026-01-01", "2026-05-21", interval="15m")
    df5["time"] = pd.to_datetime(df5["time"])
    df15["time"] = pd.to_datetime(df15["time"])

    all_dates = sorted(df5["time"].dt.date.unique())
    test_dates = all_dates[-days:] if len(all_dates) >= days else all_dates
    print(f"Test: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")

    print("Generating signals...")
    all_signals = generate_all_signals(combos_config, df5, df15, test_dates, min_confidence)
    print(f"Signals (conf>={min_confidence}): {len(all_signals)}")

    price_feed = df5[df5["time"].dt.date.isin(test_dates)].reset_index(drop=True)

    closed_trades = simulate_portfolio(
        all_signals, price_feed,
        max_contracts=max_contracts,
        flip_cooldown=flip_cooldown,
    )

    stats = compute_portfolio_stats(closed_trades, len(test_dates))
    title = f"PORTFOLIO ({len(test_dates)}-day, max {max_contracts} contracts, cooldown {flip_cooldown})"
    print_report(stats, len(test_dates), title)
    return closed_trades, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Portfolio Backtest")
    parser.add_argument("--days", type=int, default=44, help="Number of trading days")
    parser.add_argument("--max-contracts", type=int, default=3, help="Max simultaneous contracts")
    parser.add_argument("--cooldown", type=int, default=3, help="Flip cooldown bars")
    parser.add_argument("--min-conf", type=int, default=2, help="Minimum signal confidence")
    args = parser.parse_args()

    run_backtest(
        days=args.days,
        max_contracts=args.max_contracts,
        flip_cooldown=args.cooldown,
        min_confidence=args.min_conf,
    )
