"""
Labeler: Generate labeled ML dataset from backtest simulation.

Runs the CB backtest engine, then at each signal bar extracts features and labels
the trade outcome (win/loss after cost). Outputs a CSV ready for training.

Usage:
    python -m ml.labeler
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from backtest.engine import load, detect_comp, dedup, COST
from ml.features import compute_indicators, extract_features_at, META_LABEL_FEATURES
from src.strategy_config import get_combo_config, get_session_params


def generate_labeled_dataset(days=180, save_path="ml/data/cb_trades_labeled.csv"):
    """Run CB backtest and extract features + labels for each trade.

    Returns:
        DataFrame with columns: META_LABEL_FEATURES + [label, pnl, mfe, direction, session, date, time]
    """
    print("Loading 5m data...")
    df = load("5m", days=days)
    n_days = df["date"].nunique()
    print(f"  {len(df)} bars, {n_days} trading days")

    # Compute ML indicators
    print("Computing indicators...")
    df = compute_indicators(df)

    # CB config from YAML
    cb_cfg = get_combo_config("CB")
    sp = get_session_params()
    comp_cfg = cb_cfg.get("compression", {})
    ef_cfg = cb_cfg.get("entry_filter", {})

    n_bars = comp_cfg.get("n_bars", 3)
    threshold = comp_cfg.get("threshold", 0.7)
    atr_min = ef_cfg.get("atr_min", 2.5)
    atr_max = ef_cfg.get("atr_max", 4.5)
    atr_min_pm = ef_cfg.get("atr_min_pm", atr_min)
    atr_max_pm = ef_cfg.get("atr_max_pm", atr_max)
    dedup_bars = cb_cfg.get("dedup_bars", 5)

    # Dead zones
    dead_zones = []
    for slot in cb_cfg.get("dead_zones", []):
        parts = str(slot).split("-")
        if len(parts) == 2:
            def _parse(s):
                p = s.strip().split(":")
                return int(p[0]) * 60 + int(p[1])
            dead_zones.append((_parse(parts[0]), _parse(parts[1])))

    # Detect compression
    print("Detecting compression signals...")
    df = detect_comp(df, n_bars=n_bars, threshold=threshold)

    # Apply filters
    filt_am = (df["atr_14"] >= atr_min) & (df["atr_14"] <= atr_max) & (df["rsi_14"] < 70).fillna(True)
    filt_pm = (df["atr_14"] >= atr_min_pm) & (df["atr_14"] <= atr_max_pm) & (df["rsi_14"] < 70).fillna(True)

    am_time = (df["mins"] >= 9 * 60 + 15) & (df["mins"] <= 10 * 60 + 45)
    for slot_start, slot_end in dead_zones:
        am_time = am_time & ~((df["mins"] >= slot_start) & (df["mins"] < slot_end))

    pm_time = (df["mins"] >= 13 * 60 + 15) & (df["mins"] <= 14 * 60 + 15)

    mask_am = dedup(df["comp"] & am_time & filt_am, dedup_bars)
    mask_pm = dedup(df["comp"] & pm_time & filt_pm, dedup_bars)

    # Session params for trade simulation
    am_sp = sp.get("AM", {})
    pm_sp = sp.get("PM", {})

    am_sl = float(am_sp.get("sl_atr_mult", 1.5))
    am_trail = float(am_sp.get("trail_activate_pts", 7.0))
    am_mult = float(am_sp.get("trail_atr_mult", 2.5))
    am_hold = int(am_sp.get("max_hold_bars", 30))
    pm_sl = float(pm_sp.get("sl_atr_mult", 1.2))
    pm_trail = float(pm_sp.get("trail_activate_pts", 6.0))
    pm_mult = float(pm_sp.get("trail_atr_mult", 1.0))
    pm_hold = int(pm_sp.get("max_hold_bars", 8))
    be_trigger = float(am_sp.get("be_trigger_pts", 3.0))
    be_atr_min = float(am_sp.get("be_atr_min", 3.5))
    be_partial = float(am_sp.get("be_partial_pts", 1.0))

    # Simulate each trade and extract features
    print("Simulating trades and extracting features...")
    records = []

    def _simulate_and_label(idx, sl_mult, trail_activate, trail_mult, max_hold, session):
        """Simulate one trade, return (trade_result_dict, features_dict) or None."""
        row = df.iloc[idx]
        atr = float(row["atr_14"])
        if pd.isna(atr) or atr <= 0:
            return None

        if idx + 1 >= len(df):
            return None
        nxt = df.iloc[idx + 1]
        if nxt["date"] != row["date"] or nxt["session"] != row["session"]:
            return None

        direction = 1 if nxt["close"] > row["close"] else -1
        entry = float(row["close"])
        sl = entry - direction * sl_mult * atr
        best = entry
        trail_on = False
        be_done = False
        exit_p = None
        exit_r = None
        bars = 0

        for jp in range(idx + 1, min(idx + 1 + max_hold, len(df))):
            b = df.iloc[jp]
            if b["date"] != row["date"] or b["session"] != row["session"]:
                exit_p = float(df.iloc[jp - 1]["close"])
                exit_r = "SESSION"
                break
            if (b["session"] == "PM" and b["mins"] >= 14 * 60 + 25) or \
               (b["session"] == "AM" and b["mins"] >= 11 * 60 + 25):
                exit_p = float(b["close"])
                exit_r = "SESSION"
                break
            bars += 1
            if direction == 1:
                if b["low"] <= sl:
                    exit_p = sl
                    exit_r = "BE" if be_done else "SL"
                    break
                if b["high"] > best:
                    best = float(b["high"])
                mfe = best - entry
                if not be_done and mfe >= be_trigger and atr >= be_atr_min:
                    be_done = True
                    sl = max(sl, entry + be_partial)
                if mfe >= trail_activate:
                    trail_on = True
                if trail_on:
                    sl = max(sl, best - trail_mult * atr)
                    if b["low"] <= sl:
                        exit_p = sl
                        exit_r = "TRAIL"
                        break
            else:
                if b["high"] >= sl:
                    exit_p = sl
                    exit_r = "BE" if be_done else "SL"
                    break
                if b["low"] < best:
                    best = float(b["low"])
                mfe = entry - best
                if not be_done and mfe >= be_trigger and atr >= be_atr_min:
                    be_done = True
                    sl = min(sl, entry - be_partial)
                if mfe >= trail_activate:
                    trail_on = True
                if trail_on:
                    sl = min(sl, best + trail_mult * atr)
                    if b["high"] >= sl:
                        exit_p = sl
                        exit_r = "TRAIL"
                        break

        if exit_p is None:
            exit_p = entry
            exit_r = "MAX_HOLD"

        mfe_f = (best - entry) if direction == 1 else (entry - best)
        pnl = direction * (exit_p - entry) - COST

        # Extract features at signal bar
        features = extract_features_at(df, idx, direction)
        if features is None:
            return None

        features["label"] = 1 if pnl > 0 else 0
        features["pnl"] = pnl
        features["mfe"] = mfe_f
        features["direction"] = "BUY" if direction == 1 else "SELL"
        features["session"] = session
        features["exit_reason"] = exit_r
        features["date"] = str(row["date"])
        features["time"] = str(row["time"])
        features["bars_held"] = bars
        features["entry_price"] = entry

        return features

    # Process AM signals
    am_indices = df.index[mask_am].tolist()
    for idx in am_indices:
        iloc_idx = df.index.get_loc(idx)
        result = _simulate_and_label(
            iloc_idx, am_sl, am_trail, am_mult, am_hold, "AM"
        )
        if result:
            records.append(result)

    # Process PM signals
    pm_indices = df.index[mask_pm].tolist()
    for idx in pm_indices:
        iloc_idx = df.index.get_loc(idx)
        result = _simulate_and_label(
            iloc_idx, pm_sl, pm_trail, pm_mult, pm_hold, "PM"
        )
        if result:
            records.append(result)

    if not records:
        print("ERROR: No trades generated!")
        return None

    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(["date", "time"]).reset_index(drop=True)

    # Summary
    n_trades = len(result_df)
    n_wins = (result_df["label"] == 1).sum()
    wr = n_wins / n_trades * 100
    total_pnl = result_df["pnl"].sum()
    print(f"\nDataset generated:")
    print(f"  Trades: {n_trades} | WR: {wr:.1f}% | PnL: {total_pnl:+.1f} pts")
    print(f"  Label distribution: {n_wins} wins, {n_trades - n_wins} losses")
    print(f"  AM: {(result_df['session']=='AM').sum()} | PM: {(result_df['session']=='PM').sum()}")

    # Save
    result_df.to_csv(save_path, index=False)
    print(f"  Saved to: {save_path}")

    return result_df


if __name__ == "__main__":
    generate_labeled_dataset()
