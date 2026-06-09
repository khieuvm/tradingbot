"""
Regime/Session Prediction for VN30F1M.

Predicts whether a trading session (AM/PM) will be favorable for CB trades.
Uses features computed at session open to predict session outcome.

Key insight: instead of 222 per-trade predictions (too few losers),
we predict per-session quality using bar-level features (258 sessions, 9720 bars).

Applications:
  - Skip trading on predicted "bad" sessions → avoid drawdown days
  - Reduce position sizing on uncertain sessions
  - Full size on high-confidence sessions

Usage:
    python -m ml.regime_prediction
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

import pandas_ta as ta

from backtest.engine import load, detect_comp, dedup, COST
from src.strategy_config import get_combo_config, get_session_params


REPORT_PATH = Path("ml/reports/regime_prediction_results.json")


def compute_session_features(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Compute per-session features from 5m OHLCV data.

    For each session (AM/PM on each date), compute features from:
    - Previous session's price action
    - Opening bars of current session
    - Multi-bar context (last 50 bars)

    Returns DataFrame with one row per session.
    """
    df = df_5m.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    df["mins"] = df["time"].dt.hour * 60 + df["time"].dt.minute
    df["session"] = np.where(df["mins"] < 12 * 60, "AM", "PM")

    # Compute indicators
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["atr_5"] = ta.atr(df["high"], df["low"], df["close"], length=5)
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["adx"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    sma20 = ta.sma(df["close"], length=20)
    std20 = df["close"].rolling(20).std()
    df["bb_width"] = (4 * std20 / sma20).where(sma20 > 0, 0)

    sessions = []
    grouped = df.groupby(["date", "session"])

    for (date, session), group in grouped:
        if len(group) < 5:
            continue

        first_bar_iloc = df.index.get_loc(group.index[0])
        if first_bar_iloc < 20:
            continue

        # Features from bars BEFORE session starts (look-back)
        lookback = df.iloc[max(0, first_bar_iloc - 20):first_bar_iloc]
        if len(lookback) < 10:
            continue

        lb_close = lookback["close"].values
        lb_atr = lookback["atr_14"].values
        lb_high = lookback["high"].values
        lb_low = lookback["low"].values

        # Session open features (first 2-3 bars)
        open_bars = group.iloc[:min(3, len(group))]
        session_open = float(group.iloc[0]["open"])

        features = {}

        # 1. Pre-session volatility state
        features["pre_atr_14"] = float(lookback["atr_14"].iloc[-1]) if not pd.isna(lookback["atr_14"].iloc[-1]) else 3.0
        features["pre_atr_5"] = float(lookback["atr_5"].iloc[-1]) if not pd.isna(lookback["atr_5"].iloc[-1]) else 3.0
        features["pre_atr_ratio"] = features["pre_atr_5"] / features["pre_atr_14"] if features["pre_atr_14"] > 0 else 1.0
        features["pre_bb_width"] = float(lookback["bb_width"].iloc[-1]) if not pd.isna(lookback["bb_width"].iloc[-1]) else 0.02

        # 2. Pre-session momentum
        features["pre_rsi"] = float(lookback["rsi_14"].iloc[-1]) if not pd.isna(lookback["rsi_14"].iloc[-1]) else 50
        features["pre_adx"] = float(lookback["adx"].iloc[-1]) if not pd.isna(lookback["adx"].iloc[-1]) else 20
        features["pre_ret_5"] = (lb_close[-1] - lb_close[-6]) / lb_close[-6] if len(lb_close) >= 6 and lb_close[-6] > 0 else 0
        features["pre_ret_10"] = (lb_close[-1] - lb_close[-11]) / lb_close[-11] if len(lb_close) >= 11 and lb_close[-11] > 0 else 0

        # 3. Pre-session trend
        ema20_val = float(lookback["ema20"].iloc[-1]) if not pd.isna(lookback["ema20"].iloc[-1]) else lb_close[-1]
        ema50_val = float(lookback["ema50"].iloc[-1]) if not pd.isna(lookback["ema50"].iloc[-1]) else lb_close[-1]
        features["pre_ema20_dist"] = (lb_close[-1] - ema20_val) / lb_close[-1]
        features["pre_ema50_dist"] = (lb_close[-1] - ema50_val) / lb_close[-1]
        features["pre_ema_spread"] = features["pre_ema20_dist"] - features["pre_ema50_dist"]

        # 4. Pre-session range characteristics
        pre_ranges = lb_high - lb_low
        features["pre_avg_range"] = float(pre_ranges[-10:].mean()) if len(pre_ranges) >= 10 else 2.0
        features["pre_range_contraction"] = float(pre_ranges[-3:].mean() / pre_ranges[-10:].mean()) if pre_ranges[-10:].mean() > 0 else 1.0

        # 5. Session-specific temporal features
        features["is_pm"] = 1 if session == "PM" else 0
        features["day_of_week"] = pd.Timestamp(date).dayofweek

        # 6. Opening bar features (first 1-2 bars of this session)
        features["open_range"] = float(open_bars["high"].max() - open_bars["low"].min())
        features["open_body"] = float(open_bars.iloc[-1]["close"] - open_bars.iloc[0]["open"])
        features["open_vol_ratio"] = float(open_bars["volume"].mean() / max(1, lookback["volume"].mean())) if "volume" in df.columns else 1.0

        # 7. Previous session outcome (if available, from same day AM → PM)
        features["prev_session_range"] = float(lookback["high"].max() - lookback["low"].min()) if len(lookback) >= 5 else 5.0
        features["prev_session_direction"] = 1 if lb_close[-1] > lb_close[0] else -1

        # 8. Compression state entering session
        ranges_before = pre_ranges[-5:] if len(pre_ranges) >= 5 else pre_ranges
        atr_at_open = features["pre_atr_14"]
        features["compression_ratio_entering"] = float(np.min(ranges_before) / atr_at_open) if atr_at_open > 0 and len(ranges_before) > 0 else 1.0

        # Meta
        features["date"] = str(date)
        features["session"] = session
        sessions.append(features)

    return pd.DataFrame(sessions)


def label_sessions(session_df: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
    """Label each session based on CB trade outcomes in that session.

    Labels:
      - session_profitable: 1 if any CB trade in this session had PnL > 0
      - session_pnl: total PnL from all CB trades in this session
      - n_trades: number of CB trades in this session
      - good_session: 1 if session_pnl > 2 pts (meaningful profit)
      - bad_session: 1 if session_pnl < -2 pts (meaningful loss)
    """
    # Run CB simulation to get per-trade results with session info
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

    dead_zones = []
    for slot in cb_cfg.get("dead_zones", []):
        parts = str(slot).split("-")
        if len(parts) == 2:
            def _parse(s):
                p = s.strip().split(":")
                return int(p[0]) * 60 + int(p[1])
            dead_zones.append((_parse(parts[0]), _parse(parts[1])))

    df = df_5m.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    df["mins"] = df["time"].dt.hour * 60 + df["time"].dt.minute
    df["session"] = np.where(df["mins"] < 12 * 60, "AM", "PM")
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["rsi14"] = ta.rsi(df["close"], length=14)
    df["range"] = df["high"] - df["low"]
    df = detect_comp(df, n_bars=n_bars, threshold=threshold)

    # Apply filters and get trade results per session
    filt_am = (df["atr"] >= atr_min) & (df["atr"] <= atr_max) & (df["rsi14"] < 70).fillna(True)
    filt_pm = (df["atr"] >= atr_min_pm) & (df["atr"] <= atr_max_pm) & (df["rsi14"] < 70).fillna(True)
    am_time = (df["mins"] >= 9 * 60 + 15) & (df["mins"] <= 10 * 60 + 45)
    for slot_start, slot_end in dead_zones:
        am_time = am_time & ~((df["mins"] >= slot_start) & (df["mins"] < slot_end))
    pm_time = (df["mins"] >= 13 * 60 + 15) & (df["mins"] <= 14 * 60 + 15)
    mask_am = dedup(df["comp"] & am_time & filt_am, dedup_bars)
    mask_pm = dedup(df["comp"] & pm_time & filt_pm, dedup_bars)

    # Simulate trades
    am_sp = sp.get("AM", {})
    pm_sp = sp.get("PM", {})

    def sim_trade(idx, sl_mult, trail_activate, trail_mult, max_hold):
        row = df.iloc[idx]
        atr = float(row["atr"])
        if pd.isna(atr) or atr <= 0 or idx + 1 >= len(df):
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
        be_trigger = float(am_sp.get("be_trigger_pts", 3.0))
        be_atr_min = float(am_sp.get("be_atr_min", 3.5))
        be_partial = float(am_sp.get("be_partial_pts", 1.0))

        for jp in range(idx + 1, min(idx + 1 + max_hold, len(df))):
            b = df.iloc[jp]
            if b["date"] != row["date"] or b["session"] != row["session"]:
                return direction * (float(df.iloc[jp-1]["close"]) - entry) - COST
            if (b["session"] == "PM" and b["mins"] >= 14*60+25) or (b["session"] == "AM" and b["mins"] >= 11*60+25):
                return direction * (float(b["close"]) - entry) - COST
            if direction == 1:
                if b["low"] <= sl:
                    return direction * (sl - entry) - COST
                if b["high"] > best: best = float(b["high"])
                mfe = best - entry
                if not be_done and mfe >= be_trigger and atr >= be_atr_min:
                    be_done = True; sl = max(sl, entry + be_partial)
                if mfe >= trail_activate: trail_on = True
                if trail_on:
                    sl = max(sl, best - trail_mult * atr)
                    if b["low"] <= sl: return direction * (sl - entry) - COST
            else:
                if b["high"] >= sl:
                    return direction * (sl - entry) - COST
                if b["low"] < best: best = float(b["low"])
                mfe = entry - best
                if not be_done and mfe >= be_trigger and atr >= be_atr_min:
                    be_done = True; sl = min(sl, entry - be_partial)
                if mfe >= trail_activate: trail_on = True
                if trail_on:
                    sl = min(sl, best + trail_mult * atr)
                    if b["high"] >= sl: return direction * (sl - entry) - COST
        return 0.0

    # Collect PnL per session
    session_pnl = {}  # (date, session) -> list of trade PnLs

    for idx_val in df.index[mask_am]:
        iloc_idx = df.index.get_loc(idx_val)
        row = df.iloc[iloc_idx]
        key = (str(row["date"]), "AM")
        pnl = sim_trade(iloc_idx, float(am_sp.get("sl_atr_mult", 1.5)),
                       float(am_sp.get("trail_activate_pts", 7.0)),
                       float(am_sp.get("trail_atr_mult", 2.5)),
                       int(am_sp.get("max_hold_bars", 30)))
        if pnl is not None:
            session_pnl.setdefault(key, []).append(pnl)

    for idx_val in df.index[mask_pm]:
        iloc_idx = df.index.get_loc(idx_val)
        row = df.iloc[iloc_idx]
        key = (str(row["date"]), "PM")
        pnl = sim_trade(iloc_idx, float(pm_sp.get("sl_atr_mult", 1.2)),
                       float(pm_sp.get("trail_activate_pts", 6.0)),
                       float(pm_sp.get("trail_atr_mult", 1.0)),
                       int(pm_sp.get("max_hold_bars", 8)))
        if pnl is not None:
            session_pnl.setdefault(key, []).append(pnl)

    # Merge labels into session_df
    session_df = session_df.copy()
    pnl_list = []
    n_trades_list = []
    for _, row in session_df.iterrows():
        key = (row["date"], row["session"])
        trades = session_pnl.get(key, [])
        pnl_list.append(sum(trades))
        n_trades_list.append(len(trades))

    session_df["session_pnl"] = pnl_list
    session_df["n_trades"] = n_trades_list
    session_df["good_session"] = (session_df["session_pnl"] > 2.0).astype(int)
    session_df["bad_session"] = (session_df["session_pnl"] < -2.0).astype(int)
    session_df["has_trade"] = (session_df["n_trades"] > 0).astype(int)

    return session_df


def run_regime_prediction():
    """Run regime prediction experiment."""
    print("=" * 80)
    print("REGIME/SESSION PREDICTION EXPERIMENT")
    print("=" * 80)

    print("\n[1] Loading 5m data...")
    df_5m = load("5m", days=180)
    n_days = df_5m["date"].nunique()
    print(f"  {len(df_5m)} bars, {n_days} trading days")

    print("\n[2] Computing session features...")
    session_df = compute_session_features(df_5m)
    print(f"  {len(session_df)} sessions computed")

    print("\n[3] Labeling sessions from CB backtest...")
    session_df = label_sessions(session_df, df_5m)

    # Stats
    has_trade = session_df[session_df["has_trade"] == 1]
    print(f"  Sessions with CB trades: {len(has_trade)}/{len(session_df)}")
    print(f"  Good sessions (PnL > 2): {(has_trade['good_session']==1).sum()}")
    print(f"  Bad sessions (PnL < -2): {(has_trade['bad_session']==1).sum()}")
    print(f"  Neutral sessions: {len(has_trade) - (has_trade['good_session']==1).sum() - (has_trade['bad_session']==1).sum()}")

    # Feature columns
    feature_cols = [c for c in session_df.columns if c not in
                    ["date", "session", "session_pnl", "n_trades", "good_session", "bad_session", "has_trade"]]
    print(f"  Features: {len(feature_cols)}")

    # Only predict on sessions that actually have trades
    pred_df = has_trade.copy().reset_index(drop=True)
    if len(pred_df) < 50:
        print(f"  ERROR: Only {len(pred_df)} sessions with trades. Need >= 50.")
        return

    X = pred_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    dates = pd.to_datetime(pred_df["date"]).values
    pnls = pred_df["session_pnl"].values

    # Targets
    targets = {
        "bad_session": pred_df["bad_session"].values,
        "good_session": pred_df["good_session"].values,
    }

    print(f"\n[4] Running Purged LOOCV (embargo=3 days)...")

    from ml.experiment_v2 import _get_models
    models = _get_models()

    # Use only ExtraTrees and GradientBoosting (LightGBM overfits on small samples)
    models = {k: v for k, v in models.items() if k in ["ExtraTrees", "GradientBoosting"]}

    results = []
    embargo_days = 3

    for model_name, model_factory in models.items():
        for target_name, target_y in targets.items():
            print(f"  {model_name} | target={target_name}...", end=" ")

            n = len(target_y)
            predictions = np.full(n, np.nan)

            for i in range(n):
                test_date = pd.Timestamp(dates[i])
                embargo_start = test_date - timedelta(days=embargo_days)
                embargo_end = test_date + timedelta(days=embargo_days)

                train_mask = (
                    (dates < np.datetime64(embargo_start)) |
                    (dates > np.datetime64(embargo_end))
                )
                train_mask[i] = False

                if train_mask.sum() < 30:
                    continue

                X_train, y_train = X[train_mask], target_y[train_mask]
                model = model_factory(len(X_train))
                model.fit(X_train, y_train)
                predictions[i] = model.predict_proba(X[i:i+1])[0][1]

            valid = ~np.isnan(predictions)
            pred_v = predictions[valid]
            y_v = target_y[valid]
            pnl_v = pnls[valid]
            baseline_pnl = pnl_v.sum()

            best_improvement = -9999
            best_thr = None
            best_detail = None

            for thr in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
                if target_name == "bad_session":
                    # Skip sessions predicted as bad
                    skip_mask = pred_v > thr
                else:
                    # Only trade sessions predicted as good
                    skip_mask = pred_v < thr

                retained = ~skip_mask
                if retained.sum() == 0 or skip_mask.sum() == 0:
                    continue

                net_pnl = pnl_v[retained].sum()
                improvement = net_pnl - baseline_pnl
                n_skipped = skip_mask.sum()

                # Precision: of skipped sessions, how many were actually bad?
                if target_name == "bad_session":
                    precision = y_v[skip_mask].mean() * 100 if skip_mask.sum() > 0 else 0
                else:
                    precision = (1 - y_v[~retained].mean()) * 100 if (~retained).sum() > 0 else 0

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_thr = thr
                    best_detail = {
                        "retained": int(retained.sum()),
                        "skipped": int(n_skipped),
                        "net_pnl": float(net_pnl),
                        "precision": float(precision),
                    }

            print(f"best={best_improvement:+.1f}pts (thr={best_thr})")
            results.append({
                "model": model_name,
                "target": target_name,
                "n_sessions": int(valid.sum()),
                "baseline_pnl": float(baseline_pnl),
                "best_threshold": best_thr,
                "best_improvement": float(best_improvement) if best_thr else 0,
                "detail": best_detail,
            })

    # Summary
    print(f"\n{'=' * 80}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Model':<18} {'Target':<14} {'Sessions':<10} {'Baseline':<10} {'Improve':<10} {'Skip':<6} {'Precision'}")
    print(f"  {'-'*80}")

    results.sort(key=lambda r: r["best_improvement"], reverse=True)
    for r in results:
        d = r.get("detail") or {}
        print(f"  {r['model']:<18} {r['target']:<14} {r['n_sessions']:<10} "
              f"{r['baseline_pnl']:<+10.1f} {r['best_improvement']:<+10.1f} "
              f"{d.get('skipped', 0):<6} {d.get('precision', 0):.1f}%")

    # Save
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {REPORT_PATH}")

    return results


if __name__ == "__main__":
    run_regime_prediction()
