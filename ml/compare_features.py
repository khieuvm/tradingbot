"""
Compare OLD (35 features) vs NEW (50 features) standalone ML.
Walk-forward on H=6, fine threshold sweep, session/direction filters.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from backtest.engine import load
from ml.standalone_signals import compute_all_features, label_fixed_horizon, FEATURE_COLS, COST


FEATURES_OLD = [
    "atr_14", "atr_5", "atr_pct", "atr_ratio", "bb_width", "bb_pos",
    "rsi_14", "rsi_3", "rsi_7", "rsi_delta", "stoch_k", "stoch_d",
    "macd_hist", "adx", "di_plus", "di_minus",
    "ema8_dist", "ema21_dist", "ema50_dist", "ema_spread",
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_8", "ret_13",
    "body_pct", "range_pct", "upper_wick", "lower_wick", "close_vs_range",
    "vol_ratio", "hour_sin", "hour_cos", "is_pm",
]


def walk_forward(df, feature_cols, horizon=6, train_days=60, test_days=20):
    """Run walk-forward, return (test_indices, predictions, n_folds)."""
    available = [f for f in feature_cols if f in df.columns]
    X_all = df[available].values
    y_all = df["label"].values
    dates_all = df["date"].values
    unique_dates = sorted(df["date"].unique())

    all_idx, all_pred = [], []
    train_start = 0
    folds = 0

    while True:
        train_end = train_start + train_days
        test_end = train_end + test_days
        if test_end > len(unique_dates):
            break

        train_dates = set(unique_dates[train_start:train_end])
        test_dates = set(unique_dates[train_end:test_end])

        train_mask = np.array([d in train_dates for d in dates_all])
        test_mask = np.array([d in test_dates for d in dates_all])

        train_labeled = train_mask & (y_all != 0)
        X_train = X_all[train_labeled]
        y_train = (y_all[train_labeled] == 1).astype(int)

        if len(X_train) < 100:
            train_start += test_days
            continue

        model = ExtraTreesClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            max_features="sqrt", class_weight="balanced", n_jobs=-1,
        )
        model.fit(np.nan_to_num(X_train, nan=0.0), y_train)

        X_test = np.nan_to_num(X_all[test_mask], nan=0.0)
        preds = model.predict_proba(X_test)[:, 1]

        test_indices = np.where(test_mask)[0]
        all_idx.extend(test_indices.tolist())
        all_pred.extend(preds.tolist())
        folds += 1
        train_start += test_days

    return all_idx, all_pred, folds


def simulate(df, indices, predictions, threshold, horizon=6,
             session_filter=None, direction_filter=None, max_tps=3):
    """Simulate trades, return DataFrame."""
    test_df = df.iloc[indices].reset_index(drop=True)
    preds = np.array(predictions)
    c = test_df["close"].values
    dates = test_df["date"].values
    sessions = test_df["session"].values
    mins_v = test_df["mins"].values

    trades = []
    cooldown = -1
    sess_counts = {}

    for i in range(len(preds)):
        if i < cooldown:
            continue
        sk = (dates[i], sessions[i])
        if sk not in sess_counts:
            sess_counts[sk] = 0
        if sess_counts[sk] >= max_tps:
            continue
        if sessions[i] == "AM" and mins_v[i] >= 11*60+20:
            continue
        if sessions[i] == "PM" and mins_v[i] >= 14*60+20:
            continue

        prob = preds[i]
        if np.isnan(prob):
            continue

        direction = 0
        if prob > threshold:
            direction = 1
        elif prob < (1 - threshold):
            direction = -1
        if direction == 0:
            continue

        if session_filter and sessions[i] not in session_filter:
            continue
        if direction_filter and direction not in direction_filter:
            continue

        entry = c[i]
        exit_idx = None
        for j in range(i+1, min(i+horizon+1, len(test_df))):
            if dates[j] != dates[i] or sessions[j] != sessions[i]:
                exit_idx = j - 1; break
            if sessions[j] == "AM" and mins_v[j] >= 11*60+25:
                exit_idx = j; break
            if sessions[j] == "PM" and mins_v[j] >= 14*60+25:
                exit_idx = j; break
            if j == i + horizon:
                exit_idx = j; break
        if exit_idx is None:
            exit_idx = min(i + horizon, len(test_df) - 1)

        pnl = direction * (c[exit_idx] - entry) - COST
        trades.append({"pnl": pnl, "session": sessions[i],
                       "direction": "BUY" if direction == 1 else "SELL"})
        sess_counts[sk] += 1
        cooldown = i + horizon

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def metrics(trades_df, n_days):
    """Return dict of metrics or None."""
    if trades_df.empty:
        return None
    n = len(trades_df)
    wr = (trades_df["pnl"] > 0).mean() * 100
    total = trades_df["pnl"].sum()
    ppd = total / n_days
    wins = trades_df[trades_df["pnl"] > 0]["pnl"]
    losses = trades_df[trades_df["pnl"] <= 0]["pnl"]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 999
    return {"n": n, "tpd": n/n_days, "wr": wr, "ppd": ppd, "pf": pf, "total": total}


def print_comparison(label, m_old, m_new):
    """Print side-by-side comparison."""
    if m_old is None and m_new is None:
        return
    def fmt(m):
        if m is None:
            return "N/A"
        return f"N={m['n']:<4} T/d={m['tpd']:<4.1f} WR={m['wr']:<5.1f}% PnL/d={m['ppd']:<+6.2f} PF={m['pf']:<5.2f}"
    print(f"  {label:<20} OLD: {fmt(m_old)}")
    print(f"  {'':<20} NEW: {fmt(m_new)}")
    if m_old and m_new:
        delta = m_new['ppd'] - m_old['ppd']
        print(f"  {'':<20} DELTA: PnL/d {delta:+.2f}, WR {m_new['wr']-m_old['wr']:+.1f}%, PF {m_new['pf']-m_old['pf']:+.2f}")
    print()


def main():
    print("=" * 90)
    print("FEATURE COMPARISON: OLD (35) vs NEW (50) — Horizon=6, Walk-Forward")
    print("=" * 90)

    print("\n[1] Loading data & computing features...")
    df_raw = load("5m", days=180)
    df = compute_all_features(df_raw)
    df["label"] = label_fixed_horizon(df, horizon=6)
    n_days_total = df["date"].nunique()
    print(f"  {len(df)} bars, {n_days_total} days")
    print(f"  Old features: {len(FEATURES_OLD)}")
    print(f"  New features: {len(FEATURE_COLS)} ({len(FEATURE_COLS)-len(FEATURES_OLD)} added)")
    new_only = [f for f in FEATURE_COLS if f not in FEATURES_OLD]
    print(f"  Added: {new_only}")

    print("\n[2] Walk-forward with OLD features...")
    idx_old, pred_old, folds_old = walk_forward(df, FEATURES_OLD, horizon=6)
    n_days = df.iloc[idx_old]["date"].nunique()
    print(f"  {folds_old} folds, {len(pred_old)} predictions, {n_days} test days")

    print("\n[3] Walk-forward with NEW features...")
    idx_new, pred_new, folds_new = walk_forward(df, FEATURE_COLS, horizon=6)
    n_days_new = df.iloc[idx_new]["date"].nunique()
    print(f"  {folds_new} folds, {len(pred_new)} predictions, {n_days_new} test days")

    # ===== Compare at multiple thresholds =====
    print(f"\n{'=' * 90}")
    print("SECTION 1: Threshold Comparison (ALL sessions, ALL directions)")
    print(f"{'=' * 90}")
    print(f"\n  {'Thr':<6} | {'--- OLD (35 feat) ---':<42} | {'--- NEW (50 feat) ---':<42} | Delta PnL/d")
    print(f"  {'-'*105}")

    for thr in [0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60]:
        t_old = simulate(df, idx_old, pred_old, thr)
        t_new = simulate(df, idx_new, pred_new, thr)
        m_old = metrics(t_old, n_days)
        m_new = metrics(t_new, n_days_new)

        def fmt_short(m):
            if m is None: return f"{'N/A':<42}"
            return f"N={m['n']:<3} WR={m['wr']:<5.1f}% PnL/d={m['ppd']:<+6.2f} PF={m['pf']:<5.2f}"

        delta = ""
        if m_old and m_new:
            d = m_new['ppd'] - m_old['ppd']
            delta = f"{d:+.2f} {'<<<' if d > 0.3 else ''}"

        print(f"  {thr:<6.2f} | {fmt_short(m_old)} | {fmt_short(m_new)} | {delta}")

    # ===== Best configs with session/direction filters =====
    print(f"\n{'=' * 90}")
    print("SECTION 2: Best Configs with Filters (NEW features)")
    print(f"{'=' * 90}")

    configs = []
    filters = [
        ("ALL", None, None),
        ("AM+ALL", ["AM"], None),
        ("PM+ALL", ["PM"], None),
        ("ALL+SELL", None, [-1]),
        ("ALL+BUY", None, [1]),
        ("AM+SELL", ["AM"], [-1]),
        ("AM+BUY", ["AM"], [1]),
        ("PM+SELL", ["PM"], [-1]),
        ("PM+BUY", ["PM"], [1]),
    ]

    print(f"\n  {'Thr':<6} {'Filter':<12} {'N':<5} {'T/d':<5} {'WR%':<7} {'PnL/d':<8} {'PF':<6} {'Total':<8}")
    print(f"  {'-'*65}")

    for thr in [0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60]:
        for name, sf, df_filter in filters:
            t = simulate(df, idx_new, pred_new, thr, session_filter=sf, direction_filter=df_filter)
            m = metrics(t, n_days_new)
            if m and m["n"] >= 10:
                configs.append({"thr": thr, "filter": name, **m})

    configs.sort(key=lambda x: x["ppd"], reverse=True)
    for cfg in configs[:15]:
        print(f"  {cfg['thr']:<6.2f} {cfg['filter']:<12} {cfg['n']:<5} {cfg['tpd']:<5.1f} "
              f"{cfg['wr']:<7.1f} {cfg['ppd']:<+8.2f} {cfg['pf']:<6.2f} {cfg['total']:<+8.1f}")

    # ===== Feature importance from last fold =====
    print(f"\n{'=' * 90}")
    print("SECTION 3: Feature Importance (NEW model, last fold)")
    print(f"{'=' * 90}")

    available = [f for f in FEATURE_COLS if f in df.columns]
    X_all = df[available].values
    y_all = df["label"].values
    dates_all = df["date"].values
    unique_dates = sorted(df["date"].unique())

    # Train on last 60 days, just for importance
    train_dates = set(unique_dates[-80:-20])
    train_mask = np.array([d in train_dates for d in dates_all]) & (y_all != 0)
    X_train = np.nan_to_num(X_all[train_mask], nan=0.0)
    y_train = (y_all[train_mask] == 1).astype(int)

    model = ExtraTreesClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50,
        max_features="sqrt", class_weight="balanced", n_jobs=-1,
    )
    model.fit(X_train, y_train)

    importances = sorted(zip(available, model.feature_importances_),
                         key=lambda x: x[1], reverse=True)

    print(f"\n  {'#':<3} {'Feature':<22} {'Importance':<12} {'New?'}")
    print(f"  {'-'*45}")
    for i, (feat, imp) in enumerate(importances[:20]):
        is_new = "***" if feat in new_only else ""
        print(f"  {i+1:<3} {feat:<22} {imp:<12.4f} {is_new}")

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")

    # Best OLD
    best_old = None
    for thr in [0.53, 0.54, 0.55, 0.56, 0.57, 0.58]:
        t = simulate(df, idx_old, pred_old, thr, session_filter=["AM"], direction_filter=[-1])
        m = metrics(t, n_days)
        if m and (best_old is None or m["ppd"] > best_old["ppd"]):
            best_old = {**m, "thr": thr}

    # Best NEW
    best_new = configs[0] if configs else None

    print(f"\n  Best OLD: thr={best_old['thr'] if best_old else '?'}, "
          f"AM+SELL, PnL/d={best_old['ppd']:+.2f}, PF={best_old['pf']:.2f}" if best_old else "  Best OLD: N/A")
    print(f"  Best NEW: thr={best_new['thr']}, {best_new['filter']}, "
          f"PnL/d={best_new['ppd']:+.2f}, PF={best_new['pf']:.2f}" if best_new else "  Best NEW: N/A")


if __name__ == "__main__":
    main()
