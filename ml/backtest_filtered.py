"""
Backtest ML standalone with post-ML filters applied.
Compare: unfiltered SELL vs filtered SELL (ema_align <= 0).
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


def walk_forward(df, horizon=6, train_days=60, test_days=20):
    available = [f for f in FEATURE_COLS if f in df.columns]
    X_all = df[available].values
    y_all = df["label"].values
    dates_all = df["date"].values
    unique_dates = sorted(df["date"].unique())

    all_idx, all_pred = [], []
    train_start = 0
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
            max_features="sqrt", class_weight="balanced", n_jobs=-1)
        model.fit(np.nan_to_num(X_train, nan=0.0), y_train)
        X_test = np.nan_to_num(X_all[test_mask], nan=0.0)
        preds = model.predict_proba(X_test)[:, 1]
        all_idx.extend(np.where(test_mask)[0].tolist())
        all_pred.extend(preds.tolist())
        train_start += test_days
    return all_idx, all_pred


def simulate(df, indices, predictions, threshold=0.55, horizon=6,
             session_filter=None, direction_filter=None, post_filters=None):
    """Simulate with post-ML filters."""
    test_df = df.iloc[indices].reset_index(drop=True)
    preds = np.array(predictions)
    c = test_df["close"].values
    h = test_df["high"].values
    l = test_df["low"].values
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
        if sess_counts[sk] >= 2:
            continue
        if sessions[i] == "AM" and mins_v[i] >= 11*60+20:
            continue
        if sessions[i] == "PM" and mins_v[i] >= 14*60+20:
            continue
        if session_filter and sessions[i] not in session_filter:
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

        dir_str = "BUY" if direction == 1 else "SELL"
        if direction_filter and dir_str not in direction_filter:
            continue

        # Post-ML filters
        row = test_df.iloc[i]
        if post_filters:
            filt = post_filters.get(dir_str)
            if filt is None and dir_str in post_filters:
                continue
            if filt:
                if "ema_align_max" in filt:
                    if float(row.get("ema_align", 0)) > filt["ema_align_max"]:
                        continue
                if "di_spread_max" in filt:
                    if float(row.get("di_spread", 0)) > filt["di_spread_max"]:
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

        exit_price = c[exit_idx]
        pnl = direction * (exit_price - entry) - COST

        if direction == 1:
            mfe = max(h[i+1:exit_idx+1]) - entry if exit_idx > i else 0
        else:
            mfe = entry - min(l[i+1:exit_idx+1]) if exit_idx > i else 0

        trades.append({
            "pnl": pnl, "direction": dir_str, "prob": prob,
            "session": sessions[i], "date": str(dates[i]),
            "mfe": mfe, "ema_align": float(row.get("ema_align", 0)),
            "di_spread": float(row.get("di_spread", 0)),
        })
        sess_counts[sk] += 1
        cooldown = i + horizon

    return pd.DataFrame(trades)


def main():
    print("=" * 80)
    print("BACKTEST: ML Standalone with Post-ML Filters")
    print("=" * 80)

    df_raw = load("5m", days=180)
    df = compute_all_features(df_raw)
    df["label"] = label_fixed_horizon(df, horizon=6)

    print("\n[1] Walk-forward...")
    idx, pred = walk_forward(df)
    n_days = df.iloc[idx]["date"].nunique()
    print(f"  {len(pred)} predictions, {n_days} test days")

    configs = [
        ("SELL only (no filter)", ["AM"], ["SELL"], None),
        ("SELL + ema_align<=0", ["AM"], ["SELL"], {"SELL": {"ema_align_max": 0}}),
        ("SELL + ema_align<=0 + di<=5", ["AM"], ["SELL"],
         {"SELL": {"ema_align_max": 0, "di_spread_max": 5}}),
        ("SELL + di_spread<=0", ["AM"], ["SELL"], {"SELL": {"di_spread_max": 0}}),
        ("BUY+SELL (no filter)", ["AM"], ["BUY", "SELL"], None),
        ("BUY+SELL + filters", ["AM"], ["BUY", "SELL"],
         {"SELL": {"ema_align_max": 0}, "BUY": None}),
    ]

    print(f"\n  {'Config':<35} {'N':<5} {'WR%':<7} {'PnL':<10} {'PnL/d':<8} {'PF':<6}")
    print(f"  {'-'*75}")

    for name, sess, dirs, post_f in configs:
        trades = simulate(df, idx, pred, threshold=0.55, session_filter=sess,
                          direction_filter=dirs, post_filters=post_f)
        if trades.empty:
            print(f"  {name:<35} {'N/A'}")
            continue
        n = len(trades)
        wr = (trades["pnl"] > 0).mean() * 100
        pnl = trades["pnl"].sum()
        ppd = pnl / n_days
        w_sum = trades[trades["pnl"] > 0]["pnl"].sum()
        l_sum = abs(trades[trades["pnl"] <= 0]["pnl"].sum())
        pf = w_sum / l_sum if l_sum > 0 else 999
        print(f"  {name:<35} {n:<5} {wr:<7.1f} {pnl:<+10.1f} {ppd:<+8.2f} {pf:<6.2f}")

    # Per 20-day window analysis for best config
    print(f"\n\n{'='*80}")
    print("STABILITY CHECK: SELL + ema_align<=0 per 20-day window")
    print(f"{'='*80}")

    trades = simulate(df, idx, pred, threshold=0.55, session_filter=["AM"],
                      direction_filter=["SELL"], post_filters={"SELL": {"ema_align_max": 0}})
    if not trades.empty:
        trades["date_dt"] = pd.to_datetime(trades["date"])
        unique_dates = sorted(trades["date_dt"].unique())
        window = 20
        for start in range(0, len(unique_dates), window):
            end = min(start + window, len(unique_dates))
            d_start = unique_dates[start]
            d_end = unique_dates[end - 1]
            sub = trades[(trades["date_dt"] >= d_start) & (trades["date_dt"] <= d_end)]
            if len(sub) < 3:
                continue
            wr = (sub["pnl"] > 0).mean() * 100
            ppd = sub["pnl"].sum() / (end - start)
            print(f"  {str(d_start.date())} → {str(d_end.date())}: "
                  f"{len(sub)} trades, WR={wr:.0f}%, PnL/d={ppd:+.2f}")


if __name__ == "__main__":
    main()
