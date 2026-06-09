"""Test NEW features across all horizons (3, 6, 10) with filters."""
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
    """Run walk-forward with NEW features."""
    available = [f for f in FEATURE_COLS if f in df.columns]
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
    """Simulate trades."""
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


def main():
    print("=" * 90)
    print("MULTI-HORIZON TEST — NEW Features (50), Walk-Forward")
    print("=" * 90)

    df_raw = load("5m", days=180)

    filters = [
        ("ALL", None, None),
        ("AM+ALL", ["AM"], None),
        ("AM+SELL", ["AM"], [-1]),
        ("AM+BUY", ["AM"], [1]),
        ("PM+ALL", ["PM"], None),
        ("PM+SELL", ["PM"], [-1]),
        ("ALL+SELL", None, [-1]),
    ]

    for horizon in [3, 6, 10]:
        print(f"\n{'=' * 90}")
        print(f"HORIZON = {horizon} bars ({horizon*5} min)")
        print(f"{'=' * 90}")

        df = compute_all_features(df_raw.copy())
        df["label"] = label_fixed_horizon(df, horizon=horizon)

        idx, pred, folds = walk_forward(df, horizon=horizon)
        n_days = df.iloc[idx]["date"].nunique()
        print(f"  {folds} folds, {len(pred)} predictions, {n_days} test days")

        print(f"\n  {'Thr':<6} {'Filter':<12} {'N':<5} {'T/d':<5} {'WR%':<7} "
              f"{'PnL/d':<8} {'PF':<6} {'Total':<8} {'AvgW':<7} {'AvgL':<7}")
        print(f"  {'-'*80}")

        results = []
        for thr in [0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60]:
            for name, sf, df_f in filters:
                t = simulate(df, idx, pred, thr, horizon=horizon,
                            session_filter=sf, direction_filter=df_f)
                if t.empty or len(t) < 8:
                    continue
                n = len(t)
                wr = (t["pnl"] > 0).mean() * 100
                total = t["pnl"].sum()
                ppd = total / n_days
                wins = t[t["pnl"] > 0]["pnl"]
                losses = t[t["pnl"] <= 0]["pnl"]
                pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 999
                avg_w = wins.mean() if len(wins) > 0 else 0
                avg_l = abs(losses.mean()) if len(losses) > 0 else 0
                results.append({
                    "horizon": horizon, "thr": thr, "filter": name,
                    "n": n, "tpd": n/n_days, "wr": wr, "ppd": ppd,
                    "pf": pf, "total": total, "avg_w": avg_w, "avg_l": avg_l,
                })

        results.sort(key=lambda x: x["ppd"], reverse=True)
        for r in results[:12]:
            print(f"  {r['thr']:<6.2f} {r['filter']:<12} {r['n']:<5} {r['tpd']:<5.1f} "
                  f"{r['wr']:<7.1f} {r['ppd']:<+8.2f} {r['pf']:<6.2f} {r['total']:<+8.1f} "
                  f"{r['avg_w']:<+7.2f} {r['avg_l']:<7.2f}")

    # ===== FINAL SUMMARY =====
    print(f"\n{'=' * 90}")
    print("FINAL COMPARISON — Best config per horizon")
    print(f"{'=' * 90}")
    print(f"\n  Nhắc lại: CB hiện tại = WR 74.7%, PF 5.13, +6.69 pts/day")
    print(f"  ML standalone nên chạy BỔ SUNG, không thay thế CB.\n")


if __name__ == "__main__":
    main()
