"""Fine-tune standalone ML: threshold sweep + session/direction filters."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier

from backtest.engine import load, detect_comp
from ml.standalone_signals import compute_all_features, label_fixed_horizon, FEATURE_COLS, COST


def walk_forward_predictions(df, horizon=6, train_days=60, test_days=20):
    """Run walk-forward and return full prediction array."""
    df = compute_all_features(df)
    df["label"] = label_fixed_horizon(df, horizon=horizon)

    available_features = [f for f in FEATURE_COLS if f in df.columns]
    X_all = df[available_features].values
    y_all = df["label"].values
    dates_all = df["date"].values

    unique_dates = sorted(df["date"].unique())
    all_test_indices = []
    all_test_predictions = []
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

        train_labeled = train_mask & (y_all != 0)
        X_train = X_all[train_labeled]
        y_train = (y_all[train_labeled] == 1).astype(int)

        if len(X_train) < 100:
            train_start_idx += test_days
            continue

        X_train_clean = np.nan_to_num(X_train, nan=0.0)
        model = ExtraTreesClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            max_features="sqrt", class_weight="balanced", n_jobs=-1,
        )
        model.fit(X_train_clean, y_train)

        X_test = np.nan_to_num(X_all[test_mask], nan=0.0)
        predictions = model.predict_proba(X_test)[:, 1]

        test_indices = np.where(test_mask)[0]
        all_test_indices.extend(test_indices.tolist())
        all_test_predictions.extend(predictions.tolist())
        train_start_idx += test_days

    return df, all_test_indices, all_test_predictions


def simulate_filtered(df, indices, predictions, threshold, horizon=6,
                      session_filter=None, direction_filter=None, max_trades_per_session=3):
    """Simulate trades with optional session/direction filters."""
    test_df = df.iloc[indices].copy().reset_index(drop=True)
    preds = np.array(predictions)
    c = test_df["close"].values
    dates = test_df["date"].values
    sessions = test_df["session"].values
    mins_vals = test_df["mins"].values

    trades = []
    in_cooldown_until = -1
    session_trades = {}

    for i in range(len(preds)):
        if i < in_cooldown_until:
            continue

        session_key = (dates[i], sessions[i])
        if session_key not in session_trades:
            session_trades[session_key] = 0
        if session_trades[session_key] >= max_trades_per_session:
            continue

        if sessions[i] == "AM" and mins_vals[i] >= 11*60+20:
            continue
        if sessions[i] == "PM" and mins_vals[i] >= 14*60+20:
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

        # Session filter
        if session_filter and sessions[i] not in session_filter:
            continue
        # Direction filter
        if direction_filter and direction not in direction_filter:
            continue

        entry_price = c[i]
        exit_idx = None

        for j in range(i + 1, min(i + horizon + 1, len(test_df))):
            if dates[j] != dates[i] or sessions[j] != sessions[i]:
                exit_idx = j - 1
                break
            if sessions[j] == "AM" and mins_vals[j] >= 11*60+25:
                exit_idx = j
                break
            if sessions[j] == "PM" and mins_vals[j] >= 14*60+25:
                exit_idx = j
                break
            if j == i + horizon:
                exit_idx = j
                break

        if exit_idx is None:
            exit_idx = min(i + horizon, len(test_df) - 1)

        exit_price = c[exit_idx]
        pnl = direction * (exit_price - entry_price) - COST

        trades.append({
            "date": str(dates[i]),
            "session": sessions[i],
            "direction": "BUY" if direction == 1 else "SELL",
            "pnl": pnl,
            "entry": entry_price,
            "exit": exit_price,
        })
        session_trades[session_key] += 1
        in_cooldown_until = i + horizon

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def print_metrics(trades_df, n_days, label=""):
    """Print standard metrics for a set of trades."""
    if trades_df.empty:
        print(f"  {label:<20} No trades")
        return None

    n = len(trades_df)
    tpd = n / n_days
    wr = (trades_df["pnl"] > 0).mean() * 100
    total = trades_df["pnl"].sum()
    ppd = total / n_days
    avg = trades_df["pnl"].mean()
    wins = trades_df[trades_df["pnl"] > 0]["pnl"]
    losses = trades_df[trades_df["pnl"] <= 0]["pnl"]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 999
    max_dd = 0
    cum = 0
    peak = 0
    for p in trades_df["pnl"]:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    print(f"  {label:<20} N={n:<4} T/d={tpd:<5.1f} WR={wr:<5.1f}% "
          f"PnL/d={ppd:<+6.2f} AvgW={avg_win:<+5.2f} AvgL={avg_loss:<5.2f} "
          f"PF={pf:<5.2f} MaxDD={max_dd:<5.1f}")
    return {"n": n, "tpd": tpd, "wr": wr, "ppd": ppd, "pf": pf, "max_dd": max_dd,
            "avg_win": avg_win, "avg_loss": avg_loss, "total_pnl": total}


def main():
    print("=" * 90)
    print("STANDALONE ML FINE-TUNE ANALYSIS — Horizon=6 (30 min)")
    print("=" * 90)

    print("\n[1] Loading data and running walk-forward...")
    df_raw = load("5m", days=180)
    df, indices, predictions = walk_forward_predictions(df_raw, horizon=6)
    n_days = df.iloc[indices]["date"].nunique()
    print(f"  {len(predictions)} test predictions over {n_days} days")

    # ===== SECTION 1: Fine threshold sweep =====
    print(f"\n{'=' * 90}")
    print("SECTION 1: Fine Threshold Sweep (ALL sessions, ALL directions)")
    print(f"{'=' * 90}")
    print(f"  {'Threshold':<12} {'N':<5} {'T/d':<6} {'WR%':<7} {'PnL/d':<8} "
          f"{'AvgWin':<8} {'AvgLoss':<8} {'PF':<6} {'MaxDD':<6}")
    print(f"  {'-'*80}")

    best_ppd = -999
    best_thr = None
    for thr in [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.62, 0.65]:
        trades = simulate_filtered(df, indices, predictions, thr, horizon=6)
        r = print_metrics(trades, n_days, f"thr={thr:.2f}")
        if r and r["ppd"] > best_ppd:
            best_ppd = r["ppd"]
            best_thr = thr

    print(f"\n  >>> Best threshold: {best_thr} with PnL/day = {best_ppd:+.2f}")

    # ===== SECTION 2: Session filters =====
    print(f"\n{'=' * 90}")
    print("SECTION 2: Session Filters (at best threshold)")
    print(f"{'=' * 90}")

    for thr in [best_thr, 0.55, 0.58]:
        print(f"\n  --- Threshold = {thr:.2f} ---")
        trades_all = simulate_filtered(df, indices, predictions, thr)
        print_metrics(trades_all, n_days, "ALL")
        trades_am = simulate_filtered(df, indices, predictions, thr, session_filter=["AM"])
        print_metrics(trades_am, n_days, "AM only")
        trades_pm = simulate_filtered(df, indices, predictions, thr, session_filter=["PM"])
        print_metrics(trades_pm, n_days, "PM only")

    # ===== SECTION 3: Direction filters =====
    print(f"\n{'=' * 90}")
    print("SECTION 3: Direction Filters (at best threshold)")
    print(f"{'=' * 90}")

    for thr in [best_thr, 0.55, 0.58]:
        print(f"\n  --- Threshold = {thr:.2f} ---")
        trades_buy = simulate_filtered(df, indices, predictions, thr, direction_filter=[1])
        print_metrics(trades_buy, n_days, "BUY only")
        trades_sell = simulate_filtered(df, indices, predictions, thr, direction_filter=[-1])
        print_metrics(trades_sell, n_days, "SELL only")

    # ===== SECTION 4: Combined filters =====
    print(f"\n{'=' * 90}")
    print("SECTION 4: Combined Filters")
    print(f"{'=' * 90}")

    combos = [
        ("PM+SELL", ["PM"], [-1]),
        ("PM+BUY", ["PM"], [1]),
        ("AM+SELL", ["AM"], [-1]),
        ("AM+BUY", ["AM"], [1]),
        ("PM+ALL", ["PM"], None),
        ("ALL+SELL", None, [-1]),
    ]

    for thr in [0.53, 0.55, 0.57, 0.60]:
        print(f"\n  --- Threshold = {thr:.2f} ---")
        for name, sess_f, dir_f in combos:
            trades = simulate_filtered(df, indices, predictions, thr,
                                       session_filter=sess_f, direction_filter=dir_f)
            print_metrics(trades, n_days, name)

    # ===== SECTION 5: Overlap with CB =====
    print(f"\n{'=' * 90}")
    print("SECTION 5: Overlap with CB Signals")
    print(f"{'=' * 90}")

    # Detect CB compressions
    df_cb = df.copy()
    df_cb["atr"] = df_cb["atr_14"]
    df_cb["range"] = df_cb["high"] - df_cb["low"]
    df_cb = detect_comp(df_cb, n_bars=3, threshold=0.7)

    # Get ML signal bars (at best threshold)
    thr_check = best_thr
    ml_signal_indices = []
    preds_arr = np.array(predictions)
    for i, idx in enumerate(indices):
        if preds_arr[i] > thr_check or preds_arr[i] < (1 - thr_check):
            ml_signal_indices.append(idx)

    # Get CB signal bars in test period
    test_start = min(indices)
    test_end = max(indices)
    cb_signal_indices = []
    for i in range(test_start, test_end + 1):
        if i < len(df_cb) and df_cb.iloc[i].get("comp", False):
            cb_signal_indices.append(i)

    # Check overlap (within 6 bars = 30 min)
    overlap_window = 6
    overlaps = 0
    for ml_idx in ml_signal_indices:
        for cb_idx in cb_signal_indices:
            if abs(ml_idx - cb_idx) <= overlap_window:
                overlaps += 1
                break

    print(f"  ML signals (thr={thr_check}): {len(ml_signal_indices)}")
    print(f"  CB compressions in test period: {len(cb_signal_indices)}")
    print(f"  Overlaps (within {overlap_window} bars): {overlaps}")
    if len(ml_signal_indices) > 0:
        print(f"  Overlap rate: {overlaps/len(ml_signal_indices)*100:.1f}% of ML signals near CB")
    print(f"  Orthogonality: {'HIGH' if overlaps/max(len(ml_signal_indices),1) < 0.15 else 'MODERATE' if overlaps/max(len(ml_signal_indices),1) < 0.30 else 'LOW'}")

    # ===== SECTION 6: Recommendations =====
    print(f"\n{'=' * 90}")
    print("SECTION 6: RECOMMENDATIONS")
    print(f"{'=' * 90}")

    # Run best combo
    print("\n  Top 5 configurations by PnL/day:")
    configs = []
    for thr in [0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60]:
        for name, sess_f, dir_f in [("ALL", None, None)] + combos:
            trades = simulate_filtered(df, indices, predictions, thr,
                                       session_filter=sess_f, direction_filter=dir_f)
            if not trades.empty and len(trades) >= 10:
                n = len(trades)
                wr = (trades["pnl"] > 0).mean() * 100
                ppd = trades["pnl"].sum() / n_days
                pf_val = trades[trades["pnl"]>0]["pnl"].sum() / abs(trades[trades["pnl"]<=0]["pnl"].sum()) if trades[trades["pnl"]<=0]["pnl"].sum() != 0 else 999
                configs.append({"thr": thr, "filter": name, "n": n, "wr": wr, "ppd": ppd, "pf": pf_val})

    configs.sort(key=lambda x: x["ppd"], reverse=True)
    print(f"  {'#':<3} {'Thr':<6} {'Filter':<12} {'N':<5} {'WR%':<7} {'PnL/d':<8} {'PF':<6}")
    print(f"  {'-'*50}")
    for i, cfg in enumerate(configs[:10]):
        print(f"  {i+1:<3} {cfg['thr']:<6.2f} {cfg['filter']:<12} {cfg['n']:<5} "
              f"{cfg['wr']:<7.1f} {cfg['ppd']:<+8.2f} {cfg['pf']:<6.2f}")


if __name__ == "__main__":
    main()
