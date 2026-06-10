"""
Full ML analysis: BUY + SELL with direction-specific post-ML filters.
Analyze loss reasons per direction, find optimal filters.
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


def simulate_all(df, indices, predictions, threshold=0.55, horizon=6,
                 session_filter=None, max_tps=3):
    """Simulate both BUY and SELL, return full details."""
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
        if sess_counts[sk] >= max_tps:
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
            mae = entry - min(l[i+1:exit_idx+1]) if exit_idx > i else 0
        else:
            mfe = entry - min(l[i+1:exit_idx+1]) if exit_idx > i else 0
            mae = max(h[i+1:exit_idx+1]) - entry if exit_idx > i else 0

        row = test_df.iloc[i]
        trade = {
            "pnl": pnl, "direction": "BUY" if direction == 1 else "SELL",
            "prob": prob, "entry": entry, "exit": exit_price,
            "mfe": mfe, "mae": mae,
            "session": sessions[i], "mins": mins_v[i],
            "time": str(row.get("time", "")), "date": str(dates[i]),
            # Features
            "atr_14": float(row.get("atr_14", 0)),
            "rsi_14": float(row.get("rsi_14", 50)),
            "rsi_3": float(row.get("rsi_3", 50)),
            "adx": float(row.get("adx", 20)),
            "bb_pos": float(row.get("bb_pos", 0)),
            "bb_width": float(row.get("bb_width", 0.01)),
            "ema_align": float(row.get("ema_align", 0)),
            "stoch_rsi": float(row.get("stoch_rsi", 0.5)),
            "stoch_k": float(row.get("stoch_k", 50)),
            "di_spread": float(row.get("di_spread", 0)),
            "di_plus": float(row.get("di_plus", 20)),
            "di_minus": float(row.get("di_minus", 20)),
            "range_ratio_3_10": float(row.get("range_ratio_3_10", 1)),
            "macd_hist_raw": float(row.get("macd_hist_raw", 0)),
            "kc_pos": float(row.get("kc_pos", 0)),
            "price_accel": float(row.get("price_accel", 0)),
            "consec_up": float(row.get("consec_up", 0)),
            "consec_down": float(row.get("consec_down", 0)),
            "vol_spike_5": float(row.get("vol_spike_5", 1)),
            "ret_1": float(row.get("ret_1", 0)),
            "ret_3": float(row.get("ret_3", 0)),
            "ret_5": float(row.get("ret_5", 0)),
            "atr_ratio": float(row.get("atr_ratio", 1)),
            "body_pct": float(row.get("body_pct", 0)),
            "ema8_dist": float(row.get("ema8_dist", 0)),
            "ema_spread": float(row.get("ema_spread", 0)),
        }
        trades.append(trade)
        sess_counts[sk] += 1
        cooldown = i + horizon

    return pd.DataFrame(trades)


def analyze_direction(trades_df, direction, n_days):
    """Full analysis for one direction."""
    df = trades_df[trades_df["direction"] == direction].copy()
    if len(df) < 5:
        print(f"  Too few {direction} trades ({len(df)})")
        return None

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    wr = len(wins) / len(df) * 100
    total_pnl = df["pnl"].sum()
    ppd = total_pnl / n_days

    print(f"\n  {'='*80}")
    print(f"  {direction} SIGNALS: {len(df)} trades, WR={wr:.1f}%, PnL/d={ppd:+.2f}, Total={total_pnl:+.1f}")
    print(f"  Avg win: +{wins['pnl'].mean():.2f}, Avg loss: {losses['pnl'].mean():.2f}")
    print(f"  {'='*80}")

    # Feature comparison
    print(f"\n  --- Feature: Winners vs Losers ---")
    feat_cols = ["prob", "atr_14", "rsi_14", "adx", "bb_pos", "ema_align",
                 "stoch_rsi", "di_spread", "kc_pos", "consec_up", "consec_down",
                 "range_ratio_3_10", "macd_hist_raw", "vol_spike_5",
                 "ret_1", "ret_3", "bb_width", "atr_ratio", "body_pct",
                 "ema8_dist", "ema_spread", "stoch_k", "rsi_3"]

    print(f"  {'Feature':<20} {'Winners':<10} {'Losers':<10} {'Diff%':<8} {'Insight'}")
    print(f"  {'-'*70}")

    insights = []
    for feat in feat_cols:
        if feat not in df.columns:
            continue
        w_mean = wins[feat].mean() if len(wins) > 0 else 0
        l_mean = losses[feat].mean() if len(losses) > 0 else 0
        diff = abs(w_mean - l_mean) / (abs(l_mean) + 0.001) * 100

        insight = ""
        if diff > 15:
            if direction == "SELL":
                if feat == "di_spread" and l_mean > w_mean:
                    insight = "Losers sell into uptrend"
                elif feat == "ema_align" and l_mean > w_mean:
                    insight = "Losers sell above EMAs"
                elif feat == "kc_pos" and l_mean > w_mean:
                    insight = "Losers sell at KC top"
                elif feat == "bb_pos" and l_mean > w_mean:
                    insight = "Losers sell at BB top"
            elif direction == "BUY":
                if feat == "di_spread" and l_mean < w_mean:
                    insight = "Losers buy into downtrend"
                elif feat == "ema_align" and l_mean < w_mean:
                    insight = "Losers buy below EMAs"
                elif feat == "kc_pos" and l_mean < w_mean:
                    insight = "Losers buy at KC bottom"

            marker = "***" if diff > 40 else "**" if diff > 25 else "*"
        else:
            marker = ""

        if diff > 10:
            print(f"  {feat:<20} {w_mean:<10.3f} {l_mean:<10.3f} {diff:<8.1f} {marker} {insight}")
            insights.append({"feat": feat, "w": w_mean, "l": l_mean, "diff": diff})

    # Distribution analysis
    print(f"\n  --- Key Distributions ---")

    # EMA alignment
    print(f"  EMA Alignment:")
    for val in [-3, -1, 0, 1, 3]:
        sub = df[df["ema_align"] == val]
        if len(sub) >= 3:
            wr_s = (sub["pnl"] > 0).mean() * 100
            avg = sub["pnl"].mean()
            print(f"    align={val:+d}: {len(sub)} trades, WR={wr_s:.0f}%, avg={avg:+.2f}")

    # DI spread bins
    print(f"  DI Spread:")
    for lo, hi in [(-40, -10), (-10, 0), (0, 10), (10, 40)]:
        sub = df[(df["di_spread"] >= lo) & (df["di_spread"] < hi)]
        if len(sub) >= 3:
            wr_s = (sub["pnl"] > 0).mean() * 100
            avg = sub["pnl"].mean()
            print(f"    DI_sp {lo:+d} to {hi:+d}: {len(sub)} trades, WR={wr_s:.0f}%, avg={avg:+.2f}")

    # ADX
    print(f"  ADX:")
    for lo, hi in [(0, 20), (20, 30), (30, 50)]:
        sub = df[(df["adx"] >= lo) & (df["adx"] < hi)]
        if len(sub) >= 3:
            wr_s = (sub["pnl"] > 0).mean() * 100
            avg = sub["pnl"].mean()
            print(f"    ADX {lo}-{hi}: {len(sub)} trades, WR={wr_s:.0f}%, avg={avg:+.2f}")

    # RSI
    print(f"  RSI(14):")
    for lo, hi in [(20, 40), (40, 50), (50, 60), (60, 80)]:
        sub = df[(df["rsi_14"] >= lo) & (df["rsi_14"] < hi)]
        if len(sub) >= 3:
            wr_s = (sub["pnl"] > 0).mean() * 100
            avg = sub["pnl"].mean()
            print(f"    RSI {lo}-{hi}: {len(sub)} trades, WR={wr_s:.0f}%, avg={avg:+.2f}")

    # Stoch RSI
    print(f"  Stochastic RSI:")
    for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]:
        sub = df[(df["stoch_rsi"] >= lo) & (df["stoch_rsi"] < hi)]
        if len(sub) >= 3:
            wr_s = (sub["pnl"] > 0).mean() * 100
            avg = sub["pnl"].mean()
            print(f"    StRSI {lo:.1f}-{hi:.1f}: {len(sub)} trades, WR={wr_s:.0f}%, avg={avg:+.2f}")

    # MFE/MAE
    print(f"\n  --- MFE/MAE ---")
    print(f"  Winners: MFE={wins['mfe'].mean():.2f}, MAE={wins['mae'].mean():.2f}")
    print(f"  Losers:  MFE={losses['mfe'].mean():.2f}, MAE={losses['mae'].mean():.2f}")
    never_green = losses[losses["mfe"] < COST]
    had_chance = losses[losses["mfe"] >= 2.0]
    print(f"  Losers never green: {len(never_green)}/{len(losses)} ({len(never_green)/max(1,len(losses))*100:.0f}%) — completely wrong direction")
    print(f"  Losers had MFE>2: {len(had_chance)}/{len(losses)} ({len(had_chance)/max(1,len(losses))*100:.0f}%) — exit problem")

    # ===== Filter tests =====
    print(f"\n  --- Post-ML Filter Tests ({direction}) ---")
    base_pnl = df["pnl"].sum()
    base_n = len(df)
    base_wr = wr

    if direction == "SELL":
        filters = {
            "di_spread <= 5": df["di_spread"] <= 5,
            "di_spread <= 0": df["di_spread"] <= 0,
            "di_spread <= 10": df["di_spread"] <= 10,
            "ema_align <= 0": df["ema_align"] <= 0,
            "ema_align <= 1": df["ema_align"] <= 1,
            "kc_pos <= 0.5": df["kc_pos"] <= 0.5,
            "kc_pos <= 1.0": df["kc_pos"] <= 1.0,
            "bb_pos <= 0.3": df["bb_pos"] <= 0.3,
            "bb_pos <= 0.5": df["bb_pos"] <= 0.5,
            "rsi_14 <= 55": df["rsi_14"] <= 55,
            "rsi_14 <= 60": df["rsi_14"] <= 60,
            "macd_hist < 0": df["macd_hist_raw"] < 0,
            "stoch_rsi < 0.8": df["stoch_rsi"] < 0.8,
            # Combos
            "di<=5 + bb<=0.3": (df["di_spread"] <= 5) & (df["bb_pos"] <= 0.3),
            "di<=5 + ema<=1": (df["di_spread"] <= 5) & (df["ema_align"] <= 1),
            "di<=10 + kc<=0.5": (df["di_spread"] <= 10) & (df["kc_pos"] <= 0.5),
            "di<=10 + rsi<=60": (df["di_spread"] <= 10) & (df["rsi_14"] <= 60),
            "ema<=1 + kc<=0.5": (df["ema_align"] <= 1) & (df["kc_pos"] <= 0.5),
            "ema<=0 + di<=5": (df["ema_align"] <= 0) & (df["di_spread"] <= 5),
            "bb<=0.5 + di<=10": (df["bb_pos"] <= 0.5) & (df["di_spread"] <= 10),
        }
    else:  # BUY
        filters = {
            "di_spread >= -5": df["di_spread"] >= -5,
            "di_spread >= 0": df["di_spread"] >= 0,
            "di_spread >= -10": df["di_spread"] >= -10,
            "ema_align >= 0": df["ema_align"] >= 0,
            "ema_align >= -1": df["ema_align"] >= -1,
            "kc_pos >= -0.5": df["kc_pos"] >= -0.5,
            "kc_pos >= -1.0": df["kc_pos"] >= -1.0,
            "bb_pos >= -0.3": df["bb_pos"] >= -0.3,
            "bb_pos >= -0.5": df["bb_pos"] >= -0.5,
            "rsi_14 >= 40": df["rsi_14"] >= 40,
            "rsi_14 >= 45": df["rsi_14"] >= 45,
            "macd_hist > 0": df["macd_hist_raw"] > 0,
            "stoch_rsi > 0.2": df["stoch_rsi"] > 0.2,
            # Combos
            "di>=-5 + bb>=-0.3": (df["di_spread"] >= -5) & (df["bb_pos"] >= -0.3),
            "di>=-5 + ema>=-1": (df["di_spread"] >= -5) & (df["ema_align"] >= -1),
            "di>=-10 + kc>=-0.5": (df["di_spread"] >= -10) & (df["kc_pos"] >= -0.5),
            "ema>=0 + kc>=-0.5": (df["ema_align"] >= 0) & (df["kc_pos"] >= -0.5),
            "ema>=-1 + di>=0": (df["ema_align"] >= -1) & (df["di_spread"] >= 0),
            "rsi>=45 + ema>=-1": (df["rsi_14"] >= 45) & (df["ema_align"] >= -1),
            "bb>=-0.5 + di>=-10": (df["bb_pos"] >= -0.5) & (df["di_spread"] >= -10),
        }

    print(f"  Baseline: {base_n} trades, WR={base_wr:.1f}%, PnL={base_pnl:+.1f}")
    print(f"\n  {'Filter':<28} {'N':<5} {'WR%':<7} {'PnL':<10} {'PnL/d':<8} {'PF':<6} {'Imp':<8}")
    print(f"  {'-'*75}")

    results = []
    for name, mask in filters.items():
        sub = df[mask]
        if len(sub) < 5:
            continue
        n = len(sub)
        wr_s = (sub["pnl"] > 0).mean() * 100
        pnl = sub["pnl"].sum()
        ppd_s = pnl / n_days
        w_sum = sub[sub["pnl"] > 0]["pnl"].sum()
        l_sum = abs(sub[sub["pnl"] <= 0]["pnl"].sum())
        pf = w_sum / l_sum if l_sum > 0 else 999
        imp = pnl - base_pnl
        results.append({"name": name, "n": n, "wr": wr_s, "pnl": pnl,
                        "ppd": ppd_s, "pf": pf, "imp": imp})

    results.sort(key=lambda x: x["ppd"], reverse=True)
    for r in results[:15]:
        marker = " <<<" if r["pf"] > 1.5 and r["n"] >= 10 else ""
        print(f"  {r['name']:<28} {r['n']:<5} {r['wr']:<7.1f} {r['pnl']:<+10.1f} "
              f"{r['ppd']:<+8.2f} {r['pf']:<6.2f} {r['imp']:<+8.1f}{marker}")

    return results


def main():
    print("=" * 90)
    print("FULL ML ANALYSIS: BUY + SELL with Post-ML Filters")
    print("=" * 90)

    df_raw = load("5m", days=180)
    df = compute_all_features(df_raw)
    df["label"] = label_fixed_horizon(df, horizon=6)

    print("\n[1] Walk-forward...")
    idx, pred = walk_forward(df)
    n_days = df.iloc[idx]["date"].nunique()
    print(f"  {len(pred)} predictions, {n_days} test days")

    # ===== Simulate AM, both directions =====
    print("\n[2] Simulating AM trades (both BUY and SELL, thr=0.55)...")
    trades = simulate_all(df, idx, pred, threshold=0.55, session_filter=["AM"])

    buys = trades[trades["direction"] == "BUY"]
    sells = trades[trades["direction"] == "SELL"]
    print(f"  Total: {len(trades)} trades (BUY={len(buys)}, SELL={len(sells)})")
    print(f"  Overall: WR={((trades['pnl']>0).mean()*100):.1f}%, PnL/d={trades['pnl'].sum()/n_days:+.2f}")

    # ===== Analyze each direction =====
    sell_results = analyze_direction(trades, "SELL", n_days)
    buy_results = analyze_direction(trades, "BUY", n_days)

    # ===== Combined recommendation =====
    print(f"\n{'=' * 90}")
    print("FINAL RECOMMENDATION: Optimal Filter Config")
    print(f"{'=' * 90}")

    # Find best for each direction
    if sell_results:
        best_sell = max([r for r in sell_results if r["n"] >= 10], key=lambda x: x["ppd"], default=None)
        if best_sell:
            print(f"\n  SELL filter: {best_sell['name']}")
            print(f"    → {best_sell['n']} trades, WR={best_sell['wr']:.1f}%, "
                  f"PnL/d={best_sell['ppd']:+.2f}, PF={best_sell['pf']:.2f}")

    if buy_results:
        best_buy = max([r for r in buy_results if r["n"] >= 10], key=lambda x: x["ppd"], default=None)
        if best_buy:
            print(f"\n  BUY filter: {best_buy['name']}")
            print(f"    → {best_buy['n']} trades, WR={best_buy['wr']:.1f}%, "
                  f"PnL/d={best_buy['ppd']:+.2f}, PF={best_buy['pf']:.2f}")

    # Simulate combined
    print(f"\n  --- Combined (both directions with their filters) ---")
    # Apply best filters
    if sell_results and buy_results and best_sell and best_buy:
        # Reconstruct masks
        sell_mask_name = best_sell["name"]
        buy_mask_name = best_buy["name"]
        print(f"  Applying: SELL → {sell_mask_name}, BUY → {buy_mask_name}")

        # Combined PnL
        combined_ppd = best_sell["ppd"] + best_buy["ppd"]
        combined_n = best_sell["n"] + best_buy["n"]
        combined_pnl = best_sell["pnl"] + best_buy["pnl"]
        unfiltered_pnl = trades["pnl"].sum()

        print(f"\n  UNFILTERED: {len(trades)} trades, PnL/d={unfiltered_pnl/n_days:+.2f}")
        print(f"  FILTERED:   ~{combined_n} trades, PnL/d≈{combined_ppd:+.2f}")
        print(f"  Improvement: {combined_ppd - unfiltered_pnl/n_days:+.2f} pts/day")

    # Also test PM briefly
    print(f"\n\n{'=' * 90}")
    print("BONUS: PM Session Quick Check")
    print(f"{'=' * 90}")
    trades_pm = simulate_all(df, idx, pred, threshold=0.55, session_filter=["PM"])
    if not trades_pm.empty:
        pm_wr = (trades_pm["pnl"] > 0).mean() * 100
        pm_ppd = trades_pm["pnl"].sum() / n_days
        print(f"  PM ALL: {len(trades_pm)} trades, WR={pm_wr:.1f}%, PnL/d={pm_ppd:+.2f}")
        for d in ["BUY", "SELL"]:
            sub = trades_pm[trades_pm["direction"] == d]
            if len(sub) >= 5:
                wr_s = (sub["pnl"] > 0).mean() * 100
                ppd_s = sub["pnl"].sum() / n_days
                print(f"  PM {d}: {len(sub)} trades, WR={wr_s:.1f}%, PnL/d={ppd_s:+.2f}")


if __name__ == "__main__":
    main()
