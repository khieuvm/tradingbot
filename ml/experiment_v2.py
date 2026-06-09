"""
ML Experiment: LOOCV + Multi-Target + Enhanced Features.

Tests multiple configurations:
  1. Original 25 features vs Enhanced 33 features
  2. Target: win/loss vs big_loss (PnL < -3pts)
  3. Validation: Purged LOOCV (embargo 5 days)
  4. Models: LightGBM, ExtraTrees

Usage:
    python -m ml.experiment_v2
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from collections import defaultdict


DATA_PATH = Path("ml/data/cb_trades_labeled.csv")
EMBARGO_DAYS = 5
BIG_LOSS_THRESHOLD = -3.0


def _get_models():
    """Return available model factories."""
    models = {}
    try:
        import lightgbm as lgb
        models["LightGBM"] = lambda n_train: lgb.LGBMClassifier(
            n_estimators=100, max_depth=3, num_leaves=6,
            min_child_samples=max(10, int(0.07 * n_train)),
            learning_rate=0.05, colsample_bytree=0.6, subsample=0.8,
            reg_alpha=1.0, reg_lambda=3.0, class_weight="balanced",
            verbose=-1, n_jobs=-1,
        )
    except ImportError:
        pass

    from sklearn.ensemble import ExtraTreesClassifier
    models["ExtraTrees"] = lambda n_train: ExtraTreesClassifier(
        n_estimators=150, max_depth=4,
        min_samples_leaf=max(8, int(0.05 * n_train)),
        max_features="sqrt", class_weight="balanced", n_jobs=-1,
    )

    from sklearn.ensemble import GradientBoostingClassifier
    models["GradientBoosting"] = lambda n_train: GradientBoostingClassifier(
        n_estimators=100, max_depth=3, min_samples_leaf=max(10, int(0.07 * n_train)),
        learning_rate=0.05, subsample=0.8, max_features="sqrt",
    )

    return models


def purged_loocv(X, y, pnls, dates, model_factory, embargo_days=EMBARGO_DAYS):
    """Run purged Leave-One-Out CV.

    Returns: array of predicted probabilities (same length as y).
    """
    n = len(y)
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

        X_train, y_train = X[train_mask], y[train_mask]
        model = model_factory(len(X_train))
        model.fit(X_train, y_train)
        predictions[i] = model.predict_proba(X[i:i+1])[0][1]

    return predictions


def evaluate_predictions(predictions, y, pnls, label_name, model_name, feature_set):
    """Evaluate predictions at multiple thresholds."""
    valid = ~np.isnan(predictions)
    pred_v = predictions[valid]
    y_v = y[valid]
    pnl_v = pnls[valid]

    if len(pred_v) == 0:
        return None

    baseline_pnl = pnl_v.sum()
    baseline_wr = y_v.mean() * 100

    results = {
        "model": model_name,
        "features": feature_set,
        "target": label_name,
        "n_valid": int(valid.sum()),
        "baseline_wr": baseline_wr,
        "baseline_pnl": baseline_pnl,
        "thresholds": {},
    }

    best_improvement = -9999
    best_thr = None

    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        if label_name == "big_loss":
            # For big_loss: veto if P(big_loss) > thr
            veto_mask = pred_v > thr
        else:
            # For win/loss: veto if P(win) < thr
            veto_mask = pred_v < thr

        retained_mask = ~veto_mask
        n_retained = retained_mask.sum()
        n_vetoed = veto_mask.sum()

        if n_retained == 0 or n_vetoed == 0:
            continue

        net_pnl = pnl_v[retained_mask].sum()
        retained_wr = y_v[retained_mask].mean() * 100 if label_name != "big_loss" else None

        # Veto precision: what % of vetoed trades were actually bad?
        if label_name == "big_loss":
            veto_precision = y_v[veto_mask].mean() * 100  # % of vetoed that ARE big losses
        else:
            veto_precision = (1 - y_v[veto_mask].mean()) * 100  # % of vetoed that are losers

        improvement = net_pnl - baseline_pnl

        results["thresholds"][thr] = {
            "retained": int(n_retained),
            "vetoed": int(n_vetoed),
            "net_pnl": float(net_pnl),
            "improvement": float(improvement),
            "veto_precision": float(veto_precision),
            "retained_wr": float(retained_wr) if retained_wr is not None else None,
        }

        if improvement > best_improvement:
            best_improvement = improvement
            best_thr = thr

    results["best_threshold"] = best_thr
    results["best_improvement"] = float(best_improvement) if best_thr else 0
    return results


def run_experiments():
    """Run all experiment configurations."""
    # Load and regenerate dataset with new features
    print("=" * 80)
    print("ML EXPERIMENT V2: LOOCV + Multi-Target + Enhanced Features")
    print("=" * 80)

    # Regenerate dataset with V2 features
    print("\n[1] Regenerating dataset with enhanced features...")
    from ml.labeler import generate_labeled_dataset
    df = generate_labeled_dataset(days=180, save_path="ml/data/cb_trades_labeled.csv")
    if df is None:
        print("ERROR: Failed to generate dataset")
        return

    print(f"\n[2] Loading dataset...")
    df = pd.read_csv("ml/data/cb_trades_labeled.csv")
    df["date"] = pd.to_datetime(df["date"])
    dates = df["date"].values
    pnls = df["pnl"].values

    # Define feature sets
    FEATURES_V1 = [
        "atr_14", "atr_pct", "bb_width", "atr_ratio_5_14", "rsi_14", "rsi_3",
        "macd_hist", "adx", "ema50_dist", "ema_spread", "ret_3", "ret_5", "ret_13",
        "vol_ratio", "compression_depth", "pre_move_ratio", "session_minutes",
        "is_pm", "body_pct", "range_pct", "upper_wick_ratio", "bb_pos", "ret_1",
        "stoch_k", "time_to_session_end",
    ]

    FEATURES_V2 = FEATURES_V1 + [
        "trend_strength_10", "trend_consistency", "atr_acceleration",
        "bars_since_last_compression", "daily_trade_idx", "session_return",
        "high_low_ratio_5", "close_vs_range",
    ]

    # Check which features exist in data
    available_v1 = [f for f in FEATURES_V1 if f in df.columns]
    available_v2 = [f for f in FEATURES_V2 if f in df.columns]
    print(f"  V1 features available: {len(available_v1)}/{len(FEATURES_V1)}")
    print(f"  V2 features available: {len(available_v2)}/{len(FEATURES_V2)}")

    # Define targets
    targets = {
        "win_loss": df["label"].values,  # 1=win, 0=loss
        "big_loss": (df["pnl"] < BIG_LOSS_THRESHOLD).astype(int).values,  # 1=big loss, 0=ok
    }
    print(f"\n  Target 'win_loss': {(targets['win_loss']==1).sum()} wins, {(targets['win_loss']==0).sum()} losses")
    print(f"  Target 'big_loss': {(targets['big_loss']==1).sum()} big losses (PnL < {BIG_LOSS_THRESHOLD}), {(targets['big_loss']==0).sum()} others")

    # Get models
    models = _get_models()
    print(f"\n  Models: {list(models.keys())}")

    # Run experiments
    feature_sets = {"V1(25)": available_v1, "V2(33)": available_v2}
    all_results = []

    print(f"\n[3] Running Purged LOOCV (embargo={EMBARGO_DAYS}d)...")
    print("    This will take a few minutes...\n")

    total_configs = len(models) * len(feature_sets) * len(targets)
    config_i = 0

    for model_name, model_factory in models.items():
        for feat_name, feat_cols in feature_sets.items():
            X = df[feat_cols].values
            X = np.nan_to_num(X, nan=0.0)

            for target_name, target_y in targets.items():
                config_i += 1
                print(f"  [{config_i}/{total_configs}] {model_name} | {feat_name} | target={target_name}...", end=" ")

                predictions = purged_loocv(X, target_y, pnls, dates, model_factory)
                result = evaluate_predictions(predictions, target_y, pnls, target_name, model_name, feat_name)

                if result:
                    all_results.append(result)
                    best = result["best_improvement"]
                    print(f"best_improvement={best:+.1f}pts (thr={result['best_threshold']})")
                else:
                    print("FAILED")

    # Print summary table
    print(f"\n{'=' * 90}")
    print(f"RESULTS SUMMARY (sorted by best PnL improvement)")
    print(f"{'=' * 90}")
    print(f"  {'Model':<18} {'Features':<8} {'Target':<10} {'Best_Thr':<9} {'Improve':<10} {'Veto_Prec':<10} {'Retained'}")
    print(f"  {'-'*85}")

    all_results.sort(key=lambda r: r["best_improvement"], reverse=True)

    for r in all_results:
        thr = r["best_threshold"]
        if thr is None:
            continue
        thr_data = r["thresholds"].get(thr, {})
        vp = thr_data.get("veto_precision", 0)
        retained = thr_data.get("retained", 0)
        total = r["n_valid"]
        print(f"  {r['model']:<18} {r['features']:<8} {r['target']:<10} {thr:<9.2f} "
              f"{r['best_improvement']:>+8.1f}pts  {vp:>7.1f}%   {retained}/{total}")

    # Detailed breakdown for best config
    if all_results:
        best = all_results[0]
        print(f"\n{'=' * 90}")
        print(f"BEST CONFIG: {best['model']} | {best['features']} | target={best['target']}")
        print(f"{'=' * 90}")
        print(f"  Baseline: {best['n_valid']} trades, WR={best['baseline_wr']:.1f}%, PnL={best['baseline_pnl']:.1f}pts")
        print(f"\n  {'Threshold':<10} {'Retained':<10} {'Vetoed':<8} {'Net_PnL':<10} {'Improve':<10} {'Veto_Prec':<12}")
        print(f"  {'-'*65}")
        for thr in sorted(best["thresholds"].keys()):
            d = best["thresholds"][thr]
            print(f"  {thr:<10.2f} {d['retained']:<10} {d['vetoed']:<8} "
                  f"{d['net_pnl']:<+10.1f} {d['improvement']:<+10.1f} {d['veto_precision']:<12.1f}")

    # Save results
    import json
    report_path = Path("ml/reports/experiment_v2_results.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {report_path}")

    return all_results


if __name__ == "__main__":
    run_experiments()
