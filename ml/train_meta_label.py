"""
Train meta-labeling model for CB signal filtering.

Uses purged walk-forward CV to evaluate model performance on out-of-sample data.
Trains LightGBM classifier to predict P(CB trade is profitable).

Usage:
    python -m ml.train_meta_label
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ml.features import META_LABEL_FEATURES


MODEL_PATH = Path("ml/models/meta_label_latest.pkl")
REPORT_PATH = Path("ml/reports/validation_report.json")
DATA_PATH = Path("ml/data/cb_trades_labeled.csv")


def _try_import_lgb():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        print("LightGBM not installed. Install with: pip install lightgbm")
        print("Falling back to sklearn ExtraTreesClassifier...")
        return None


def train_and_validate(
    data_path=DATA_PATH,
    train_days=120,
    test_days=30,
    embargo_days=5,
    threshold=0.45,
):
    """Train model with purged walk-forward validation.

    Args:
        data_path: Path to labeled CSV.
        train_days: Training window in calendar days.
        test_days: Test window in calendar days.
        embargo_days: Gap between train and test to prevent leakage.
        threshold: P(win) threshold below which trades are vetoed.

    Returns:
        dict with validation metrics.
    """
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df)} trades from {data_path}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  WR: {(df['label']==1).mean()*100:.1f}% | Wins: {(df['label']==1).sum()} | Losses: {(df['label']==0).sum()}")

    # Check feature completeness
    missing = [f for f in META_LABEL_FEATURES if f not in df.columns]
    if missing:
        print(f"ERROR: Missing features: {missing}")
        return None

    X = df[META_LABEL_FEATURES].values
    y = df["label"].values
    dates = df["date"].values

    # Fill NaN in features
    X = np.nan_to_num(X, nan=0.0)

    # =========================================================================
    # PURGED WALK-FORWARD CROSS-VALIDATION
    # =========================================================================
    print(f"\n--- Purged Walk-Forward CV (train={train_days}d, test={test_days}d, embargo={embargo_days}d) ---")

    min_date = pd.Timestamp(dates.min())
    max_date = pd.Timestamp(dates.max())
    total_days = (max_date - min_date).days

    folds = []
    fold_start = min_date

    while True:
        train_end = fold_start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_days)

        if test_end > max_date:
            break

        train_mask = (dates >= np.datetime64(fold_start)) & (dates < np.datetime64(train_end))
        test_mask = (dates >= np.datetime64(test_start)) & (dates < np.datetime64(test_end))

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train >= 30 and n_test >= 5:
            folds.append((train_mask, test_mask, fold_start, test_start, test_end))

        fold_start += timedelta(days=test_days)

    if not folds:
        print("Not enough data for walk-forward CV. Using Leave-One-Out with purging.")
        return _purged_loocv(X, y, dates, df, embargo_days, threshold)

    print(f"  Folds: {len(folds)}")

    all_predictions = []
    all_actuals = []
    all_pnls = []

    lgb = _try_import_lgb()

    for fold_i, (train_mask, test_mask, fold_start, test_start, test_end) in enumerate(folds):
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_pnl = df.loc[test_mask, "pnl"].values

        if lgb:
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                num_leaves=6,
                min_child_samples=max(15, int(0.07 * len(X_train))),
                learning_rate=0.05,
                colsample_bytree=0.6,
                subsample=0.8,
                reg_alpha=1.0,
                reg_lambda=3.0,
                class_weight="balanced",
                verbose=-1,
                n_jobs=-1,
            )
        else:
            from sklearn.ensemble import ExtraTreesClassifier
            model = ExtraTreesClassifier(
                n_estimators=150,
                max_depth=4,
                min_samples_leaf=max(10, int(0.07 * len(X_train))),
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
            )

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        all_predictions.extend(proba.tolist())
        all_actuals.extend(y_test.tolist())
        all_pnls.extend(test_pnl.tolist())

        fold_wr = y_test.mean() * 100
        print(f"  Fold {fold_i+1}: train={train_mask.sum()}, test={test_mask.sum()}, "
              f"test_WR={fold_wr:.1f}%, mean_proba={proba.mean():.3f}")

    return _evaluate(all_predictions, all_actuals, all_pnls, threshold, X, y, lgb, df)


def _purged_loocv(X, y, dates, df, embargo_days, threshold):
    """Purged Leave-One-Out CV for small datasets."""
    print("  Running Purged LOOCV (this may take a moment)...")

    lgb = _try_import_lgb()
    predictions = np.zeros(len(y))
    valid_mask = np.ones(len(y), dtype=bool)

    for i in range(len(y)):
        test_date = pd.Timestamp(dates[i])
        embargo_start = test_date - timedelta(days=embargo_days)
        embargo_end = test_date + timedelta(days=embargo_days)

        train_mask = (
            (dates < np.datetime64(embargo_start)) |
            (dates > np.datetime64(embargo_end))
        )
        train_mask[i] = False

        if train_mask.sum() < 30:
            valid_mask[i] = False
            continue

        X_train, y_train = X[train_mask], y[train_mask]

        if lgb:
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                num_leaves=6,
                min_child_samples=max(10, int(0.07 * len(X_train))),
                learning_rate=0.05,
                colsample_bytree=0.6,
                subsample=0.8,
                reg_alpha=1.0,
                reg_lambda=3.0,
                class_weight="balanced",
                verbose=-1,
                n_jobs=-1,
            )
        else:
            from sklearn.ensemble import ExtraTreesClassifier
            model = ExtraTreesClassifier(
                n_estimators=150,
                max_depth=4,
                min_samples_leaf=max(10, int(0.07 * len(X_train))),
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
            )

        model.fit(X_train, y_train)
        predictions[i] = model.predict_proba(X[i:i+1])[0][1]

    valid_preds = predictions[valid_mask].tolist()
    valid_actuals = y[valid_mask].tolist()
    valid_pnls = df.loc[valid_mask, "pnl"].values.tolist()

    print(f"  LOOCV completed: {sum(valid_mask)} valid predictions out of {len(y)}")
    return _evaluate(valid_preds, valid_actuals, valid_pnls, threshold, X, y, lgb, df)


def _evaluate(predictions, actuals, pnls, threshold, X_all, y_all, lgb, df):
    """Evaluate predictions and train final model."""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    pnls = np.array(pnls)

    # Metrics at various thresholds
    print(f"\n--- Threshold Analysis ---")
    print(f"  {'Threshold':<10} {'Retained':<10} {'Vetoed':<8} {'Ret_WR':<10} "
          f"{'Veto_Precision':<15} {'Net_PnL':<10} {'Baseline_PnL':<12}")
    print("  " + "-" * 80)

    baseline_pnl = pnls.sum()
    best_threshold = threshold
    best_net_pnl = -9999

    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        retained_mask = predictions >= thr
        vetoed_mask = ~retained_mask

        n_retained = retained_mask.sum()
        n_vetoed = vetoed_mask.sum()

        if n_retained == 0 or n_vetoed == 0:
            continue

        retained_wr = actuals[retained_mask].mean() * 100
        veto_precision = (1 - actuals[vetoed_mask].mean()) * 100  # % of vetoes that were actually losers
        net_pnl = pnls[retained_mask].sum()

        if net_pnl > best_net_pnl:
            best_net_pnl = net_pnl
            best_threshold = thr

        print(f"  {thr:<10.2f} {n_retained:<10} {n_vetoed:<8} {retained_wr:<10.1f} "
              f"{veto_precision:<15.1f} {net_pnl:<10.1f} {baseline_pnl:<12.1f}")

    # Summary with recommended threshold
    retained_at_best = predictions >= best_threshold
    retained_wr = actuals[retained_at_best].mean() * 100 if retained_at_best.sum() > 0 else 0
    retention_rate = retained_at_best.mean() * 100

    print(f"\n--- Recommendation ---")
    print(f"  Best threshold: {best_threshold:.2f}")
    print(f"  Retained WR: {retained_wr:.1f}% (baseline: {actuals.mean()*100:.1f}%)")
    print(f"  Retention rate: {retention_rate:.1f}%")
    print(f"  Net PnL: {best_net_pnl:.1f} pts (baseline: {baseline_pnl:.1f} pts)")
    print(f"  PnL improvement: {best_net_pnl - baseline_pnl:+.1f} pts")

    # Train final model on ALL data
    print(f"\n--- Training Final Model (all {len(X_all)} samples) ---")
    X_clean = np.nan_to_num(X_all, nan=0.0)

    if lgb:
        import lightgbm
        final_model = lightgbm.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            num_leaves=6,
            min_child_samples=max(15, int(0.07 * len(X_clean))),
            learning_rate=0.05,
            colsample_bytree=0.6,
            subsample=0.8,
            reg_alpha=1.0,
            reg_lambda=3.0,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import ExtraTreesClassifier
        final_model = ExtraTreesClassifier(
            n_estimators=150,
            max_depth=4,
            min_samples_leaf=max(10, int(0.07 * len(X_clean))),
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
        )

    final_model.fit(X_clean, y_all)

    # Feature importance
    if hasattr(final_model, "feature_importances_"):
        importances = final_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print(f"\n  Top 10 features:")
        for rank, idx in enumerate(sorted_idx[:10]):
            print(f"    {rank+1}. {META_LABEL_FEATURES[idx]:<25} {importances[idx]:.4f}")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": final_model,
            "features": META_LABEL_FEATURES,
            "threshold": best_threshold,
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X_all),
            "baseline_wr": float(actuals.mean() * 100),
            "validated_wr": float(retained_wr),
        }, f)
    print(f"  Model saved: {MODEL_PATH}")

    # Save report
    report = {
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(X_all),
        "baseline_wr": float(actuals.mean() * 100),
        "best_threshold": float(best_threshold),
        "retained_wr": float(retained_wr),
        "retention_rate": float(retention_rate),
        "baseline_pnl": float(baseline_pnl),
        "net_pnl": float(best_net_pnl),
        "pnl_improvement": float(best_net_pnl - baseline_pnl),
        "n_folds": "LOOCV" if len(predictions) == len(X_all) else "walk-forward",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {REPORT_PATH}")

    return report


if __name__ == "__main__":
    # First generate dataset if not exists
    if not DATA_PATH.exists():
        print("No labeled dataset found. Generating...")
        from ml.labeler import generate_labeled_dataset
        generate_labeled_dataset()

    if DATA_PATH.exists():
        train_and_validate()
    else:
        print("Failed to generate dataset. Check data availability.")
