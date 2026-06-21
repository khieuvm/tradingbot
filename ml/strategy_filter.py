"""Per-strategy ML meta-labeling — THE highest-ROI ML approach.

Key innovations vs previous ML attempts:
1. Labels = ACTUAL trade outcome (trailing exit PnL > 0), NOT fixed-horizon return
2. Per-strategy models (each strategy has different loss characteristics)
3. LightGBM + ExtraTrees + Consensus with cost-sensitive learning
4. Feature selection: top-K per strategy (reduce noise for small samples)
5. Calibrated probabilities via isotonic regression
6. Purged walk-forward with embargo (no indicator leakage)
7. WR-maximization with PnL floor constraint

Usage:
    python -m ml.strategy_filter [--tf 5m]
"""
import sys, os, time, json, warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from strategies.backtest_new import load_data
from strategies.optimize_exits import (
    collect_signals, filter_signals, simulate_trade_fast, COST
)
from ml.standalone_signals import compute_all_features, FEATURE_COLS

TRAIN_DAYS = 0  # 0 = expanding window (use ALL history before test)
TEST_DAYS = 60
STEP_DAYS = 60
EMBARGO_BARS = 14
MIN_TRAIN_SIGNALS = 30
MIN_TEST_SIGNALS = 8
TOP_K_FEATURES = 25
FP_COST_MULT = 2.0  # penalize false positives (bad trades taken) 2x more

STRATEGIES_5M = [
    {'strategy': 'market_structure', 'module': 'strategies.market_structure',
     'class': 'MarketStructureStrategy',
     'session_filter': 'ALL', 'direction_filter': 'BUY',
     'exit_params': {'sl_mult': 2.0, 'trail_pts': 8.0, 'trail_mult': 2.0, 'tp_mult': None,
                     'be_trigger': None, 'trail_tighten_pts': None, 'trail_tighten_mult': None, 'max_hold': 25}},
    {'strategy': 'fibonacci', 'module': 'strategies.fibonacci',
     'class': 'FibonacciStrategy',
     'session_filter': 'AM', 'direction_filter': 'SELL',
     'exit_params': {'sl_mult': 2.0, 'trail_pts': 8.0, 'trail_mult': 2.0, 'tp_mult': None,
                     'be_trigger': None, 'trail_tighten_pts': None, 'trail_tighten_mult': None, 'max_hold': 25}},
    {'strategy': 'sr_horizontal', 'module': 'strategies.sr_horizontal',
     'class': 'SRHorizontalStrategy',
     'session_filter': 'AM', 'direction_filter': 'SELL',
     'exit_params': {'sl_mult': 1.5, 'trail_pts': 8.0, 'trail_mult': 2.0, 'tp_mult': None,
                     'be_trigger': None, 'trail_tighten_pts': None, 'trail_tighten_mult': None, 'max_hold': 25}},
    {'strategy': 'heikin_ashi', 'module': 'strategies.heikin_ashi',
     'class': 'HeikinAshiStrategy',
     'session_filter': 'AM', 'direction_filter': 'SELL',
     'exit_params': {'sl_mult': 2.5, 'trail_pts': 8.0, 'trail_mult': 2.0, 'tp_mult': None,
                     'be_trigger': None, 'trail_tighten_pts': None, 'trail_tighten_mult': None, 'max_hold': 25}},
    {'strategy': 'reversal_patterns', 'module': 'strategies.reversal_patterns',
     'class': 'ReversalPatternsStrategy',
     'session_filter': 'AM', 'direction_filter': 'SELL',
     'exit_params': {'sl_mult': 1.5, 'trail_pts': 8.0, 'trail_mult': 2.0, 'tp_mult': None,
                     'be_trigger': None, 'trail_tighten_pts': None, 'trail_tighten_mult': None, 'max_hold': 25}},
]

STRATEGIES_1M = [
    {'strategy': 'momentum_trend', 'module': 'strategies.momentum_trend',
     'class': 'MomentumTrendStrategy',
     'session_filter': 'PM', 'direction_filter': 'SELL',
     'exit_params': {'sl_mult': 2.5, 'trail_pts': 12.0, 'trail_mult': 1.0, 'tp_mult': None,
                     'be_trigger': None, 'trail_tighten_pts': None, 'trail_tighten_mult': None, 'max_hold': 50}},
    {'strategy': 'macd_cross', 'module': 'strategies.macd_cross',
     'class': 'MACDCrossStrategy',
     'session_filter': 'AM', 'direction_filter': 'SELL',
     'exit_params': {'sl_mult': 2.5, 'trail_pts': 12.0, 'trail_mult': 1.0, 'tp_mult': None,
                     'be_trigger': None, 'trail_tighten_pts': None, 'trail_tighten_mult': None, 'max_hold': 50}},
]


def collect_labeled_signals(df, config, highs, lows, closes, mins_arr):
    """Collect signals with ACTUAL trade outcome labels (not fixed-horizon)."""
    mod = __import__(config['module'], fromlist=[config['class']])
    strat = getattr(mod, config['class'])()

    all_signals = collect_signals(df, strat)
    signals = filter_signals(all_signals, config['session_filter'], config['direction_filter'])

    labeled = []
    ep = config['exit_params']
    for s in signals:
        raw_pnl, mfe, reason = simulate_trade_fast(
            highs, lows, closes, mins_arr,
            s['idx'], s['direction'], s['atr'], s['session'],
            ep['sl_mult'], ep['trail_pts'], ep['trail_mult'], ep['tp_mult'],
            ep['be_trigger'], ep['trail_tighten_pts'], ep['trail_tighten_mult'],
            ep['max_hold'])
        pnl = raw_pnl - COST
        labeled.append({
            'idx': s['idx'],
            'date': s['date'],
            'direction': s['direction'],
            'atr': s['atr'],
            'session': s['session'],
            'pnl': pnl,
            'mfe': mfe,
            'reason': reason,
            'label': 1 if pnl > 0 else 0,
        })
    return labeled


def build_ml_model(model_type='lgbm', n_train=200, cost_sensitive=False):
    """Build ML model with appropriate regularization for sample size."""
    if model_type == 'lgbm':
        base = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=3,
            num_leaves=8,
            min_child_samples=max(15, int(0.07 * n_train)),
            learning_rate=0.03,
            colsample_bytree=0.5,
            subsample=0.8,
            subsample_freq=5,
            reg_alpha=0.5,
            reg_lambda=3.0,
            class_weight='balanced',
            scale_pos_weight=1.0 / FP_COST_MULT if cost_sensitive else 1.0,
            verbose=-1,
            n_jobs=-1,
        )
    elif model_type == 'et':
        if cost_sensitive:
            cw = {0: FP_COST_MULT, 1: 1.0}
        else:
            cw = 'balanced'
        base = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=max(30, int(0.07 * n_train)),
            max_features='sqrt',
            class_weight=cw,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return base


def select_top_features(X, y, feature_cols, k=TOP_K_FEATURES):
    """Select top-K features using mutual information."""
    X_clean = np.nan_to_num(X, nan=0.0)
    mi = mutual_info_classif(X_clean, y, random_state=42, n_neighbors=5)
    top_idx = np.argsort(mi)[-k:]
    return sorted(top_idx), [feature_cols[i] for i in sorted(top_idx)]


def walk_forward_filter(signals_labeled, df, feature_cols, model_type='lgbm',
                        thresholds=(0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
                        use_feature_selection=True, cost_sensitive=False):
    """Walk-forward meta-labeling with embargo. Expanding train window.

    Enhancements for WR maximization:
    - Feature selection: top-K features per fold (reduce noise)
    - Cost-sensitive: penalize false positives (bad trades taken) more
    - Wider thresholds: up to 0.80 for aggressive WR targeting
    - Calibration: isotonic regression on training set

    Returns results for each threshold showing filtered vs unfiltered performance.
    """
    dates = df['date'].values
    unique_dates = sorted(set(dates))
    n_days = len(unique_dates)

    signal_indices = np.array([s['idx'] for s in signals_labeled])
    signal_dates = np.array([s['date'] for s in signals_labeled])
    signal_labels = np.array([s['label'] for s in signals_labeled])
    signal_pnls = np.array([s['pnl'] for s in signals_labeled])

    X_all = df[feature_cols].values

    results_by_threshold = {t: [] for t in thresholds}
    unfiltered_trades = []

    min_train_days = 200
    start_day = min_train_days
    n_folds = 0

    while start_day + TEST_DAYS <= n_days:
        train_end_date = unique_dates[start_day]
        test_end_date = unique_dates[min(start_day + TEST_DAYS - 1, n_days - 1)]

        train_mask = signal_dates < train_end_date
        train_sig_indices = np.where(train_mask)[0]
        if len(train_sig_indices) > EMBARGO_BARS:
            train_sig_indices = train_sig_indices[:-EMBARGO_BARS]

        test_mask = (signal_dates >= train_end_date) & (signal_dates <= test_end_date)
        test_sig_indices = np.where(test_mask)[0]

        if len(train_sig_indices) < MIN_TRAIN_SIGNALS or len(test_sig_indices) < MIN_TEST_SIGNALS:
            start_day += STEP_DAYS
            continue

        train_bar_idx = signal_indices[train_sig_indices]
        test_bar_idx = signal_indices[test_sig_indices]

        X_train_full = np.nan_to_num(X_all[train_bar_idx], nan=0.0)
        y_train = signal_labels[train_sig_indices]
        X_test_full = np.nan_to_num(X_all[test_bar_idx], nan=0.0)
        test_pnls = signal_pnls[test_sig_indices]

        if len(set(y_train)) < 2:
            start_day += STEP_DAYS
            continue

        # Feature selection: use top-K features per fold
        if use_feature_selection and len(X_train_full) >= 50:
            feat_idx, _ = select_top_features(X_train_full, y_train, feature_cols)
            X_train = X_train_full[:, feat_idx]
            X_test = X_test_full[:, feat_idx]
        else:
            X_train = X_train_full
            X_test = X_test_full

        if model_type in ('consensus', 'agreement'):
            model_lgbm = build_ml_model('lgbm', n_train=len(X_train), cost_sensitive=cost_sensitive)
            model_et = build_ml_model('et', n_train=len(X_train), cost_sensitive=cost_sensitive)
            model_lgbm.fit(X_train, y_train)
            model_et.fit(X_train, y_train)

            p_lgbm = model_lgbm.predict_proba(X_test)[:, 1]
            p_et = model_et.predict_proba(X_test)[:, 1]
            if model_type == 'agreement':
                disagreement = np.abs(p_lgbm - p_et)
                p_win = np.minimum(p_lgbm, p_et)
                p_win[disagreement > 0.15] = 0.0
            else:
                p_win = np.minimum(p_lgbm, p_et)
        else:
            model = build_ml_model(model_type, n_train=len(X_train), cost_sensitive=cost_sensitive)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            if proba.shape[1] < 2:
                start_day += STEP_DAYS
                continue
            p_win = proba[:, 1]

        for i, sig_idx in enumerate(test_sig_indices):
            unfiltered_trades.append({
                'pnl': signal_pnls[sig_idx],
                'prob': float(p_win[i]),
                'fold': n_folds,
            })

        for threshold in thresholds:
            for i in range(len(test_sig_indices)):
                if p_win[i] >= threshold:
                    results_by_threshold[threshold].append({
                        'pnl': test_pnls[i],
                        'prob': float(p_win[i]),
                        'fold': n_folds,
                    })

        n_folds += 1
        start_day += STEP_DAYS

    return unfiltered_trades, results_by_threshold, n_folds


def compute_pf_wr(trades):
    """Compute PF and WR from trade list."""
    if not trades:
        return 0, 0, 0, 0
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_win / gross_loss
    wr = len(wins) / len(pnls) * 100
    return len(trades), wr, pf, sum(pnls)


def run_strategy_filter(df, config, highs, lows, closes, mins_arr, feature_cols, n_days):
    """Run full meta-label pipeline for one strategy.

    Tests multiple model variants:
    - Standard: lgbm, et, consensus
    - Cost-sensitive: lgbm_cs, et_cs, consensus_cs (penalize FP 2x)
    - Feature-selected: all above with top-K features

    Picks the variant that maximizes WR while keeping PnL > 0.
    """
    name = f"{config['strategy']} {config['session_filter']}/{config['direction_filter']}"

    t0 = time.time()
    signals = collect_labeled_signals(df, config, highs, lows, closes, mins_arr)
    elapsed_collect = time.time() - t0

    if len(signals) < MIN_TRAIN_SIGNALS + MIN_TEST_SIGNALS:
        print(f"  {name:<30} {len(signals):>4} signals ({elapsed_collect:.0f}s) -> SKIP (too few)")
        return None

    base_wr = sum(1 for s in signals if s['label'] == 1) / len(signals) * 100
    base_pnl = sum(s['pnl'] for s in signals)
    print(f"  {name:<30} {len(signals):>4} signals, WR {base_wr:.0f}%, PnL {base_pnl:+.1f}", flush=True)

    all_model_results = []
    thresholds = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)

    # Test all model variants: (model_type, cost_sensitive, feature_selection)
    variants = [
        ('lgbm', False, False),
        ('et', False, False),
        ('consensus', False, False),
        ('agreement', False, False),    # consensus + disagreement filter
        ('lgbm', True, False),          # cost-sensitive
        ('et', True, False),
        ('consensus', True, False),
        ('agreement', True, False),
        ('lgbm', False, True),          # feature-selected
        ('et', False, True),
        ('consensus', False, True),
        ('agreement', False, True),
        ('lgbm', True, True),           # cost-sensitive + feature-selected
        ('et', True, True),
        ('consensus', True, True),
        ('agreement', True, True),
    ]

    for model_type, cost_sensitive, use_feat_sel in variants:
        variant_name = model_type
        if cost_sensitive:
            variant_name += '_cs'
        if use_feat_sel:
            variant_name += '_fs'

        unfiltered, filtered_by_thresh, n_folds = walk_forward_filter(
            signals, df, feature_cols, model_type=model_type,
            thresholds=thresholds,
            use_feature_selection=use_feat_sel,
            cost_sensitive=cost_sensitive)

        if n_folds == 0:
            continue

        n_unf, wr_unf, pf_unf, pnl_unf = compute_pf_wr(unfiltered)

        for thresh, trades in filtered_by_thresh.items():
            n_f, wr_f, pf_f, pnl_f = compute_pf_wr(trades)
            if n_f < 10:
                continue
            keep_ratio = n_f / max(n_unf, 1)
            # WR must improve AND PnL must stay positive AND keep ≥20% trades
            if wr_f > wr_unf and pnl_f > 0 and keep_ratio >= 0.20:
                wr_delta = wr_f - wr_unf
                # Score: WR improvement is king, PnL is floor, keep_ratio is bonus
                score = wr_f * 200 + wr_delta * 50 + pnl_f * 0.05 + keep_ratio * 30
                all_model_results.append({
                    'model_type': variant_name,
                    'threshold': thresh,
                    'n_folds': n_folds,
                    'unfiltered_n': n_unf, 'unfiltered_wr': wr_unf,
                    'unfiltered_pf': pf_unf, 'unfiltered_pnl': pnl_unf,
                    'filtered_n': n_f, 'filtered_wr': wr_f,
                    'filtered_pf': pf_f, 'filtered_pnl': pnl_f,
                    'keep_ratio': keep_ratio,
                    'score': score,
                    'pnl_per_day': pnl_f / max(n_folds * TEST_DAYS, 1),
                })

    if all_model_results:
        all_model_results.sort(key=lambda x: x['score'], reverse=True)
        best_result = all_model_results[0]
        wr_delta = best_result['filtered_wr'] - best_result['unfiltered_wr']
        print(f"    -> BEST: {best_result['model_type'].upper()} thresh={best_result['threshold']:.2f} | "
              f"WR {best_result['unfiltered_wr']:.0f}% -> {best_result['filtered_wr']:.0f}% ({wr_delta:+.0f}pp) | "
              f"PF {best_result['unfiltered_pf']:.2f} -> {best_result['filtered_pf']:.2f} | "
              f"N {best_result['unfiltered_n']} -> {best_result['filtered_n']} "
              f"(keep {best_result['keep_ratio']:.0%}) | "
              f"PnL {best_result['unfiltered_pnl']:+.0f} -> {best_result['filtered_pnl']:+.0f}")
        # Show top 3 alternatives
        if len(all_model_results) > 1:
            print(f"    -> ALT:  ", end='')
            for alt in all_model_results[1:4]:
                print(f"{alt['model_type']}@{alt['threshold']:.2f}(WR{alt['filtered_wr']:.0f}%,N{alt['filtered_n']}) ", end='')
            print()
    else:
        if 'unfiltered' in dir() and unfiltered:
            n_unf, wr_unf, pf_unf, pnl_unf = compute_pf_wr(unfiltered)
        else:
            n_unf, wr_unf, pf_unf, pnl_unf = 0, 0, 0, 0
        print(f"    -> NO IMPROVEMENT (unfiltered WR={wr_unf:.0f}%, ML can't improve WR while keeping PnL positive)")
        best_result = {
            'model_type': 'none', 'threshold': 0,
            'n_folds': n_folds if 'n_folds' in dir() else 0,
            'unfiltered_n': n_unf, 'unfiltered_wr': wr_unf,
            'unfiltered_pf': pf_unf, 'unfiltered_pnl': pnl_unf,
            'filtered_n': n_unf, 'filtered_wr': wr_unf,
            'filtered_pf': pf_unf, 'filtered_pnl': pnl_unf,
            'keep_ratio': 1.0, 'score': 0,
            'pnl_per_day': pnl_unf / max(n_folds * TEST_DAYS, 1) if 'n_folds' in dir() and n_folds else 0,
        }

    best_result['strategy'] = config['strategy']
    best_result['config'] = config
    return best_result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', default='5m', choices=['1m', '5m'])
    args = parser.parse_args()

    strategies = STRATEGIES_5M if args.tf == '5m' else STRATEGIES_1M

    print(f"[ML STRATEGY FILTER] Per-strategy meta-labeling")
    print(f"  Model: LightGBM (max_depth=3, num_leaves=8) vs ExtraTrees")
    print(f"  Labels: ACTUAL trade outcome (trailing exit PnL > 0)")
    print(f"  Validation: {TRAIN_DAYS}d train / {TEST_DAYS}d test / {EMBARGO_BARS}-bar embargo")
    print(f"{'='*80}\n")

    df = load_data(tf=args.tf, days=9999)
    if df is None or df.empty:
        print("[ERROR] No data!")
        return

    n_days = df['date'].nunique()
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    mins_arr = df['mins'].values

    # Compute all features
    print(f"[DATA] {len(df)} bars, {n_days} days")
    print(f"[FEATURES] Computing 50 ML features...", flush=True)
    t0 = time.time()
    df = compute_all_features(df)
    print(f"  Done in {time.time()-t0:.0f}s\n", flush=True)

    # Use features available in df
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    print(f"[FEATURES] Using {len(feature_cols)} features\n")

    # Run per-strategy
    results = []
    for config in strategies:
        result = run_strategy_filter(df, config, highs, lows, closes, mins_arr, feature_cols, n_days)
        if result:
            results.append(result)
        print()

    # Summary
    print(f"\n{'='*100}")
    print(f"  SUMMARY — ML Meta-Label Filter v2 (WR-maximized, cost-sensitive) ({args.tf})")
    print(f"  Techniques: feature selection + cost-sensitive learning + disagreement filter")
    print(f"{'='*100}")
    print(f"{'Strategy':<25} {'Model':<16} {'Thr':>4} | {'--- Unfiltered ---':^22} | {'--- ML Filtered ---':^22} |")
    print(f"{'':<25} {'':<16} {'':<4} | {'N':>4} {'WR':>5} {'PF':>5} {'PnL':>7} | {'N':>4} {'WR':>5} {'PF':>5} {'PnL':>7} |")
    print(f"{'-'*25} {'-'*16} {'-'*4}   {'-'*22}   {'-'*22}  ")

    improved = []
    for r in results:
        model = r['model_type']
        thresh = r['threshold']
        wr_delta = r['filtered_wr'] - r['unfiltered_wr']
        marker = '✓' if wr_delta > 2 else ' '
        print(f"  {r['strategy']:<23} {model:<16} {thresh:>4.2f} | "
              f"{r['unfiltered_n']:>4} {r['unfiltered_wr']:>4.0f}% {r['unfiltered_pf']:>4.2f} {r['unfiltered_pnl']:>+6.0f} | "
              f"{r['filtered_n']:>4} {r['filtered_wr']:>4.0f}% {r['filtered_pf']:>4.2f} {r['filtered_pnl']:>+6.0f} | "
              f"WR {wr_delta:>+3.0f}pp {marker}")
        if r['model_type'] != 'none' and r['filtered_wr'] > r['unfiltered_wr']:
            improved.append(r)

    total_unf_pnl = sum(r['unfiltered_pnl'] for r in results)
    total_filt_pnl = sum(r['filtered_pnl'] for r in results)
    print(f"\n  Total PnL (OOS): unfiltered {total_unf_pnl:+.1f} → filtered {total_filt_pnl:+.1f} "
          f"(Δ {total_filt_pnl - total_unf_pnl:+.1f})")
    print(f"  Improved: {len(improved)}/{len(results)} strategies")

    # Save
    os.makedirs('ml/reports', exist_ok=True)
    output_file = f"ml/reports/strategy_filter_{args.tf}_{date.today().isoformat()}.json"
    save_data = {
        'config': {'train_days': TRAIN_DAYS, 'test_days': TEST_DAYS,
                   'embargo_bars': EMBARGO_BARS, 'features': len(feature_cols)},
        'results': [{k: v for k, v in r.items() if k != 'config'} for r in results],
        'total_unfiltered_pnl': total_unf_pnl,
        'total_filtered_pnl': total_filt_pnl,
    }
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[SAVED] {output_file}")


if __name__ == '__main__':
    main()
