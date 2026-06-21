"""ML Ensemble using strategy signals as features for VN30F1M.

Combines all strategy detect() outputs as features for an ExtraTrees model.
Walk-forward validated: 60d train / 20d test / step 20d.

Usage:
    python -m strategies.ml_ensemble [--tf 5m]
"""
import sys
import os
import json
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from strategies.base import COST
from strategies.backtest_new import load_data
from strategies.optimize_exits import load_all_strategies

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score


HORIZON = 6
THRESHOLD = 0.60
TRAIN_DAYS = 60
TEST_DAYS = 20
STEP_DAYS = 20
MIN_TRADES_PER_FOLD = 10


def compute_signal_matrix(df, strategies):
    """Run all strategies and build signal matrix (N_bars x N_strategies)."""
    n = len(df)
    n_strats = len(strategies)
    signal_matrix = np.zeros((n, n_strats), dtype=np.int8)

    # Skip strategies known to be very slow (>300s) for per-bar detection
    SLOW_STRATEGIES = {'role_reversal'}

    for s_idx, strat in enumerate(strategies):
        if strat.name in SLOW_STRATEGIES:
            print(f"    {strat.name}: SKIPPED (too slow for per-bar)")
            continue
        t0 = time.time()
        try:
            strat.prepare(df)
            for i in range(60, n - 2):
                sig = strat.detect(df, i)
                if sig != 0:
                    signal_matrix[i, s_idx] = sig
        except Exception as e:
            print(f"  [WARN] {strat.name}: {e}")
        elapsed = time.time() - t0
        print(f"    {strat.name}: {elapsed:.0f}s ({np.sum(signal_matrix[:, s_idx] != 0)} signals)")

    return signal_matrix


def compute_context_features(df):
    """Compute market context features per bar."""
    n = len(df)
    features = np.zeros((n, 10))

    atr = df['atr'].values
    rsi = df['rsi'].values

    ema8 = df['ema8'].values
    ema21 = df['ema21'].values
    ema50 = df['ema50'].values
    close = df['close'].values

    features[:, 0] = atr
    features[:, 1] = rsi
    features[:, 2] = df.get('adx', pd.Series(np.full(n, 20))).values if 'adx' in df.columns else 20
    features[:, 3] = df['stoch_k'].values if 'stoch_k' in df.columns else 50

    # EMA alignment (-3 to +3)
    ema_align = np.sign(close - ema8) + np.sign(close - ema21) + np.sign(close - ema50)
    features[:, 4] = ema_align

    # BB position
    if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
        bb_range = df['bb_upper'].values - df['bb_lower'].values
        bb_range = np.where(bb_range > 0, bb_range, 1)
        features[:, 5] = (close - df['bb_lower'].values) / bb_range
    else:
        features[:, 5] = 0.5

    # DI spread
    if 'di_plus' in df.columns:
        features[:, 6] = df['di_plus'].values - df['di_minus'].values
    else:
        features[:, 6] = 0

    # Session progress (0-1)
    mins = df['mins'].values
    features[:, 7] = np.where(mins < 780,
                              (mins - 540) / 150,  # AM: 9:00-11:30
                              (mins - 780) / 90)   # PM: 13:00-14:30
    features[:, 7] = np.clip(features[:, 7], 0, 1)

    # Is PM
    features[:, 8] = (mins >= 780).astype(float)

    # MACD hist
    if 'macd_line' in df.columns:
        features[:, 9] = df['macd_line'].values - df['macd_signal'].values
    else:
        features[:, 9] = 0

    return features


def build_features_and_labels(df, signal_matrix, context_features, strategy_names):
    """Build feature matrix and labels for ML model."""
    n = len(df)
    closes = df['close'].values
    sessions = df['session'].values
    mins = df['mins'].values

    # Labels: 6-bar forward return
    labels = np.full(n, 0, dtype=np.int8)
    for i in range(n - HORIZON):
        ret = closes[i + HORIZON] - closes[i]
        if ret > COST:
            labels[i] = 1
        elif ret < -COST:
            labels[i] = -1

    # Build per-bar features
    n_strats = signal_matrix.shape[1]
    n_context = context_features.shape[1]

    # Signal agreement features
    buy_count = np.sum(signal_matrix == 1, axis=1)
    sell_count = np.sum(signal_matrix == -1, axis=1)
    net_signal = buy_count - sell_count

    # Bars since last signal (any strategy)
    last_buy = np.full(n, 999)
    last_sell = np.full(n, 999)
    for i in range(1, n):
        if np.any(signal_matrix[i-1] == 1):
            last_buy[i] = 1
        else:
            last_buy[i] = min(last_buy[i-1] + 1, 999)
        if np.any(signal_matrix[i-1] == -1):
            last_sell[i] = 1
        else:
            last_sell[i] = min(last_sell[i-1] + 1, 999)

    # Combine all features
    # [signal_matrix, buy_count, sell_count, net_signal, last_buy, last_sell, context]
    X = np.column_stack([
        signal_matrix,
        buy_count.reshape(-1, 1),
        sell_count.reshape(-1, 1),
        net_signal.reshape(-1, 1),
        np.minimum(last_buy, 50).reshape(-1, 1),
        np.minimum(last_sell, 50).reshape(-1, 1),
        context_features
    ])

    feature_names = (
        strategy_names +
        ['buy_count', 'sell_count', 'net_signal', 'bars_since_buy', 'bars_since_sell'] +
        ['atr', 'rsi', 'adx', 'stoch_k', 'ema_align', 'bb_pos', 'di_spread',
         'session_pct', 'is_pm', 'macd_hist']
    )

    return X, labels, feature_names


def walk_forward_ensemble(df, X, labels, feature_names):
    """Walk-forward validation: 60d train / 20d test / step 20d."""
    dates = df['date'].values
    unique_dates = sorted(set(dates))
    sessions = df['session'].values
    mins = df['mins'].values

    all_trades = []
    fold_results = []

    # Valid bars for trading signals
    valid_mask = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if sessions[i] == 'AM' and not (555 <= mins[i] <= 645):
            valid_mask[i] = False
        elif sessions[i] == 'PM' and not (795 <= mins[i] <= 855):
            valid_mask[i] = False
        elif sessions[i] not in ('AM', 'PM'):
            valid_mask[i] = False

    n_folds = 0
    start = TRAIN_DAYS

    while start + TEST_DAYS <= len(unique_dates):
        train_end_date = unique_dates[start]
        test_end_date = unique_dates[min(start + TEST_DAYS, len(unique_dates) - 1)]
        train_start_date = unique_dates[max(0, start - TRAIN_DAYS)]

        train_mask = (dates >= train_start_date) & (dates < train_end_date)
        test_mask = (dates >= train_end_date) & (dates <= test_end_date)

        # Filter to bars with labels != 0 for training
        train_idx = np.where(train_mask & (labels != 0))[0]
        test_idx = np.where(test_mask & valid_mask)[0]

        if len(train_idx) < 100 or len(test_idx) < 20:
            start += STEP_DAYS
            continue

        X_train = X[train_idx]
        y_train = labels[train_idx]

        # Convert to binary: 1 = long profitable, 0 = short profitable
        y_binary = (y_train == 1).astype(int)

        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X[test_idx], nan=0.0)

        # Train
        model = ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=30,
            class_weight='balanced', random_state=42, n_jobs=-1)
        model.fit(X_train, y_binary)

        # Predict on test
        proba = model.predict_proba(X_test)
        if proba.shape[1] < 2:
            start += STEP_DAYS
            continue

        p_long = proba[:, 1]

        # Generate signals with threshold + dedup
        closes = df['close'].values
        last_trade_idx = -10

        for k, idx in enumerate(test_idx):
            if idx - last_trade_idx < 5:
                continue

            atr = df['atr'].iloc[idx]
            if pd.isna(atr) or atr < 1.5:
                continue

            if p_long[k] > THRESHOLD:
                direction = 1
            elif p_long[k] < (1 - THRESHOLD):
                direction = -1
            else:
                continue

            # Simple fixed-horizon exit
            exit_idx = min(idx + HORIZON, len(closes) - 1)
            pnl = direction * (closes[exit_idx] - closes[idx]) - COST

            all_trades.append({
                'date': str(dates[idx]),
                'session': sessions[idx],
                'direction': 'BUY' if direction == 1 else 'SELL',
                'pnl': pnl,
                'prob': float(p_long[k]),
                'fold': n_folds,
            })
            last_trade_idx = idx

        n_folds += 1
        start += STEP_DAYS

    return all_trades, n_folds, model, feature_names


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', default='5m', choices=['1m', '3m', '5m'])
    parser.add_argument('--days', type=int, default=9999)
    args = parser.parse_args()

    print(f"[ML ENSEMBLE] Strategy signals as features, walk-forward validated")
    print(f"{'='*70}")

    df = load_data(tf=args.tf, days=args.days)
    if df is None or df.empty:
        print("[ERROR] No data!")
        return

    n_days = df['date'].nunique()
    print(f"[DATA] {len(df)} bars, {n_days} days\n")

    # Load all strategies
    strategies = load_all_strategies()
    strategy_names = [s.name for s in strategies]
    print(f"[STRATEGIES] {len(strategies)} loaded: {', '.join(strategy_names[:5])}...\n")

    # Compute signal matrix
    print("[COMPUTING] Signal matrix (all strategies x all bars)...")
    t0 = time.time()
    signal_matrix = compute_signal_matrix(df, strategies)
    print(f"  Done in {time.time()-t0:.0f}s")

    signals_per_strat = np.sum(signal_matrix != 0, axis=0)
    for i, name in enumerate(strategy_names):
        if signals_per_strat[i] > 0:
            print(f"    {name}: {signals_per_strat[i]} signals")

    # Compute context features
    print("\n[COMPUTING] Context features...")
    context_features = compute_context_features(df)

    # Build ML dataset
    print("[BUILDING] Feature matrix + labels...")
    X, labels, feature_names = build_features_and_labels(df, signal_matrix, context_features, strategy_names)
    print(f"  Shape: {X.shape}, labels: +1={np.sum(labels==1)}, -1={np.sum(labels==-1)}, 0={np.sum(labels==0)}")

    # Walk-forward
    print(f"\n[WALK-FORWARD] {TRAIN_DAYS}d train / {TEST_DAYS}d test / step {STEP_DAYS}d")
    print(f"  Threshold: P > {THRESHOLD} for LONG, P < {1-THRESHOLD} for SHORT")
    t0 = time.time()
    trades, n_folds, last_model, feature_names = walk_forward_ensemble(df, X, labels, feature_names)
    elapsed = time.time() - t0
    print(f"  Completed {n_folds} folds in {elapsed:.0f}s")
    print(f"  Total trades: {len(trades)}")

    if not trades:
        print("\n[RESULT] No trades generated!")
        return

    # Results
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_win / gross_loss
    wr = len(wins) / len(pnls) * 100
    total_pnl = sum(pnls)
    trade_dates = set(t['date'] for t in trades)
    pnl_per_day = total_pnl / max(len(trade_dates), 1)

    print(f"\n{'='*70}")
    print(f"  ML ENSEMBLE RESULTS (OOS walk-forward)")
    print(f"{'='*70}")
    print(f"  Trades:   {len(trades)}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  PF:       {pf:.2f}")
    print(f"  Total PnL: {total_pnl:+.1f} pts")
    print(f"  PnL/day:  {pnl_per_day:+.2f} pts")
    print(f"  Avg trade: {total_pnl/len(trades):+.2f} pts")

    # Breakdown by direction
    buy_trades = [t for t in trades if t['direction'] == 'BUY']
    sell_trades = [t for t in trades if t['direction'] == 'SELL']
    if buy_trades:
        buy_pnls = [t['pnl'] for t in buy_trades]
        buy_wr = len([p for p in buy_pnls if p > 0]) / len(buy_pnls) * 100
        print(f"\n  BUY:  {len(buy_trades)} trades, WR {buy_wr:.0f}%, PnL {sum(buy_pnls):+.1f}")
    if sell_trades:
        sell_pnls = [t['pnl'] for t in sell_trades]
        sell_wr = len([p for p in sell_pnls if p > 0]) / len(sell_pnls) * 100
        print(f"  SELL: {len(sell_trades)} trades, WR {sell_wr:.0f}%, PnL {sum(sell_pnls):+.1f}")

    # By session
    am_trades = [t for t in trades if t['session'] == 'AM']
    pm_trades = [t for t in trades if t['session'] == 'PM']
    if am_trades:
        am_pnls = [t['pnl'] for t in am_trades]
        am_wr = len([p for p in am_pnls if p > 0]) / len(am_pnls) * 100
        print(f"\n  AM:   {len(am_trades)} trades, WR {am_wr:.0f}%, PnL {sum(am_pnls):+.1f}")
    if pm_trades:
        pm_pnls = [t['pnl'] for t in pm_trades]
        pm_wr = len([p for p in pm_pnls if p > 0]) / len(pm_pnls) * 100
        print(f"  PM:   {len(pm_trades)} trades, WR {pm_wr:.0f}%, PnL {sum(pm_pnls):+.1f}")

    # Feature importance
    if last_model is not None:
        importances = last_model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        print(f"\n  Top 15 features:")
        for i, idx in enumerate(top_idx):
            print(f"    {i+1:>2}. {feature_names[idx]:<25} {importances[idx]:.4f}")

    # Save
    os.makedirs('strategies/results', exist_ok=True)
    output_file = f"strategies/results/ml_ensemble_{args.tf}_{date.today().isoformat()}.json"
    save_data = {
        'config': {'horizon': HORIZON, 'threshold': THRESHOLD,
                   'train_days': TRAIN_DAYS, 'test_days': TEST_DAYS},
        'results': {'n_trades': len(trades), 'wr': wr, 'pf': pf,
                    'total_pnl': total_pnl, 'pnl_per_day': pnl_per_day},
        'by_direction': {
            'BUY': {'n': len(buy_trades), 'pnl': sum(t['pnl'] for t in buy_trades)} if buy_trades else {},
            'SELL': {'n': len(sell_trades), 'pnl': sum(t['pnl'] for t in sell_trades)} if sell_trades else {},
        },
        'feature_importance': {feature_names[i]: float(importances[i]) for i in top_idx} if last_model else {},
        'n_folds': n_folds,
    }
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[SAVED] {output_file}")


if __name__ == '__main__':
    main()
