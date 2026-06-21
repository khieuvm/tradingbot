r"""Backtest new strategies from E:\Strategy documents + MACD analysis.

Usage:
    python -m strategies.backtest_new [--tf 5m] [--days 0]
"""
import sys
import os
import json
import time
from datetime import date

import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies.base import compute_metrics, run_backtest

COST = 0.96
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_data(tf='5m', days=0):
    parquet_file = os.path.join(DATA_DIR, f"vn30f1m_{tf}.parquet")
    print(f"[DATA] Loading {tf} from parquet cache...")
    df = pd.read_parquet(parquet_file)
    df['time'] = pd.to_datetime(df['time'])
    if days and days > 0 and days < 9000:
        cutoff_date = (date.today() - pd.Timedelta(days=days))
        df = df[df['time'].dt.date >= cutoff_date].reset_index(drop=True)

    df = df.reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    if 'time' not in df.columns and 'datetime' in df.columns:
        df.rename(columns={'datetime': 'time'}, inplace=True)

    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute

    am_mask = (df['mins'] >= 540) & (df['mins'] <= 690)
    pm_mask = (df['mins'] >= 780) & (df['mins'] <= 870)
    df = df[am_mask | pm_mask].reset_index(drop=True)
    df['session'] = 'AM'
    df.loc[df['mins'] >= 780, 'session'] = 'PM'

    # Core indicators
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['range'] = df['high'] - df['low']
    df['ema8'] = ta.ema(df['close'], length=8)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema50'] = ta.ema(df['close'], length=50)

    bb = ta.bbands(df['close'], length=20, std=2.0)
    if bb is not None and len(bb.columns) >= 3:
        df['bb_lower'] = bb.iloc[:, 0]
        df['bb_mid'] = bb.iloc[:, 1]
        df['bb_upper'] = bb.iloc[:, 2]
    else:
        df['bb_lower'] = df['bb_mid'] = df['bb_upper'] = np.nan

    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is not None:
        df['adx'] = adx_df.iloc[:, 0]
        df['di_plus'] = adx_df.iloc[:, 1]
        df['di_minus'] = adx_df.iloc[:, 2]
    else:
        df['adx'] = df['di_plus'] = df['di_minus'] = np.nan

    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd_df is not None:
        df['macd_line'] = macd_df.iloc[:, 0]
        df['macd_signal'] = macd_df.iloc[:, 2]
    else:
        df['macd_line'] = df['macd_signal'] = np.nan

    stoch_df = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    if stoch_df is not None:
        df['stoch_k'] = stoch_df.iloc[:, 0]
        df['stoch_d'] = stoch_df.iloc[:, 1]
    else:
        df['stoch_k'] = df['stoch_d'] = np.nan

    df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    df['obv'] = ta.obv(df['close'], df['volume'])

    n_days = df['date'].nunique()
    print(f"[DATA] Loaded {len(df)} bars, {n_days} trading days")
    print(f"[DATA] Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    return df


def get_new_strategies():
    """Load strategies from PDF documents + MACD analysis."""
    strategies = []

    try:
        from strategies.macd_cross import MACDCrossStrategy
        strategies.append(MACDCrossStrategy())
    except Exception as e:
        print(f"  [SKIP] MACD Cross: {e}")

    try:
        from strategies.ma_crossover import MACrossoverStrategy
        strategies.append(MACrossoverStrategy())
    except Exception as e:
        print(f"  [SKIP] MA Crossover: {e}")

    try:
        from strategies.bb_squeeze import BBSqueezeStrategy
        strategies.append(BBSqueezeStrategy())
    except Exception as e:
        print(f"  [SKIP] BB Squeeze: {e}")

    try:
        from strategies.narrow_range import NarrowRangeStrategy
        strategies.append(NarrowRangeStrategy())
    except Exception as e:
        print(f"  [SKIP] Narrow Range: {e}")

    try:
        from strategies.role_reversal import RoleReversalStrategy
        strategies.append(RoleReversalStrategy())
    except Exception as e:
        print(f"  [SKIP] Role Reversal: {e}")

    try:
        from strategies.rsi2 import RSI2Strategy
        strategies.append(RSI2Strategy())
    except Exception as e:
        print(f"  [SKIP] RSI2: {e}")

    try:
        from strategies.ha_stoch import HAStochStrategy
        strategies.append(HAStochStrategy())
    except Exception as e:
        print(f"  [SKIP] HA+Stoch: {e}")

    return strategies


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', default='5m', choices=['1m', '3m', '5m'])
    parser.add_argument('--days', type=int, default=0)
    args = parser.parse_args()

    df = load_data(tf=args.tf, days=args.days)
    if df is None:
        return

    n_days = df['date'].nunique()
    strategies = get_new_strategies()
    print(f"\n[STRATEGIES] Loaded {len(strategies)} new strategies\n")

    print(f"{'#':<3} {'Strategy':<20} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL/d':>7} {'MFE':>6} {'AM_PnL/d':>9} {'PM_PnL/d':>9}")
    print(f"{'-'*3} {'-'*20} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*9} {'-'*9}")

    results = []
    for strat in strategies:
        t0 = time.time()
        try:
            trades = run_backtest(df, strat, dedup_bars=5)
        except Exception as e:
            print(f"  {strat.name}: ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        trades_am = [t for t in trades if t['session'] == 'AM']
        trades_pm = [t for t in trades if t['session'] == 'PM']
        m_all = compute_metrics(trades)
        m_am = compute_metrics(trades_am)
        m_pm = compute_metrics(trades_pm)
        elapsed = time.time() - t0

        print(f" {len(results)+1:<2} {strat.name:<20} {m_all['n']:>6} {m_all['wr']:>6.1f}% {m_all['pf']:>6.2f} "
              f"{m_all['pnl_per_day']:>+6.2f} {m_all['avg_mfe']:>5.1f} "
              f"{m_am['pnl_per_day']:>+8.2f} {m_pm['pnl_per_day']:>+8.2f}  ({elapsed:.1f}s)")

        results.append({
            'name': strat.name, 'exit_type': strat.exit_type, 'tf': args.tf,
            'metrics_all': m_all, 'metrics_am': m_am, 'metrics_pm': m_pm,
            'trades': trades
        })

    # Detailed breakdown for profitable ones
    print(f"\n{'='*80}")
    print(f"  DETAILED BREAKDOWN")
    print(f"{'='*80}")

    for r in sorted(results, key=lambda x: x['metrics_all']['pnl_per_day'], reverse=True):
        m_all = r['metrics_all']
        m_am = r['metrics_am']
        m_pm = r['metrics_pm']
        if m_all['n'] == 0:
            continue

        print(f"\n  {r['name']} (exit: {r['exit_type']})")
        print(f"  {'':>6} {'Trades':>7} {'WR%':>7} {'PF':>7} {'PnL':>8} {'PnL/d':>7} {'MFE':>6}")
        print(f"  {'ALL:':>6} {m_all['n']:>7} {m_all['wr']:>6.1f}% {m_all['pf']:>6.2f} {m_all['pnl']:>+7.1f} {m_all['pnl_per_day']:>+6.2f} {m_all['avg_mfe']:>5.1f}")
        print(f"  {'AM:':>6} {m_am['n']:>7} {m_am['wr']:>6.1f}% {m_am['pf']:>6.2f} {m_am['pnl']:>+7.1f} {m_am['pnl_per_day']:>+6.2f} {m_am['avg_mfe']:>5.1f}")
        print(f"  {'PM:':>6} {m_pm['n']:>7} {m_pm['wr']:>6.1f}% {m_pm['pf']:>6.2f} {m_pm['pnl']:>+7.1f} {m_pm['pnl_per_day']:>+6.2f} {m_pm['avg_mfe']:>5.1f}")

        trades = r['trades']
        if trades:
            reasons = {}
            for t in trades:
                reasons[t['reason']] = reasons.get(t['reason'], 0) + 1
            reason_str = ', '.join(f"{k}:{v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
            print(f"  {'Exits:':>6} {reason_str}")

    # Save
    os.makedirs('strategies/results', exist_ok=True)
    output_file = f"strategies/results/backtest_new_{args.tf}_{date.today().isoformat()}.json"
    save_data = []
    for r in results:
        save_data.append({
            'name': r['name'], 'exit_type': r['exit_type'], 'tf': r['tf'],
            'metrics_all': r['metrics_all'], 'metrics_am': r['metrics_am'],
            'metrics_pm': r['metrics_pm'], 'n_trades': len(r['trades']),
        })
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[SAVED] {output_file}")


if __name__ == '__main__':
    main()
