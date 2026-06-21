"""Scan ALL 1m strategy combinations with 5m HTF filter.

Philosophy: Generate MANY signals on 1m (relax constraints), then let the
5m HTF filter do quality control. Previously "mediocre" combos may become
profitable with proper HTF alignment.

Tests: 7 strategies × 2 sessions × 2 directions = 28 combinations on 1m.

Usage:
    python -m ml.scan_all_1m_combos
"""
import sys, os, time, json, warnings
from datetime import date
from collections import defaultdict

import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from strategies.backtest_new import load_data
from strategies.optimize_exits import (
    collect_signals, filter_signals, simulate_trade_fast, COST
)
from ml.mtf_1m_with_5m import compute_5m_from_1m, compute_5m_indicators, map_1m_to_5m
from ml.mtf_indicator_discovery import analyze_indicator_discrimination

# ALL available strategies
ALL_STRATEGIES = [
    ('market_structure', 'strategies.market_structure', 'MarketStructureStrategy'),
    ('fibonacci', 'strategies.fibonacci', 'FibonacciStrategy'),
    ('sr_horizontal', 'strategies.sr_horizontal', 'SRHorizontalStrategy'),
    ('heikin_ashi', 'strategies.heikin_ashi', 'HeikinAshiStrategy'),
    ('reversal_patterns', 'strategies.reversal_patterns', 'ReversalPatternsStrategy'),
    ('momentum_trend', 'strategies.momentum_trend', 'MomentumTrendStrategy'),
    ('macd_cross', 'strategies.macd_cross', 'MACDCrossStrategy'),
]

# Exit params for 1m (proven working)
EXIT_1M = {
    'sl_mult': 2.5, 'trail_pts': 12.0, 'trail_mult': 1.0, 'tp_mult': None,
    'be_trigger': None, 'trail_tighten_pts': None, 'trail_tighten_mult': None,
    'max_hold': 50
}

SESSIONS = ['AM', 'PM', 'ALL']
DIRECTIONS = ['BUY', 'SELL', 'ALL']


def run_combo(df_1m, strategy_name, module, cls_name, session, direction, highs, lows, closes, mins_arr):
    """Run one strategy combo and return labeled signals."""
    mod = __import__(module, fromlist=[cls_name])
    strat = getattr(mod, cls_name)()

    all_signals = collect_signals(df_1m, strat)
    signals = filter_signals(all_signals, session, direction)

    if len(signals) < 20:
        return None

    labeled = []
    ep = EXIT_1M
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


def find_best_5m_filter(signals, df_5m, mapping, htf_cols, n_days):
    """Find the best single 5m indicator filter for signals."""
    signals_with_features = []
    for s in signals:
        htf_idx = mapping.get(s['idx'])
        if htf_idx is not None and htf_idx < len(df_5m):
            row = df_5m.iloc[htf_idx]
            htf_feats = {col: float(row[col]) if col in row.index and not pd.isna(row[col]) else np.nan
                         for col in htf_cols}
        else:
            htf_feats = {col: np.nan for col in htf_cols}
        signals_with_features.append({**s, 'htf_features': htf_feats})

    winners = [s for s in signals_with_features if s['label'] == 1]
    losers = [s for s in signals_with_features if s['label'] == 0]

    if len(winners) < 5 or len(losers) < 5:
        return None

    best = None
    for col in htf_cols:
        w_vals = [s['htf_features'][col] for s in winners]
        l_vals = [s['htf_features'][col] for s in losers]
        result = analyze_indicator_discrimination(w_vals, l_vals, col)
        if result and result['score'] > 0:
            if best is None or result['score'] > best['score']:
                best = result

    if best is None:
        return None

    # Apply the best filter and get stats
    passed = []
    for s in signals_with_features:
        val = s['htf_features'].get(best['indicator'], np.nan)
        if np.isnan(val):
            continue
        if best['best_direction'] == 'above' and val > best['best_thresh']:
            passed.append(s)
        elif best['best_direction'] == 'below' and val < best['best_thresh']:
            passed.append(s)

    if len(passed) < 10:
        return None

    filt_wr = sum(1 for s in passed if s['label'] == 1) / len(passed)
    filt_pnl = sum(s['pnl'] for s in passed)

    return {
        'indicator': best['indicator'],
        'direction': best['best_direction'],
        'threshold': best['best_thresh'],
        'filtered_n': len(passed),
        'filtered_wr': filt_wr,
        'filtered_pnl': filt_pnl,
        'keep_ratio': len(passed) / len(signals_with_features),
        'trades_per_day': len(passed) / n_days,
    }


def main():
    print(f"[SCAN ALL 1m COMBOS] 6 strategies x 3 sessions x 3 directions = 54 combos")
    print(f"  + 5m HTF filter to rescue 'mediocre' combos")
    print(f"{'='*100}\n")

    df_1m = load_data(tf='1m', days=9999)
    if df_1m is None or df_1m.empty:
        print("[ERROR] No 1m data!")
        return

    # Skip slow strategies — market_structure has O(n²) detect() on 163k bars
    FAST_STRATEGIES = [s for s in ALL_STRATEGIES if s[0] != 'market_structure']

    n_days = df_1m['date'].nunique()
    highs = df_1m['high'].values
    lows = df_1m['low'].values
    closes = df_1m['close'].values
    mins_arr = df_1m['mins'].values
    print(f"[DATA] {len(df_1m)} 1m bars, {n_days} days")

    # Compute 5m HTF
    print(f"[5m] Aggregating and computing indicators...", flush=True)
    df_5m = compute_5m_from_1m(df_1m)
    df_5m = compute_5m_indicators(df_5m)
    htf_cols = [c for c in df_5m.columns if c not in
                ('date', 'bar_group', 'open', 'high', 'low', 'close', 'volume', 'mins')]
    print(f"  {len(df_5m)} 5m bars, {len(htf_cols)} indicators")

    print(f"[MAP] Building 1m->5m mapping...", flush=True)
    mapping = map_1m_to_5m(df_1m, df_5m)
    print(f"  Done\n", flush=True)

    # Scan all combos
    results = []
    print(f"{'Strategy':<20} {'Sess':<4} {'Dir':<5} | {'N':>5} {'N/day':>5} {'WR':>4} {'PnL':>7} | "
          f"{'5m Filter':<25} {'N_f':>4} {'WR_f':>5} {'PnL_f':>7} {'Keep%':>5}")
    print(f"{'-'*20} {'-'*4} {'-'*5}   {'-'*5} {'-'*5} {'-'*4} {'-'*7}   "
          f"{'-'*25} {'-'*4} {'-'*5} {'-'*7} {'-'*5}")

    for strat_name, module, cls_name in FAST_STRATEGIES:
        print(f"  [scanning {strat_name}...]", flush=True)
        # Collect signals ONCE per strategy, then filter by session/direction
        mod = __import__(module, fromlist=[cls_name])
        strat = getattr(mod, cls_name)()
        all_signals = collect_signals(df_1m, strat)
        print(f"    {len(all_signals)} raw signals", flush=True)

        for session in SESSIONS:
            for direction in DIRECTIONS:
                signals_f = filter_signals(all_signals, session, direction)
                if len(signals_f) < 20:
                    continue

                # Simulate trades
                labeled = []
                ep = EXIT_1M
                for s in signals_f:
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
                signals = labeled
                if not signals:
                    continue

                n = len(signals)
                wr = sum(1 for s in signals if s['label'] == 1) / n
                pnl = sum(s['pnl'] for s in signals)
                n_per_day = n / n_days

                # Skip if fewer than 30 signals total
                if n < 30:
                    continue

                # Find best 5m filter
                best_filter = find_best_5m_filter(signals, df_5m, mapping, htf_cols, n_days)

                if best_filter and best_filter['filtered_wr'] > 0.50 and best_filter['filtered_pnl'] > 0:
                    filt_str = f"{best_filter['indicator']} {best_filter['direction']} {best_filter['threshold']:.2f}"
                    print(f"  {strat_name:<18} {session:<4} {direction:<5} | "
                          f"{n:>5} {n_per_day:>4.2f} {wr*100:>3.0f}% {pnl:>+6.0f} | "
                          f"{filt_str:<25} {best_filter['filtered_n']:>4} {best_filter['filtered_wr']*100:>4.0f}% "
                          f"{best_filter['filtered_pnl']:>+6.0f} {best_filter['keep_ratio']*100:>4.0f}%")

                    results.append({
                        'strategy': strat_name,
                        'session': session,
                        'direction': direction,
                        'n_signals': n,
                        'n_per_day': n_per_day,
                        'base_wr': wr,
                        'base_pnl': pnl,
                        'filter': best_filter,
                    })
                elif pnl > 0 and wr >= 0.45:
                    # Profitable even without filter
                    print(f"  {strat_name:<18} {session:<4} {direction:<5} | "
                          f"{n:>5} {n_per_day:>4.2f} {wr*100:>3.0f}% {pnl:>+6.0f} | "
                          f"{'(no improvement found)':<25} {'':>4} {'':>5} {'':>7} {'':>5}")
                    results.append({
                        'strategy': strat_name,
                        'session': session,
                        'direction': direction,
                        'n_signals': n,
                        'n_per_day': n_per_day,
                        'base_wr': wr,
                        'base_pnl': pnl,
                        'filter': None,
                    })

    # Summary: top combos by filtered PnL/day
    print(f"\n\n{'='*100}")
    print(f"  TOP COMBOS — Sorted by PnL (with 5m filter)")
    print(f"{'='*100}")
    print(f"  {'#':>2} {'Strategy':<20} {'S/D':<8} | {'Base':^18} | {'With 5m Filter':^28} | {'Trades'}")
    print(f"  {'':>2} {'':>20} {'':<8} | {'N':>4} {'WR':>4} {'PnL':>7} | {'N':>4} {'WR':>5} {'PnL':>7} {'Filter':<15} | {'/day':>5} {'/wk':>4}")
    print(f"  {'-'*2} {'-'*20} {'-'*8}   {'-'*18}   {'-'*28}   {'-'*10}")

    # Sort by filtered PnL (or base PnL if no filter)
    results.sort(key=lambda x: x['filter']['filtered_pnl'] if x['filter'] else x['base_pnl'], reverse=True)

    total_filt_n = 0
    total_filt_pnl = 0

    for i, r in enumerate(results[:25]):
        f = r['filter']
        if f:
            filt_name = f['indicator'][:14]
            filt_n = f['filtered_n']
            filt_wr = f['filtered_wr'] * 100
            filt_pnl = f['filtered_pnl']
            filt_pd = f['trades_per_day']
            total_filt_n += filt_n
            total_filt_pnl += filt_pnl
        else:
            filt_name = 'none'
            filt_n = r['n_signals']
            filt_wr = r['base_wr'] * 100
            filt_pnl = r['base_pnl']
            filt_pd = r['n_per_day']
            total_filt_n += filt_n
            total_filt_pnl += filt_pnl

        s_d = f"{r['session']}/{r['direction']}"
        print(f"  {i+1:>2} {r['strategy']:<20} {s_d:<8} | "
              f"{r['n_signals']:>4} {r['base_wr']*100:>3.0f}% {r['base_pnl']:>+6.0f} | "
              f"{filt_n:>4} {filt_wr:>4.0f}% {filt_pnl:>+6.0f} {filt_name:<15} | "
              f"{filt_pd:>4.2f} {filt_pd*5:>4.1f}")

    print(f"\n  TOTALS (top 25 combos):")
    print(f"    Filtered trades: {total_filt_n} over {n_days} days = {total_filt_n/n_days:.2f}/day = {total_filt_n/n_days*5:.1f}/week")
    print(f"    Filtered PnL: {total_filt_pnl:+.1f} pts ({total_filt_pnl/n_days:+.2f}/day)")

    # Save
    os.makedirs('ml/reports', exist_ok=True)
    output_file = f"ml/reports/all_1m_combos_{date.today().isoformat()}.json"
    save_data = [{k: v for k, v in r.items()} for r in results[:25]]
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[SAVED] {output_file}")


if __name__ == '__main__':
    main()
