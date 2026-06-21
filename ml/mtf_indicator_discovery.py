"""Multi-Timeframe Indicator Discovery for Signal Filtering.

Approach: compute 15m/30m indicators at the time of each 5m/1m signal,
then compare indicator distributions between WINNERS and LOSERS.
Find the best discriminating indicators and optimal thresholds.

Goal: KEEP more winners + REJECT more losers (better than pure ML).

Usage:
    python -m ml.mtf_indicator_discovery [--tf 5m]
"""
import sys, os, time, json, warnings
from datetime import date
from pathlib import Path
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
from ml.strategy_filter import STRATEGIES_5M, STRATEGIES_1M, collect_labeled_signals


def compute_15m_bars(df_5m):
    """Aggregate 5m bars into 15m bars, aligned to session boundaries."""
    df = df_5m.copy()
    df['bar_group'] = df.groupby('date').cumcount() // 3

    df_15m = df.groupby(['date', 'bar_group']).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'mins': 'last',
    }).reset_index()

    return df_15m


def compute_htf_indicators(df_15m):
    """Compute indicators on 15m timeframe."""
    c = df_15m['close']
    h = df_15m['high']
    l = df_15m['low']
    o = df_15m['open']

    # Trend indicators
    df_15m['ema8_15m'] = ta.ema(c, length=8)
    df_15m['ema21_15m'] = ta.ema(c, length=21)
    df_15m['ema50_15m'] = ta.ema(c, length=50)

    # EMA slopes (rate of change)
    df_15m['ema8_slope_15m'] = df_15m['ema8_15m'].diff(1) / c
    df_15m['ema21_slope_15m'] = df_15m['ema21_15m'].diff(1) / c

    # EMA alignment on 15m (-3 to +3)
    df_15m['ema_align_15m'] = (
        np.sign(c - df_15m['ema8_15m']).fillna(0) +
        np.sign(c - df_15m['ema21_15m']).fillna(0) +
        np.sign(c - df_15m['ema50_15m']).fillna(0)
    )

    # RSI on 15m
    df_15m['rsi_15m'] = ta.rsi(c, length=14)
    df_15m['rsi_7_15m'] = ta.rsi(c, length=7)

    # MACD on 15m
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        df_15m['macd_hist_15m'] = macd.iloc[:, 1]
        df_15m['macd_line_15m'] = macd.iloc[:, 0]
        df_15m['macd_signal_15m'] = macd.iloc[:, 2]
    else:
        df_15m['macd_hist_15m'] = 0.0
        df_15m['macd_line_15m'] = 0.0
        df_15m['macd_signal_15m'] = 0.0

    # MACD histogram direction (rising/falling)
    df_15m['macd_hist_dir_15m'] = np.sign(df_15m['macd_hist_15m'].diff(1))

    # ADX on 15m
    adx_df = ta.adx(h, l, c, length=14)
    if adx_df is not None:
        df_15m['adx_15m'] = adx_df['ADX_14']
        df_15m['di_plus_15m'] = adx_df['DMP_14']
        df_15m['di_minus_15m'] = adx_df['DMN_14']
    else:
        df_15m['adx_15m'] = 20.0
        df_15m['di_plus_15m'] = 20.0
        df_15m['di_minus_15m'] = 20.0

    df_15m['di_spread_15m'] = df_15m['di_plus_15m'] - df_15m['di_minus_15m']

    # ATR on 15m
    df_15m['atr_15m'] = ta.atr(h, l, c, length=14)

    # Stochastic on 15m
    stoch = ta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        df_15m['stoch_k_15m'] = stoch.iloc[:, 0]
        df_15m['stoch_d_15m'] = stoch.iloc[:, 1]
    else:
        df_15m['stoch_k_15m'] = 50.0
        df_15m['stoch_d_15m'] = 50.0

    # Bollinger Band position on 15m
    sma20 = ta.sma(c, length=20)
    std20 = c.rolling(20).std()
    df_15m['bb_pos_15m'] = ((c - sma20) / (2 * std20)).where(std20 > 0, 0)

    # Keltner Channel position on 15m
    ema20 = ta.ema(c, length=20)
    df_15m['kc_pos_15m'] = ((c - ema20) / (1.5 * df_15m['atr_15m'])).where(df_15m['atr_15m'] > 0, 0)

    # Candle structure on 15m
    bar_range = h - l
    df_15m['body_pct_15m'] = (c - o) / bar_range.replace(0, np.nan)
    df_15m['range_15m'] = bar_range

    # Higher high / Lower low pattern
    df_15m['hh_15m'] = (h > h.shift(1)).astype(int) + (h > h.shift(2)).astype(int)
    df_15m['ll_15m'] = (l < l.shift(1)).astype(int) + (l < l.shift(2)).astype(int)

    # Volume indicators on 15m
    v = df_15m.get('volume')
    if v is not None and not v.isna().all() and (v > 0).any():
        vol_sma = v.rolling(10).mean()
        df_15m['vol_ratio_15m'] = (v / vol_sma).where(vol_sma > 0, 1.0)
        # VWAP approximation (cumulative within day)
        typical_price = (h + l + c) / 3
        df_15m['vwap_dist_15m'] = 0.0  # placeholder — computed per day below
    else:
        df_15m['vol_ratio_15m'] = 1.0
        df_15m['vwap_dist_15m'] = 0.0

    # Distance from recent high/low (15m)
    df_15m['dist_high_5_15m'] = (c - h.rolling(5).max()) / df_15m['atr_15m'].replace(0, 1)
    df_15m['dist_low_5_15m'] = (c - l.rolling(5).min()) / df_15m['atr_15m'].replace(0, 1)

    # Momentum: close vs N bars ago
    df_15m['mom_3_15m'] = c - c.shift(3)
    df_15m['mom_5_15m'] = c - c.shift(5)

    # Trend strength: price above/below EMA for how many bars
    above_ema = (c > df_15m['ema21_15m']).astype(int)
    df_15m['bars_above_ema21_15m'] = above_ema.rolling(10).sum()

    return df_15m


def map_5m_to_15m(df_5m, df_15m):
    """Create mapping from 5m bar index to corresponding 15m bar."""
    # Each 5m bar belongs to a 15m bar (group of 3)
    # Map: 5m idx → 15m row that is COMPLETED before this 5m bar

    df_5m_copy = df_5m.copy()
    df_5m_copy['_5m_idx'] = range(len(df_5m_copy))
    df_5m_copy['_15m_group'] = df_5m_copy.groupby('date').cumcount() // 3

    # For signal on 5m bar i, we use the 15m bar that ENDED before bar i
    # That's the 15m bar at group (i//3 - 1) within the same day, or the last group of previous day
    mapping = {}

    # Build index of 15m bars by (date, group)
    df_15m_indexed = df_15m.reset_index(drop=True)
    htf_by_date_group = {}
    for idx, row in df_15m_indexed.iterrows():
        htf_by_date_group[(row['date'], row['bar_group'])] = idx

    for i in range(len(df_5m_copy)):
        date_val = df_5m_copy.iloc[i]['date']
        group = df_5m_copy.iloc[i]['_15m_group']
        # Use the PREVIOUS completed 15m bar (avoid look-ahead)
        target_group = group - 1
        if target_group >= 0:
            key = (date_val, target_group)
        else:
            # First 15m bar of day — use last bar of previous day
            key = None
            # Find previous date
            date_idx = df_5m_copy.iloc[i].name
            if i > 0:
                prev_date = df_5m_copy.iloc[i-1]['date']
                if prev_date != date_val:
                    # Get last group of previous date
                    prev_groups = [k for k in htf_by_date_group if k[0] == prev_date]
                    if prev_groups:
                        key = max(prev_groups, key=lambda x: x[1])

        if key and key in htf_by_date_group:
            mapping[i] = htf_by_date_group[key]
        else:
            mapping[i] = None

    return mapping, df_15m_indexed


def get_htf_features_for_signal(signal_idx, mapping, df_15m_indexed, htf_cols):
    """Get 15m features for a signal at 5m bar index."""
    htf_idx = mapping.get(signal_idx)
    if htf_idx is None or htf_idx >= len(df_15m_indexed):
        return {col: np.nan for col in htf_cols}

    row = df_15m_indexed.iloc[htf_idx]
    return {col: float(row[col]) if col in row.index and not pd.isna(row[col]) else np.nan
            for col in htf_cols}


def analyze_indicator_discrimination(winners_features, losers_features, indicator_name):
    """Compute how well an indicator separates winners from losers."""
    w_vals = [f for f in winners_features if not np.isnan(f)]
    l_vals = [f for f in losers_features if not np.isnan(f)]

    if len(w_vals) < 10 or len(l_vals) < 10:
        return None

    w_mean = np.mean(w_vals)
    l_mean = np.mean(l_vals)
    w_std = np.std(w_vals) + 1e-8
    l_std = np.std(l_vals) + 1e-8
    pooled_std = np.sqrt((w_std**2 + l_std**2) / 2)

    # Cohen's d (effect size)
    cohens_d = (w_mean - l_mean) / pooled_std

    # Find optimal threshold via sweep
    all_vals = sorted(w_vals + l_vals)
    best_score = 0
    best_thresh = None
    best_direction = None
    best_wr = 0
    best_keep = 0

    n_total = len(w_vals) + len(l_vals)
    base_wr = len(w_vals) / n_total

    for pct in range(10, 91, 5):
        thresh = np.percentile(all_vals, pct)

        # Direction 1: indicator > thresh means TAKE trade
        w_pass = sum(1 for v in w_vals if v > thresh)
        l_pass = sum(1 for v in l_vals if v > thresh)
        total_pass = w_pass + l_pass
        if total_pass >= 10:
            wr = w_pass / total_pass
            keep_ratio = total_pass / n_total
            if wr > base_wr and keep_ratio >= 0.30:
                score = (wr - base_wr) * 100 + keep_ratio * 20
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
                    best_direction = 'above'
                    best_wr = wr
                    best_keep = keep_ratio

        # Direction 2: indicator < thresh means TAKE trade
        w_pass = sum(1 for v in w_vals if v < thresh)
        l_pass = sum(1 for v in l_vals if v < thresh)
        total_pass = w_pass + l_pass
        if total_pass >= 10:
            wr = w_pass / total_pass
            keep_ratio = total_pass / n_total
            if wr > base_wr and keep_ratio >= 0.30:
                score = (wr - base_wr) * 100 + keep_ratio * 20
                if score > best_score:
                    best_score = score
                    best_thresh = thresh
                    best_direction = 'below'
                    best_wr = wr
                    best_keep = keep_ratio

    return {
        'indicator': indicator_name,
        'w_mean': w_mean,
        'l_mean': l_mean,
        'cohens_d': cohens_d,
        'best_thresh': best_thresh,
        'best_direction': best_direction,
        'best_wr': best_wr,
        'base_wr': base_wr,
        'wr_improvement': best_wr - base_wr if best_wr else 0,
        'keep_ratio': best_keep,
        'score': best_score,
    }


def test_combined_filters(signals_with_features, htf_cols, top_indicators, n_top=5):
    """Test combinations of top indicators as filters."""
    print(f"\n  COMBINED FILTER TEST (top {n_top} indicators):")
    print(f"  {'Combination':<50} | {'WR':>5} | {'N':>4} | {'Keep%':>5} | {'PnL':>8} | {'WR Δ':>5}")
    print(f"  {'-'*50}   {'-'*5}   {'-'*4}   {'-'*5}   {'-'*8}   {'-'*5}")

    base_wr = sum(1 for s in signals_with_features if s['label'] == 1) / len(signals_with_features)
    base_pnl = sum(s['pnl'] for s in signals_with_features)
    base_n = len(signals_with_features)

    results = []

    # Test individual filters
    for ind in top_indicators[:n_top]:
        name = ind['indicator']
        thresh = ind['best_thresh']
        direction = ind['best_direction']

        if thresh is None:
            continue

        passed = []
        for s in signals_with_features:
            val = s['htf_features'].get(name, np.nan)
            if np.isnan(val):
                continue
            if direction == 'above' and val > thresh:
                passed.append(s)
            elif direction == 'below' and val < thresh:
                passed.append(s)

        if len(passed) < 10:
            continue

        wr = sum(1 for s in passed if s['label'] == 1) / len(passed)
        pnl = sum(s['pnl'] for s in passed)
        keep = len(passed) / base_n
        wr_delta = wr - base_wr

        cond = f"{name} {direction} {thresh:.3f}"
        print(f"  {cond:<50} | {wr*100:>4.0f}% | {len(passed):>4} | {keep*100:>4.0f}% | {pnl:>+7.1f} | {wr_delta*100:>+4.0f}%")

        results.append({
            'filter': cond,
            'indicators': [(name, direction, thresh)],
            'wr': wr,
            'n': len(passed),
            'keep': keep,
            'pnl': pnl,
            'wr_delta': wr_delta,
        })

    # Test pairs
    print(f"\n  PAIR COMBINATIONS:")
    for i in range(min(n_top, len(top_indicators))):
        for j in range(i+1, min(n_top, len(top_indicators))):
            ind_i = top_indicators[i]
            ind_j = top_indicators[j]

            if ind_i['best_thresh'] is None or ind_j['best_thresh'] is None:
                continue

            passed = []
            for s in signals_with_features:
                val_i = s['htf_features'].get(ind_i['indicator'], np.nan)
                val_j = s['htf_features'].get(ind_j['indicator'], np.nan)
                if np.isnan(val_i) or np.isnan(val_j):
                    continue

                pass_i = (ind_i['best_direction'] == 'above' and val_i > ind_i['best_thresh']) or \
                         (ind_i['best_direction'] == 'below' and val_i < ind_i['best_thresh'])
                pass_j = (ind_j['best_direction'] == 'above' and val_j > ind_j['best_thresh']) or \
                         (ind_j['best_direction'] == 'below' and val_j < ind_j['best_thresh'])

                if pass_i and pass_j:
                    passed.append(s)

            if len(passed) < 10:
                continue

            wr = sum(1 for s in passed if s['label'] == 1) / len(passed)
            pnl = sum(s['pnl'] for s in passed)
            keep = len(passed) / base_n
            wr_delta = wr - base_wr

            if wr_delta > 0:
                cond = f"{ind_i['indicator']}+{ind_j['indicator']}"
                print(f"  {cond:<50} | {wr*100:>4.0f}% | {len(passed):>4} | {keep*100:>4.0f}% | {pnl:>+7.1f} | {wr_delta*100:>+4.0f}%")
                results.append({
                    'filter': cond,
                    'indicators': [
                        (ind_i['indicator'], ind_i['best_direction'], ind_i['best_thresh']),
                        (ind_j['indicator'], ind_j['best_direction'], ind_j['best_thresh']),
                    ],
                    'wr': wr,
                    'n': len(passed),
                    'keep': keep,
                    'pnl': pnl,
                    'wr_delta': wr_delta,
                })

    return results


def run_analysis(df_5m, config, df_15m_indexed, mapping, htf_cols):
    """Run full MTF indicator analysis for one strategy."""
    name = f"{config['strategy']} {config['session_filter']}/{config['direction_filter']}"

    highs = df_5m['high'].values
    lows = df_5m['low'].values
    closes = df_5m['close'].values
    mins_arr = df_5m['mins'].values

    signals = collect_labeled_signals(df_5m, config, highs, lows, closes, mins_arr)

    if len(signals) < 30:
        print(f"  {name}: {len(signals)} signals — too few, skipping")
        return None

    # Get 15m features for each signal
    signals_with_features = []
    for s in signals:
        htf_feats = get_htf_features_for_signal(s['idx'], mapping, df_15m_indexed, htf_cols)
        signals_with_features.append({**s, 'htf_features': htf_feats})

    winners = [s for s in signals_with_features if s['label'] == 1]
    losers = [s for s in signals_with_features if s['label'] == 0]

    base_wr = len(winners) / len(signals_with_features) * 100
    base_pnl = sum(s['pnl'] for s in signals_with_features)

    print(f"\n{'='*90}")
    print(f"  {name} — MTF INDICATOR DISCOVERY")
    print(f"  {len(signals_with_features)} signals, WR {base_wr:.0f}%, PnL {base_pnl:+.1f}")
    print(f"{'='*90}")

    # Analyze each 15m indicator
    indicator_results = []
    for col in htf_cols:
        w_vals = [s['htf_features'][col] for s in winners]
        l_vals = [s['htf_features'][col] for s in losers]

        result = analyze_indicator_discrimination(w_vals, l_vals, col)
        if result and result['score'] > 0:
            indicator_results.append(result)

    # Sort by score (best discrimination)
    indicator_results.sort(key=lambda x: x['score'], reverse=True)

    # Print top indicators
    print(f"\n  TOP 15m INDICATORS (by WR improvement):")
    print(f"  {'Indicator':<25} | {'Winners':>8} | {'Losers':>8} | {'Cohen d':>7} | {'Thresh':>8} | {'Dir':>5} | {'WR':>5} | {'Keep%':>5} | {'WR Δ':>5}")
    print(f"  {'-'*25}   {'-'*8}   {'-'*8}   {'-'*7}   {'-'*8}   {'-'*5}   {'-'*5}   {'-'*5}   {'-'*5}")

    for r in indicator_results[:20]:
        print(f"  {r['indicator']:<25} | {r['w_mean']:>+8.3f} | {r['l_mean']:>+8.3f} | {r['cohens_d']:>+6.3f} | "
              f"{r['best_thresh']:>8.3f} | {r['best_direction']:>5} | {r['best_wr']*100:>4.0f}% | "
              f"{r['keep_ratio']*100:>4.0f}% | {r['wr_improvement']*100:>+4.1f}%")

    # Test combined filters
    combined_results = test_combined_filters(signals_with_features, htf_cols, indicator_results, n_top=7)

    return {
        'strategy': config['strategy'],
        'direction': config['direction_filter'],
        'session': config['session_filter'],
        'n_signals': len(signals_with_features),
        'base_wr': base_wr,
        'base_pnl': base_pnl,
        'top_indicators': indicator_results[:10],
        'combined_results': combined_results[:5] if combined_results else [],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', default='5m', choices=['1m', '5m'])
    args = parser.parse_args()

    strategies = STRATEGIES_5M if args.tf == '5m' else STRATEGIES_1M

    print(f"[MTF INDICATOR DISCOVERY] Finding 15m indicators that separate winners from losers")
    print(f"  Goal: take MORE winners while rejecting MORE losers")
    print(f"{'='*90}\n")

    # Load base timeframe data
    df = load_data(tf=args.tf, days=9999)
    if df is None or df.empty:
        print("[ERROR] No data!")
        return

    n_days = df['date'].nunique()
    print(f"[DATA] {len(df)} bars ({args.tf}), {n_days} days")

    # Compute 15m bars
    print(f"[15m] Computing 15m bars from {args.tf}...")
    if args.tf == '5m':
        df_15m = compute_15m_bars(df)
    else:
        # For 1m, first aggregate to 5m then to 15m
        df_temp = df.copy()
        df_temp['bar_group_5m'] = df_temp.groupby('date').cumcount() // 5
        df_5m_agg = df_temp.groupby(['date', 'bar_group_5m']).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum', 'mins': 'last',
        }).reset_index().rename(columns={'bar_group_5m': 'bar_group_base'})

        df_5m_agg['bar_group'] = df_5m_agg.groupby('date').cumcount() // 3
        df_15m = df_5m_agg.groupby(['date', 'bar_group']).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum', 'mins': 'last',
        }).reset_index()

    print(f"[15m] {len(df_15m)} 15m bars")

    # Compute 15m indicators
    print(f"[15m] Computing indicators on 15m timeframe...", flush=True)
    t0 = time.time()
    df_15m = compute_htf_indicators(df_15m)
    print(f"  Done in {time.time()-t0:.1f}s")

    # HTF indicator columns
    htf_cols = [c for c in df_15m.columns if c not in
                ('date', 'bar_group', 'open', 'high', 'low', 'close', 'volume', 'mins')]
    print(f"[15m] {len(htf_cols)} HTF indicators computed\n")

    # Map 5m bars to 15m bars
    print(f"[MAP] Mapping {args.tf} signals to 15m bars...", flush=True)
    mapping, df_15m_indexed = map_5m_to_15m(df, df_15m)
    print(f"  Done\n")

    # Run analysis per strategy
    all_results = []
    for config in strategies:
        result = run_analysis(df, config, df_15m_indexed, mapping, htf_cols)
        if result:
            all_results.append(result)

    # Final summary
    print(f"\n\n{'='*90}")
    print(f"  FINAL SUMMARY — BEST 15m FILTERS PER STRATEGY")
    print(f"{'='*90}")

    for r in all_results:
        print(f"\n  {r['strategy']} {r['session']}/{r['direction']} (base WR={r['base_wr']:.0f}%, N={r['n_signals']}):")
        if r['top_indicators']:
            top = r['top_indicators'][0]
            print(f"    Best single: {top['indicator']} {top['best_direction']} {top['best_thresh']:.3f} → "
                  f"WR {top['best_wr']*100:.0f}% (Δ{top['wr_improvement']*100:+.0f}%), keep {top['keep_ratio']*100:.0f}%")
        if r['combined_results']:
            best_combo = max(r['combined_results'], key=lambda x: x['wr_delta'])
            print(f"    Best combo:  {best_combo['filter']} → "
                  f"WR {best_combo['wr']*100:.0f}% (Δ{best_combo['wr_delta']*100:+.0f}%), "
                  f"N={best_combo['n']}, PnL {best_combo['pnl']:+.1f}")

    # Save results
    os.makedirs('ml/reports', exist_ok=True)
    output_file = f"ml/reports/mtf_discovery_{args.tf}_{date.today().isoformat()}.json"
    save_data = {
        'timeframe': args.tf,
        'htf': '15m',
        'n_indicators': len(htf_cols),
        'results': [{k: v for k, v in r.items() if k != 'combined_results'} for r in all_results],
    }
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[SAVED] {output_file}")


if __name__ == '__main__':
    main()
