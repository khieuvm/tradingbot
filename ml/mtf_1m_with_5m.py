"""MTF Indicator Discovery for 1m strategies using 5m as higher timeframe.

For 1m signals, the natural HTF is 5m (5:1 ratio, same as 5m→15m).
Compute 5m indicators and find which ones best separate winners from losers.

Usage:
    python -m ml.mtf_1m_with_5m
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
from ml.strategy_filter import STRATEGIES_1M, collect_labeled_signals
from ml.mtf_indicator_discovery import (
    analyze_indicator_discrimination, test_combined_filters
)


def compute_5m_from_1m(df_1m):
    """Aggregate 1m bars into 5m bars."""
    df = df_1m.copy()
    df['bar_group'] = df.groupby('date').cumcount() // 5

    df_5m = df.groupby(['date', 'bar_group']).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'mins': 'last',
    }).reset_index()

    return df_5m


def compute_5m_indicators(df_5m):
    """Compute indicators on 5m timeframe (HTF for 1m signals)."""
    c = df_5m['close']
    h = df_5m['high']
    l = df_5m['low']
    o = df_5m['open']

    # Trend
    df_5m['ema8_5m'] = ta.ema(c, length=8)
    df_5m['ema21_5m'] = ta.ema(c, length=21)
    df_5m['ema50_5m'] = ta.ema(c, length=50)

    df_5m['ema8_slope_5m'] = df_5m['ema8_5m'].diff(1) / c
    df_5m['ema21_slope_5m'] = df_5m['ema21_5m'].diff(1) / c

    df_5m['ema_align_5m'] = (
        np.sign(c - df_5m['ema8_5m']).fillna(0) +
        np.sign(c - df_5m['ema21_5m']).fillna(0) +
        np.sign(c - df_5m['ema50_5m']).fillna(0)
    )

    # RSI
    df_5m['rsi_5m'] = ta.rsi(c, length=14)
    df_5m['rsi_7_5m'] = ta.rsi(c, length=7)

    # MACD
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        df_5m['macd_hist_5m'] = macd.iloc[:, 1]
        df_5m['macd_line_5m'] = macd.iloc[:, 0]
        df_5m['macd_signal_5m'] = macd.iloc[:, 2]
    else:
        df_5m['macd_hist_5m'] = 0.0
        df_5m['macd_line_5m'] = 0.0
        df_5m['macd_signal_5m'] = 0.0

    df_5m['macd_hist_dir_5m'] = np.sign(df_5m['macd_hist_5m'].diff(1))

    # ADX
    adx_df = ta.adx(h, l, c, length=14)
    if adx_df is not None:
        df_5m['adx_5m'] = adx_df['ADX_14']
        df_5m['di_plus_5m'] = adx_df['DMP_14']
        df_5m['di_minus_5m'] = adx_df['DMN_14']
    else:
        df_5m['adx_5m'] = 20.0
        df_5m['di_plus_5m'] = 20.0
        df_5m['di_minus_5m'] = 20.0

    df_5m['di_spread_5m'] = df_5m['di_plus_5m'] - df_5m['di_minus_5m']

    # ATR
    df_5m['atr_5m'] = ta.atr(h, l, c, length=14)

    # Stochastic
    stoch = ta.stoch(h, l, c, k=14, d=3)
    if stoch is not None:
        df_5m['stoch_k_5m'] = stoch.iloc[:, 0]
        df_5m['stoch_d_5m'] = stoch.iloc[:, 1]
    else:
        df_5m['stoch_k_5m'] = 50.0
        df_5m['stoch_d_5m'] = 50.0

    # Bollinger Band position
    sma20 = ta.sma(c, length=20)
    std20 = c.rolling(20).std()
    df_5m['bb_pos_5m'] = ((c - sma20) / (2 * std20)).where(std20 > 0, 0)

    # Keltner Channel position
    ema20 = ta.ema(c, length=20)
    df_5m['kc_pos_5m'] = ((c - ema20) / (1.5 * df_5m['atr_5m'])).where(df_5m['atr_5m'] > 0, 0)

    # Candle structure
    bar_range = h - l
    df_5m['body_pct_5m'] = (c - o) / bar_range.replace(0, np.nan)
    df_5m['range_5m'] = bar_range

    # Higher high / Lower low
    df_5m['hh_5m'] = (h > h.shift(1)).astype(int) + (h > h.shift(2)).astype(int)
    df_5m['ll_5m'] = (l < l.shift(1)).astype(int) + (l < l.shift(2)).astype(int)

    # Volume
    v = df_5m.get('volume')
    if v is not None and not v.isna().all() and (v > 0).any():
        vol_sma = v.rolling(10).mean()
        df_5m['vol_ratio_5m'] = (v / vol_sma).where(vol_sma > 0, 1.0)
    else:
        df_5m['vol_ratio_5m'] = 1.0

    # Distance from recent high/low
    df_5m['dist_high_5_5m'] = (c - h.rolling(5).max()) / df_5m['atr_5m'].replace(0, 1)
    df_5m['dist_low_5_5m'] = (c - l.rolling(5).min()) / df_5m['atr_5m'].replace(0, 1)

    # Momentum
    df_5m['mom_3_5m'] = c - c.shift(3)
    df_5m['mom_5_5m'] = c - c.shift(5)

    # Bars above/below EMA21
    above_ema = (c > df_5m['ema21_5m']).astype(int)
    df_5m['bars_above_ema21_5m'] = above_ema.rolling(10).sum()

    return df_5m


def map_1m_to_5m(df_1m, df_5m):
    """Map 1m bar index to corresponding COMPLETED 5m bar (vectorized)."""
    htf_by_date_group = {}
    for idx, row in df_5m.iterrows():
        htf_by_date_group[(row['date'], row['bar_group'])] = idx

    dates_arr = df_1m['date'].values
    groups_1m = (df_1m.groupby('date').cumcount() // 5).values

    # Precompute last group per date for cross-day lookback
    last_group_per_date = {}
    for (d, g), idx in htf_by_date_group.items():
        if d not in last_group_per_date or g > last_group_per_date[d][0]:
            last_group_per_date[d] = (g, idx)

    mapping = {}
    prev_date = None
    prev_date_last_idx = None

    for i in range(len(df_1m)):
        date_val = dates_arr[i]
        group = groups_1m[i]
        target_group = group - 1

        if target_group >= 0:
            key = (date_val, target_group)
            mapping[i] = htf_by_date_group.get(key)
        else:
            # First 5 bars of day → use previous day's last 5m bar
            if date_val != prev_date:
                prev_date = date_val
                # Find the previous trading date
                if i > 0:
                    pd_val = dates_arr[i - 1]
                    if pd_val != date_val and pd_val in last_group_per_date:
                        prev_date_last_idx = last_group_per_date[pd_val][1]
                    else:
                        prev_date_last_idx = None
                else:
                    prev_date_last_idx = None
            mapping[i] = prev_date_last_idx

    return mapping


def main():
    print(f"[MTF 1m->5m] Finding 5m indicators to filter 1m strategy signals")
    print(f"  Goal: keep more winners, reject losers")
    print(f"{'='*90}\n")

    df_1m = load_data(tf='1m', days=9999)
    if df_1m is None or df_1m.empty:
        print("[ERROR] No 1m data!")
        return

    n_days = df_1m['date'].nunique()
    print(f"[DATA] {len(df_1m)} 1m bars, {n_days} days")

    print(f"[5m] Aggregating to 5m bars...")
    df_5m = compute_5m_from_1m(df_1m)
    print(f"[5m] {len(df_5m)} 5m bars")

    print(f"[5m] Computing indicators...", flush=True)
    t0 = time.time()
    df_5m = compute_5m_indicators(df_5m)
    print(f"  Done in {time.time()-t0:.1f}s")

    htf_cols = [c for c in df_5m.columns if c not in
                ('date', 'bar_group', 'open', 'high', 'low', 'close', 'volume', 'mins')]
    print(f"[5m] {len(htf_cols)} HTF indicators\n")

    print(f"[MAP] Mapping 1m -> 5m...", flush=True)
    mapping = map_1m_to_5m(df_1m, df_5m)
    print(f"  Done\n")

    highs = df_1m['high'].values
    lows = df_1m['low'].values
    closes = df_1m['close'].values
    mins_arr = df_1m['mins'].values

    for config in STRATEGIES_1M:
        name = f"{config['strategy']} {config['session_filter']}/{config['direction_filter']}"

        signals = collect_labeled_signals(df_1m, config, highs, lows, closes, mins_arr)
        if len(signals) < 30:
            print(f"  {name}: {len(signals)} signals, too few")
            continue

        # Get 5m features for each signal
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

        base_wr = len(winners) / len(signals_with_features) * 100
        base_pnl = sum(s['pnl'] for s in signals_with_features)

        print(f"\n{'='*90}")
        print(f"  {name} — 5m INDICATOR DISCOVERY")
        print(f"  {len(signals_with_features)} signals, WR {base_wr:.0f}%, PnL {base_pnl:+.1f}")
        print(f"  Trades/day: {len(signals_with_features)/n_days:.2f}, Trades/week: {len(signals_with_features)/n_days*5:.1f}")
        print(f"{'='*90}")

        # Analyze each indicator
        indicator_results = []
        for col in htf_cols:
            w_vals = [s['htf_features'][col] for s in winners]
            l_vals = [s['htf_features'][col] for s in losers]
            result = analyze_indicator_discrimination(w_vals, l_vals, col)
            if result and result['score'] > 0:
                indicator_results.append(result)

        indicator_results.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n  TOP 5m INDICATORS (by WR improvement):")
        print(f"  {'Indicator':<22} | {'Winners':>8} | {'Losers':>8} | {'Cohen d':>7} | {'Thresh':>8} | {'Dir':>5} | {'WR':>5} | {'Keep%':>5} | {'WR +':>5}")
        print(f"  {'-'*22}   {'-'*8}   {'-'*8}   {'-'*7}   {'-'*8}   {'-'*5}   {'-'*5}   {'-'*5}   {'-'*5}")

        for r in indicator_results[:20]:
            print(f"  {r['indicator']:<22} | {r['w_mean']:>+8.3f} | {r['l_mean']:>+8.3f} | {r['cohens_d']:>+6.3f} | "
                  f"{r['best_thresh']:>8.3f} | {r['best_direction']:>5} | {r['best_wr']*100:>4.0f}% | "
                  f"{r['keep_ratio']*100:>4.0f}% | {r['wr_improvement']*100:>+4.1f}%")

        # Combined filter test
        combined = test_combined_filters(signals_with_features, htf_cols, indicator_results, n_top=7)

        # Summary
        if indicator_results:
            top = indicator_results[0]
            print(f"\n  BEST SINGLE FILTER: {top['indicator']} {top['best_direction']} {top['best_thresh']:.3f}")
            print(f"    WR: {base_wr:.0f}% -> {top['best_wr']*100:.0f}% (+{top['wr_improvement']*100:.0f}pp)")
            print(f"    Keep: {top['keep_ratio']*100:.0f}% of trades")

            # Compute PnL for best filter
            if top['best_direction'] == 'above':
                passed = [s for s in signals_with_features
                          if not np.isnan(s['htf_features'].get(top['indicator'], np.nan))
                          and s['htf_features'][top['indicator']] > top['best_thresh']]
            else:
                passed = [s for s in signals_with_features
                          if not np.isnan(s['htf_features'].get(top['indicator'], np.nan))
                          and s['htf_features'][top['indicator']] < top['best_thresh']]

            if passed:
                filt_pnl = sum(s['pnl'] for s in passed)
                filt_n = len(passed)
                filt_pd = filt_n / n_days
                print(f"    Trades: {filt_n} ({filt_pd:.2f}/day, {filt_pd*5:.1f}/week)")
                print(f"    PnL: {base_pnl:+.1f} -> {filt_pnl:+.1f} (per trade: {filt_pnl/filt_n:+.2f})")


if __name__ == '__main__':
    main()
