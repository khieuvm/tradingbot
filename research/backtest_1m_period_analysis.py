"""Deep analysis of 1m strategies + 5m filters: full period vs recent 30d.

Questions:
1. Do strategies degrade over time (regime drift)?
2. How does SL behave - too tight, too loose, regime-dependent?
3. Which combos are stable vs which only worked in a specific window?

Usage:
    python -m research.backtest_1m_period_analysis
"""
import sys
import os
import json
import time
from datetime import date, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from strategies.base import BaseStrategy, EXIT_PRESETS, COST, AM_CUTOFF, PM_CUTOFF

# ══════════════════════════════════════════════════════════════════════════════
# TOP COMBOS FROM scan_all_1m_combos (filtered by PnL > 100 and meaningful WR)
# ══════════════════════════════════════════════════════════════════════════════
TOP_COMBOS = [
    # (strategy_name, session_filter, direction_filter, htf_indicator, htf_direction, htf_threshold)
    ('momentum_trend', 'PM', 'SELL', 'ema8_5m', 'above', 1885.76),
    ('momentum_trend', 'PM', 'BUY', 'ema21_slope_5m', 'above', 0.000427),
    ('macd_cross', 'PM', 'BUY', 'ema21_slope_5m', 'above', 0.000413),
    ('macd_cross', 'ALL', 'BUY', 'ema8_slope_5m', 'above', 0.000678),
    ('fibonacci', 'PM', 'SELL', 'macd_hist_5m', 'above', 0.0237),
    ('macd_cross', 'AM', 'SELL', 'll_5m', 'above', 1.0),
    ('heikin_ashi', 'AM', 'SELL', 'macd_line_5m', 'below', 5.948),
    ('heikin_ashi', 'AM', 'BUY', 'macd_signal_5m', 'above', 3.227),
    ('fibonacci', 'AM', 'SELL', 'stoch_k_5m', 'above', 57.34),
    ('macd_cross', 'PM', 'SELL', 'macd_signal_5m', 'below', -2.358),
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_1m_data():
    """Load 1m data with all required indicators."""
    parquet_file = os.path.join(DATA_DIR, "vn30f1m_1m.parquet")
    print(f"[DATA] Loading 1m from parquet...", flush=True)
    df = pd.read_parquet(parquet_file)
    df['time'] = pd.to_datetime(df['time'])
    df.columns = [c.lower() for c in df.columns]
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

    n_days = df['date'].nunique()
    print(f"[DATA] {len(df)} bars, {n_days} days ({df['date'].iloc[0]} to {df['date'].iloc[-1]})", flush=True)
    return df


def compute_5m_from_1m(df_1m):
    """Aggregate 1m bars into 5m bars."""
    df_1m = df_1m.copy()
    df_1m['bar_group'] = df_1m.groupby('date').cumcount() // 5

    df_5m = df_1m.groupby(['date', 'bar_group']).agg({
        'time': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'mins': 'first',
        'session': 'first',
    }).reset_index()

    return df_5m


def compute_5m_indicators(df_5m):
    """Compute all 5m indicators needed for HTF filtering."""
    df_5m['atr_5m'] = ta.atr(df_5m['high'], df_5m['low'], df_5m['close'], length=14)
    df_5m['ema8_5m'] = ta.ema(df_5m['close'], length=8)
    df_5m['ema21_5m'] = ta.ema(df_5m['close'], length=21)
    df_5m['ema8_slope_5m'] = df_5m['ema8_5m'].diff() / df_5m['ema8_5m'].shift(1)
    df_5m['ema21_slope_5m'] = df_5m['ema21_5m'].diff() / df_5m['ema21_5m'].shift(1)

    macd_df = ta.macd(df_5m['close'], fast=12, slow=26, signal=9)
    if macd_df is not None:
        df_5m['macd_hist_5m'] = macd_df.iloc[:, 1]
        df_5m['macd_line_5m'] = macd_df.iloc[:, 0]
        df_5m['macd_signal_5m'] = macd_df.iloc[:, 2]
    else:
        df_5m['macd_hist_5m'] = df_5m['macd_line_5m'] = df_5m['macd_signal_5m'] = np.nan

    stoch_df = ta.stoch(df_5m['high'], df_5m['low'], df_5m['close'], k=14, d=3)
    if stoch_df is not None:
        df_5m['stoch_k_5m'] = stoch_df.iloc[:, 0]
    else:
        df_5m['stoch_k_5m'] = np.nan

    # Lower lows count (for ll_5m filter)
    df_5m['ll_5m'] = 0.0
    lows = df_5m['low'].values
    for i in range(2, len(lows)):
        count = 0
        if lows[i] < lows[i-1]:
            count += 1
        if i >= 2 and lows[i-1] < lows[i-2]:
            count += 1
        df_5m.iloc[i, df_5m.columns.get_loc('ll_5m')] = count

    return df_5m


def map_1m_to_5m(df_1m, df_5m):
    """Map each 1m bar to the most recently COMPLETED 5m bar (vectorized)."""
    htf_by_date_group = {}
    for idx in range(len(df_5m)):
        d = df_5m['date'].iloc[idx]
        g = df_5m['bar_group'].iloc[idx]
        htf_by_date_group[(d, g)] = idx

    dates_arr = df_1m['date'].values
    groups_1m = (df_1m.groupby('date').cumcount() // 5).values

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
            if date_val != prev_date:
                prev_date = date_val
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


def get_strategy(name):
    """Load a strategy by name."""
    if name == 'momentum_trend':
        from strategies.momentum_trend import MomentumTrendStrategy
        return MomentumTrendStrategy()
    elif name == 'macd_cross':
        from strategies.macd_cross import MACDCrossStrategy
        return MACDCrossStrategy()
    elif name == 'fibonacci':
        from strategies.fibonacci import FibonacciStrategy
        return FibonacciStrategy()
    elif name == 'heikin_ashi':
        from strategies.heikin_ashi import HeikinAshiStrategy
        return HeikinAshiStrategy()
    else:
        raise ValueError(f"Unknown strategy: {name}")


def simulate_trade_detailed(df, entry_idx, direction, atr, session, exit_params):
    """Simulate trade and return detailed info including SL distance and what happened."""
    entry = df['close'].iloc[entry_idx]
    sl_mult = exit_params['sl_mult']
    trail_pts = exit_params['trail_pts']
    trail_mult = exit_params['trail_mult']
    tp_mult = exit_params['tp_mult']
    max_hold = exit_params['max_hold']
    be_trigger = exit_params.get('be_trigger')
    trail_tighten_pts = exit_params.get('trail_tighten_pts')
    trail_tighten_mult = exit_params.get('trail_tighten_mult')
    cutoff = AM_CUTOFF if session == 'AM' else PM_CUTOFF

    sl_initial = entry - direction * sl_mult * atr
    sl = sl_initial
    tp = entry + direction * tp_mult * atr if tp_mult else None
    best_price = entry
    mfe = 0.0
    mae = 0.0  # maximum adverse excursion
    trail_active = False
    be_active = False
    bars_to_mfe = 0
    sl_distance_pts = sl_mult * atr

    for j in range(1, min(max_hold + 1, len(df) - entry_idx)):
        bar_idx = entry_idx + j
        bar = df.iloc[bar_idx]
        bar_h = bar['high']
        bar_l = bar['low']
        bar_mins = bar['mins']

        if bar_mins >= cutoff:
            exit_price = bar['close']
            pnl = direction * (exit_price - entry)
            return {
                'pnl_raw': pnl, 'mfe': mfe, 'mae': mae, 'reason': 'SESSION',
                'bars': j, 'sl_dist': sl_distance_pts, 'bars_to_mfe': bars_to_mfe,
                'trail_active': trail_active, 'be_active': be_active
            }

        if direction == 1:
            cur_mfe = bar_h - entry
            cur_mae = entry - bar_l
            if bar_h > best_price:
                best_price = bar_h
        else:
            cur_mfe = entry - bar_l
            cur_mae = bar_h - entry
            if bar_l < best_price:
                best_price = bar_l

        if cur_mfe > mfe:
            mfe = cur_mfe
            bars_to_mfe = j
        mae = max(mae, cur_mae)

        if be_trigger and not be_active and mfe >= be_trigger:
            be_active = True
            sl = entry if direction == 1 else entry

        if mfe >= trail_pts:
            trail_active = True
            tm = trail_tighten_mult if (trail_tighten_pts and mfe >= trail_tighten_pts) else trail_mult
            if direction == 1:
                sl = max(sl, best_price - tm * atr)
            else:
                sl = min(sl, best_price + tm * atr)

        if direction == 1 and bar_l <= sl:
            reason = 'TRAIL' if trail_active else ('BE' if be_active else 'SL')
            pnl = direction * (sl - entry)
            return {
                'pnl_raw': pnl, 'mfe': mfe, 'mae': mae, 'reason': reason,
                'bars': j, 'sl_dist': sl_distance_pts, 'bars_to_mfe': bars_to_mfe,
                'trail_active': trail_active, 'be_active': be_active
            }
        if direction == -1 and bar_h >= sl:
            reason = 'TRAIL' if trail_active else ('BE' if be_active else 'SL')
            pnl = direction * (sl - entry)
            return {
                'pnl_raw': pnl, 'mfe': mfe, 'mae': mae, 'reason': reason,
                'bars': j, 'sl_dist': sl_distance_pts, 'bars_to_mfe': bars_to_mfe,
                'trail_active': trail_active, 'be_active': be_active
            }

        if tp:
            if direction == 1 and bar_h >= tp:
                pnl = tp - entry
                return {
                    'pnl_raw': pnl, 'mfe': mfe, 'mae': mae, 'reason': 'TP',
                    'bars': j, 'sl_dist': sl_distance_pts, 'bars_to_mfe': bars_to_mfe,
                    'trail_active': trail_active, 'be_active': be_active
                }
            if direction == -1 and bar_l <= tp:
                pnl = entry - tp
                return {
                    'pnl_raw': pnl, 'mfe': mfe, 'mae': mae, 'reason': 'TP',
                    'bars': j, 'sl_dist': sl_distance_pts, 'bars_to_mfe': bars_to_mfe,
                    'trail_active': trail_active, 'be_active': be_active
                }

    last_idx = min(entry_idx + max_hold, len(df) - 1)
    exit_price = df['close'].iloc[last_idx]
    pnl = direction * (exit_price - entry)
    return {
        'pnl_raw': pnl, 'mfe': mfe, 'mae': mae, 'reason': 'MAX_HOLD',
        'bars': max_hold, 'sl_dist': sl_distance_pts, 'bars_to_mfe': bars_to_mfe,
        'trail_active': trail_active, 'be_active': be_active
    }


def collect_signals_with_filter(df_1m, df_5m, mapping, strategy, session_filter, direction_filter, htf_indicator, htf_direction, htf_threshold):
    """Collect signals from strategy and apply HTF filter, returning detailed trades."""
    strategy_obj = get_strategy(strategy)
    strategy_obj.prepare(df_1m)

    trades = []
    last_idx = -10

    for i in range(60, len(df_1m) - 2):
        if i - last_idx < 5:
            continue

        session = df_1m['session'].iloc[i]
        if session not in ('AM', 'PM'):
            continue
        if session_filter != 'ALL' and session != session_filter:
            continue

        mins = df_1m['mins'].iloc[i]
        if session == 'AM' and not (555 <= mins <= 645):
            continue
        if session == 'PM' and not (795 <= mins <= 855):
            continue

        atr = df_1m['atr'].iloc[i]
        if pd.isna(atr) or atr < 0.5:
            continue

        direction = strategy_obj.detect(df_1m, i)
        if direction == 0:
            continue
        if direction_filter == 'BUY' and direction != 1:
            continue
        if direction_filter == 'SELL' and direction != -1:
            continue

        # Apply HTF filter
        htf_idx = mapping.get(i)
        if htf_idx is None:
            continue
        htf_val = df_5m[htf_indicator].iloc[htf_idx]
        if pd.isna(htf_val):
            continue
        if htf_direction == 'above' and htf_val < htf_threshold:
            continue
        if htf_direction == 'below' and htf_val > htf_threshold:
            continue

        # Simulate trade
        exit_params = strategy_obj.get_exit_params(session)
        result = simulate_trade_detailed(df_1m, i, direction, atr, session, exit_params)

        trades.append({
            'date': df_1m['date'].iloc[i],
            'time': df_1m['time'].iloc[i],
            'session': session,
            'mins': mins,
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry': df_1m['close'].iloc[i],
            'atr': atr,
            'pnl': result['pnl_raw'] - COST,
            'pnl_raw': result['pnl_raw'],
            'mfe': result['mfe'],
            'mae': result['mae'],
            'reason': result['reason'],
            'bars': result['bars'],
            'sl_dist': result['sl_dist'],
            'bars_to_mfe': result['bars_to_mfe'],
            'trail_active': result['trail_active'],
            'be_active': result['be_active'],
        })
        last_idx = i

    return trades


def analyze_period(trades, label):
    """Analyze a set of trades for a specific period."""
    if not trades:
        return None

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n = len(trades)
    total_pnl = sum(pnls)
    dates = set(t['date'] for t in trades)
    n_days = max(len(dates), 1)
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001

    # SL analysis
    sl_trades = [t for t in trades if t['reason'] == 'SL']
    trail_trades = [t for t in trades if t['reason'] == 'TRAIL']
    session_trades = [t for t in trades if t['reason'] == 'SESSION']
    maxhold_trades = [t for t in trades if t['reason'] == 'MAX_HOLD']
    be_trades = [t for t in trades if t['reason'] == 'BE']
    tp_trades = [t for t in trades if t['reason'] == 'TP']

    # SL hit analysis
    sl_hit_mfe = [t['mfe'] for t in sl_trades] if sl_trades else []
    sl_hit_mae = [t['mae'] for t in sl_trades] if sl_trades else []
    sl_hit_bars = [t['bars'] for t in sl_trades] if sl_trades else []

    # All trades MFE/MAE
    all_mfe = [t['mfe'] for t in trades]
    all_mae = [t['mae'] for t in trades]
    all_sl_dist = [t['sl_dist'] for t in trades]

    # Losers that had positive MFE (went in our direction first, then reversed)
    losers_with_mfe = [t for t in trades if t['pnl'] <= 0 and t['mfe'] > 1.0]

    return {
        'label': label,
        'n': n,
        'n_days': n_days,
        'wr': len(wins) / n * 100,
        'pf': gross_win / gross_loss,
        'total_pnl': total_pnl,
        'pnl_per_day': total_pnl / n_days,
        'avg_pnl': total_pnl / n,
        'avg_mfe': np.mean(all_mfe),
        'avg_mae': np.mean(all_mae),
        'avg_sl_dist': np.mean(all_sl_dist),
        'exit_breakdown': {
            'SL': len(sl_trades),
            'TRAIL': len(trail_trades),
            'SESSION': len(session_trades),
            'MAX_HOLD': len(maxhold_trades),
            'BE': len(be_trades),
            'TP': len(tp_trades),
        },
        'sl_pct': len(sl_trades) / n * 100 if n else 0,
        'sl_avg_mfe_before_hit': np.mean(sl_hit_mfe) if sl_hit_mfe else 0,
        'sl_avg_mae': np.mean(sl_hit_mae) if sl_hit_mae else 0,
        'sl_avg_bars': np.mean(sl_hit_bars) if sl_hit_bars else 0,
        'losers_had_mfe_gt1': len(losers_with_mfe),
        'losers_had_mfe_gt1_pct': len(losers_with_mfe) / max(len(losses), 1) * 100,
    }


def print_period_comparison(full, recent, combo_name):
    """Print side-by-side comparison."""
    if full is None:
        print(f"  {combo_name}: NO TRADES in full period")
        return
    if recent is None:
        recent = {'n': 0, 'wr': 0, 'pf': 0, 'pnl_per_day': 0, 'sl_pct': 0,
                  'avg_mfe': 0, 'avg_mae': 0, 'avg_sl_dist': 0}

    print(f"\n{'='*80}")
    print(f"  {combo_name}")
    print(f"{'='*80}")
    print(f"  {'Metric':<25} {'FULL PERIOD':>15} {'LAST 30 DAYS':>15} {'DELTA':>12}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")

    def row(label, v1, v2, fmt='.1f', suffix=''):
        if isinstance(v2, (int, float)) and isinstance(v1, (int, float)):
            delta = v2 - v1
            delta_s = f"{delta:+{fmt}}{suffix}"
        else:
            delta_s = "N/A"
        print(f"  {label:<25} {v1:>14{fmt}}{suffix} {v2:>14{fmt}}{suffix} {delta_s:>12}")

    row('Trades', full['n'], recent.get('n', 0), 'd')
    row('Win Rate %', full['wr'], recent.get('wr', 0), '.1f', '%')
    row('Profit Factor', full['pf'], recent.get('pf', 0), '.2f')
    row('PnL/day', full['pnl_per_day'], recent.get('pnl_per_day', 0), '.2f')
    row('Avg MFE', full['avg_mfe'], recent.get('avg_mfe', 0), '.2f')
    row('Avg MAE', full['avg_mae'], recent.get('avg_mae', 0), '.2f')
    row('Avg SL distance', full['avg_sl_dist'], recent.get('avg_sl_dist', 0), '.2f')
    row('SL hit %', full['sl_pct'], recent.get('sl_pct', 0), '.1f', '%')

    # Exit breakdown
    print(f"\n  Exit Breakdown (full):")
    for reason, count in sorted(full['exit_breakdown'].items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / full['n'] * 100
            print(f"    {reason:<12} {count:>4} ({pct:.1f}%)")

    if recent.get('n', 0) > 0:
        print(f"\n  Exit Breakdown (last 30d):")
        for reason, count in sorted(recent['exit_breakdown'].items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / recent['n'] * 100
                print(f"    {reason:<12} {count:>4} ({pct:.1f}%)")

    # SL deep analysis
    print(f"\n  SL ANALYSIS:")
    print(f"    SL trades had avg MFE before hit: {full['sl_avg_mfe_before_hit']:.2f} pts")
    print(f"    SL trades avg MAE (how far wrong): {full['sl_avg_mae']:.2f} pts")
    print(f"    SL trades avg bars to hit: {full['sl_avg_bars']:.1f}")
    print(f"    Losers that had MFE>1pt first: {full['losers_had_mfe_gt1']} ({full['losers_had_mfe_gt1_pct']:.1f}% of losers)")


def rolling_window_analysis(trades, window_days=7):
    """Analyze performance in rolling windows to detect regime drift."""
    if not trades:
        return []

    trades_sorted = sorted(trades, key=lambda t: t['date'])
    all_dates = sorted(set(t['date'] for t in trades_sorted))
    if len(all_dates) < window_days * 2:
        return []

    windows = []
    for i in range(0, len(all_dates) - window_days + 1, window_days):
        window_dates = set(all_dates[i:i+window_days])
        window_trades = [t for t in trades_sorted if t['date'] in window_dates]
        if len(window_trades) >= 3:
            pnls = [t['pnl'] for t in window_trades]
            wins = [p for p in pnls if p > 0]
            wr = len(wins) / len(pnls) * 100
            total = sum(pnls)
            windows.append({
                'start': min(window_dates),
                'end': max(window_dates),
                'n': len(window_trades),
                'wr': wr,
                'pnl': total,
                'pnl_per_day': total / window_days,
                'sl_pct': len([t for t in window_trades if t['reason'] == 'SL']) / len(window_trades) * 100,
            })
    return windows


def sl_optimality_analysis(trades):
    """Analyze if SL is too tight or too loose.

    Key questions:
    - What % of SL hits later would have been winners (SL too tight)?
    - What % of losses had MFE > 2*SL distance (could have exited profitably)?
    - What's the optimal SL in hindsight?
    """
    if not trades:
        return

    sl_trades = [t for t in trades if t['reason'] == 'SL']
    all_losers = [t for t in trades if t['pnl'] <= 0]

    print(f"\n  SL OPTIMALITY ANALYSIS:")
    print(f"  {'─'*50}")

    if not sl_trades:
        print(f"    No SL exits found")
        return

    # How many SL hits had MFE > 0 (price went our way first)
    sl_had_mfe = [t for t in sl_trades if t['mfe'] > 0.5]
    print(f"    SL hits that went profitable first (MFE>0.5): {len(sl_had_mfe)}/{len(sl_trades)} ({len(sl_had_mfe)/len(sl_trades)*100:.1f}%)")

    # Distribution of MAE at SL hit — does it cluster near the SL level?
    sl_dists = [t['sl_dist'] for t in sl_trades]
    maes = [t['mae'] for t in sl_trades]
    mfes = [t['mfe'] for t in sl_trades]

    print(f"    Avg SL distance: {np.mean(sl_dists):.2f} pts")
    print(f"    Avg MAE when SL hit: {np.mean(maes):.2f} pts")
    print(f"    Avg MFE before SL hit: {np.mean(mfes):.2f} pts")

    # Optimal SL hindsight: what if SL was 1.5x or 2x wider?
    all_mfes = [t['mfe'] for t in trades]
    all_maes = [t['mae'] for t in trades]

    # What % of all trades have MAE < various thresholds
    for mult_label, mult in [('0.5x', 0.5), ('1.0x', 1.0), ('1.5x', 1.5), ('2.0x', 2.0), ('2.5x', 2.5), ('3.0x', 3.0)]:
        # Hypothetical SL at mult * avg_atr
        avg_atr = np.mean([t['atr'] for t in trades])
        sl_thresh = mult * avg_atr
        would_survive = len([t for t in trades if t['mae'] <= sl_thresh])
        pct = would_survive / len(trades) * 100
        # Net PnL if only trades that survive this SL → use their actual outcome
        survivors = [t for t in trades if t['mae'] <= sl_thresh]
        stopped = [t for t in trades if t['mae'] > sl_thresh]
        survivor_pnl = sum(t['pnl'] for t in survivors)
        stopped_loss = sum(mult * t['atr'] + COST for t in stopped)
        net_pnl = survivor_pnl - stopped_loss
        print(f"    SL={mult_label}×ATR: {pct:.0f}% survive, est net PnL={net_pnl:.1f}")


def main():
    t0 = time.time()

    # Load data
    df_1m = load_1m_data()

    # Compute 5m bars and indicators
    print("[HTF] Computing 5m bars from 1m...", flush=True)
    df_5m = compute_5m_from_1m(df_1m)
    df_5m = compute_5m_indicators(df_5m)
    print(f"[HTF] {len(df_5m)} 5m bars computed", flush=True)

    # Map 1m to 5m
    print("[HTF] Building 1m→5m mapping...", flush=True)
    mapping = map_1m_to_5m(df_1m, df_5m)
    print(f"[HTF] Mapping done ({len(mapping)} entries)", flush=True)

    # Define period split
    all_dates = sorted(df_1m['date'].unique())
    cutoff_30d = date.today() - timedelta(days=30)
    recent_dates = set(d for d in all_dates if d >= cutoff_30d)

    print(f"\n[PERIODS] Full: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")
    print(f"[PERIODS] Recent 30d: {cutoff_30d} onward ({len(recent_dates)} days)")

    # ══════════════════════════════════════════════════════════════════════════
    # RUN EACH COMBO
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  RUNNING TOP 10 COMBOS: FULL vs LAST 30 DAYS")
    print(f"{'═'*80}")

    all_results = []

    for combo in TOP_COMBOS:
        strategy_name, session_filter, direction_filter, htf_indicator, htf_direction, htf_threshold = combo
        combo_label = f"{strategy_name} {session_filter}/{direction_filter} [{htf_indicator} {htf_direction} {htf_threshold}]"

        print(f"\n[COMBO] {combo_label}...", flush=True)

        trades = collect_signals_with_filter(
            df_1m, df_5m, mapping,
            strategy_name, session_filter, direction_filter,
            htf_indicator, htf_direction, htf_threshold
        )

        if not trades:
            print(f"  → 0 trades, skipping")
            continue

        # Split into full period and recent 30d
        trades_full = trades
        trades_recent = [t for t in trades if t['date'] >= cutoff_30d]

        # Analyze both periods
        full_stats = analyze_period(trades_full, 'FULL')
        recent_stats = analyze_period(trades_recent, 'RECENT_30D') if trades_recent else None

        # Print comparison
        print_period_comparison(full_stats, recent_stats, combo_label)

        # Rolling window drift analysis
        windows = rolling_window_analysis(trades_full, window_days=7)
        if windows:
            print(f"\n  ROLLING 7-DAY WINDOWS:")
            print(f"    {'Period':<25} {'Trades':>6} {'WR%':>7} {'PnL/d':>8} {'SL%':>6}")
            for w in windows:
                print(f"    {str(w['start'])} - {str(w['end'])[-5:]}  {w['n']:>6} {w['wr']:>6.1f}% {w['pnl_per_day']:>+7.2f} {w['sl_pct']:>5.1f}%")

            # Trend detection
            if len(windows) >= 3:
                first_half = windows[:len(windows)//2]
                second_half = windows[len(windows)//2:]
                pnl_first = np.mean([w['pnl_per_day'] for w in first_half])
                pnl_second = np.mean([w['pnl_per_day'] for w in second_half])
                if pnl_second < pnl_first * 0.5:
                    print(f"    ⚠ DEGRADING: first half avg +{pnl_first:.2f}/d → second half +{pnl_second:.2f}/d")
                elif pnl_second > pnl_first * 1.5:
                    print(f"    ✓ IMPROVING: first half +{pnl_first:.2f}/d → second half +{pnl_second:.2f}/d")
                else:
                    print(f"    ~ STABLE: first half +{pnl_first:.2f}/d → second half +{pnl_second:.2f}/d")

        # SL optimality
        sl_optimality_analysis(trades_full)

        all_results.append({
            'combo': combo_label,
            'full': full_stats,
            'recent': recent_stats,
            'trades': trades,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'═'*100}")
    print(f"  SUMMARY: ALL COMBOS — FULL vs RECENT 30 DAYS")
    print(f"{'═'*100}")
    print(f"  {'Combo':<50} {'Full WR':>7} {'Full PD':>7} {'Rec WR':>7} {'Rec PD':>7} {'Trend':>8}")
    print(f"  {'-'*50} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    for r in sorted(all_results, key=lambda x: x['full']['pnl_per_day'], reverse=True):
        f = r['full']
        rec = r['recent'] if r['recent'] else {'wr': 0, 'pnl_per_day': 0}
        trend = '↗' if rec['pnl_per_day'] > f['pnl_per_day'] * 1.2 else ('↘' if rec['pnl_per_day'] < f['pnl_per_day'] * 0.5 else '→')
        combo_short = r['combo'][:48]
        print(f"  {combo_short:<50} {f['wr']:>6.1f}% {f['pnl_per_day']:>+6.2f} {rec['wr']:>6.1f}% {rec['pnl_per_day']:>+6.2f} {trend:>8}")

    # ══════════════════════════════════════════════════════════════════════════
    # SL SENSITIVITY ANALYSIS (aggregate across all trades)
    # ══════════════════════════════════════════════════════════════════════════
    all_trades = []
    for r in all_results:
        all_trades.extend(r['trades'])

    if all_trades:
        print(f"\n\n{'═'*80}")
        print(f"  AGGREGATE SL SENSITIVITY ({len(all_trades)} total trades)")
        print(f"{'═'*80}")

        avg_atr = np.mean([t['atr'] for t in all_trades])
        print(f"  Average ATR at entry: {avg_atr:.3f} pts")
        print(f"  Current SL configs:")
        print(f"    momentum_trend: trend preset → 2.0×ATR = {2.0*avg_atr:.2f} pts")
        print(f"    macd_cross:     trend preset → 2.0×ATR = {2.0*avg_atr:.2f} pts")
        print(f"    fibonacci:      mean_rev preset → 1.0×ATR = {1.0*avg_atr:.2f} pts")
        print(f"    heikin_ashi:    trend preset → 2.0×ATR = {2.0*avg_atr:.2f} pts")

        print(f"\n  MAE DISTRIBUTION (all trades):")
        maes = [t['mae'] for t in all_trades]
        for pct in [25, 50, 75, 90, 95]:
            print(f"    P{pct}: {np.percentile(maes, pct):.2f} pts")

        print(f"\n  MAE by trade outcome:")
        winners = [t for t in all_trades if t['pnl'] > 0]
        losers = [t for t in all_trades if t['pnl'] <= 0]
        if winners:
            print(f"    Winners MAE: mean={np.mean([t['mae'] for t in winners]):.2f}, P75={np.percentile([t['mae'] for t in winners], 75):.2f}")
        if losers:
            print(f"    Losers  MAE: mean={np.mean([t['mae'] for t in losers]):.2f}, P75={np.percentile([t['mae'] for t in losers], 75):.2f}")

        # SL sweep: what multiplier maximizes PnL?
        print(f"\n  SL MULTIPLIER SWEEP (net PnL at each SL level):")
        print(f"    {'SL mult':>8} {'Survive%':>9} {'Net WR':>7} {'Est PnL':>8} {'PnL/d':>8}")
        for sl_mult in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]:
            survived = []
            stopped = []
            for t in all_trades:
                if t['mae'] <= sl_mult * t['atr']:
                    survived.append(t)
                else:
                    stopped.append(t)

            survivor_pnl = sum(t['pnl_raw'] for t in survived)
            stopped_loss = sum(sl_mult * t['atr'] for t in stopped)
            net_pnl = survivor_pnl - stopped_loss - COST * len(all_trades)
            n_days = max(len(set(t['date'] for t in all_trades)), 1)
            net_wins = len([t for t in survived if t['pnl_raw'] > 0])
            net_wr = net_wins / len(all_trades) * 100 if all_trades else 0
            survive_pct = len(survived) / len(all_trades) * 100
            print(f"    {sl_mult:>7.1f}x {survive_pct:>8.1f}% {net_wr:>6.1f}% {net_pnl:>+7.1f} {net_pnl/n_days:>+7.2f}")

        # SL per strategy
        print(f"\n  SL HIT RATE BY STRATEGY:")
        by_strat = defaultdict(list)
        for t in all_trades:
            # Extract strategy from combo
            for combo in TOP_COMBOS:
                if combo[0] in str(t.get('_combo', '')):
                    by_strat[combo[0]].append(t)
                    break

        # Group trades by strategy based on exit_type pattern
        for r in all_results:
            strat_name = r['combo'].split()[0]
            sl_count = len([t for t in r['trades'] if t['reason'] == 'SL'])
            n = len(r['trades'])
            if n > 0:
                avg_sl_dist = np.mean([t['sl_dist'] for t in r['trades']])
                avg_mae_sl = np.mean([t['mae'] for t in r['trades'] if t['reason'] == 'SL']) if sl_count > 0 else 0
                print(f"    {r['combo'][:45]:<47} SL:{sl_count}/{n} ({sl_count/n*100:.0f}%) avg_dist={avg_sl_dist:.2f}")

    elapsed = time.time() - t0
    print(f"\n[DONE] Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
