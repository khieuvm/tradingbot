"""SL Sweep vs Reversal Analysis + Re-Entry Hyperopt for 1m strategies.

Questions:
1. Of all SL hits, what % are "sweeps" (price returns to original direction after)?
2. Can we re-enter after a sweep and profit?
3. What's the optimal SL distance + re-entry rule to capture 50% of real moves?

Methodology:
- For each SL trade, track price action for N bars AFTER the SL hit
- If price moves >= X pts in original direction after SL → it was a SWEEP
- If price continues against → it was a REAL REVERSAL
- Then: hyperopt (SL width, re-entry delay, re-entry threshold, new SL/TP)

Usage:
    python -m research.sl_sweep_analysis
"""
import sys
import os
import time
from datetime import date, timedelta
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from strategies.base import BaseStrategy, EXIT_PRESETS, COST, AM_CUTOFF, PM_CUTOFF

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIES TO ANALYZE (highest signal frequency)
# ══════════════════════════════════════════════════════════════════════════════
STRATEGIES = [
    ('momentum_trend', 'PM', 'SELL'),
    ('momentum_trend', 'ALL', 'SELL'),
    ('macd_cross', 'AM', 'SELL'),
    ('macd_cross', 'PM', 'BUY'),
    ('heikin_ashi', 'AM', 'SELL'),
    ('heikin_ashi', 'AM', 'BUY'),
    ('fibonacci', 'PM', 'SELL'),
    ('fibonacci', 'AM', 'SELL'),
]


def load_1m_data():
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


def get_strategy(name):
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
        raise ValueError(f"Unknown: {name}")


def collect_raw_signals(df, strategy_name, session_filter, direction_filter, dedup_bars=5):
    """Collect signals without any HTF filter — raw signal quality."""
    strategy = get_strategy(strategy_name)
    strategy.prepare(df)
    signals = []
    last_idx = -dedup_bars - 1

    for i in range(60, len(df) - 2):
        if i - last_idx < dedup_bars:
            continue
        session = df['session'].iloc[i]
        if session not in ('AM', 'PM'):
            continue
        if session_filter != 'ALL' and session != session_filter:
            continue
        mins = df['mins'].iloc[i]
        if session == 'AM' and not (555 <= mins <= 645):
            continue
        if session == 'PM' and not (795 <= mins <= 855):
            continue
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < 0.5:
            continue

        direction = strategy.detect(df, i)
        if direction == 0:
            continue
        if direction_filter == 'BUY' and direction != 1:
            continue
        if direction_filter == 'SELL' and direction != -1:
            continue

        signals.append({
            'idx': i,
            'direction': direction,
            'session': session,
            'atr': atr,
            'entry_price': df['close'].iloc[i],
            'date': df['date'].iloc[i],
            'mins': mins,
        })
        last_idx = i

    return signals


def simulate_with_post_sl_tracking(df, signals, sl_mult, max_hold, trail_pts, trail_mult,
                                    be_trigger=None, tp_mult=None, track_bars_after_sl=30):
    """Simulate trades AND track what happens after SL is hit.

    Returns: list of trade results with post-SL price path.
    """
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    mins_arr = df['mins'].values
    sessions = df['session'].values

    results = []

    for sig in signals:
        i = sig['idx']
        direction = sig['direction']
        atr = sig['atr']
        session = sig['session']
        entry = closes[i]
        cutoff = AM_CUTOFF if session == 'AM' else PM_CUTOFF

        sl = entry - direction * sl_mult * atr
        tp = entry + direction * tp_mult * atr if tp_mult else None
        best_price = entry
        mfe = 0.0
        mae = 0.0
        trail_active = False
        be_active = False
        exit_bar = None
        exit_reason = None
        exit_price = None

        end_idx = min(i + max_hold + 1, len(closes))
        for j_abs in range(i + 1, end_idx):
            bar_h = highs[j_abs]
            bar_l = lows[j_abs]

            if mins_arr[j_abs] >= cutoff:
                exit_bar = j_abs
                exit_price = closes[j_abs]
                exit_reason = 'SESSION'
                break

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
            mfe = max(mfe, cur_mfe)
            mae = max(mae, cur_mae)

            if be_trigger and not be_active and mfe >= be_trigger:
                be_active = True
                sl = entry

            if mfe >= trail_pts:
                trail_active = True
                if direction == 1:
                    sl = max(sl, best_price - trail_mult * atr)
                else:
                    sl = min(sl, best_price + trail_mult * atr)

            if direction == 1 and bar_l <= sl:
                exit_bar = j_abs
                exit_price = sl
                exit_reason = 'TRAIL' if trail_active else ('BE' if be_active else 'SL')
                break
            if direction == -1 and bar_h >= sl:
                exit_bar = j_abs
                exit_price = sl
                exit_reason = 'TRAIL' if trail_active else ('BE' if be_active else 'SL')
                break

            if tp:
                if direction == 1 and bar_h >= tp:
                    exit_bar = j_abs
                    exit_price = tp
                    exit_reason = 'TP'
                    break
                if direction == -1 and bar_l <= tp:
                    exit_bar = j_abs
                    exit_price = tp
                    exit_reason = 'TP'
                    break

        if exit_bar is None:
            exit_bar = min(i + max_hold, len(closes) - 1)
            exit_price = closes[exit_bar]
            exit_reason = 'MAX_HOLD'

        pnl = direction * (exit_price - entry) - COST
        bars_held = exit_bar - i

        # ═══ POST-SL TRACKING ═══
        # Track what happens AFTER the SL hit
        post_sl_mfe = 0.0  # max favorable move after SL in original direction
        post_sl_mae = 0.0  # max adverse move after SL
        post_sl_bars_to_recover = None
        is_sweep = False

        if exit_reason == 'SL':
            track_end = min(exit_bar + track_bars_after_sl, len(closes))
            for k in range(exit_bar + 1, track_end):
                if mins_arr[k] >= cutoff and session == sessions[k]:
                    break
                # Price movement after SL in ORIGINAL direction
                if direction == 1:
                    move_favorable = highs[k] - exit_price
                    move_adverse = exit_price - lows[k]
                else:
                    move_favorable = exit_price - lows[k]
                    move_adverse = highs[k] - exit_price

                post_sl_mfe = max(post_sl_mfe, move_favorable)
                post_sl_mae = max(post_sl_mae, move_adverse)

                # Did price recover to entry level?
                if post_sl_bars_to_recover is None:
                    if direction == 1 and highs[k] >= entry:
                        post_sl_bars_to_recover = k - exit_bar
                    elif direction == -1 and lows[k] <= entry:
                        post_sl_bars_to_recover = k - exit_bar

            # Classify: SWEEP if post-SL MFE > SL distance (price went back significantly)
            sl_distance = sl_mult * atr
            is_sweep = post_sl_mfe >= sl_distance * 0.7  # recovered at least 70% of SL dist

        results.append({
            'idx': i,
            'date': sig['date'],
            'session': session,
            'direction': direction,
            'entry': entry,
            'exit': exit_price,
            'atr': atr,
            'pnl': pnl,
            'mfe': mfe,
            'mae': mae,
            'reason': exit_reason,
            'bars': bars_held,
            'sl_dist': sl_mult * atr,
            'post_sl_mfe': post_sl_mfe,
            'post_sl_mae': post_sl_mae,
            'post_sl_bars_to_recover': post_sl_bars_to_recover,
            'is_sweep': is_sweep,
        })

    return results


def simulate_reentry(df, sl_trades_sweeps, reentry_delay, reentry_sl_mult, reentry_tp_mult, reentry_max_hold):
    """Simulate re-entry trades after detected SL sweeps.

    Re-entry logic: after SL is hit and price starts moving back in original direction,
    wait `reentry_delay` bars then enter in same direction with new SL/TP.
    """
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    mins_arr = df['mins'].values

    reentry_trades = []

    for t in sl_trades_sweeps:
        exit_bar = t['idx'] + t['bars']  # bar where SL was hit
        direction = t['direction']
        session = t['session']
        cutoff = AM_CUTOFF if session == 'AM' else PM_CUTOFF

        # Wait reentry_delay bars after SL
        reentry_bar = exit_bar + reentry_delay
        if reentry_bar >= len(closes) - 2:
            continue
        if mins_arr[reentry_bar] >= cutoff:
            continue

        # Re-enter at close of reentry_bar
        reentry_price = closes[reentry_bar]
        atr = t['atr']
        new_sl = reentry_price - direction * reentry_sl_mult * atr
        new_tp = reentry_price + direction * reentry_tp_mult * atr if reentry_tp_mult else None

        # Simulate new trade
        end_idx = min(reentry_bar + reentry_max_hold + 1, len(closes))
        re_exit_price = None
        re_exit_reason = None
        re_mfe = 0.0

        for j in range(reentry_bar + 1, end_idx):
            if mins_arr[j] >= cutoff:
                re_exit_price = closes[j]
                re_exit_reason = 'SESSION'
                break

            if direction == 1:
                cur_mfe = highs[j] - reentry_price
            else:
                cur_mfe = reentry_price - lows[j]
            re_mfe = max(re_mfe, cur_mfe)

            # SL check
            if direction == 1 and lows[j] <= new_sl:
                re_exit_price = new_sl
                re_exit_reason = 'SL'
                break
            if direction == -1 and highs[j] >= new_sl:
                re_exit_price = new_sl
                re_exit_reason = 'SL'
                break

            # TP check
            if new_tp:
                if direction == 1 and highs[j] >= new_tp:
                    re_exit_price = new_tp
                    re_exit_reason = 'TP'
                    break
                if direction == -1 and lows[j] <= new_tp:
                    re_exit_price = new_tp
                    re_exit_reason = 'TP'
                    break

        if re_exit_price is None:
            re_exit_price = closes[min(reentry_bar + reentry_max_hold, len(closes) - 1)]
            re_exit_reason = 'MAX_HOLD'

        re_pnl = direction * (re_exit_price - reentry_price) - COST

        reentry_trades.append({
            'date': t['date'],
            'session': session,
            'direction': direction,
            'pnl': re_pnl,
            'mfe': re_mfe,
            'reason': re_exit_reason,
            'original_sl_bar': exit_bar,
        })

    return reentry_trades


def hyperopt_reentry(df, all_sl_sweeps):
    """Grid search for best re-entry parameters after SL sweep."""
    print(f"\n{'═'*80}")
    print(f"  HYPEROPT: RE-ENTRY AFTER SL SWEEP ({len(all_sl_sweeps)} sweep events)")
    print(f"{'═'*80}")

    # Parameter grid
    reentry_delays = [2, 3, 5, 7, 10]
    reentry_sl_mults = [1.0, 1.5, 2.0, 2.5, 3.0]
    reentry_tp_mults = [2.0, 3.0, 4.0, 5.0]
    reentry_max_holds = [15, 20, 30]

    best_pnl_per_day = -999
    best_params = None
    all_combos = list(product(reentry_delays, reentry_sl_mults, reentry_tp_mults, reentry_max_holds))
    print(f"  Testing {len(all_combos)} parameter combinations...", flush=True)

    results_grid = []
    for delay, sl_m, tp_m, max_h in all_combos:
        trades = simulate_reentry(df, all_sl_sweeps, delay, sl_m, tp_m, max_h)
        if len(trades) < 30:
            continue
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        total = sum(pnls)
        n_days = max(len(set(t['date'] for t in trades)), 1)
        pnl_per_day = total / n_days
        wr = len(wins) / len(pnls) * 100
        gross_win = sum(wins) if wins else 0
        gross_loss = abs(sum(p for p in pnls if p <= 0)) if any(p <= 0 for p in pnls) else 0.001
        pf = gross_win / gross_loss

        results_grid.append({
            'delay': delay, 'sl_m': sl_m, 'tp_m': tp_m, 'max_h': max_h,
            'n': len(trades), 'wr': wr, 'pf': pf, 'pnl': total, 'pnl_per_day': pnl_per_day,
        })

        if pnl_per_day > best_pnl_per_day:
            best_pnl_per_day = pnl_per_day
            best_params = (delay, sl_m, tp_m, max_h)

    # Sort by PnL/day and show top 20
    results_grid.sort(key=lambda x: x['pnl_per_day'], reverse=True)
    print(f"\n  TOP 20 RE-ENTRY CONFIGS:")
    print(f"  {'Delay':>5} {'SL_m':>5} {'TP_m':>5} {'MaxH':>5} {'Trades':>7} {'WR%':>6} {'PF':>6} {'PnL/d':>7}")
    print(f"  {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")
    for r in results_grid[:20]:
        print(f"  {r['delay']:>5} {r['sl_m']:>5.1f} {r['tp_m']:>5.1f} {r['max_h']:>5} "
              f"{r['n']:>7} {r['wr']:>5.1f}% {r['pf']:>5.2f} {r['pnl_per_day']:>+6.2f}")

    return results_grid, best_params


def hyperopt_full_strategy(df, signals, target_capture_pct=0.50):
    """Hyperopt the FULL strategy (SL + trail + re-entry) targeting 50% opportunity capture.

    'Opportunity' = price moves >= 3 pts in signal direction within 30 bars.
    Target: capture at least 50% of these moves profitably.
    """
    print(f"\n{'═'*80}")
    print(f"  HYPEROPT: FULL STRATEGY (target {target_capture_pct*100:.0f}% opportunity capture)")
    print(f"{'═'*80}")

    # First: identify all "real opportunities" — moves of >= 3 pts in direction
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    mins_arr = df['mins'].values

    opportunities = []
    for sig in signals:
        i = sig['idx']
        direction = sig['direction']
        session = sig['session']
        cutoff = AM_CUTOFF if session == 'AM' else PM_CUTOFF
        entry = closes[i]

        max_move = 0.0
        end_bar = min(i + 30, len(closes))
        for j in range(i + 1, end_bar):
            if mins_arr[j] >= cutoff:
                break
            if direction == 1:
                move = highs[j] - entry
            else:
                move = entry - lows[j]
            max_move = max(max_move, move)

        opportunities.append({
            'idx': i,
            'direction': direction,
            'session': session,
            'atr': sig['atr'],
            'max_move': max_move,
            'is_real_opportunity': max_move >= 3.0,  # >= 3 pts move available
        })

    n_opportunities = sum(1 for o in opportunities if o['is_real_opportunity'])
    n_total = len(opportunities)
    print(f"  Total signals: {n_total}")
    print(f"  Real opportunities (MFE >= 3pts in 30 bars): {n_opportunities} ({n_opportunities/n_total*100:.1f}%)")
    print(f"  Target: capture {int(n_opportunities * target_capture_pct)} of these profitably")

    # Parameter grid for full strategy
    sl_mults = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    trail_pts_list = [2.0, 3.0, 4.0, 5.0]
    trail_mults = [0.8, 1.0, 1.5, 2.0]
    max_holds = [15, 20, 30, 45]
    be_triggers = [None, 2.0, 3.0]
    tp_mults = [None, 3.0, 4.0]

    print(f"  Grid: {len(sl_mults)}×{len(trail_pts_list)}×{len(trail_mults)}×{len(max_holds)}×{len(be_triggers)}×{len(tp_mults)} = {len(sl_mults)*len(trail_pts_list)*len(trail_mults)*len(max_holds)*len(be_triggers)*len(tp_mults)} combos")
    print(f"  Running...", flush=True)

    best_score = -999
    best_config = None
    results = []
    t0 = time.time()

    for sl_m, trail_p, trail_m, max_h, be_t, tp_m in product(
        sl_mults, trail_pts_list, trail_mults, max_holds, be_triggers, tp_mults
    ):
        # Skip illogical combinations
        if trail_p and tp_m and tp_m <= trail_p * 0.5:
            continue
        if be_t and be_t >= trail_p:
            continue

        trades = simulate_with_post_sl_tracking(
            df, signals, sl_m, max_h, trail_p, trail_m,
            be_trigger=be_t, tp_mult=tp_m, track_bars_after_sl=0
        )

        if len(trades) < 50:
            continue

        pnls = [t['pnl'] for t in trades]
        wins = [t for t in trades if t['pnl'] > 0]
        total_pnl = sum(pnls)
        n_days = max(len(set(t['date'] for t in trades)), 1)
        pnl_per_day = total_pnl / n_days
        wr = len(wins) / len(trades) * 100

        # Calculate opportunity capture rate
        captured = 0
        for t, opp in zip(trades, opportunities):
            if opp['is_real_opportunity'] and t['pnl'] > 0:
                captured += 1

        capture_rate = captured / max(n_opportunities, 1)

        gross_win = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
        pf = gross_win / max(gross_loss, 0.001)

        # Score: PnL/day * min(capture_rate / target, 1.0)
        # Penalize if capture_rate < target
        capture_score = min(capture_rate / target_capture_pct, 1.0)
        score = pnl_per_day * capture_score

        if score > best_score and pf > 1.0 and capture_rate >= target_capture_pct * 0.7:
            best_score = score
            best_config = {
                'sl_m': sl_m, 'trail_p': trail_p, 'trail_m': trail_m,
                'max_h': max_h, 'be_t': be_t, 'tp_m': tp_m
            }

        if pf > 1.0 and len(trades) >= 100:
            results.append({
                'sl_m': sl_m, 'trail_p': trail_p, 'trail_m': trail_m,
                'max_h': max_h, 'be_t': be_t, 'tp_m': tp_m,
                'n': len(trades), 'wr': wr, 'pf': pf,
                'pnl_per_day': pnl_per_day, 'capture_rate': capture_rate,
                'score': score,
            })

    elapsed = time.time() - t0
    print(f"  Hyperopt done in {elapsed:.1f}s", flush=True)

    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n  TOP 15 CONFIGS (PF>1.0, min 100 trades):")
    print(f"  {'SL':>4} {'Trail':>5} {'TrM':>4} {'MaxH':>4} {'BE':>4} {'TP':>4} "
          f"{'N':>5} {'WR%':>6} {'PF':>5} {'P/D':>6} {'Cap%':>5} {'Score':>6}")
    print(f"  {'-'*4} {'-'*5} {'-'*4} {'-'*4} {'-'*4} {'-'*4} "
          f"{'-'*5} {'-'*6} {'-'*5} {'-'*6} {'-'*5} {'-'*6}")
    for r in results[:15]:
        be_s = f"{r['be_t']:.0f}" if r['be_t'] else '-'
        tp_s = f"{r['tp_m']:.0f}" if r['tp_m'] else '-'
        print(f"  {r['sl_m']:>4.1f} {r['trail_p']:>5.1f} {r['trail_m']:>4.1f} {r['max_h']:>4} {be_s:>4} {tp_s:>4} "
              f"{r['n']:>5} {r['wr']:>5.1f}% {r['pf']:>4.2f} {r['pnl_per_day']:>+5.2f} {r['capture_rate']*100:>4.0f}% {r['score']:>+5.2f}")

    if best_config:
        print(f"\n  BEST CONFIG: SL={best_config['sl_m']}x, Trail@{best_config['trail_p']}pts/{best_config['trail_m']}x, "
              f"MaxHold={best_config['max_h']}, BE={best_config['be_t']}, TP={best_config['tp_m']}")

    return results, best_config


def main():
    t0 = time.time()
    df = load_1m_data()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Classify SL hits as SWEEP vs REVERSAL
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  PHASE 1: SL SWEEP vs REVERSAL CLASSIFICATION")
    print(f"{'═'*80}")

    all_sl_sweeps = []
    all_sl_reversals = []
    all_signals_combined = []

    for strat_name, session_filter, direction_filter in STRATEGIES:
        combo_label = f"{strat_name} {session_filter}/{direction_filter}"
        print(f"\n  [{combo_label}]", flush=True)

        signals = collect_raw_signals(df, strat_name, session_filter, direction_filter)
        if len(signals) < 20:
            print(f"    Only {len(signals)} signals, skipping")
            continue

        all_signals_combined.extend(signals)

        # Use the strategy's default exit preset
        strategy = get_strategy(strat_name)
        preset = EXIT_PRESETS[strategy.exit_type]
        sl_mult = preset['sl_mult']
        trail_pts = preset['trail_pts']
        trail_mult = preset['trail_mult']
        max_hold = preset['max_hold_am']  # use AM (longer)
        be_trigger = preset.get('be_trigger')
        tp_mult = preset.get('tp_mult')

        trades = simulate_with_post_sl_tracking(
            df, signals, sl_mult, max_hold, trail_pts, trail_mult,
            be_trigger=be_trigger, tp_mult=tp_mult, track_bars_after_sl=30
        )

        sl_trades = [t for t in trades if t['reason'] == 'SL']
        sweeps = [t for t in sl_trades if t['is_sweep']]
        reversals = [t for t in sl_trades if not t['is_sweep']]

        all_sl_sweeps.extend(sweeps)
        all_sl_reversals.extend(reversals)

        n_sl = len(sl_trades)
        n_sweeps = len(sweeps)
        n_rev = len(reversals)

        print(f"    Signals: {len(signals)} | Trades: {len(trades)} | SL hits: {n_sl}")
        if n_sl > 0:
            print(f"    SWEEPS: {n_sweeps} ({n_sweeps/n_sl*100:.1f}%) — price returned after SL")
            print(f"    REVERSALS: {n_rev} ({n_rev/n_sl*100:.1f}%) — price kept going against")

            if sweeps:
                avg_post_mfe = np.mean([t['post_sl_mfe'] for t in sweeps])
                avg_recovery_bars = np.mean([t['post_sl_bars_to_recover'] for t in sweeps
                                            if t['post_sl_bars_to_recover'] is not None])
                n_recovered = sum(1 for t in sweeps if t['post_sl_bars_to_recover'] is not None)
                print(f"    Sweep avg post-SL MFE: {avg_post_mfe:.2f} pts")
                print(f"    Recovered to entry: {n_recovered}/{n_sweeps} ({n_recovered/n_sweeps*100:.1f}%)")
                if not np.isnan(avg_recovery_bars):
                    print(f"    Avg bars to recover: {avg_recovery_bars:.1f}")

            if reversals:
                avg_post_mae = np.mean([t['post_sl_mae'] for t in reversals])
                print(f"    Reversal avg post-SL adverse move: {avg_post_mae:.2f} pts (SL was correct)")

    # Aggregate stats
    total_sl = len(all_sl_sweeps) + len(all_sl_reversals)
    print(f"\n{'─'*80}")
    print(f"  AGGREGATE: {total_sl} total SL hits")
    print(f"    SWEEPS:    {len(all_sl_sweeps)} ({len(all_sl_sweeps)/total_sl*100:.1f}%)")
    print(f"    REVERSALS: {len(all_sl_reversals)} ({len(all_sl_reversals)/total_sl*100:.1f}%)")

    if all_sl_sweeps:
        print(f"\n  SWEEP CHARACTERISTICS:")
        mfes = [t['post_sl_mfe'] for t in all_sl_sweeps]
        print(f"    Post-SL MFE: P25={np.percentile(mfes,25):.2f}, P50={np.percentile(mfes,50):.2f}, P75={np.percentile(mfes,75):.2f}")
        recoveries = [t['post_sl_bars_to_recover'] for t in all_sl_sweeps if t['post_sl_bars_to_recover'] is not None]
        if recoveries:
            print(f"    Bars to recover: P25={np.percentile(recoveries,25):.0f}, P50={np.percentile(recoveries,50):.0f}, P75={np.percentile(recoveries,75):.0f}")
        n_rec = len(recoveries)
        print(f"    Recovered to entry: {n_rec}/{len(all_sl_sweeps)} ({n_rec/len(all_sl_sweeps)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: HYPEROPT RE-ENTRY AFTER SWEEP
    # ══════════════════════════════════════════════════════════════════════════
    if len(all_sl_sweeps) >= 50:
        reentry_results, best_reentry = hyperopt_reentry(df, all_sl_sweeps)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: HYPEROPT FULL STRATEGY (target 50% opportunity capture)
    # ══════════════════════════════════════════════════════════════════════════
    # Use the top 3 strategies with most signals
    print(f"\n\n{'═'*80}")
    print(f"  PHASE 3: FULL STRATEGY HYPEROPT PER COMBO")
    print(f"{'═'*80}")

    for strat_name, session_filter, direction_filter in STRATEGIES[:4]:
        combo_label = f"{strat_name} {session_filter}/{direction_filter}"
        signals = collect_raw_signals(df, strat_name, session_filter, direction_filter)
        if len(signals) < 100:
            print(f"\n  [{combo_label}] — only {len(signals)} signals, need 100+. Skipping.")
            continue

        print(f"\n  [{combo_label}] — {len(signals)} signals")
        hyperopt_full_strategy(df, signals, target_capture_pct=0.50)

    elapsed = time.time() - t0
    print(f"\n[DONE] Total: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
