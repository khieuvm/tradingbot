"""Sweep-Confirm Strategy: detect SL sweep → wait for confirmation → re-enter.

New approach:
- Instead of predicting direction at entry, WAIT for the market to sweep stops
- Sweep = false breakout in wrong direction → then price reverses
- Confirmation = price crosses back above/below entry level after sweep

This captures 50% of moves because:
1. Many real moves start with a stop sweep (market makers hunt stops first)
2. The sweep itself IS the confirmation that the move is real
3. We enter AFTER the noise, with better risk/reward

Also tests: wider SL from start (avoiding sweep altogether) on recent data.

Usage:
    python -m research.sweep_confirm_strategy
"""
import sys
import os
import time
from datetime import date, timedelta
from itertools import product

import numpy as np
import pandas as pd
import pandas_ta as ta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from strategies.base import EXIT_PRESETS, COST, AM_CUTOFF, PM_CUTOFF

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_1m_data(days=0):
    parquet_file = os.path.join(DATA_DIR, "data", "vn30f1m_1m.parquet")
    print(f"[DATA] Loading 1m...", flush=True)
    df = pd.read_parquet(parquet_file)
    df['time'] = pd.to_datetime(df['time'])
    df.columns = [c.lower() for c in df.columns]
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute

    if days > 0:
        cutoff = date.today() - timedelta(days=days)
        df = df[df['date'] >= cutoff]

    am_mask = (df['mins'] >= 540) & (df['mins'] <= 690)
    pm_mask = (df['mins'] >= 780) & (df['mins'] <= 870)
    df = df[am_mask | pm_mask].reset_index(drop=True)
    df['session'] = 'AM'
    df.loc[df['mins'] >= 780, 'session'] = 'PM'

    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema8'] = ta.ema(df['close'], length=8)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema50'] = ta.ema(df['close'], length=50)

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

    df['range'] = df['high'] - df['low']
    n_days = df['date'].nunique()
    print(f"[DATA] {len(df)} bars, {n_days} days ({df['date'].iloc[0]} to {df['date'].iloc[-1]})", flush=True)
    return df


def detect_sweep_setups(df, lookback=10, sweep_atr_mult=1.0, min_adx=20):
    """Detect potential sweep setups: recent swing high/low broken then reversed.

    Logic:
    1. Find recent swing low (lowest low in lookback bars)
    2. Current bar breaks below it (potential sweep)
    3. Current bar CLOSES above it (rejection / sweep confirmation)
    → LONG signal (stops were hunted below, now reversing up)

    Mirror for SHORT.
    """
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values
    atr = df['atr'].values
    adx = df['adx'].values
    mins_arr = df['mins'].values
    sessions = df['session'].values

    signals = []

    for i in range(lookback + 2, len(df) - 2):
        if pd.isna(atr[i]) or atr[i] < 0.5:
            continue
        if pd.isna(adx[i]) or adx[i] < min_adx:
            continue

        session = sessions[i]
        if session not in ('AM', 'PM'):
            continue
        mins = mins_arr[i]
        if session == 'AM' and not (555 <= mins <= 645):
            continue
        if session == 'PM' and not (795 <= mins <= 855):
            continue

        # Recent swing low and swing high in lookback
        window_lows = lows[i - lookback:i]
        window_highs = highs[i - lookback:i]
        swing_low = np.min(window_lows)
        swing_high = np.max(window_highs)

        # LONG SWEEP: bar penetrates below swing low then closes above
        if lows[i] < swing_low and closes[i] > swing_low:
            penetration = swing_low - lows[i]
            if penetration >= sweep_atr_mult * atr[i] * 0.3:  # minimum penetration
                if closes[i] > opens[i]:  # bullish close (confirmation)
                    signals.append({
                        'idx': i, 'direction': 1, 'session': session,
                        'atr': atr[i], 'entry_price': closes[i],
                        'date': df['date'].iloc[i], 'mins': mins,
                        'penetration': penetration, 'type': 'sweep_long'
                    })

        # SHORT SWEEP: bar penetrates above swing high then closes below
        elif highs[i] > swing_high and closes[i] < swing_high:
            penetration = highs[i] - swing_high
            if penetration >= sweep_atr_mult * atr[i] * 0.3:
                if closes[i] < opens[i]:  # bearish close
                    signals.append({
                        'idx': i, 'direction': -1, 'session': session,
                        'atr': atr[i], 'entry_price': closes[i],
                        'date': df['date'].iloc[i], 'mins': mins,
                        'penetration': penetration, 'type': 'sweep_short'
                    })

    return signals


def detect_strategy_sweep_reentry(df, strategy_name, session_filter, direction_filter,
                                   initial_sl_mult=2.0, confirm_bars=3, confirm_threshold=0.5):
    """Two-phase approach:
    Phase 1: Strategy fires signal → set wide initial SL
    Phase 2: If SL hit within confirm_bars → mark as potential sweep
    Phase 3: Wait confirm_bars more → if price returns past entry → RE-ENTER

    This captures the 72.6% of sweeps.
    """
    from strategies.momentum_trend import MomentumTrendStrategy
    from strategies.macd_cross import MACDCrossStrategy
    from strategies.heikin_ashi import HeikinAshiStrategy
    from strategies.fibonacci import FibonacciStrategy

    strats = {
        'momentum_trend': MomentumTrendStrategy,
        'macd_cross': MACDCrossStrategy,
        'heikin_ashi': HeikinAshiStrategy,
        'fibonacci': FibonacciStrategy,
    }

    strategy = strats[strategy_name]()
    strategy.prepare(df)

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    mins_arr = df['mins'].values
    sessions_arr = df['session'].values

    signals = []
    last_idx = -10

    for i in range(60, len(df) - 2):
        if i - last_idx < 5:
            continue
        session = sessions_arr[i]
        if session not in ('AM', 'PM'):
            continue
        if session_filter != 'ALL' and session != session_filter:
            continue
        mins = mins_arr[i]
        if session == 'AM' and not (555 <= mins <= 645):
            continue
        if session == 'PM' and not (795 <= mins <= 855):
            continue
        atr_val = df['atr'].iloc[i]
        if pd.isna(atr_val) or atr_val < 0.5:
            continue

        direction = strategy.detect(df, i)
        if direction == 0:
            continue
        if direction_filter == 'BUY' and direction != 1:
            continue
        if direction_filter == 'SELL' and direction != -1:
            continue

        entry = closes[i]
        sl_level = entry - direction * initial_sl_mult * atr_val
        cutoff = AM_CUTOFF if session == 'AM' else PM_CUTOFF

        # Check if SL would be hit within confirm_bars
        sl_hit_bar = None
        for j in range(i + 1, min(i + confirm_bars + 1, len(df))):
            if mins_arr[j] >= cutoff:
                break
            if direction == 1 and lows[j] <= sl_level:
                sl_hit_bar = j
                break
            if direction == -1 and highs[j] >= sl_level:
                sl_hit_bar = j
                break

        if sl_hit_bar is None:
            # No sweep — this is a normal signal, trade as usual
            signals.append({
                'idx': i, 'direction': direction, 'session': session,
                'atr': atr_val, 'entry_price': entry,
                'date': df['date'].iloc[i], 'mins': mins,
                'type': 'direct',  # no sweep occurred
            })
        else:
            # SL was hit — wait for price to return (sweep confirmation)
            # Look for price crossing back in direction within confirm_bars after SL hit
            for k in range(sl_hit_bar + 1, min(sl_hit_bar + confirm_bars + 1, len(df))):
                if mins_arr[k] >= cutoff:
                    break
                # Price returned past entry + threshold
                if direction == 1 and closes[k] >= entry + confirm_threshold * atr_val:
                    signals.append({
                        'idx': k, 'direction': direction, 'session': session,
                        'atr': atr_val, 'entry_price': closes[k],
                        'date': df['date'].iloc[k], 'mins': mins_arr[k],
                        'type': 'sweep_reentry',
                    })
                    break
                if direction == -1 and closes[k] <= entry - confirm_threshold * atr_val:
                    signals.append({
                        'idx': k, 'direction': direction, 'session': session,
                        'atr': atr_val, 'entry_price': closes[k],
                        'date': df['date'].iloc[k], 'mins': mins_arr[k],
                        'type': 'sweep_reentry',
                    })
                    break

        last_idx = i

    return signals


def simulate_trades(df, signals, sl_mult, trail_pts, trail_mult, max_hold, be_trigger=None, tp_mult=None):
    """Simulate trades from signals."""
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    mins_arr = df['mins'].values

    trades = []
    for sig in signals:
        i = sig['idx']
        direction = sig['direction']
        atr = sig['atr']
        session = sig['session']
        entry = sig['entry_price']
        cutoff = AM_CUTOFF if session == 'AM' else PM_CUTOFF

        sl = entry - direction * sl_mult * atr
        tp = entry + direction * tp_mult * atr if tp_mult else None
        best_price = entry
        mfe = 0.0
        trail_active = False
        be_active = False
        exit_price = None
        exit_reason = None

        end_idx = min(i + max_hold + 1, len(closes))
        for j in range(i + 1, end_idx):
            bar_h = highs[j]
            bar_l = lows[j]

            if mins_arr[j] >= cutoff:
                exit_price = closes[j]
                exit_reason = 'SESSION'
                break

            if direction == 1:
                cur_mfe = bar_h - entry
                if bar_h > best_price: best_price = bar_h
            else:
                cur_mfe = entry - bar_l
                if bar_l < best_price: best_price = bar_l
            mfe = max(mfe, cur_mfe)

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
                exit_price = sl
                exit_reason = 'TRAIL' if trail_active else ('BE' if be_active else 'SL')
                break
            if direction == -1 and bar_h >= sl:
                exit_price = sl
                exit_reason = 'TRAIL' if trail_active else ('BE' if be_active else 'SL')
                break

            if tp:
                if direction == 1 and bar_h >= tp:
                    exit_price = tp; exit_reason = 'TP'; break
                if direction == -1 and bar_l <= tp:
                    exit_price = tp; exit_reason = 'TP'; break

        if exit_price is None:
            exit_price = closes[min(i + max_hold, len(closes) - 1)]
            exit_reason = 'MAX_HOLD'

        pnl = direction * (exit_price - entry) - COST
        trades.append({
            'pnl': pnl, 'mfe': mfe, 'reason': exit_reason,
            'date': sig['date'], 'session': session, 'type': sig.get('type', 'unknown'),
        })

    return trades


def compute_metrics(trades):
    if not trades:
        return None
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n = len(trades)
    total = sum(pnls)
    n_days = max(len(set(t['date'] for t in trades)), 1)
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    return {
        'n': n, 'wr': len(wins)/n*100, 'pf': gross_win/gross_loss,
        'pnl': total, 'pnl_per_day': total/n_days,
        'avg_mfe': np.mean([t['mfe'] for t in trades]),
    }


def main():
    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 1: Pure sweep detection strategy (no underlying strategy needed)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"{'═'*80}")
    print(f"  TEST 1: PURE SWEEP DETECTION (swing break + rejection)")
    print(f"{'═'*80}")

    for days_label, days in [("FULL (682d)", 0), ("6 MONTHS", 180), ("3 MONTHS", 90), ("1 MONTH", 30)]:
        df = load_1m_data(days=days)
        n_days = df['date'].nunique()

        # Grid over sweep params
        best_score = -999
        best_result = None

        for lookback in [5, 8, 10, 15]:
            for min_adx in [15, 20, 25]:
                signals = detect_sweep_setups(df, lookback=lookback, min_adx=min_adx)
                if len(signals) < 30:
                    continue

                # Test with multiple exit configs
                for sl_m, trail_p, trail_m, max_h, tp_m in [
                    (1.5, 3.0, 1.0, 20, 3.0),
                    (2.0, 4.0, 1.5, 25, 4.0),
                    (2.5, 5.0, 2.0, 30, None),
                    (1.0, 2.0, 0.8, 15, 2.5),
                    (1.5, 3.0, 1.0, 20, None),
                    (2.0, 3.0, 1.0, 30, 4.0),
                    (3.0, 4.0, 1.5, 30, 5.0),
                    (2.0, 5.0, 1.5, 45, None),
                ]:
                    trades = simulate_trades(df, signals, sl_m, trail_p, trail_m, max_h, tp_mult=tp_m)
                    m = compute_metrics(trades)
                    if m and m['n'] >= 30 and m['pf'] > 1.0:
                        score = m['pnl_per_day']
                        if score > best_score:
                            best_score = score
                            best_result = {**m, 'lookback': lookback, 'min_adx': min_adx,
                                          'sl_m': sl_m, 'trail_p': trail_p, 'trail_m': trail_m,
                                          'max_h': max_h, 'tp_m': tp_m, 'n_signals': len(signals)}

        print(f"\n  [{days_label}] ({n_days} days)")
        if best_result:
            r = best_result
            print(f"    BEST: lb={r['lookback']}, ADX>{r['min_adx']}, SL={r['sl_m']}x, Trail@{r['trail_p']}/{r['trail_m']}x, MaxH={r['max_h']}, TP={r['tp_m']}")
            print(f"    {r['n_signals']} signals → {r['n']} trades | WR {r['wr']:.1f}% | PF {r['pf']:.2f} | +{r['pnl_per_day']:.2f}/d")
        else:
            print(f"    No profitable config found (PF > 1.0 with 30+ trades)")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 2: Strategy + Sweep Re-entry (combined approach)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'═'*80}")
    print(f"  TEST 2: STRATEGY + SWEEP RE-ENTRY COMBINED")
    print(f"{'═'*80}")

    df_full = load_1m_data(days=0)
    df_6m = load_1m_data(days=180)

    combos = [
        ('momentum_trend', 'PM', 'SELL'),
        ('momentum_trend', 'ALL', 'SELL'),
        ('macd_cross', 'PM', 'BUY'),
        ('heikin_ashi', 'AM', 'SELL'),
    ]

    for strat, sess, dirn in combos:
        combo_label = f"{strat} {sess}/{dirn}"

        for df_test, period_label in [(df_full, "FULL"), (df_6m, "6M")]:
            n_days = df_test['date'].nunique()

            best_score = -999
            best = None

            # Grid over sweep-reentry params and exit params
            for initial_sl in [1.5, 2.0, 2.5]:
                for confirm_bars in [3, 5, 7]:
                    for confirm_thresh in [0.0, 0.3, 0.5]:
                        signals = detect_strategy_sweep_reentry(
                            df_test, strat, sess, dirn,
                            initial_sl_mult=initial_sl,
                            confirm_bars=confirm_bars,
                            confirm_threshold=confirm_thresh
                        )
                        if len(signals) < 30:
                            continue

                        # Split by type
                        direct_signals = [s for s in signals if s['type'] == 'direct']
                        reentry_signals = [s for s in signals if s['type'] == 'sweep_reentry']

                        for sl_m, trail_p, trail_m, max_h, tp_m in [
                            (2.0, 3.0, 1.0, 20, 3.0),
                            (2.5, 4.0, 1.5, 25, 4.0),
                            (3.0, 5.0, 2.0, 30, None),
                            (2.0, 4.0, 1.5, 30, None),
                            (1.5, 3.0, 1.0, 20, None),
                            (3.0, 4.0, 1.5, 30, 5.0),
                        ]:
                            trades = simulate_trades(df_test, signals, sl_m, trail_p, trail_m, max_h, tp_mult=tp_m)
                            m = compute_metrics(trades)
                            if m and m['n'] >= 30 and m['pf'] > 1.0:
                                if m['pnl_per_day'] > best_score:
                                    best_score = m['pnl_per_day']
                                    best = {**m,
                                           'initial_sl': initial_sl, 'confirm_bars': confirm_bars,
                                           'confirm_thresh': confirm_thresh,
                                           'sl_m': sl_m, 'trail_p': trail_p, 'trail_m': trail_m,
                                           'max_h': max_h, 'tp_m': tp_m,
                                           'n_direct': len(direct_signals),
                                           'n_reentry': len(reentry_signals)}

            print(f"\n  [{combo_label}] {period_label} ({n_days}d)")
            if best:
                r = best
                print(f"    Config: init_SL={r['initial_sl']}x, confirm={r['confirm_bars']}bars/{r['confirm_thresh']}x")
                print(f"    Exit: SL={r['sl_m']}x, Trail@{r['trail_p']}/{r['trail_m']}x, MaxH={r['max_h']}, TP={r['tp_m']}")
                print(f"    Direct: {r['n_direct']} | Re-entry: {r['n_reentry']} | Total trades: {r['n']}")
                print(f"    WR {r['wr']:.1f}% | PF {r['pf']:.2f} | +{r['pnl_per_day']:.2f}/d | Avg MFE {r['avg_mfe']:.2f}")
            else:
                print(f"    No profitable config (PF>1.0, 30+ trades)")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 3: WIDE SL approach (avoid sweep entirely)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'═'*80}")
    print(f"  TEST 3: WIDE SL (avoid sweep, let trades breathe)")
    print(f"{'═'*80}")
    print(f"  Since 72.6% of SL hits are sweeps, what if we just use 4-5x ATR SL?")

    from strategies.momentum_trend import MomentumTrendStrategy
    from strategies.macd_cross import MACDCrossStrategy

    for df_test, period_label in [(df_full, "FULL"), (df_6m, "6M")]:
        n_days = df_test['date'].nunique()
        print(f"\n  === {period_label} ({n_days} days) ===")

        for strat_class, strat_name, sess, dirn in [
            (MomentumTrendStrategy, 'momentum_trend', 'PM', 'SELL'),
            (MomentumTrendStrategy, 'momentum_trend', 'ALL', 'SELL'),
            (MACDCrossStrategy, 'macd_cross', 'PM', 'BUY'),
        ]:
            strategy = strat_class()
            strategy.prepare(df_test)

            # Collect raw signals
            signals = []
            last_idx = -10
            sessions_arr = df_test['session'].values
            mins_arr = df_test['mins'].values

            for i in range(60, len(df_test) - 2):
                if i - last_idx < 5:
                    continue
                session = sessions_arr[i]
                if session not in ('AM', 'PM'):
                    continue
                if sess != 'ALL' and session != sess:
                    continue
                mins = mins_arr[i]
                if session == 'AM' and not (555 <= mins <= 645):
                    continue
                if session == 'PM' and not (795 <= mins <= 855):
                    continue
                atr = df_test['atr'].iloc[i]
                if pd.isna(atr) or atr < 0.5:
                    continue

                direction = strategy.detect(df_test, i)
                if direction == 0:
                    continue
                if dirn == 'BUY' and direction != 1:
                    continue
                if dirn == 'SELL' and direction != -1:
                    continue

                signals.append({
                    'idx': i, 'direction': direction, 'session': session,
                    'atr': atr, 'entry_price': df_test['close'].iloc[i],
                    'date': df_test['date'].iloc[i], 'mins': mins,
                })
                last_idx = i

            if len(signals) < 30:
                continue

            # Test wide SL configs
            print(f"\n  {strat_name} {sess}/{dirn} ({len(signals)} signals)")
            print(f"    {'SL':>4} {'Trail':>5} {'TrM':>4} {'MaxH':>4} {'TP':>4} {'N':>5} {'WR%':>6} {'PF':>5} {'P/D':>7} {'SL%':>5}")

            configs = [
                # Wide SL, aggressive trail
                (3.0, 3.0, 1.0, 30, 4.0),
                (4.0, 4.0, 1.5, 30, 5.0),
                (5.0, 5.0, 2.0, 45, None),
                (4.0, 3.0, 1.0, 30, None),
                # Very wide SL, tight trail
                (5.0, 3.0, 0.8, 20, None),
                (4.0, 2.0, 0.8, 20, 3.0),
                # Medium SL, no trail (TP only)
                (2.0, 99.0, 1.0, 30, 3.0),
                (3.0, 99.0, 1.0, 30, 4.0),
                (4.0, 99.0, 1.0, 30, 5.0),
                # Current config for reference
                (2.0, 5.0, 2.0, 30, None),
            ]

            for sl_m, trail_p, trail_m, max_h, tp_m in configs:
                trades = simulate_trades(df_test, signals, sl_m, trail_p, trail_m, max_h, tp_mult=tp_m)
                m = compute_metrics(trades)
                if m and m['n'] >= 20:
                    sl_pct = len([t for t in trades if t['reason'] == 'SL']) / m['n'] * 100
                    tp_s = f"{tp_m:.0f}" if tp_m else '-'
                    marker = " ←" if m['pf'] > 1.0 else ""
                    print(f"    {sl_m:>4.1f} {trail_p:>5.1f} {trail_m:>4.1f} {max_h:>4} {tp_s:>4} "
                          f"{m['n']:>5} {m['wr']:>5.1f}% {m['pf']:>4.2f} {m['pnl_per_day']:>+6.2f} {sl_pct:>4.0f}%{marker}")

    elapsed = time.time() - t0
    print(f"\n[DONE] {elapsed:.1f}s")


if __name__ == '__main__':
    main()
