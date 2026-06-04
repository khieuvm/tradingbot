"""
Multi-TF Backtest: CB + MI on 1m, 3m, 5m
Exact same logic as sim_day.py / research_strategy.py
Cost: 0.96 pts round-trip (slippage 0.5 + commission 0.46)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta

COST = 0.96


# ===================================================================
# DATA LOADING
# ===================================================================

def load_data(days=180, tf='5m'):
    from src.data_fetcher import DataFetcher
    fetcher = DataFetcher()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    df = fetcher.get_futures_ohlcv("VN30F1M", start, end, interval=tf)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    h = df['time'].dt.hour
    m = df['time'].dt.minute
    mins = h * 60 + m
    in_session = ((mins >= 9*60) & (mins < 11*60+30)) | ((mins >= 13*60) & (mins < 14*60+30))
    df = df[in_session].reset_index(drop=True)

    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    df['ema9'] = ta.ema(df['close'], length=9)

    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['mins_in_day'] = df['hour'] * 60 + df['minute']
    df['date'] = df['time'].dt.date
    df['session'] = np.where(df['mins_in_day'] < 12*60, 'AM', 'PM')
    df['range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['bar_dir'] = np.where(df['close'] > df['open'], 1, -1)
    df['vol_sma20'] = df['volume'].rolling(20).mean()

    return df


# ===================================================================
# COMPRESSION DETECTION (exact replica of research_strategy.py)
# ===================================================================

def detect_compression(df, n_bars=3, threshold=0.7):
    """Same logic as research_strategy.detect_compression()"""
    ranges = df['range'].values
    atr = df['atr'].values
    compressed = np.zeros(len(df), dtype=bool)
    for i in range(n_bars, len(df)):
        max_range = max(ranges[i - n_bars:i])
        if atr[i] > 0 and max_range < threshold * atr[i]:
            compressed[i] = True
    df['compressed'] = compressed
    return df


def dedup_signals(mask, min_bars=5):
    """Remove signals within min_bars of each other."""
    result = mask.copy()
    last_signal = -999
    for i in range(len(result)):
        if result.iloc[i]:
            if i - last_signal < min_bars:
                result.iloc[i] = False
            else:
                last_signal = i
    return result


# ===================================================================
# CB BACKTEST — exact same logic as sim_one_trade() in sim_day.py
# ===================================================================

def backtest_cb_full(df, signals_am, signals_pm,
                     # AM params
                     am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=24,
                     # PM params
                     pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=12,
                     # Adaptive (when ATR >= be_atr_min)
                     be_trigger=4.0, be_atr_min=3.5,
                     trail_tighten_at=8.0, trail_tighten_mult=1.2):
    """
    Simulate CB trades with full AM/PM split and adaptive exits.
    Direction determined from NEXT bar (same as original).
    """
    trades = []

    def sim_one(idx, sl_mult, trail_activate, trail_atr_mult, max_hold):
        row = df.loc[idx]
        atr = row['atr']
        if pd.isna(atr) or atr <= 0:
            return None

        # Direction from NEXT bar
        next_pos = df.index.get_loc(idx) + 1
        if next_pos >= len(df):
            return None
        next_bar = df.iloc[next_pos]
        if next_bar['date'] != row['date'] or next_bar['session'] != row['session']:
            return None

        direction = 1 if next_bar['close'] > row['close'] else -1
        entry_price = row['close']

        sl_price = entry_price - direction * sl_mult * atr
        best_price = entry_price
        trail_active = False
        be_applied = False
        use_be = (be_trigger > 0 and atr >= be_atr_min)
        use_tighten = (trail_tighten_at > 0 and atr >= be_atr_min)
        exit_price = None
        exit_reason = None
        bars_held = 0

        i_pos = df.index.get_loc(idx)
        for j_pos in range(i_pos + 1, min(i_pos + 1 + max_hold, len(df))):
            bar = df.iloc[j_pos]

            if bar['date'] != row['date'] or bar['session'] != row['session']:
                exit_price = df.iloc[j_pos - 1]['close']
                exit_reason = 'SESSION_CLOSE'
                break
            if (bar['session'] == 'PM' and bar['mins_in_day'] >= 14*60+25) or \
               (bar['session'] == 'AM' and bar['mins_in_day'] >= 11*60+25):
                exit_price = bar['close']
                exit_reason = 'SESSION_CLOSE'
                break

            bars_held += 1
            cur_trail = trail_atr_mult

            if direction == 1:
                if bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'BE' if be_applied else 'SL'
                    break
                if bar['high'] > best_price:
                    best_price = bar['high']
                mfe = best_price - entry_price
                if use_be and not be_applied and mfe >= be_trigger:
                    be_applied = True
                    sl_price = max(sl_price, entry_price)
                if mfe >= trail_activate and not trail_active:
                    trail_active = True
                if use_tighten and mfe >= trail_tighten_at:
                    cur_trail = trail_tighten_mult
                if trail_active:
                    sl_price = max(sl_price, best_price - cur_trail * atr)
                    if bar['low'] <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'TRAIL'
                        break
            else:
                if bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'BE' if be_applied else 'SL'
                    break
                if bar['low'] < best_price:
                    best_price = bar['low']
                mfe = entry_price - best_price
                if use_be and not be_applied and mfe >= be_trigger:
                    be_applied = True
                    sl_price = min(sl_price, entry_price)
                if mfe >= trail_activate and not trail_active:
                    trail_active = True
                if use_tighten and mfe >= trail_tighten_at:
                    cur_trail = trail_tighten_mult
                if trail_active:
                    sl_price = min(sl_price, best_price + cur_trail * atr)
                    if bar['high'] >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'TRAIL'
                        break

        if exit_price is None:
            exit_price = entry_price
            exit_reason = 'MAX_HOLD'

        mfe_val = (best_price - entry_price) if direction == 1 else (entry_price - best_price)
        pnl = direction * (exit_price - entry_price) - COST

        return {
            'pnl': pnl, 'mfe': mfe_val, 'exit_reason': exit_reason,
            'date': row['date'], 'time': row['time'].strftime('%H:%M'),
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry': entry_price, 'exit': exit_price, 'atr': atr,
            'bars_held': bars_held, 'session': row['session'],
        }

    # Run AM signals
    for idx in df.index[signals_am]:
        t = sim_one(idx, am_sl, am_trail_activate, am_trail_mult, am_max_hold)
        if t:
            trades.append(t)

    # Run PM signals
    for idx in df.index[signals_pm]:
        t = sim_one(idx, pm_sl, pm_trail_activate, pm_trail_mult, pm_max_hold)
        if t:
            trades.append(t)

    if not trades:
        return None

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf['pnl'] > 0]
    losses = tdf[tdf['pnl'] <= 0]
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 999
    return {
        'trades': len(tdf), 'win_rate': (tdf['pnl'] > 0).mean() * 100,
        'pf': pf, 'total_pnl': tdf['pnl'].sum(), 'avg_mfe': tdf['mfe'].mean(),
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        'exits': tdf['exit_reason'].value_counts().to_dict(),
        'df': tdf,
    }


def run_cb(df, n_bars=3, threshold=0.7, atr_max=4.5, atr_min=2.5, rsi_max=70,
           dedup_bars=5,
           am_start=9*60+15, am_end=10*60+45,
           pm_start=13*60+15, pm_end=14*60+15,
           **bt_kw):
    """Full pipeline: detect compression → build AM/PM masks → backtest."""
    df = detect_compression(df.copy(), n_bars=n_bars, threshold=threshold)

    filt = (df['atr'] <= atr_max) & (df['atr'] >= atr_min)
    if 'rsi14' in df.columns:
        filt = filt & (df['rsi14'] < rsi_max).fillna(True)

    maskAM = df['compressed'] & (df['mins_in_day'] >= am_start) & (df['mins_in_day'] <= am_end) & filt
    maskPM = df['compressed'] & (df['mins_in_day'] >= pm_start) & (df['mins_in_day'] <= pm_end) & filt

    maskAM = dedup_signals(maskAM.fillna(False), min_bars=dedup_bars)
    maskPM = dedup_signals(maskPM.fillna(False), min_bars=dedup_bars)

    return backtest_cb_full(df, maskAM, maskPM, **bt_kw)


# ===================================================================
# MI BACKTEST
# ===================================================================

def run_mi(df, range_mult=2.0, body_ratio=0.6, atr_min=2.0, atr_max=5.0,
           dedup_bars=6, sl_atr_mult=1.0, trail_mode='ema9',
           max_hold=10, be_trigger=2.0):
    """Momentum Ignition: big single bar, EMA9 trail."""
    trades = []

    sigs = pd.Series(0, index=df.index)
    for i in range(14, len(df)):
        row = df.iloc[i]
        atr = row['atr']
        if pd.isna(atr) or atr < atr_min or atr > atr_max:
            continue

        mins = row['mins_in_day']
        if not ((mins >= 9*60+10 and mins <= 11*60+10) or (mins >= 13*60+5 and mins <= 14*60+15)):
            continue

        bar_range = row['range']
        if bar_range < range_mult * atr:
            continue
        if row['body'] < body_ratio * bar_range:
            continue

        sigs.iloc[i] = 1 if row['close'] > row['open'] else -1

    buy_mask = dedup_signals(sigs == 1, min_bars=dedup_bars)
    sell_mask = dedup_signals(sigs == -1, min_bars=dedup_bars)
    sigs = pd.Series(0, index=df.index)
    sigs[buy_mask] = 1
    sigs[sell_mask] = -1

    for i in df.index[sigs != 0]:
        i_pos = df.index.get_loc(i)
        row = df.iloc[i_pos]
        direction = int(sigs.iloc[i_pos])
        atr = row['atr']
        entry_price = row['close']

        if pd.isna(atr) or atr <= 0:
            continue

        sl_price = entry_price - direction * sl_atr_mult * atr
        best_price = entry_price
        be_applied = False
        exit_price = None
        exit_reason = None
        bars_held = 0

        for j_pos in range(i_pos + 1, min(i_pos + 1 + max_hold, len(df))):
            bar = df.iloc[j_pos]
            if bar['date'] != row['date'] or bar['session'] != row['session']:
                exit_price = df.iloc[j_pos - 1]['close']
                exit_reason = 'SESSION_END'
                break
            if (bar['session'] == 'PM' and bar['mins_in_day'] >= 14*60+25) or \
               (bar['session'] == 'AM' and bar['mins_in_day'] >= 11*60+25):
                exit_price = bar['close']
                exit_reason = 'SESSION_END'
                break

            bars_held += 1
            if direction == 1:
                if bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'BE' if be_applied else 'SL'
                    break
                if bar['high'] > best_price:
                    best_price = bar['high']
                mfe = best_price - entry_price
                if be_trigger > 0 and not be_applied and mfe >= be_trigger:
                    be_applied = True
                    sl_price = max(sl_price, entry_price)
                if trail_mode == 'ema9' and be_applied and pd.notna(bar.get('ema9')):
                    ema_sl = bar['ema9'] - 0.3 * atr
                    sl_price = max(sl_price, ema_sl)
                    if bar['low'] <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'EMA_TRAIL'
                        break
            else:
                if bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'BE' if be_applied else 'SL'
                    break
                if bar['low'] < best_price:
                    best_price = bar['low']
                mfe = entry_price - best_price
                if be_trigger > 0 and not be_applied and mfe >= be_trigger:
                    be_applied = True
                    sl_price = min(sl_price, entry_price)
                if trail_mode == 'ema9' and be_applied and pd.notna(bar.get('ema9')):
                    ema_sl = bar['ema9'] + 0.3 * atr
                    sl_price = min(sl_price, ema_sl)
                    if bar['high'] >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'EMA_TRAIL'
                        break

        if exit_price is None:
            exit_price = entry_price
            exit_reason = 'MAX_HOLD'

        mfe_val = (best_price - entry_price) if direction == 1 else (entry_price - best_price)
        pnl = direction * (exit_price - entry_price) - COST

        trades.append({
            'pnl': pnl, 'mfe': mfe_val, 'exit_reason': exit_reason,
            'date': row['date'], 'direction': 'BUY' if direction == 1 else 'SELL',
            'entry': entry_price, 'exit': exit_price, 'atr': atr, 'bars_held': bars_held,
        })

    if not trades:
        return None
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf['pnl'] > 0]
    losses = tdf[tdf['pnl'] <= 0]
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 999
    return {
        'trades': len(tdf), 'win_rate': (tdf['pnl'] > 0).mean() * 100,
        'pf': pf, 'total_pnl': tdf['pnl'].sum(), 'avg_mfe': tdf['mfe'].mean(),
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        'exits': tdf['exit_reason'].value_counts().to_dict(),
    }


def print_result(label, r, n_days, min_trades=5):
    if r and r['trades'] >= min_trades:
        print(f'  {label:<60} {r["trades"]:>4} {r["win_rate"]:>5.1f}% {r["pf"]:>5.2f} '
              f'{r["total_pnl"]:>+8.1f} {r["total_pnl"]/n_days:>+6.2f}/d {r["avg_mfe"]:>5.1f}')
    elif r:
        print(f'  {label:<60} {r["trades"]:>4} (too few trades)')
    else:
        print(f'  {label:<60}   -- no trades --')


# ===================================================================
# MAIN
# ===================================================================

if __name__ == '__main__':
    print(f'COST = {COST} pts/trade (slippage 0.5 + commission 0.46)\n')

    header = f"  {'Config':<60} {'T':>4} {'WR%':>6} {'PF':>5} {'PnL':>8} {'P/D':>8} {'MFE':>5}"

    # =====================================================================
    # 5m — validated params (IS 180d)
    # =====================================================================
    print("Loading 5m 180-day data...")
    df5 = load_data(days=180, tf='5m')
    n5 = df5['date'].nunique()
    print(f"  5m: {len(df5)} bars, {n5} days\n")

    print('=' * 100)
    print(f'CB on 5m ({n5}d) — VALIDATED PARAMS (from proven_edges.md)')
    print('=' * 100)
    print(header)
    print('  ' + '-' * 94)

    # Exact params from proven_edges.md
    cb5_configs = [
        ('5m CB [PROVEN] 3bar/<0.7x ATR2.5-4.5 RSI<70 | AM:SL1.2/T@5/2x H24 | PM:SL1.0/T@4/1.5x H12',
         dict(n_bars=3, threshold=0.7, atr_min=2.5, atr_max=4.5, rsi_max=70, dedup_bars=5),
         dict(am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=24,
              pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=12,
              be_trigger=4.0, be_atr_min=3.5, trail_tighten_at=8.0, trail_tighten_mult=1.2)),
        ('5m CB 3bar/<0.7x ATR2.5-4.5 RSI<70 | AM:SL1.2/T@5/2x H24 | PM:SL1.0/T@4/1.5x H12 [no tighten]',
         dict(n_bars=3, threshold=0.7, atr_min=2.5, atr_max=4.5, rsi_max=70, dedup_bars=5),
         dict(am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=24,
              pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=12,
              be_trigger=4.0, be_atr_min=3.5, trail_tighten_at=0, trail_tighten_mult=1.2)),
        ('5m CB 3bar/<0.8x ATR2.5-4.5 RSI<70 | AM:SL1.2/T@5/2x H24 | PM:SL1.0/T@4/1.5x H12',
         dict(n_bars=3, threshold=0.8, atr_min=2.5, atr_max=4.5, rsi_max=70, dedup_bars=5),
         dict(am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=24,
              pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=12,
              be_trigger=4.0, be_atr_min=3.5, trail_tighten_at=8.0, trail_tighten_mult=1.2)),
    ]

    for label, sig_kw, bt_kw in cb5_configs:
        r = run_cb(df5, **sig_kw, **bt_kw)
        print_result(label, r, n5)

    # =====================================================================
    # 3m — adapt params
    # =====================================================================
    print("\nLoading 3m data...")
    df3 = load_data(days=180, tf='3m')
    n3 = df3['date'].nunique()
    print(f"  3m: {len(df3)} bars, {n3} days\n")

    print(f'\n{"="*100}')
    print(f'CB on 3m ({n3}d)')
    print('=' * 100)
    print(header)
    print('  ' + '-' * 94)

    # 3m params: 1 3m bar ≈ 0.6x ATR of 5m
    # Compression: 5 bars @ 3m = 15 min (same as 3 bars @ 5m = 15 min)
    # ATR thresholds scale down (1.5-3.0)
    # Trail activation in pts stays same (5 pts), max_hold scales up
    cb3_configs = [
        ('3m CB 5bar/<0.7x ATR1.5-3.0 | AM:SL1.2/T@5/2x H40 | PM:SL1.0/T@4/1.5x H20',
         dict(n_bars=5, threshold=0.7, atr_min=1.5, atr_max=3.0, rsi_max=70, dedup_bars=8),
         dict(am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=40,
              pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=20,
              be_trigger=4.0, be_atr_min=2.5, trail_tighten_at=8.0, trail_tighten_mult=1.2)),
        ('3m CB 4bar/<0.7x ATR1.5-3.0 | AM:SL1.2/T@5/2x H40 | PM:SL1.0/T@4/1.5x H20',
         dict(n_bars=4, threshold=0.7, atr_min=1.5, atr_max=3.0, rsi_max=70, dedup_bars=7),
         dict(am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=40,
              pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=20,
              be_trigger=4.0, be_atr_min=2.5, trail_tighten_at=8.0, trail_tighten_mult=1.2)),
        ('3m CB 3bar/<0.7x ATR1.5-3.0 | AM:SL1.2/T@5/2x H40 | PM:SL1.0/T@4/1.5x H20',
         dict(n_bars=3, threshold=0.7, atr_min=1.5, atr_max=3.0, rsi_max=70, dedup_bars=5),
         dict(am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=40,
              pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=20,
              be_trigger=4.0, be_atr_min=2.5, trail_tighten_at=8.0, trail_tighten_mult=1.2)),
        ('3m CB 5bar/<0.8x ATR1.5-3.5 | AM:SL1.2/T@5/2x H40 | PM:SL1.0/T@4/1.5x H20',
         dict(n_bars=5, threshold=0.8, atr_min=1.5, atr_max=3.5, rsi_max=70, dedup_bars=8),
         dict(am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=40,
              pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=20,
              be_trigger=4.0, be_atr_min=2.5, trail_tighten_at=8.0, trail_tighten_mult=1.2)),
        ('3m CB 5bar/<0.7x ATR1.2-2.5 | AM:SL1.5/T@4/2x H35 | PM:SL1.2/T@3/1.5x H18',
         dict(n_bars=5, threshold=0.7, atr_min=1.2, atr_max=2.5, rsi_max=70, dedup_bars=8),
         dict(am_sl=1.5, am_trail_activate=4.0, am_trail_mult=2.0, am_max_hold=35,
              pm_sl=1.2, pm_trail_activate=3.0, pm_trail_mult=1.5, pm_max_hold=18,
              be_trigger=3.0, be_atr_min=2.0, trail_tighten_at=7.0, trail_tighten_mult=1.2)),
    ]

    for label, sig_kw, bt_kw in cb3_configs:
        r = run_cb(df3, **sig_kw, **bt_kw)
        print_result(label, r, n3)

    # =====================================================================
    # 1m — adapt params
    # =====================================================================
    print("\nLoading 1m data...")
    df1 = load_data(days=180, tf='1m')
    n1 = df1['date'].nunique()
    print(f"  1m: {len(df1)} bars, {n1} days\n")

    print(f'\n{"="*100}')
    print(f'CB on 1m ({n1}d)')
    print('=' * 100)
    print(header)
    print('  ' + '-' * 94)

    # 1m params: ATR ≈ 0.3x of 5m
    # Compression: 15 bars @ 1m = 15 min (same as 3 bars @ 5m)
    # ATR thresholds: 0.7-1.5
    cb1_configs = [
        ('1m CB 15bar/<0.7x ATR0.7-1.5 | AM:SL1.2/T@5/2x H90 | PM:SL1.0/T@4/1.5x H45',
         dict(n_bars=15, threshold=0.7, atr_min=0.7, atr_max=1.5, rsi_max=70, dedup_bars=25),
         dict(am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=90,
              pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=45,
              be_trigger=4.0, be_atr_min=1.0, trail_tighten_at=8.0, trail_tighten_mult=1.2)),
        ('1m CB 10bar/<0.7x ATR0.7-1.5 | AM:SL1.2/T@5/2x H90 | PM:SL1.0/T@4/1.5x H45',
         dict(n_bars=10, threshold=0.7, atr_min=0.7, atr_max=1.5, rsi_max=70, dedup_bars=15),
         dict(am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=90,
              pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=45,
              be_trigger=4.0, be_atr_min=1.0, trail_tighten_at=8.0, trail_tighten_mult=1.2)),
        ('1m CB 15bar/<0.7x ATR0.5-1.2 | AM:SL1.2/T@4/2x H80 | PM:SL1.0/T@3/1.5x H40',
         dict(n_bars=15, threshold=0.7, atr_min=0.5, atr_max=1.2, rsi_max=70, dedup_bars=25),
         dict(am_sl=1.2, am_trail_activate=4.0, am_trail_mult=2.0, am_max_hold=80,
              pm_sl=1.0, pm_trail_activate=3.0, pm_trail_mult=1.5, pm_max_hold=40,
              be_trigger=3.0, be_atr_min=0.8, trail_tighten_at=7.0, trail_tighten_mult=1.2)),
        ('1m CB 20bar/<0.7x ATR0.7-1.5 | AM:SL1.2/T@5/2x H90 | PM:SL1.0/T@4/1.5x H45',
         dict(n_bars=20, threshold=0.7, atr_min=0.7, atr_max=1.5, rsi_max=70, dedup_bars=30),
         dict(am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=90,
              pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=45,
              be_trigger=4.0, be_atr_min=1.0, trail_tighten_at=8.0, trail_tighten_mult=1.2)),
    ]

    for label, sig_kw, bt_kw in cb1_configs:
        r = run_cb(df1, **sig_kw, **bt_kw)
        print_result(label, r, n1)

    # =====================================================================
    # MI on all TFs
    # =====================================================================
    for tf_name, df_tf, n_d in [('5m', df5, n5), ('3m', df3, n3), ('1m', df1, n1)]:
        print(f'\n{"="*100}')
        print(f'MI on {tf_name} ({n_d}d)')
        print('=' * 100)
        print(header)
        print('  ' + '-' * 94)

        if tf_name == '5m':
            mi_cfgs = [
                ('R>2.0x Body>60% ATR2-5 SL1.0 EMA9 H10 BE2', dict(range_mult=2.0, atr_min=2.0, atr_max=5.0, dedup_bars=6, sl_atr_mult=1.0, max_hold=10, be_trigger=2.0)),
                ('R>1.8x Body>60% ATR2-5 SL1.2 EMA9 H12 BE2', dict(range_mult=1.8, atr_min=2.0, atr_max=5.0, dedup_bars=6, sl_atr_mult=1.2, max_hold=12, be_trigger=2.0)),
                ('R>2.5x Body>60% ATR2-5 SL0.8 EMA9 H8 BE1.5', dict(range_mult=2.5, atr_min=2.0, atr_max=5.0, dedup_bars=6, sl_atr_mult=0.8, max_hold=8, be_trigger=1.5)),
            ]
        elif tf_name == '3m':
            mi_cfgs = [
                ('R>2.0x Body>60% ATR1.5-4 SL1.0 EMA9 H15 BE2', dict(range_mult=2.0, atr_min=1.5, atr_max=4.0, dedup_bars=8, sl_atr_mult=1.0, max_hold=15, be_trigger=2.0)),
                ('R>1.8x Body>60% ATR1.5-4 SL1.2 EMA9 H18 BE2', dict(range_mult=1.8, atr_min=1.5, atr_max=4.0, dedup_bars=8, sl_atr_mult=1.2, max_hold=18, be_trigger=2.0)),
                ('R>2.5x Body>60% ATR1.5-4 SL0.8 EMA9 H12 BE1.5', dict(range_mult=2.5, atr_min=1.5, atr_max=4.0, dedup_bars=8, sl_atr_mult=0.8, max_hold=12, be_trigger=1.5)),
            ]
        else:
            mi_cfgs = [
                ('R>2.5x Body>60% ATR0.7-2 SL1.0 EMA9 H20 BE1.5', dict(range_mult=2.5, atr_min=0.7, atr_max=2.0, dedup_bars=15, sl_atr_mult=1.0, max_hold=20, be_trigger=1.5)),
                ('R>2.0x Body>60% ATR0.7-2 SL1.2 EMA9 H25 BE2', dict(range_mult=2.0, atr_min=0.7, atr_max=2.0, dedup_bars=15, sl_atr_mult=1.2, max_hold=25, be_trigger=2.0)),
                ('R>3.0x Body>60% ATR0.7-2 SL0.8 EMA9 H15 BE1.5', dict(range_mult=3.0, atr_min=0.7, atr_max=2.0, dedup_bars=15, sl_atr_mult=0.8, max_hold=15, be_trigger=1.5)),
            ]

        for label, kw in mi_cfgs:
            r = run_mi(df_tf, **kw)
            print_result(f'{tf_name} MI {label}', r, n_d)

    # =====================================================================
    # 30-DAY OOS VALIDATION
    # =====================================================================
    print(f'\n\n{"="*100}')
    print('30-DAY OUT-OF-SAMPLE VALIDATION')
    print('=' * 100)

    print("\nLoading 30-day OOS data...")
    df5_oos = load_data(days=30, tf='5m')
    df3_oos = load_data(days=30, tf='3m')
    df1_oos = load_data(days=30, tf='1m')
    n_oos5 = df5_oos['date'].nunique()
    n_oos3 = df3_oos['date'].nunique()
    n_oos1 = df1_oos['date'].nunique()
    print(f"  5m: {len(df5_oos)} bars, {n_oos5}d | 3m: {len(df3_oos)} bars, {n_oos3}d | 1m: {len(df1_oos)} bars, {n_oos1}d\n")

    print(header)
    print('  ' + '-' * 94)

    # OOS: CB proven params on 5m
    r = run_cb(df5_oos,
               n_bars=3, threshold=0.7, atr_min=2.5, atr_max=4.5, rsi_max=70, dedup_bars=5,
               am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=24,
               pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=12,
               be_trigger=4.0, be_atr_min=3.5, trail_tighten_at=8.0, trail_tighten_mult=1.2)
    print_result('OOS 5m CB [PROVEN] 3bar/<0.7x AM:SL1.2/T@5/2x | PM:SL1.0/T@4/1.5x', r, n_oos5, min_trades=3)

    # OOS: CB best 3m config
    r = run_cb(df3_oos,
               n_bars=5, threshold=0.7, atr_min=1.5, atr_max=3.0, rsi_max=70, dedup_bars=8,
               am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=40,
               pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=20,
               be_trigger=4.0, be_atr_min=2.5, trail_tighten_at=8.0, trail_tighten_mult=1.2)
    print_result('OOS 3m CB 5bar/<0.7x AM:SL1.2/T@5/2x H40 | PM:SL1.0/T@4/1.5x H20', r, n_oos3, min_trades=3)

    # OOS: CB best 1m config
    r = run_cb(df1_oos,
               n_bars=15, threshold=0.7, atr_min=0.7, atr_max=1.5, rsi_max=70, dedup_bars=25,
               am_sl=1.2, am_trail_activate=5.0, am_trail_mult=2.0, am_max_hold=90,
               pm_sl=1.0, pm_trail_activate=4.0, pm_trail_mult=1.5, pm_max_hold=45,
               be_trigger=4.0, be_atr_min=1.0, trail_tighten_at=8.0, trail_tighten_mult=1.2)
    print_result('OOS 1m CB 15bar/<0.7x AM:SL1.2/T@5/2x H90 | PM:SL1.0/T@4/1.5x H45', r, n_oos1, min_trades=3)

    # OOS: MI on each TF
    r = run_mi(df5_oos, range_mult=2.0, atr_min=2.0, atr_max=5.0, dedup_bars=6,
               sl_atr_mult=1.0, max_hold=10, be_trigger=2.0)
    print_result('OOS 5m MI R>2.0x SL1.0 EMA9 H10 BE2', r, n_oos5, min_trades=3)

    r = run_mi(df3_oos, range_mult=2.0, atr_min=1.5, atr_max=4.0, dedup_bars=8,
               sl_atr_mult=1.0, max_hold=15, be_trigger=2.0)
    print_result('OOS 3m MI R>2.0x SL1.0 EMA9 H15 BE2', r, n_oos3, min_trades=3)

    r = run_mi(df1_oos, range_mult=2.5, atr_min=0.7, atr_max=2.0, dedup_bars=15,
               sl_atr_mult=1.0, max_hold=20, be_trigger=1.5)
    print_result('OOS 1m MI R>2.5x SL1.0 EMA9 H20 BE1.5', r, n_oos1, min_trades=3)

    print(f'\n\nDone. All results with COST = {COST} pts/trade.')
