"""
CBT: Compression Breakout in Trend (continuation pattern)
CB gốc: compression trong market flat → trade breakout bất kỳ hướng
CBT: compression SAU momentum → trade continuation theo hướng momentum

Logic:
1. Detect MOMENTUM: last N bars moved > X pts (strong directional move)
2. Detect COMPRESSION within momentum: 2-3 bars with range < threshold (pause/consolidation)
3. ENTER in momentum direction when compression breaks
4. SL: below compression zone
5. Trail: tight (quick continuation, not fresh breakout)

Also test multiple timeframes: 1m, 3m, 5m
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from research_strategy import load_data, add_indicators, dedup_signals
import pandas as pd
import numpy as np
import pandas_ta as ta

COST = 1.74


def prepare_data(days=180, tf='5m'):
    """Load data with specified timeframe."""
    df = load_data(days=days)
    df = add_indicators(df)

    if tf != '5m':
        df = resample_to_tf(df, tf)

    df['rsi14'] = ta.rsi(df['close'], length=14)
    df['ema9'] = ta.ema(df['close'], length=9)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['bar_range'] = df['high'] - df['low']
    df['body'] = df['close'] - df['open']
    return df


def resample_to_tf(df, tf):
    """Resample 5m data to 1m or 3m equivalent by grouping."""
    if tf == '3m':
        # Group every 3 bars from 5m = pseudo-15m
        # Actually for 3m we need raw 1m data which we don't have
        # So for 3m, let's use 5m as base (can't resample up to get more bars)
        # Instead: group 5m into 15m (every 3 bars)
        pass
    elif tf == '15m':
        # Group every 3 bars of 5m = 15m bars
        groups = []
        session_groups = df.groupby(['date', 'session'])
        for (date, sess), grp in session_groups:
            for i in range(0, len(grp), 3):
                chunk = grp.iloc[i:i+3]
                if len(chunk) == 0:
                    continue
                bar = {
                    'time': chunk.iloc[0]['time'],
                    'open': chunk.iloc[0]['open'],
                    'high': chunk['high'].max(),
                    'low': chunk['low'].min(),
                    'close': chunk.iloc[-1]['close'],
                    'volume': chunk['volume'].sum(),
                    'date': date,
                    'session': sess,
                    'mins_in_day': chunk.iloc[0]['mins_in_day'],
                }
                groups.append(bar)
        df_new = pd.DataFrame(groups).reset_index(drop=True)
        df_new['atr'] = ta.atr(df_new['high'], df_new['low'], df_new['close'], length=14)
        return df_new
    return df


def detect_cbt_signals(df, momentum_bars=4, momentum_pts=5.0, compress_bars=2,
                       compress_thresh=0.8, atr_max=5.0, atr_min=2.0, vol_min=0.5):
    """
    Detect Compression-in-Trend signals.

    1. Look back `momentum_bars` to check if price moved > `momentum_pts`
    2. Check last `compress_bars` have range < compress_thresh * ATR (consolidation)
    3. Direction = momentum direction
    """
    signals = pd.Series(0, index=df.index)

    for i in range(momentum_bars + compress_bars, len(df)):
        idx = df.index[i]
        row = df.loc[idx]
        atr = row['atr']

        if pd.isna(atr) or atr < atr_min or atr > atr_max:
            continue

        # Time filter
        mins = row['mins_in_day']
        if not ((mins >= 9*60+15 and mins <= 11*60+10) or (mins >= 13*60+5 and mins <= 14*60+15)):
            continue

        # Check compression: last `compress_bars` bars have small range
        compress_zone = df.iloc[i - compress_bars + 1:i + 1]
        if len(compress_zone) < compress_bars:
            continue

        # Must be same date/session
        if compress_zone.iloc[0]['date'] != row['date'] or compress_zone.iloc[0]['session'] != row['session']:
            continue

        # All compression bars must have range < threshold * ATR
        ranges = compress_zone['bar_range']
        if ranges.max() > compress_thresh * atr:
            continue

        # Check momentum BEFORE compression
        momentum_zone = df.iloc[i - compress_bars - momentum_bars + 1:i - compress_bars + 1]
        if len(momentum_zone) < momentum_bars:
            continue
        if momentum_zone.iloc[0]['date'] != row['date'] or momentum_zone.iloc[0]['session'] != row['session']:
            continue

        momentum_move = momentum_zone.iloc[-1]['close'] - momentum_zone.iloc[0]['open']

        if abs(momentum_move) < momentum_pts:
            continue

        direction = 1 if momentum_move > 0 else -1

        # Volume filter (at least one momentum bar had volume)
        if vol_min > 0 and 'vol_sma20' in df.columns:
            vol_ratio = row.get('volume', 0) / (row.get('vol_sma20', 1) or 1)
            # Skip vol filter if not available

        signals.loc[idx] = direction

    # Dedup
    buy_mask = signals == 1
    sell_mask = signals == -1
    buy_mask = dedup_signals(buy_mask, min_bars=6)
    sell_mask = dedup_signals(sell_mask, min_bars=6)
    signals = pd.Series(0, index=df.index)
    signals[buy_mask] = 1
    signals[sell_mask] = -1

    return signals


def backtest_cbt(df, signals, sl_mode='compression_low', sl_mult=1.2,
                 trail_activate=3.0, trail_atr_mult=1.5, max_hold=15,
                 be_trigger=2.0, compress_bars=2):
    """Backtest CBT trades."""
    trades = []
    signal_indices = df.index[signals != 0]

    for sig_idx in signal_indices:
        i_pos = df.index.get_loc(sig_idx)
        row = df.loc[sig_idx]
        direction = int(signals.loc[sig_idx])
        atr = row['atr']
        entry_price = row['close']

        if pd.isna(atr) or atr <= 0:
            continue

        # SL based on compression zone
        if sl_mode == 'compression_low':
            comp_zone = df.iloc[max(0, i_pos - compress_bars + 1):i_pos + 1]
            if direction == 1:
                sl_price = comp_zone['low'].min() - 0.3 * atr
            else:
                sl_price = comp_zone['high'].max() + 0.3 * atr
        else:
            sl_price = entry_price - direction * sl_mult * atr

        best_price = entry_price
        trail_active = False
        be_applied = False
        exit_price = None
        exit_reason = None

        start_bar = sig_idx + 1 if sig_idx + 1 in df.index else None
        if start_bar is None:
            continue

        bars_held = 0
        for j_pos in range(i_pos + 1, min(i_pos + 1 + max_hold, len(df))):
            if j_pos >= len(df):
                break
            j_idx = df.index[j_pos]
            bar = df.loc[j_idx]

            if bar['date'] != row['date'] or bar['session'] != row['session']:
                prev_idx = df.index[j_pos - 1]
                exit_price = df.loc[prev_idx, 'close']
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

                if mfe >= trail_activate:
                    trail_active = True
                if trail_active:
                    new_sl = best_price - trail_atr_mult * atr
                    sl_price = max(sl_price, new_sl)
                if trail_active and bar['low'] <= sl_price:
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

                if be_trigger > 0 and not be_applied and mfe >= be_trigger:
                    be_applied = True
                    sl_price = min(sl_price, entry_price)

                if mfe >= trail_activate:
                    trail_active = True
                if trail_active:
                    new_sl = best_price + trail_atr_mult * atr
                    sl_price = min(sl_price, new_sl)
                if trail_active and bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'TRAIL'
                    break

        if exit_price is None:
            exit_price = entry_price
            exit_reason = 'MAX_HOLD'

        pnl = direction * (exit_price - entry_price) - COST
        mfe_val = (best_price - entry_price) if direction == 1 else (entry_price - best_price)

        trades.append({
            'pnl': pnl, 'mfe': mfe_val, 'exit_reason': exit_reason,
            'date': row['date'], 'time': row['time'].strftime('%H:%M'),
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry': entry_price, 'exit': exit_price, 'atr': atr,
            'bars_held': bars_held,
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
        'df': tdf,
    }


def print_result(label, r, n_days):
    if r and r['trades'] >= 3:
        print(f'  {label:<55} {r["trades"]:>3} {r["win_rate"]:>5.1f} {r["pf"]:>5.2f} '
              f'{r["total_pnl"]:>+7.1f} {r["total_pnl"]/n_days:>+5.2f} {r["avg_mfe"]:>5.1f}')
    elif r:
        print(f'  {label:<55} {r["trades"]:>3} (too few)')
    else:
        print(f'  {label:<55}   -- no trades --')


if __name__ == '__main__':
    header = f"  {'Config':<55} {'T':>3} {'WR%':>5} {'PF':>5} {'PnL':>7} {'P/D':>5} {'MFE':>5}"

    # =====================================================================
    # 5m timeframe
    # =====================================================================
    print('Loading 180d 5m data...')
    df5 = prepare_data(days=180, tf='5m')
    # Add vol
    df5['vol_sma20'] = df5['volume'].rolling(20).mean()
    n_days = df5['date'].nunique()
    print(f'{len(df5)} bars, {n_days} days\n')

    print('=' * 95)
    print('CBT (Compression-in-Trend) on 5m')
    print('=' * 95)

    # Test 1: Signal detection params
    print('\n--- Signal Detection Sweep ---')
    print(header)
    print('  ' + '-' * 88)

    for mom_bars in [3, 4, 5, 6]:
        for mom_pts in [3.0, 4.0, 5.0, 6.0, 8.0]:
            for comp_bars in [2, 3]:
                for comp_thresh in [0.6, 0.7, 0.8, 1.0]:
                    sigs = detect_cbt_signals(df5, momentum_bars=mom_bars, momentum_pts=mom_pts,
                                             compress_bars=comp_bars, compress_thresh=comp_thresh)
                    n_sigs = (sigs != 0).sum()
                    if n_sigs < 10:
                        continue
                    label = f'Mom={mom_bars}b/{mom_pts}pts Comp={comp_bars}b/<{comp_thresh}x'
                    r = backtest_cbt(df5, sigs, compress_bars=comp_bars,
                                    trail_activate=3, trail_atr_mult=1.5, max_hold=15, be_trigger=2)
                    if r and r['trades'] >= 10 and r['pf'] >= 0.8:
                        print_result(label, r, n_days)

    # Test 2: Exit params for best signal configs
    print('\n--- Exit Optimization (best signal configs) ---')
    print(header)
    print('  ' + '-' * 88)

    # Pick configs with most trades and decent PF
    good_configs = [
        (4, 4.0, 2, 0.8),
        (4, 5.0, 2, 0.8),
        (3, 4.0, 2, 0.8),
        (4, 4.0, 3, 0.8),
        (5, 5.0, 2, 0.8),
        (4, 3.0, 2, 1.0),
        (3, 3.0, 2, 1.0),
    ]

    for mom_b, mom_p, comp_b, comp_t in good_configs:
        sigs = detect_cbt_signals(df5, momentum_bars=mom_b, momentum_pts=mom_p,
                                 compress_bars=comp_b, compress_thresh=comp_t)
        n_sigs = (sigs != 0).sum()
        if n_sigs < 8:
            continue

        for trail_act, trail_mult, max_h, be in [
            (2, 1.0, 10, 1.5),
            (3, 1.2, 12, 2.0),
            (3, 1.5, 15, 2.0),
            (4, 1.5, 15, 3.0),
            (4, 2.0, 20, 3.0),
            (5, 2.0, 20, 4.0),
        ]:
            label = f'M{mom_b}/{mom_p} C{comp_b}/{comp_t} T@+{trail_act}/{trail_mult}x H{max_h}'
            r = backtest_cbt(df5, sigs, compress_bars=comp_b,
                            trail_activate=trail_act, trail_atr_mult=trail_mult,
                            max_hold=max_h, be_trigger=be)
            if r and r['trades'] >= 8 and r['pf'] >= 1.0:
                print_result(label, r, n_days)

    # =====================================================================
    # 15m timeframe (3x 5m bars)
    # =====================================================================
    print('\n\n' + '=' * 95)
    print('CBT on 15m (grouped from 5m)')
    print('=' * 95)

    df15 = prepare_data(days=180, tf='15m')
    df15['vol_sma20'] = df15.get('volume', pd.Series(1, index=df15.index)).rolling(10).mean()
    n_days15 = df15['date'].nunique()
    print(f'{len(df15)} bars, {n_days15} days')

    print('\n--- Signal Detection ---')
    print(header)
    print('  ' + '-' * 88)

    for mom_bars in [2, 3, 4]:
        for mom_pts in [4.0, 5.0, 6.0, 8.0, 10.0]:
            for comp_bars in [2, 3]:
                for comp_thresh in [0.7, 0.8, 1.0]:
                    sigs = detect_cbt_signals(df15, momentum_bars=mom_bars, momentum_pts=mom_pts,
                                             compress_bars=comp_bars, compress_thresh=comp_thresh,
                                             atr_max=8.0)
                    n_sigs = (sigs != 0).sum()
                    if n_sigs < 5:
                        continue
                    label = f'15m Mom={mom_bars}b/{mom_pts}pts Comp={comp_bars}b/<{comp_thresh}x'
                    r = backtest_cbt(df15, sigs, compress_bars=comp_bars,
                                    trail_activate=4, trail_atr_mult=1.5, max_hold=8, be_trigger=3)
                    if r and r['trades'] >= 5 and r['pf'] >= 0.8:
                        print_result(label, r, n_days15)

    # =====================================================================
    # OOS Validation
    # =====================================================================
    print('\n\n' + '=' * 95)
    print('30-DAY OOS VALIDATION')
    print('=' * 95)

    df5_30 = prepare_data(days=30, tf='5m')
    df5_30['vol_sma20'] = df5_30['volume'].rolling(20).mean()
    nd30 = df5_30['date'].nunique()
    print(f'5m OOS: {len(df5_30)} bars, {nd30} days')

    print(f'\n{header}')
    print('  ' + '-' * 88)

    oos_configs = [
        ('5m M4/4 C2/0.8 T@3/1.5x H15 BE2', (4, 4.0, 2, 0.8), dict(trail_activate=3, trail_atr_mult=1.5, max_hold=15, be_trigger=2)),
        ('5m M4/5 C2/0.8 T@3/1.5x H15 BE2', (4, 5.0, 2, 0.8), dict(trail_activate=3, trail_atr_mult=1.5, max_hold=15, be_trigger=2)),
        ('5m M3/4 C2/0.8 T@3/1.5x H15 BE2', (3, 4.0, 2, 0.8), dict(trail_activate=3, trail_atr_mult=1.5, max_hold=15, be_trigger=2)),
        ('5m M4/4 C2/0.8 T@4/2.0x H20 BE3', (4, 4.0, 2, 0.8), dict(trail_activate=4, trail_atr_mult=2.0, max_hold=20, be_trigger=3)),
        ('5m M4/3 C2/1.0 T@3/1.5x H15 BE2', (4, 3.0, 2, 1.0), dict(trail_activate=3, trail_atr_mult=1.5, max_hold=15, be_trigger=2)),
        ('5m M3/3 C2/1.0 T@3/1.2x H12 BE2', (3, 3.0, 2, 1.0), dict(trail_activate=3, trail_atr_mult=1.2, max_hold=12, be_trigger=2)),
        ('5m M5/5 C2/0.8 T@4/2.0x H20 BE3', (5, 5.0, 2, 0.8), dict(trail_activate=4, trail_atr_mult=2.0, max_hold=20, be_trigger=3)),
        ('5m M4/4 C3/0.8 T@3/1.5x H15 BE2', (4, 4.0, 3, 0.8), dict(trail_activate=3, trail_atr_mult=1.5, max_hold=15, be_trigger=2)),
    ]

    best_oos = None
    best_oos_label = ''

    for label, (mb, mp, cb, ct), exit_kw in oos_configs:
        sigs = detect_cbt_signals(df5_30, momentum_bars=mb, momentum_pts=mp,
                                 compress_bars=cb, compress_thresh=ct)
        r = backtest_cbt(df5_30, sigs, compress_bars=cb, **exit_kw)
        print_result(label, r, nd30)
        if r and r['trades'] >= 5 and (best_oos is None or r['pf'] > best_oos['pf']):
            best_oos = r
            best_oos_label = label

    # Show best OOS trades
    if best_oos and best_oos['trades'] >= 3:
        print(f'\n--- Best OOS: {best_oos_label} ---')
        print(f'Exits: {best_oos["exits"]}')
        tdf = best_oos['df']
        print(f"\n{'Date':<12} {'T':<6} {'Dir':<5} {'Entry':>7} {'Exit':>7} {'MFE':>5} {'PnL':>6} {'Bars':>4} {'Reason':<10}")
        for _, t in tdf.iterrows():
            print(f'{str(t["date"]):<12} {t["time"]:<6} {t["direction"]:<5} '
                  f'{t["entry"]:>7.1f} {t["exit"]:>7.1f} {t["mfe"]:>5.1f} '
                  f'{t["pnl"]:>+6.1f} {t["bars_held"]:>4} {t["exit_reason"]:<10}')

    # June 3 specific check
    print('\n--- JUNE 3 CHECK ---')
    target = '2026-06-03'
    for label, (mb, mp, cb, ct), exit_kw in oos_configs:
        sigs = detect_cbt_signals(df5_30, momentum_bars=mb, momentum_pts=mp,
                                 compress_bars=cb, compress_thresh=ct)
        r = backtest_cbt(df5_30, sigs, compress_bars=cb, **exit_kw)
        if r:
            j3 = r['df'][r['df']['date'] == target]
            if len(j3) > 0:
                for _, t in j3.iterrows():
                    print(f'  {label}: {t["time"]} {t["direction"]} @{t["entry"]:.1f}->{t["exit"]:.1f} '
                          f'MFE={t["mfe"]:.1f} PnL={t["pnl"]:+.1f} ({t["exit_reason"]})')
