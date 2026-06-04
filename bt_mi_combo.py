"""
Backtest MI (Momentum Ignition + Pullback) combo for trending days.

Logic:
1. DETECT: 5m bar with range > K * ATR(14) = "ignition bar"
2. CONFIRM: bar closes in direction (close near high for BUY, close near low for SELL)
3. WAIT: 1-3 bars for pullback toward EMA9 or 50% of ignition bar
4. ENTER: when price holds above (BUY) ignition bar midpoint after pullback
5. SL: below pullback low (or ignition bar low for BUY)
6. TRAIL: EMA9 as trailing stop (close below EMA9 = exit)
7. EXIT: session end or trail hit

Filters:
- ADX(14) >= threshold (confirms trending, not random spike)
- Volume on ignition bar >= vol_ratio * SMA(20)
- Not within 15 min of session end
- Dedup: only first ignition per session direction
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from research_strategy import load_data, add_indicators, dedup_signals
import pandas as pd
import numpy as np
import pandas_ta as ta

COST = 1.74

def prepare_data(days=180):
    df = load_data(days=days)
    df = add_indicators(df)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    df['ema9'] = ta.ema(df['close'], length=9)
    df['ema21'] = ta.ema(df['close'], length=21)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14']
    df['dmp'] = adx_df['DMP_14']
    df['dmn'] = adx_df['DMN_14']
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20']
    df['bar_range'] = df['high'] - df['low']
    df['range_atr_ratio'] = df['bar_range'] / df['atr'].replace(0, np.nan)
    df['body'] = df['close'] - df['open']
    df['body_ratio'] = abs(df['body']) / df['bar_range'].replace(0, np.nan)
    df['bar_mid'] = (df['high'] + df['low']) / 2
    return df


def detect_ignition(df, range_mult=2.5, body_ratio_min=0.5, adx_min=20, vol_min=1.0):
    """Detect momentum ignition bars."""
    ignition = (
        (df['range_atr_ratio'] >= range_mult) &
        (df['body_ratio'] >= body_ratio_min) &
        (df['adx'] >= adx_min) &
        (df['vol_ratio'] >= vol_min) &
        (df['atr'] >= 2.0) &
        (df['atr'] <= 5.0)
    )
    return ignition.fillna(False)


def backtest_mi(df, range_mult=2.5, body_ratio_min=0.5, adx_min=20, vol_min=1.0,
                pullback_bars=3, pullback_depth=0.5, sl_mode='pullback_low',
                sl_atr_mult=1.5, trail_mode='ema9', max_hold=20,
                session_filter=True):
    """
    Backtest MI strategy.

    Entry modes:
    - 'immediate': enter at close of ignition bar (aggressive)
    - 'pullback': wait for pullback then enter when price resumes

    Trail modes:
    - 'ema9': exit when close < EMA9 (for longs)
    - 'atr': standard ATR trailing
    """
    ignition = detect_ignition(df, range_mult, body_ratio_min, adx_min, vol_min)

    # Time window filter
    if session_filter:
        time_ok = (
            ((df['mins_in_day'] >= 9*60+15) & (df['mins_in_day'] <= 11*60)) |
            ((df['mins_in_day'] >= 13*60) & (df['mins_in_day'] <= 14*60))
        )
        ignition = ignition & time_ok

    # Dedup: max 1 signal per direction per session
    ignition = dedup_signals(ignition, min_bars=10)

    trades = []
    entry_indices = df.index[ignition]

    for sig_idx in entry_indices:
        row = df.loc[sig_idx]
        atr = row['atr']
        direction = 1 if row['body'] > 0 else -1

        # Ignition bar properties
        ig_high = row['high']
        ig_low = row['low']
        ig_mid = row['bar_mid']
        ig_close = row['close']

        # Find pullback entry within next N bars
        entry_price = None
        entry_bar_idx = None
        pullback_low_price = ig_close if direction == 1 else ig_close

        if pullback_bars == 0:
            # Immediate entry at ignition bar close
            entry_price = ig_close
            entry_bar_idx = sig_idx
            pullback_low_price = ig_low if direction == 1 else ig_high
        else:
            # Wait for pullback
            found_pullback = False
            pb_extreme = ig_close  # track pullback depth

            for pb in range(1, pullback_bars + 1):
                pb_idx = sig_idx + pb
                if pb_idx not in df.index:
                    break
                pb_bar = df.loc[pb_idx]
                if pb_bar['date'] != row['date'] or pb_bar['session'] != row['session']:
                    break

                if direction == 1:
                    # For BUY: pullback = bar with lower low or lower close
                    if pb_bar['low'] < pb_extreme:
                        pb_extreme = pb_bar['low']

                    # Check pullback depth: must retrace at least to midpoint or less
                    retrace = ig_close - pb_extreme

                    # Entry when bar closes above EMA9 or above ignition midpoint after pullback
                    if retrace >= 0.5 * atr:  # meaningful pullback
                        found_pullback = True

                    if found_pullback and pb_bar['close'] > pb_bar['open'] and pb_bar['close'] >= ig_mid:
                        entry_price = pb_bar['close']
                        entry_bar_idx = pb_idx
                        pullback_low_price = pb_extreme
                        break
                else:
                    # For SELL: pullback = bar with higher high
                    if pb_bar['high'] > pb_extreme:
                        pb_extreme = pb_bar['high']

                    retrace = pb_extreme - ig_close

                    if retrace >= 0.5 * atr:
                        found_pullback = True

                    if found_pullback and pb_bar['close'] < pb_bar['open'] and pb_bar['close'] <= ig_mid:
                        entry_price = pb_bar['close']
                        entry_bar_idx = pb_idx
                        pullback_low_price = pb_extreme
                        break

            # If no pullback found within window, try immediate entry at next bar
            if entry_price is None and pullback_bars > 0:
                # Fallback: enter at bar+1 close if it continues in direction
                next_idx = sig_idx + 1
                if next_idx in df.index:
                    next_bar = df.loc[next_idx]
                    if next_bar['date'] == row['date'] and next_bar['session'] == row['session']:
                        if direction == 1 and next_bar['close'] > ig_mid:
                            entry_price = next_bar['close']
                            entry_bar_idx = next_idx
                            pullback_low_price = min(ig_low, next_bar['low'])
                        elif direction == -1 and next_bar['close'] < ig_mid:
                            entry_price = next_bar['close']
                            entry_bar_idx = next_idx
                            pullback_low_price = max(ig_high, next_bar['high'])

        if entry_price is None:
            continue

        # Set stop loss
        if sl_mode == 'pullback_low':
            if direction == 1:
                sl_price = pullback_low_price - 0.3 * atr
            else:
                sl_price = pullback_low_price + 0.3 * atr
        else:  # atr mode
            sl_price = entry_price - direction * sl_atr_mult * atr

        # Simulate trade
        best_price = entry_price
        exit_price = None
        exit_reason = None
        trail_active = False

        start_bar = entry_bar_idx + 1
        for j in range(start_bar, min(start_bar + max_hold, len(df))):
            if j not in df.index:
                break
            bar = df.loc[j]
            if bar['date'] != row['date'] or bar['session'] != row['session']:
                exit_price = df.loc[j-1, 'close'] if j-1 in df.index else entry_price
                exit_reason = 'SESSION_END'
                break
            if (bar['session'] == 'PM' and bar['mins_in_day'] >= 14*60+25) or \
               (bar['session'] == 'AM' and bar['mins_in_day'] >= 11*60+25):
                exit_price = bar['close']
                exit_reason = 'SESSION_END'
                break

            if direction == 1:
                if bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    break
                if bar['high'] > best_price:
                    best_price = bar['high']

                # EMA9 trailing
                if trail_mode == 'ema9':
                    mfe = best_price - entry_price
                    if mfe >= 3.0:  # activate trail after +3
                        trail_active = True
                    if trail_active:
                        ema9_val = bar.get('ema9', entry_price)
                        if not pd.isna(ema9_val):
                            new_sl = ema9_val - 0.5 * atr
                            sl_price = max(sl_price, new_sl)
                    if trail_active and bar['close'] < df.loc[j, 'ema9']:
                        exit_price = bar['close']
                        exit_reason = 'EMA9_TRAIL'
                        break
                else:  # ATR trail
                    mfe = best_price - entry_price
                    if mfe >= 4.0:
                        trail_active = True
                    if trail_active:
                        new_sl = best_price - 1.5 * atr
                        sl_price = max(sl_price, new_sl)
                    if trail_active and bar['low'] <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'ATR_TRAIL'
                        break
            else:
                if bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    break
                if bar['low'] < best_price:
                    best_price = bar['low']

                if trail_mode == 'ema9':
                    mfe = entry_price - best_price
                    if mfe >= 3.0:
                        trail_active = True
                    if trail_active:
                        ema9_val = bar.get('ema9', entry_price)
                        if not pd.isna(ema9_val):
                            new_sl = ema9_val + 0.5 * atr
                            sl_price = min(sl_price, new_sl)
                    if trail_active and bar['close'] > df.loc[j, 'ema9']:
                        exit_price = bar['close']
                        exit_reason = 'EMA9_TRAIL'
                        break
                else:
                    mfe = entry_price - best_price
                    if mfe >= 4.0:
                        trail_active = True
                    if trail_active:
                        new_sl = best_price + 1.5 * atr
                        sl_price = min(sl_price, new_sl)
                    if trail_active and bar['high'] >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'ATR_TRAIL'
                        break

        if exit_price is None:
            exit_price = entry_price
            exit_reason = 'MAX_HOLD'

        pnl = direction * (exit_price - entry_price) - COST
        mfe = (best_price - entry_price) if direction == 1 else (entry_price - best_price)

        trades.append({
            'signal_time': row['time'].strftime('%H:%M'),
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'mfe': mfe,
            'exit_reason': exit_reason,
            'atr': atr,
            'range_ratio': row['range_atr_ratio'],
            'adx': row['adx'],
            'date': row['date'],
        })

    if not trades:
        return None

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf['pnl'] > 0]
    losses = tdf[tdf['pnl'] <= 0]
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 999

    return {
        'trades': len(tdf),
        'win_rate': (tdf['pnl'] > 0).mean() * 100,
        'pf': pf,
        'total_pnl': tdf['pnl'].sum(),
        'avg_mfe': tdf['mfe'].mean(),
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        'exit_reasons': tdf['exit_reason'].value_counts().to_dict(),
        'trades_df': tdf,
    }


# ============================================================
# MAIN BACKTEST
# ============================================================
if __name__ == '__main__':
    print('Loading 180-day data...')
    df = prepare_data(days=180)
    n_days = df['date'].nunique()
    print(f'Data: {len(df)} bars, {n_days} days')
    print(f'ATR range: {df["atr"].min():.2f} - {df["atr"].max():.2f}')
    print()

    # ============================================================
    print('='*80)
    print('MI COMBO: MOMENTUM IGNITION + PULLBACK')
    print('='*80)

    # Test 1: Signal detection parameters
    print('\n--- TEST 1: IGNITION DETECTION (immediate entry) ---')
    print(f'  {"Config":<45} {"T":>4} {"WR%":>5} {"PF":>5} {"PnL":>8} {"PnL/D":>6} {"MFE":>5}')
    print('  ' + '-'*80)

    configs_detect = [
        ('Range>2.0 ADX>20 Vol>0.8 Body>0.4', dict(range_mult=2.0, adx_min=20, vol_min=0.8, body_ratio_min=0.4)),
        ('Range>2.0 ADX>20 Vol>1.0 Body>0.5', dict(range_mult=2.0, adx_min=20, vol_min=1.0, body_ratio_min=0.5)),
        ('Range>2.0 ADX>25 Vol>1.0 Body>0.5', dict(range_mult=2.0, adx_min=25, vol_min=1.0, body_ratio_min=0.5)),
        ('Range>2.5 ADX>20 Vol>0.8 Body>0.4', dict(range_mult=2.5, adx_min=20, vol_min=0.8, body_ratio_min=0.4)),
        ('Range>2.5 ADX>20 Vol>1.0 Body>0.5', dict(range_mult=2.5, adx_min=20, vol_min=1.0, body_ratio_min=0.5)),
        ('Range>2.5 ADX>25 Vol>1.0 Body>0.5', dict(range_mult=2.5, adx_min=25, vol_min=1.0, body_ratio_min=0.5)),
        ('Range>3.0 ADX>20 Vol>0.8 Body>0.4', dict(range_mult=3.0, adx_min=20, vol_min=0.8, body_ratio_min=0.4)),
        ('Range>3.0 ADX>20 Vol>1.0 Body>0.5', dict(range_mult=3.0, adx_min=20, vol_min=1.0, body_ratio_min=0.5)),
        ('Range>3.0 ADX>25 Vol>1.0 Body>0.5', dict(range_mult=3.0, adx_min=25, vol_min=1.0, body_ratio_min=0.5)),
    ]

    for label, kwargs in configs_detect:
        r = backtest_mi(df, pullback_bars=0, trail_mode='ema9', max_hold=20, **kwargs)
        if r:
            print(f'  {label:<45} {r["trades"]:>4} {r["win_rate"]:>5.1f} {r["pf"]:>5.2f} '
                  f'{r["total_pnl"]:>+8.1f} {r["total_pnl"]/n_days:>+6.2f} {r["avg_mfe"]:>5.1f}')
        else:
            print(f'  {label:<45}   -- no trades --')

    # Test 2: Pullback entry vs immediate
    print('\n--- TEST 2: ENTRY MODE (pullback vs immediate) ---')
    print(f'  {"Config":<45} {"T":>4} {"WR%":>5} {"PF":>5} {"PnL":>8} {"PnL/D":>6} {"MFE":>5}')
    print('  ' + '-'*80)

    best_detect = dict(range_mult=2.5, adx_min=20, vol_min=0.8, body_ratio_min=0.4)

    for pb_bars in [0, 1, 2, 3]:
        for trail in ['ema9', 'atr']:
            label = f'PB={pb_bars}bars Trail={trail}'
            r = backtest_mi(df, pullback_bars=pb_bars, trail_mode=trail, max_hold=20, **best_detect)
            if r:
                print(f'  {label:<45} {r["trades"]:>4} {r["win_rate"]:>5.1f} {r["pf"]:>5.2f} '
                      f'{r["total_pnl"]:>+8.1f} {r["total_pnl"]/n_days:>+6.2f} {r["avg_mfe"]:>5.1f}')

    # Test 3: SL mode
    print('\n--- TEST 3: SL MODE ---')
    print(f'  {"Config":<45} {"T":>4} {"WR%":>5} {"PF":>5} {"PnL":>8} {"PnL/D":>6}')
    print('  ' + '-'*75)

    for sl_mode in ['pullback_low', 'atr']:
        for sl_mult in [1.0, 1.5, 2.0]:
            label = f'SL={sl_mode} mult={sl_mult}'
            r = backtest_mi(df, pullback_bars=1, trail_mode='ema9', max_hold=20,
                           sl_mode=sl_mode, sl_atr_mult=sl_mult, **best_detect)
            if r:
                print(f'  {label:<45} {r["trades"]:>4} {r["win_rate"]:>5.1f} {r["pf"]:>5.2f} '
                      f'{r["total_pnl"]:>+8.1f} {r["total_pnl"]/n_days:>+6.2f}')

    # Test 4: Max hold bars
    print('\n--- TEST 4: MAX HOLD BARS ---')
    print(f'  {"Config":<45} {"T":>4} {"WR%":>5} {"PF":>5} {"PnL":>8} {"PnL/D":>6}')
    print('  ' + '-'*75)

    for max_h in [8, 12, 15, 20, 30]:
        label = f'MaxHold={max_h} bars'
        r = backtest_mi(df, pullback_bars=1, trail_mode='ema9', max_hold=max_h,
                       sl_mode='pullback_low', **best_detect)
        if r:
            print(f'  {label:<45} {r["trades"]:>4} {r["win_rate"]:>5.1f} {r["pf"]:>5.2f} '
                  f'{r["total_pnl"]:>+8.1f} {r["total_pnl"]/n_days:>+6.2f}')

    # ============================================================
    # BEST CONFIG: detailed analysis
    print('\n' + '='*80)
    print('BEST CONFIG DETAILED ANALYSIS')
    print('='*80)

    # Run with best params
    best_result = backtest_mi(df, range_mult=2.5, body_ratio_min=0.4, adx_min=20, vol_min=0.8,
                              pullback_bars=1, trail_mode='ema9', max_hold=20,
                              sl_mode='pullback_low')

    if best_result:
        tdf = best_result['trades_df']
        print(f'\nBest config: Range>2.5x, ADX>20, Vol>0.8, Body>0.4, PB=1, Trail=EMA9')
        print(f'Trades: {best_result["trades"]} | WR: {best_result["win_rate"]:.1f}% | PF: {best_result["pf"]:.2f}')
        print(f'Total PnL: {best_result["total_pnl"]:+.1f} pts | Per day: {best_result["total_pnl"]/n_days:+.2f}')
        print(f'Avg Win: {best_result["avg_win"]:+.1f} | Avg Loss: {best_result["avg_loss"]:+.1f}')
        print(f'Avg MFE: {best_result["avg_mfe"]:.1f} pts')
        print(f'Exit reasons: {best_result["exit_reasons"]}')

        # Direction breakdown
        buys = tdf[tdf['direction'] == 'BUY']
        sells = tdf[tdf['direction'] == 'SELL']
        print(f'\nBUY:  {len(buys)} trades, WR {(buys["pnl"]>0).mean()*100:.0f}%, PnL {buys["pnl"].sum():+.1f}')
        print(f'SELL: {len(sells)} trades, WR {(sells["pnl"]>0).mean()*100:.0f}%, PnL {sells["pnl"].sum():+.1f}')

        # Daily distribution
        daily = tdf.groupby('date')['pnl'].sum()
        print(f'\nDaily: {len(daily)} active days, {(daily>0).sum()} profitable')
        print(f'Best day: {daily.max():+.1f} | Worst day: {daily.min():+.1f}')

        # Show some sample trades
        print(f'\n--- SAMPLE TRADES ---')
        print(f'{"Date":<12} {"Time":<6} {"Dir":<5} {"Entry":>7} {"Exit":>7} {"MFE":>5} {"PnL":>6} {"Reason":<12}')
        for _, t in tdf.head(15).iterrows():
            print(f'{t["date"]:<12} {t["signal_time"]:<6} {t["direction"]:<5} '
                  f'{t["entry_price"]:>7.1f} {t["exit_price"]:>7.1f} {t["mfe"]:>5.1f} '
                  f'{t["pnl"]:>+6.1f} {t["exit_reason"]:<12}')

    # ============================================================
    # 30-day OOS validation
    print('\n\n' + '='*80)
    print('30-DAY OUT-OF-SAMPLE VALIDATION')
    print('='*80)

    df30 = prepare_data(days=30)
    nd30 = df30['date'].nunique()
    print(f'OOS data: {len(df30)} bars, {nd30} days')

    configs_oos = [
        ('Range>2.0 ADX>20 Vol>0.8 PB=0 EMA9', dict(range_mult=2.0, adx_min=20, vol_min=0.8, body_ratio_min=0.4, pullback_bars=0, trail_mode='ema9')),
        ('Range>2.0 ADX>20 Vol>0.8 PB=1 EMA9', dict(range_mult=2.0, adx_min=20, vol_min=0.8, body_ratio_min=0.4, pullback_bars=1, trail_mode='ema9')),
        ('Range>2.5 ADX>20 Vol>0.8 PB=0 EMA9', dict(range_mult=2.5, adx_min=20, vol_min=0.8, body_ratio_min=0.4, pullback_bars=0, trail_mode='ema9')),
        ('Range>2.5 ADX>20 Vol>0.8 PB=1 EMA9', dict(range_mult=2.5, adx_min=20, vol_min=0.8, body_ratio_min=0.4, pullback_bars=1, trail_mode='ema9')),
        ('Range>2.5 ADX>20 Vol>1.0 PB=1 EMA9', dict(range_mult=2.5, adx_min=20, vol_min=1.0, body_ratio_min=0.5, pullback_bars=1, trail_mode='ema9')),
        ('Range>3.0 ADX>20 Vol>0.8 PB=0 EMA9', dict(range_mult=3.0, adx_min=20, vol_min=0.8, body_ratio_min=0.4, pullback_bars=0, trail_mode='ema9')),
        ('Range>2.5 ADX>20 Vol>0.8 PB=1 ATR', dict(range_mult=2.5, adx_min=20, vol_min=0.8, body_ratio_min=0.4, pullback_bars=1, trail_mode='atr')),
    ]

    print(f'\n  {"Config":<45} {"T":>4} {"WR%":>5} {"PF":>5} {"PnL":>8} {"PnL/D":>6}')
    print('  ' + '-'*75)

    for label, kwargs in configs_oos:
        r = backtest_mi(df30, max_hold=20, sl_mode='pullback_low', **kwargs)
        if r:
            print(f'  {label:<45} {r["trades"]:>4} {r["win_rate"]:>5.1f} {r["pf"]:>5.2f} '
                  f'{r["total_pnl"]:>+8.1f} {r["total_pnl"]/nd30:>+6.2f}')
        else:
            print(f'  {label:<45}   -- no trades --')

    # ============================================================
    # Check if June 3 would have been caught
    print('\n\n' + '='*80)
    print('JUNE 3 SIMULATION (does MI catch the trend day?)')
    print('='*80)

    target_date = '2026-06-03'
    june3_trades = best_result['trades_df'][best_result['trades_df']['date'] == target_date] if best_result else pd.DataFrame()

    if len(june3_trades) > 0:
        print(f'\nJune 3 trades:')
        for _, t in june3_trades.iterrows():
            print(f'  {t["signal_time"]} {t["direction"]} @ {t["entry_price"]:.1f} -> {t["exit_price"]:.1f} '
                  f'MFE={t["mfe"]:.1f} PnL={t["pnl"]:+.1f} ({t["exit_reason"]})')
        print(f'  Total: {june3_trades["pnl"].sum():+.1f} pts')
    else:
        print('\nNo MI trades on June 3 with best config.')
        # Try with looser params
        print('Trying looser params (Range>2.0)...')
        r2 = backtest_mi(df, range_mult=2.0, body_ratio_min=0.4, adx_min=20, vol_min=0.8,
                         pullback_bars=0, trail_mode='ema9', max_hold=20, sl_mode='pullback_low')
        if r2:
            j3 = r2['trades_df'][r2['trades_df']['date'] == target_date]
            if len(j3) > 0:
                print(f'With Range>2.0:')
                for _, t in j3.iterrows():
                    print(f'  {t["signal_time"]} {t["direction"]} @ {t["entry_price"]:.1f} -> {t["exit_price"]:.1f} '
                          f'MFE={t["mfe"]:.1f} PnL={t["pnl"]:+.1f} ({t["exit_reason"]})')
