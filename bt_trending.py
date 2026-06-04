"""
Backtest 3 trending-day strategies from GitHub research on VN30F1M 5m:
1. Dual SuperTrend (10,3)+(21,2) + ADX filter
2. EMA 9/21 Crossover + ADX filter
3. VWAP Breakout + 2-bar confirmation

All designed for TRENDING days when CB (compression breakout) doesn't fire.
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

    # EMAs
    df['ema9'] = ta.ema(df['close'], length=9)
    df['ema21'] = ta.ema(df['close'], length=21)

    # ADX
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14']
    df['dmp'] = adx_df['DMP_14']
    df['dmn'] = adx_df['DMN_14']

    # SuperTrend fast (10,3) and slow (21,2)
    st_fast = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
    df['st_fast_dir'] = st_fast['SUPERTd_10_3.0']  # 1=up, -1=down
    df['st_fast_val'] = st_fast['SUPERT_10_3.0']

    st_slow = ta.supertrend(df['high'], df['low'], df['close'], length=21, multiplier=2.0)
    df['st_slow_dir'] = st_slow['SUPERTd_21_2.0']
    df['st_slow_val'] = st_slow['SUPERT_21_2.0']

    # Volume
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20']

    # VWAP (per session)
    df['vwap'] = _calc_session_vwap(df)
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['atr']

    return df


def _calc_session_vwap(df):
    """Calculate VWAP resetting each session."""
    vwap = pd.Series(index=df.index, dtype=float)
    groups = df.groupby(['date', 'session'])
    for (date, sess), grp in groups:
        typical = (grp['high'] + grp['low'] + grp['close']) / 3
        cum_tp_vol = (typical * grp['volume']).cumsum()
        cum_vol = grp['volume'].cumsum()
        vwap.loc[grp.index] = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap


def simulate_trades(df, signals, sl_mode='atr', sl_mult=1.5, trail_mode='ema9',
                    trail_activate=4.0, trail_atr_mult=2.0, max_hold=20, be_trigger=3.0):
    """Generic trade simulator. signals = Series with 1 (BUY) or -1 (SELL)."""
    trades = []
    signal_indices = df.index[signals != 0]

    for sig_idx in signal_indices:
        row = df.loc[sig_idx]
        direction = int(signals.loc[sig_idx])
        atr = row['atr']
        entry_price = row['close']

        if pd.isna(atr) or atr <= 0:
            continue

        # SL
        if sl_mode == 'atr':
            sl_price = entry_price - direction * sl_mult * atr
        elif sl_mode == 'supertrend':
            if direction == 1:
                sl_price = row['st_fast_val'] - 0.3 * atr
            else:
                sl_price = row['st_fast_val'] + 0.3 * atr
        else:
            sl_price = entry_price - direction * sl_mult * atr

        best_price = entry_price
        trail_active = False
        be_applied = False
        exit_price = None
        exit_reason = None

        start_bar = sig_idx + 1
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
                    exit_reason = 'BE' if be_applied else 'SL'
                    break
                if bar['high'] > best_price:
                    best_price = bar['high']
                mfe = best_price - entry_price

                if be_trigger > 0 and not be_applied and mfe >= be_trigger:
                    be_applied = True
                    sl_price = max(sl_price, entry_price)

                if trail_mode == 'ema9':
                    if mfe >= trail_activate:
                        trail_active = True
                    if trail_active and not pd.isna(bar.get('ema9', np.nan)):
                        new_sl = bar['ema9'] - 0.3 * atr
                        sl_price = max(sl_price, new_sl)
                elif trail_mode == 'atr':
                    if mfe >= trail_activate:
                        trail_active = True
                    if trail_active:
                        new_sl = best_price - trail_atr_mult * atr
                        sl_price = max(sl_price, new_sl)
                elif trail_mode == 'supertrend':
                    st_val = bar.get('st_fast_val', entry_price - sl_mult * atr)
                    if not pd.isna(st_val) and st_val > sl_price:
                        sl_price = st_val
                        trail_active = True

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

                if trail_mode == 'ema9':
                    if mfe >= trail_activate:
                        trail_active = True
                    if trail_active and not pd.isna(bar.get('ema9', np.nan)):
                        new_sl = bar['ema9'] + 0.3 * atr
                        sl_price = min(sl_price, new_sl)
                elif trail_mode == 'atr':
                    if mfe >= trail_activate:
                        trail_active = True
                    if trail_active:
                        new_sl = best_price + trail_atr_mult * atr
                        sl_price = min(sl_price, new_sl)
                elif trail_mode == 'supertrend':
                    st_val = bar.get('st_fast_val', entry_price + sl_mult * atr)
                    if not pd.isna(st_val) and st_val < sl_price:
                        sl_price = st_val
                        trail_active = True

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
        })

    return _summarize(trades)


def _summarize(trades):
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


# =====================================================================
# STRATEGY 1: DUAL SUPERTREND
# =====================================================================
def gen_dual_supertrend_signals(df, adx_min=22, vol_min=0.8, atr_max=5.0):
    """
    Signal when BOTH SuperTrends flip to same direction.
    Fast ST (10,3) captures quick moves, slow ST (21,2) confirms trend.
    """
    signals = pd.Series(0, index=df.index)

    # Detect flips: when fast ST flips AND slow ST already agrees
    for i in range(1, len(df)):
        idx = df.index[i]
        prev_idx = df.index[i-1]
        row = df.loc[idx]

        if pd.isna(row['adx']) or row['adx'] < adx_min:
            continue
        if pd.isna(row['atr']) or row['atr'] > atr_max or row['atr'] < 2.0:
            continue
        if row['vol_ratio'] < vol_min:
            continue

        # Time filter
        mins = row['mins_in_day']
        if not ((mins >= 9*60+15 and mins <= 11*60) or (mins >= 13*60+5 and mins <= 14*60+10)):
            continue

        # Fast ST flips up AND slow ST is already up
        prev_fast = df.loc[prev_idx, 'st_fast_dir']
        curr_fast = row['st_fast_dir']
        curr_slow = row['st_slow_dir']

        if prev_fast == -1 and curr_fast == 1 and curr_slow == 1:
            signals.loc[idx] = 1
        elif prev_fast == 1 and curr_fast == -1 and curr_slow == -1:
            signals.loc[idx] = -1

    # Dedup
    buy_mask = signals == 1
    sell_mask = signals == -1
    buy_mask = dedup_signals(buy_mask, min_bars=10)
    sell_mask = dedup_signals(sell_mask, min_bars=10)
    signals = pd.Series(0, index=df.index)
    signals[buy_mask] = 1
    signals[sell_mask] = -1

    return signals


# =====================================================================
# STRATEGY 2: EMA 9/21 CROSSOVER
# =====================================================================
def gen_ema_cross_signals(df, adx_min=20, vol_min=0.8, atr_max=5.0, dmp_dmn_diff=5):
    """
    Signal when EMA9 crosses EMA21 + ADX confirms trend strength.
    Additional filter: DI+ - DI- > threshold for direction confirmation.
    """
    signals = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        idx = df.index[i]
        prev_idx = df.index[i-1]
        row = df.loc[idx]

        if pd.isna(row['adx']) or row['adx'] < adx_min:
            continue
        if pd.isna(row['atr']) or row['atr'] > atr_max or row['atr'] < 2.0:
            continue
        if row['vol_ratio'] < vol_min:
            continue

        mins = row['mins_in_day']
        if not ((mins >= 9*60+15 and mins <= 11*60) or (mins >= 13*60+5 and mins <= 14*60+10)):
            continue

        prev_ema9 = df.loc[prev_idx, 'ema9']
        prev_ema21 = df.loc[prev_idx, 'ema21']
        curr_ema9 = row['ema9']
        curr_ema21 = row['ema21']

        if pd.isna(prev_ema9) or pd.isna(curr_ema9):
            continue

        # Bullish cross: EMA9 crosses above EMA21
        if prev_ema9 <= prev_ema21 and curr_ema9 > curr_ema21:
            if row['dmp'] - row['dmn'] >= dmp_dmn_diff:
                signals.loc[idx] = 1
        # Bearish cross
        elif prev_ema9 >= prev_ema21 and curr_ema9 < curr_ema21:
            if row['dmn'] - row['dmp'] >= dmp_dmn_diff:
                signals.loc[idx] = -1

    buy_mask = signals == 1
    sell_mask = signals == -1
    buy_mask = dedup_signals(buy_mask, min_bars=8)
    sell_mask = dedup_signals(sell_mask, min_bars=8)
    signals = pd.Series(0, index=df.index)
    signals[buy_mask] = 1
    signals[sell_mask] = -1

    return signals


# =====================================================================
# STRATEGY 3: VWAP BREAKOUT + CONFIRMATION
# =====================================================================
def gen_vwap_breakout_signals(df, vwap_dist_min=1.5, confirm_bars=2, adx_min=20,
                              vol_min=0.8, atr_max=5.0):
    """
    Signal when price breaks significantly from VWAP AND confirms direction.
    Logic:
    - Price moves > vwap_dist_min * ATR from VWAP
    - Next N bars confirm (close in same direction, above/below VWAP)
    - ADX + volume filter
    """
    signals = pd.Series(0, index=df.index)

    for i in range(confirm_bars, len(df)):
        idx = df.index[i]
        row = df.loc[idx]

        if pd.isna(row['adx']) or row['adx'] < adx_min:
            continue
        if pd.isna(row['atr']) or row['atr'] > atr_max or row['atr'] < 2.0:
            continue
        if pd.isna(row['vwap_dist']):
            continue

        mins = row['mins_in_day']
        if not ((mins >= 9*60+15 and mins <= 11*60) or (mins >= 13*60+5 and mins <= 14*60+10)):
            continue

        # Check VWAP distance
        vwap_d = row['vwap_dist']

        if abs(vwap_d) < vwap_dist_min:
            continue

        # Check confirmation: last N bars all on same side of VWAP and trending
        direction = 1 if vwap_d > 0 else -1
        confirmed = True
        for cb in range(1, confirm_bars + 1):
            prev_i = i - cb
            if prev_i < 0:
                confirmed = False
                break
            prev_idx = df.index[prev_i]
            prev_row = df.loc[prev_idx]
            if prev_row['date'] != row['date'] or prev_row['session'] != row['session']:
                confirmed = False
                break
            prev_vwap_d = prev_row['vwap_dist']
            if pd.isna(prev_vwap_d):
                confirmed = False
                break
            # All confirmation bars must be on same side and increasing distance
            if direction == 1 and prev_vwap_d <= 0:
                confirmed = False
                break
            elif direction == -1 and prev_vwap_d >= 0:
                confirmed = False
                break

        if not confirmed:
            continue

        # Volume on signal bar
        if row['vol_ratio'] < vol_min:
            continue

        signals.loc[idx] = direction

    buy_mask = signals == 1
    sell_mask = signals == -1
    buy_mask = dedup_signals(buy_mask, min_bars=10)
    sell_mask = dedup_signals(sell_mask, min_bars=10)
    signals = pd.Series(0, index=df.index)
    signals[buy_mask] = 1
    signals[sell_mask] = -1

    return signals


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    print('Loading 180d data...')
    df = prepare_data(days=180)
    n_days = df['date'].nunique()
    print(f'{len(df)} bars, {n_days} days\n')

    def print_result(label, r, n_days):
        if r:
            print(f'  {label:<52} {r["trades"]:>3} {r["win_rate"]:>5.1f} {r["pf"]:>5.2f} '
                  f'{r["total_pnl"]:>+7.1f} {r["total_pnl"]/n_days:>+5.2f} {r["avg_mfe"]:>5.1f}')
        else:
            print(f'  {label:<52}   -- no trades --')

    header = f"  {'Config':<52} {'T':>3} {'WR%':>5} {'PF':>5} {'PnL':>7} {'P/D':>5} {'MFE':>5}"

    # =================================================================
    print('=' * 90)
    print('STRATEGY 1: DUAL SUPERTREND (10,3)+(21,2)')
    print('=' * 90)
    print(header)
    print('  ' + '-' * 85)

    for adx_min in [18, 20, 22, 25]:
        for vol_min in [0.5, 0.8, 1.0]:
            sigs = gen_dual_supertrend_signals(df, adx_min=adx_min, vol_min=vol_min)
            label = f'ADX>{adx_min} Vol>{vol_min}'
            # Test with SuperTrend as trailing stop
            r = simulate_trades(df, sigs, sl_mode='supertrend', trail_mode='supertrend', max_hold=25)
            print_result(f'{label} trail=ST', r, n_days)

    # Best config with different exits
    print('\n  --- Exit optimization (ADX>20 Vol>0.8) ---')
    sigs = gen_dual_supertrend_signals(df, adx_min=20, vol_min=0.8)
    for trail, sl_m, tact, tatr in [
        ('supertrend', 'supertrend', 0, 0),
        ('ema9', 'atr', 4, 0),
        ('atr', 'atr', 4, 1.5),
        ('atr', 'atr', 5, 2.0),
        ('atr', 'atr', 6, 2.5),
    ]:
        r = simulate_trades(df, sigs, sl_mode=sl_m, trail_mode=trail,
                           trail_activate=tact, trail_atr_mult=tatr, sl_mult=1.5, max_hold=25)
        print_result(f'SL={sl_m} Trail={trail} @+{tact}/{tatr}x', r, n_days)

    # =================================================================
    print('\n' + '=' * 90)
    print('STRATEGY 2: EMA 9/21 CROSSOVER + ADX')
    print('=' * 90)
    print(header)
    print('  ' + '-' * 85)

    for adx_min in [18, 20, 22, 25, 30]:
        for di_diff in [3, 5, 8, 10]:
            sigs = gen_ema_cross_signals(df, adx_min=adx_min, dmp_dmn_diff=di_diff)
            label = f'ADX>{adx_min} DI_diff>{di_diff}'
            r = simulate_trades(df, sigs, sl_mode='atr', sl_mult=1.5,
                               trail_mode='ema9', trail_activate=3, max_hold=20)
            if r and r['trades'] >= 5:
                print_result(label, r, n_days)

    # Best config exits
    print('\n  --- Exit optimization (ADX>20 DI>5) ---')
    sigs = gen_ema_cross_signals(df, adx_min=20, dmp_dmn_diff=5)
    for trail, sl_mult, tact, tatr in [
        ('ema9', 1.5, 3, 0),
        ('ema9', 2.0, 3, 0),
        ('atr', 1.5, 4, 1.5),
        ('atr', 1.5, 5, 2.0),
        ('atr', 2.0, 5, 2.0),
    ]:
        r = simulate_trades(df, sigs, sl_mode='atr', sl_mult=sl_mult,
                           trail_mode=trail, trail_activate=tact, trail_atr_mult=tatr, max_hold=20)
        print_result(f'SL={sl_mult}xATR Trail={trail} @+{tact}/{tatr}x', r, n_days)

    # =================================================================
    print('\n' + '=' * 90)
    print('STRATEGY 3: VWAP BREAKOUT + CONFIRMATION')
    print('=' * 90)
    print(header)
    print('  ' + '-' * 85)

    for vwap_d in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for confirm in [1, 2, 3]:
            sigs = gen_vwap_breakout_signals(df, vwap_dist_min=vwap_d, confirm_bars=confirm)
            label = f'VWAP>{vwap_d}xATR Confirm={confirm}bars'
            r = simulate_trades(df, sigs, sl_mode='atr', sl_mult=1.5,
                               trail_mode='ema9', trail_activate=3, max_hold=20)
            if r and r['trades'] >= 3:
                print_result(label, r, n_days)

    # Best VWAP config exits
    print('\n  --- Exit optimization (VWAP>1.5 Confirm=2) ---')
    sigs = gen_vwap_breakout_signals(df, vwap_dist_min=1.5, confirm_bars=2)
    for trail, sl_mult, tact, tatr in [
        ('ema9', 1.5, 3, 0),
        ('ema9', 2.0, 3, 0),
        ('atr', 1.5, 4, 1.5),
        ('atr', 1.5, 5, 2.0),
        ('atr', 2.0, 5, 2.0),
        ('atr', 2.0, 6, 2.5),
    ]:
        r = simulate_trades(df, sigs, sl_mode='atr', sl_mult=sl_mult,
                           trail_mode=trail, trail_activate=tact, trail_atr_mult=tatr, max_hold=20)
        print_result(f'SL={sl_mult}xATR Trail={trail} @+{tact}/{tatr}x', r, n_days)

    # =================================================================
    # 30-DAY OOS
    # =================================================================
    print('\n\n' + '=' * 90)
    print('30-DAY OUT-OF-SAMPLE VALIDATION')
    print('=' * 90)

    df30 = prepare_data(days=30)
    nd30 = df30['date'].nunique()
    print(f'OOS: {len(df30)} bars, {nd30} days\n')
    print(header)
    print('  ' + '-' * 85)

    # Best from each strategy
    oos_tests = [
        ('ST: ADX>20 Vol>0.8 trail=ST',
         lambda d: gen_dual_supertrend_signals(d, adx_min=20, vol_min=0.8),
         dict(sl_mode='supertrend', trail_mode='supertrend', max_hold=25)),
        ('ST: ADX>22 Vol>0.8 trail=ST',
         lambda d: gen_dual_supertrend_signals(d, adx_min=22, vol_min=0.8),
         dict(sl_mode='supertrend', trail_mode='supertrend', max_hold=25)),
        ('EMA: ADX>20 DI>5 trail=EMA9',
         lambda d: gen_ema_cross_signals(d, adx_min=20, dmp_dmn_diff=5),
         dict(sl_mode='atr', sl_mult=1.5, trail_mode='ema9', trail_activate=3, max_hold=20)),
        ('EMA: ADX>22 DI>5 trail=EMA9',
         lambda d: gen_ema_cross_signals(d, adx_min=22, dmp_dmn_diff=5),
         dict(sl_mode='atr', sl_mult=1.5, trail_mode='ema9', trail_activate=3, max_hold=20)),
        ('EMA: ADX>20 DI>8 trail=ATR@5/2x',
         lambda d: gen_ema_cross_signals(d, adx_min=20, dmp_dmn_diff=8),
         dict(sl_mode='atr', sl_mult=1.5, trail_mode='atr', trail_activate=5, trail_atr_mult=2.0, max_hold=20)),
        ('VWAP: >1.5x C=2 trail=EMA9',
         lambda d: gen_vwap_breakout_signals(d, vwap_dist_min=1.5, confirm_bars=2),
         dict(sl_mode='atr', sl_mult=1.5, trail_mode='ema9', trail_activate=3, max_hold=20)),
        ('VWAP: >2.0x C=2 trail=EMA9',
         lambda d: gen_vwap_breakout_signals(d, vwap_dist_min=2.0, confirm_bars=2),
         dict(sl_mode='atr', sl_mult=1.5, trail_mode='ema9', trail_activate=3, max_hold=20)),
        ('VWAP: >1.5x C=2 trail=ATR@5/2x',
         lambda d: gen_vwap_breakout_signals(d, vwap_dist_min=1.5, confirm_bars=2),
         dict(sl_mode='atr', sl_mult=2.0, trail_mode='atr', trail_activate=5, trail_atr_mult=2.0, max_hold=20)),
    ]

    for label, sig_fn, kwargs in oos_tests:
        sigs = sig_fn(df30)
        r = simulate_trades(df30, sigs, **kwargs)
        print_result(label, r, nd30)

    # June 3 check
    print('\n\n--- JUNE 3 CHECK ---')
    target = '2026-06-03'
    for label, sig_fn, kwargs in oos_tests:
        sigs = sig_fn(df30)
        r = simulate_trades(df30, sigs, **kwargs)
        if r:
            j3 = r['df'][r['df']['date'] == target]
            if len(j3) > 0:
                for _, t in j3.iterrows():
                    print(f'  {label}: {t["time"]} {t["direction"]} @{t["entry"]:.1f}->{t["exit"]:.1f} '
                          f'MFE={t["mfe"]:.1f} PnL={t["pnl"]:+.1f} ({t["exit_reason"]})')
