"""Research: Opening strategies for VN30F1M AM session.

Tests on 23 days of 1m data (2026-05-18 to 2026-06-17):
1. Crabel Stretch (open +/- SMA10 of daily range)
2. Initial Balance Breakout (60min range, trade after 10:00)
3. Opening Drive (strong first 5min bar -> continuation)
4. Opening Reversal (gap + first bar reversal)
5. Narrow ORB (tight opening range -> expect expansion)
6. Day-of-week analysis (all strategies split by weekday)
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import pandas_ta as ta
from src.data_fetcher import DataFetcher

COST = 0.96


def load_data():
    fetcher = DataFetcher()
    df = fetcher.get_futures_ohlcv('VN30F1M', '2026-05-02', '2026-06-17', interval='1m')
    if df is None or len(df) == 0:
        print('No data')
        sys.exit(1)

    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['session'] = df['mins'].apply(lambda m: 'AM' if m < 720 else 'PM')
    df['weekday'] = df['time'].dt.weekday  # 0=Mon, 4=Fri
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema8'] = ta.ema(df['close'], length=8)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema50'] = ta.ema(df['close'], length=50)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14']
    df['di_plus'] = adx_df['DMP_14']
    df['di_minus'] = adx_df['DMN_14']
    df['rsi'] = ta.rsi(df['close'], length=14)

    print(f'[DATA] {len(df)} bars, {df["date"].nunique()} days')
    print(f'  Range: {df["time"].iloc[0].date()} to {df["time"].iloc[-1].date()}')
    return df


def simulate_exit(df, entry_idx, direction, sl_price):
    """Simulate: hold until SL hit or session end."""
    row = df.iloc[entry_idx]
    session = row['session']
    date = row['date']

    for j in range(entry_idx + 1, min(entry_idx + 200, len(df))):
        bar = df.iloc[j]
        if bar['date'] != date or bar['session'] != session:
            return df.iloc[j - 1]['close'], 'SESSION', df.iloc[j - 1]['time']
        if session == 'AM' and bar['mins'] >= 685:
            return bar['close'], 'SESSION', bar['time']
        if session == 'PM' and bar['mins'] >= 865:
            return bar['close'], 'SESSION', bar['time']
        if direction == 1 and bar['low'] <= sl_price:
            return sl_price, 'SL', bar['time']
        if direction == -1 and bar['high'] >= sl_price:
            return sl_price, 'SL', bar['time']

    return df.iloc[min(entry_idx + 199, len(df) - 1)]['close'], 'MAX', df.iloc[min(entry_idx + 199, len(df) - 1)]['time']


def report(trades, label, n_days=None):
    if not trades:
        print(f'  {label}: No trades')
        print()
        return
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n = len(pnls)
    wr = len(wins) / n * 100
    gw = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0.001
    pf = gw / gl
    total = sum(pnls)
    days = n_days or 23
    per_day = total / days

    print(f'  {label}:')
    print(f'    N={n}, WR={wr:.1f}%, PF={pf:.2f}, Total={total:+.1f}pts, /day={per_day:+.2f}')
    if wins:
        print(f'    Avg W={np.mean(wins):.2f}, Avg L={np.mean(losses):.2f}, '
              f'Best={max(pnls):+.1f}, Worst={min(pnls):+.1f}')
    print()


def daily_ranges(df):
    """Compute daily session ranges for Crabel Stretch."""
    ranges = {}
    for date in sorted(df['date'].unique()):
        day_df = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 690)]
        if len(day_df) < 10:
            continue
        ranges[date] = day_df['high'].max() - day_df['low'].min()
    return ranges


def get_prev_close(df, date):
    """Previous session's last close."""
    dates = sorted(df['date'].unique())
    idx = list(dates).index(date) if date in dates else -1
    if idx <= 0:
        return None
    prev_date = dates[idx - 1]
    prev_bars = df[(df['date'] == prev_date)]
    if len(prev_bars) == 0:
        return None
    return prev_bars['close'].iloc[-1]


def main():
    df = load_data()
    dates = sorted(df['date'].unique())
    n_days = len(dates)

    # Pre-compute daily ranges for Stretch
    d_ranges = daily_ranges(df)
    dates_list = sorted(d_ranges.keys())

    print()
    print('=' * 75)
    print(f'  OPENING STRATEGIES RESEARCH ({n_days} days)')
    print('=' * 75)
    print()

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 1: Crabel Stretch
    # Entry = open +/- SMA(10) of daily AM ranges
    # ═══════════════════════════════════════════════════════════════════
    print('--- 1. CRABEL STRETCH (open +/- avg daily range) ---')
    print()

    for stretch_lookback in [5, 10]:
        for direction_filter in ['BOTH', 'SELL', 'BUY']:
            trades = []
            for i, date in enumerate(dates_list):
                if i < stretch_lookback:
                    continue
                # Compute stretch = SMA of last N daily ranges
                recent_ranges = [d_ranges[dates_list[k]] for k in range(i - stretch_lookback, i)]
                stretch = np.mean(recent_ranges)

                am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
                if len(am) < 20:
                    continue

                am_open = am['open'].iloc[0]
                atr = am['atr'].iloc[10] if len(am) > 10 and not pd.isna(am['atr'].iloc[10]) else None
                if atr is None or atr <= 0:
                    continue

                # Look for price crossing open +/- stretch
                long_trigger = am_open + stretch * 0.5  # Use 50% of stretch as entry
                short_trigger = am_open - stretch * 0.5

                traded = False
                for j in range(5, len(am)):
                    if traded:
                        break
                    bar = am.iloc[j]

                    if direction_filter != 'BUY' and bar['low'] < short_trigger:
                        # SELL trigger
                        traded = True
                        entry_price = short_trigger
                        sl = am_open  # Stop at today's open
                        exit_p, exit_r, exit_t = simulate_exit(df, am.index[j], -1, sl)
                        pnl = (entry_price - exit_p) - COST
                        trades.append({'date': str(date), 'pnl': pnl, 'direction': 'SELL',
                                       'entry': entry_price, 'exit': exit_p, 'exit_reason': exit_r,
                                       'weekday': am.iloc[j]['weekday']})

                    elif direction_filter != 'SELL' and bar['high'] > long_trigger:
                        # BUY trigger
                        traded = True
                        entry_price = long_trigger
                        sl = am_open
                        exit_p, exit_r, exit_t = simulate_exit(df, am.index[j], 1, sl)
                        pnl = (exit_p - entry_price) - COST
                        trades.append({'date': str(date), 'pnl': pnl, 'direction': 'BUY',
                                       'entry': entry_price, 'exit': exit_p, 'exit_reason': exit_r,
                                       'weekday': am.iloc[j]['weekday']})

            if trades:
                report(trades, f'Stretch(SMA{stretch_lookback}, 50%) {direction_filter}', n_days)

    # Also test with wider SL (3.0xATR instead of open)
    print('  --- Stretch with SL=3.0xATR (wider) ---')
    trades_wide = []
    for i, date in enumerate(dates_list):
        if i < 10:
            continue
        recent_ranges = [d_ranges[dates_list[k]] for k in range(i - 10, i)]
        stretch = np.mean(recent_ranges)

        am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
        if len(am) < 20:
            continue

        am_open = am['open'].iloc[0]
        atr = am['atr'].iloc[10] if len(am) > 10 and not pd.isna(am['atr'].iloc[10]) else None
        if atr is None or atr <= 0:
            continue

        short_trigger = am_open - stretch * 0.5

        traded = False
        for j in range(5, len(am)):
            if traded:
                break
            bar = am.iloc[j]
            if bar['low'] < short_trigger:
                traded = True
                entry_price = short_trigger
                sl = entry_price + 3.0 * atr
                exit_p, exit_r, exit_t = simulate_exit(df, am.index[j], -1, sl)
                pnl = (entry_price - exit_p) - COST
                trades_wide.append({'date': str(date), 'pnl': pnl, 'direction': 'SELL',
                                    'entry': entry_price, 'exit': exit_p, 'exit_reason': exit_r,
                                    'weekday': am.iloc[j]['weekday']})

    report(trades_wide, 'Stretch SELL (SMA10, 50%, SL=3xATR)', n_days)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 2: Initial Balance Breakout (first 60min range)
    # ═══════════════════════════════════════════════════════════════════
    print()
    print('--- 2. INITIAL BALANCE BREAKOUT (60min range, trade after 10:00) ---')
    print()

    for ib_filter in ['none', 'narrow', 'wide']:
        for direction_filter in ['BOTH', 'SELL', 'BUY']:
            trades = []
            for date in dates:
                am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
                if len(am) < 80:
                    continue

                # IB = first 60 bars (9:00-9:59)
                ib = am[am['mins'] < 600]
                if len(ib) < 50:
                    continue

                ib_high = ib['high'].max()
                ib_low = ib['low'].min()
                ib_range = ib_high - ib_low
                atr = ib['atr'].iloc[-1]

                if pd.isna(atr) or atr <= 0:
                    continue

                # IB filter
                if ib_filter == 'narrow' and ib_range >= 1.0 * atr:
                    continue
                if ib_filter == 'wide' and ib_range < 1.0 * atr:
                    continue

                # Post-IB bars (10:00 - 11:25)
                post_ib = am[am['mins'] >= 600]
                if len(post_ib) < 10:
                    continue

                traded = False
                for j in range(len(post_ib)):
                    if traded:
                        break
                    bar = post_ib.iloc[j]

                    if direction_filter != 'BUY' and bar['close'] < ib_low:
                        traded = True
                        entry_price = ib_low
                        sl = ib_high  # Stop at opposite end
                        exit_p, exit_r, exit_t = simulate_exit(df, post_ib.index[j], -1, sl)
                        pnl = (entry_price - exit_p) - COST
                        trades.append({'date': str(date), 'pnl': pnl, 'direction': 'SELL',
                                       'exit_reason': exit_r, 'weekday': bar['weekday'],
                                       'ib_range': ib_range, 'atr': atr})

                    elif direction_filter != 'SELL' and bar['close'] > ib_high:
                        traded = True
                        entry_price = ib_high
                        sl = ib_low
                        exit_p, exit_r, exit_t = simulate_exit(df, post_ib.index[j], 1, sl)
                        pnl = (exit_p - entry_price) - COST
                        trades.append({'date': str(date), 'pnl': pnl, 'direction': 'BUY',
                                       'exit_reason': exit_r, 'weekday': bar['weekday'],
                                       'ib_range': ib_range, 'atr': atr})

            if trades and len(trades) >= 5:
                report(trades, f'IB Breakout ({ib_filter} IB) {direction_filter}', n_days)

    # IB with 3.0xATR SL (same as RAW exit approach)
    print('  --- IB Breakout with SL=3.0xATR ---')
    for ib_filter in ['narrow']:
        for direction_filter in ['BOTH', 'SELL']:
            trades = []
            for date in dates:
                am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
                if len(am) < 80:
                    continue
                ib = am[am['mins'] < 600]
                if len(ib) < 50:
                    continue
                ib_high = ib['high'].max()
                ib_low = ib['low'].min()
                ib_range = ib_high - ib_low
                atr = ib['atr'].iloc[-1]
                if pd.isna(atr) or atr <= 0:
                    continue
                if ib_filter == 'narrow' and ib_range >= 1.0 * atr:
                    continue

                post_ib = am[am['mins'] >= 600]
                if len(post_ib) < 10:
                    continue

                traded = False
                for j in range(len(post_ib)):
                    if traded:
                        break
                    bar = post_ib.iloc[j]
                    if direction_filter != 'BUY' and bar['close'] < ib_low:
                        traded = True
                        entry_price = ib_low
                        sl = entry_price + 3.0 * atr
                        exit_p, exit_r, exit_t = simulate_exit(df, post_ib.index[j], -1, sl)
                        pnl = (entry_price - exit_p) - COST
                        trades.append({'date': str(date), 'pnl': pnl, 'direction': 'SELL',
                                       'exit_reason': exit_r, 'weekday': bar['weekday']})
                    elif direction_filter != 'SELL' and bar['close'] > ib_high:
                        traded = True
                        entry_price = ib_high
                        sl = entry_price - 3.0 * atr
                        exit_p, exit_r, exit_t = simulate_exit(df, post_ib.index[j], 1, sl)
                        pnl = (exit_p - entry_price) - COST
                        trades.append({'date': str(date), 'pnl': pnl, 'direction': 'BUY',
                                       'exit_reason': exit_r, 'weekday': bar['weekday']})

            if trades and len(trades) >= 3:
                report(trades, f'IB({ib_filter}) {direction_filter} SL=3xATR', n_days)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 3: Opening Drive (strong first 5min bar -> continuation)
    # ═══════════════════════════════════════════════════════════════════
    print()
    print('--- 3. OPENING DRIVE (strong first bar -> continuation) ---')
    print()

    for bar_count in [5, 10]:
        for body_threshold in [0.5, 0.7]:
            trades = []
            for date in dates:
                am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
                if len(am) < 30:
                    continue

                first_bars = am.iloc[:bar_count]
                if len(first_bars) < bar_count:
                    continue

                body = first_bars['close'].iloc[-1] - first_bars['open'].iloc[0]
                rng = first_bars['high'].max() - first_bars['low'].min()
                atr = am['atr'].iloc[bar_count] if len(am) > bar_count and not pd.isna(am['atr'].iloc[bar_count]) else None
                if atr is None or atr <= 0 or rng == 0:
                    continue

                body_ratio = abs(body) / rng
                body_atr_ratio = abs(body) / atr

                # Strong directional first bar: body > threshold*ATR and body/range > 60%
                if body_atr_ratio < body_threshold or body_ratio < 0.6:
                    continue

                direction = 1 if body > 0 else -1
                entry_idx = am.index[bar_count]
                entry_price = am.iloc[bar_count]['close']
                sl = entry_price - direction * 3.0 * atr

                exit_p, exit_r, exit_t = simulate_exit(df, entry_idx, direction, sl)
                pnl = (exit_p - entry_price) * direction - COST
                trades.append({'date': str(date), 'pnl': pnl,
                               'direction': 'BUY' if direction == 1 else 'SELL',
                               'body_atr': body_atr_ratio, 'exit_reason': exit_r,
                               'weekday': am.iloc[bar_count]['weekday']})

            if trades and len(trades) >= 5:
                report(trades, f'Opening Drive ({bar_count}bar, body>{body_threshold}xATR)', n_days)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 4: Opening Reversal (gap + first bar reverses)
    # ═══════════════════════════════════════════════════════════════════
    print()
    print('--- 4. OPENING REVERSAL (gap + first bar in opposite direction) ---')
    print()

    for gap_threshold in [0.3, 0.5, 1.0]:
        trades = []
        for date in dates:
            prev_close = get_prev_close(df, date)
            if prev_close is None:
                continue

            am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
            if len(am) < 20:
                continue

            am_open = am['open'].iloc[0]
            atr = am['atr'].iloc[5] if len(am) > 5 and not pd.isna(am['atr'].iloc[5]) else None
            if atr is None or atr <= 0:
                continue

            gap = am_open - prev_close
            gap_atr = gap / atr

            if abs(gap_atr) < gap_threshold:
                continue

            # First 5 bars direction
            first5_close = am['close'].iloc[4] if len(am) > 4 else am['close'].iloc[-1]
            first5_ret = first5_close - am_open

            # Reversal: gap up but first bars close below open (or vice versa)
            if gap > 0 and first5_ret >= 0:
                continue  # No reversal
            if gap < 0 and first5_ret <= 0:
                continue  # No reversal

            # Trade opposite to gap (fade the gap)
            direction = -1 if gap > 0 else 1
            entry_idx = am.index[5]
            entry_price = am.iloc[5]['close']
            sl = entry_price - direction * 3.0 * atr

            exit_p, exit_r, exit_t = simulate_exit(df, entry_idx, direction, sl)
            pnl = (exit_p - entry_price) * direction - COST
            trades.append({'date': str(date), 'pnl': pnl,
                           'direction': 'BUY' if direction == 1 else 'SELL',
                           'gap_atr': gap_atr, 'exit_reason': exit_r,
                           'weekday': am.iloc[5]['weekday']})

        if trades and len(trades) >= 3:
            report(trades, f'Opening Reversal (gap>{gap_threshold}xATR + 5bar fade)', n_days)

    # ═══════════════════════════════════════════════════════════════════
    # STRATEGY 5: Narrow ORB (tight opening range -> expansion)
    # ═══════════════════════════════════════════════════════════════════
    print()
    print('--- 5. NARROW ORB (tight 15min range -> breakout) ---')
    print()

    for range_threshold in [0.5, 0.7, 1.0]:
        for direction_filter in ['BOTH', 'SELL']:
            trades = []
            for date in dates:
                am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
                if len(am) < 30:
                    continue

                # First 15 bars
                opening = am.iloc[:15]
                or_high = opening['high'].max()
                or_low = opening['low'].min()
                or_range = or_high - or_low
                atr = opening['atr'].iloc[-1]

                if pd.isna(atr) or atr <= 0:
                    continue

                # Narrow filter: OR range < threshold * ATR
                if or_range >= range_threshold * atr:
                    continue

                # Wait for breakout
                post = am.iloc[15:]
                if len(post) < 5:
                    continue

                traded = False
                for j in range(len(post)):
                    if traded:
                        break
                    bar = post.iloc[j]
                    if bar['mins'] > 645:
                        break  # Don't enter after 10:45

                    if direction_filter != 'BUY' and bar['close'] < or_low:
                        traded = True
                        entry_price = or_low
                        sl = entry_price + 3.0 * atr
                        exit_p, exit_r, exit_t = simulate_exit(df, post.index[j], -1, sl)
                        pnl = (entry_price - exit_p) - COST
                        trades.append({'date': str(date), 'pnl': pnl, 'direction': 'SELL',
                                       'or_range': or_range, 'exit_reason': exit_r,
                                       'weekday': bar['weekday']})

                    elif direction_filter != 'SELL' and bar['close'] > or_high:
                        traded = True
                        entry_price = or_high
                        sl = entry_price - 3.0 * atr
                        exit_p, exit_r, exit_t = simulate_exit(df, post.index[j], 1, sl)
                        pnl = (exit_p - entry_price) - COST
                        trades.append({'date': str(date), 'pnl': pnl, 'direction': 'BUY',
                                       'or_range': or_range, 'exit_reason': exit_r,
                                       'weekday': bar['weekday']})

            if trades and len(trades) >= 3:
                report(trades, f'Narrow ORB (OR<{range_threshold}xATR) {direction_filter}', n_days)

    # ═══════════════════════════════════════════════════════════════════
    # DAY-OF-WEEK ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print()
    print('=' * 75)
    print('  DAY-OF-WEEK ANALYSIS')
    print('=' * 75)
    print()

    # Collect all Opening Range Breakdown trades (best strategy from earlier)
    all_orb_trades = []
    for date in dates:
        am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
        if len(am) < 20:
            continue
        opening = am.iloc[:15]
        or_low = opening['low'].min()
        or_high = opening['high'].max()
        atr = opening['atr'].iloc[-1]
        if pd.isna(atr) or atr <= 0:
            continue

        post = am.iloc[15:]
        if len(post) < 5:
            continue

        traded = False
        for j in range(len(post)):
            if traded:
                break
            bar = post.iloc[j]
            if bar['mins'] > 645:
                break

            direction = 0
            if bar['close'] < or_low:
                direction = -1
                entry_price = or_low
            elif bar['close'] > or_high:
                direction = 1
                entry_price = or_high

            if direction != 0:
                traded = True
                sl = entry_price - direction * 3.0 * atr
                exit_p, exit_r, exit_t = simulate_exit(df, post.index[j], direction, sl)
                pnl = (exit_p - entry_price) * direction - COST
                all_orb_trades.append({
                    'date': str(date), 'pnl': pnl,
                    'direction': 'BUY' if direction == 1 else 'SELL',
                    'weekday': bar['weekday'],
                    'weekday_name': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][bar['weekday']],
                })

    # Also: AM session net move by weekday
    print('AM Session Net Move by Weekday:')
    print(f'  {"Day":<5} {"N":>3} {"Avg Net":>8} {"Avg Range":>9} {"Drop>3":>7} {"Rise>3":>7}')
    print(f'  {"-"*45}')

    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    for dow in range(5):
        day_dates = [d for d in dates if pd.Timestamp(d).weekday() == dow]
        if not day_dates:
            continue
        nets = []
        ranges_ = []
        drops = 0
        rises = 0
        for date in day_dates:
            am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
            if len(am) < 10:
                continue
            net = am['close'].iloc[-1] - am['open'].iloc[0]
            rng = am['high'].max() - am['low'].min()
            nets.append(net)
            ranges_.append(rng)
            if net < -3:
                drops += 1
            if net > 3:
                rises += 1
        if nets:
            print(f'  {dow_names[dow]:<5} {len(nets):>3} {np.mean(nets):>+8.2f} {np.mean(ranges_):>9.1f} '
                  f'{drops:>7} {rises:>7}')

    print()

    # ORB trades by weekday
    if all_orb_trades:
        print('Opening Range Breakout by Weekday:')
        print(f'  {"Day":<5} {"N":>3} {"WR":>5} {"PF":>5} {"PnL":>7}')
        print(f'  {"-"*30}')

        for dow in range(5):
            dow_trades = [t for t in all_orb_trades if t['weekday'] == dow]
            if not dow_trades:
                continue
            pnls = [t['pnl'] for t in dow_trades]
            wins = [p for p in pnls if p > 0]
            gl = abs(sum([p for p in pnls if p <= 0])) or 0.001
            pf = sum(wins) / gl if wins else 0
            wr = len(wins) / len(pnls) * 100
            print(f'  {dow_names[dow]:<5} {len(pnls):>3} {wr:>5.0f}% {pf:>5.2f} {sum(pnls):>+7.1f}')

        # Direction by weekday
        print()
        print('  Breakdown by direction:')
        for dow in range(5):
            dow_sell = [t for t in all_orb_trades if t['weekday'] == dow and t['direction'] == 'SELL']
            dow_buy = [t for t in all_orb_trades if t['weekday'] == dow and t['direction'] == 'BUY']
            if dow_sell or dow_buy:
                sell_pnl = sum(t['pnl'] for t in dow_sell) if dow_sell else 0
                buy_pnl = sum(t['pnl'] for t in dow_buy) if dow_buy else 0
                print(f'    {dow_names[dow]}: SELL n={len(dow_sell)} PnL={sell_pnl:+.1f} | '
                      f'BUY n={len(dow_buy)} PnL={buy_pnl:+.1f}')

    # ═══════════════════════════════════════════════════════════════════
    # COMBINED BEST: Narrow ORB SELL on specific weekdays
    # ═══════════════════════════════════════════════════════════════════
    print()
    print('=' * 75)
    print('  BEST COMBINED STRATEGY (top strategy + best weekday filter)')
    print('=' * 75)
    print()

    # Find which weekdays work best for each top strategy
    # Combine with direction filter


if __name__ == '__main__':
    main()
