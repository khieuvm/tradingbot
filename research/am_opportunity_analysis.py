"""AM Opportunity Gap Analysis — what moves are missed and what precedes them.

Goal: Understand the 82% gap in opportunity coverage.
- What do AM moves >= 5pts look like?
- What indicators are present before them?
- Can we find viable AM entry signals?

Raw approach: SL=3.0xATR, no trail, session exit.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import pandas_ta as ta

COST = 0.96


def load_data():
    print('[LOAD] Loading 1m data...')
    df = pd.read_parquet('data/vn30f1m_1m.parquet')
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['session'] = df['mins'].apply(lambda m: 'AM' if m < 720 else 'PM')

    # Indicators
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ema8'] = ta.ema(df['close'], length=8)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema50'] = ta.ema(df['close'], length=50)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14']
    df['di_plus'] = adx_df['DMP_14']
    df['di_minus'] = adx_df['DMN_14']
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd_hist'] = macd['MACDh_12_26_9']
    bb = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    df['bb_mid'] = bb['BBM_20_2.0']

    print(f'[DATA] {len(df)} bars, {df["date"].nunique()} days')
    return df


def find_am_moves(df):
    """Find AM sessions with significant directional moves."""
    results = []

    for date in df['date'].unique():
        am_mask = (df['date'] == date) & (df['mins'] >= 555) & (df['mins'] <= 685)
        am = df[am_mask].reset_index(drop=True)
        if len(am) < 20:
            continue

        # Find the actual directional move (max drawdown from start)
        prices = am['close'].values
        cummax = np.maximum.accumulate(prices)
        cummin = np.minimum.accumulate(prices)

        # BUY move: low point → subsequent high point
        low_idx = np.argmin(prices[:len(prices)//2+5]) if len(prices) > 5 else 0
        remaining_high = np.max(prices[low_idx:]) if low_idx < len(prices) else prices[-1]
        buy_move = remaining_high - prices[low_idx]

        # SELL move: high point → subsequent low point
        high_idx = np.argmax(prices[:len(prices)//2+5]) if len(prices) > 5 else 0
        remaining_low = np.min(prices[high_idx:]) if high_idx < len(prices) else prices[-1]
        sell_move = prices[high_idx] - remaining_low

        # Total session range
        total_range = am['high'].max() - am['low'].min()

        # Opening indicators (first 3 bars = 09:15-09:17)
        open_bars = am.iloc[:5] if len(am) >= 5 else am
        open_atr = open_bars['atr'].mean()
        open_adx = open_bars['adx'].iloc[-1] if not pd.isna(open_bars['adx'].iloc[-1]) else 0
        open_rsi = open_bars['rsi'].iloc[-1] if not pd.isna(open_bars['rsi'].iloc[-1]) else 50
        open_di_spread = (open_bars['di_plus'].iloc[-1] - open_bars['di_minus'].iloc[-1]) if not pd.isna(open_bars['di_plus'].iloc[-1]) else 0
        open_macd = open_bars['macd_hist'].iloc[-1] if not pd.isna(open_bars['macd_hist'].iloc[-1]) else 0
        open_ema_spread = (open_bars['ema8'].iloc[-1] - open_bars['ema21'].iloc[-1]) if not pd.isna(open_bars['ema8'].iloc[-1]) else 0

        # Price vs EMA50 at open
        price_vs_ema50 = 0
        if not pd.isna(open_bars['ema50'].iloc[-1]) and open_atr > 0:
            price_vs_ema50 = (open_bars['close'].iloc[-1] - open_bars['ema50'].iloc[-1]) / open_atr

        # First 15 min momentum (direction of opening)
        first_15 = am[am['mins'] <= 570]
        if len(first_15) >= 10:
            first_15_ret = first_15['close'].iloc[-1] - first_15['open'].iloc[0]
        else:
            first_15_ret = 0

        results.append({
            'date': str(date),
            'buy_move': buy_move,
            'sell_move': sell_move,
            'total_range': total_range,
            'best_direction': 'BUY' if buy_move > sell_move else 'SELL',
            'best_move': max(buy_move, sell_move),
            'open_atr': open_atr,
            'open_adx': open_adx,
            'open_rsi': open_rsi,
            'open_di_spread': open_di_spread,
            'open_macd': open_macd,
            'open_ema_spread': open_ema_spread,
            'price_vs_ema50': price_vs_ema50,
            'first_15_ret': first_15_ret,
            'n_bars': len(am),
        })

    return pd.DataFrame(results)


def analyze_move_patterns(moves_df):
    """What indicators precede large AM moves?"""
    print()
    print('=' * 70)
    print('  AM OPPORTUNITY GAP ANALYSIS')
    print('=' * 70)
    print()

    big = moves_df[moves_df['best_move'] >= 5.0]
    small = moves_df[moves_df['best_move'] < 3.0]

    print(f'Total AM sessions: {len(moves_df)}')
    print(f'Sessions with >= 5pt move: {len(big)} ({len(big)/len(moves_df)*100:.1f}%)')
    print(f'Sessions with < 3pt move: {len(small)} ({len(small)/len(moves_df)*100:.1f}%)')
    print(f'Avg best_move: {moves_df["best_move"].mean():.2f} pts')
    print()

    # Direction distribution
    big_buy = big[big['best_direction'] == 'BUY']
    big_sell = big[big['best_direction'] == 'SELL']
    print(f'Big moves direction: BUY={len(big_buy)} ({len(big_buy)/len(big)*100:.0f}%) '
          f'SELL={len(big_sell)} ({len(big_sell)/len(big)*100:.0f}%)')
    print()

    # Compare indicators: big vs small moves
    print('INDICATOR COMPARISON: Big moves (>=5pt) vs Small moves (<3pt)')
    print(f'  {"Indicator":<20} {"Big (mean)":<12} {"Small (mean)":<12} {"Diff":>8} {"Cohen d":>8}')
    print(f'  {"-"*60}')

    indicators = ['open_atr', 'open_adx', 'open_rsi', 'open_di_spread',
                  'open_macd', 'open_ema_spread', 'price_vs_ema50', 'first_15_ret']

    for ind in indicators:
        big_vals = big[ind].dropna()
        small_vals = small[ind].dropna()
        if len(big_vals) < 5 or len(small_vals) < 5:
            continue
        big_mean = big_vals.mean()
        small_mean = small_vals.mean()
        diff = big_mean - small_mean
        pooled_std = np.sqrt((big_vals.std()**2 + small_vals.std()**2) / 2)
        d = diff / pooled_std if pooled_std > 0 else 0
        print(f'  {ind:<20} {big_mean:>10.3f}  {small_mean:>10.3f}  {diff:>8.3f} {d:>8.3f}')

    # First 15-min momentum as predictor
    print()
    print('FIRST 15-MIN MOMENTUM AS DIRECTION PREDICTOR:')
    correct = 0
    total = 0
    for _, row in big.iterrows():
        if abs(row['first_15_ret']) < 0.5:
            continue
        total += 1
        momentum_dir = 'BUY' if row['first_15_ret'] > 0 else 'SELL'
        if momentum_dir == row['best_direction']:
            correct += 1
    if total > 0:
        print(f'  First 15min agrees with best direction: {correct}/{total} = {correct/total*100:.1f}%')
    print()

    # Opening momentum entry test
    print('OPENING MOMENTUM ENTRY TEST (trade in direction of first 15-min move):')
    return big


def backtest_opening_momentum(df, moves_df, min_first_15=1.0):
    """Backtest: enter in direction of first 15-min momentum if > threshold."""
    trades = []

    for _, row in moves_df.iterrows():
        date = pd.Timestamp(row['date']).date()
        am_mask = (df['date'] == date) & (df['mins'] >= 555) & (df['mins'] <= 685)
        am = df[am_mask].reset_index(drop=True)
        if len(am) < 30:
            continue

        # Wait for first 15 mins (09:15 - 09:30 = mins 555-570)
        first_15 = am[am['mins'] <= 570]
        if len(first_15) < 10:
            continue

        first_15_ret = first_15['close'].iloc[-1] - first_15['open'].iloc[0]
        if abs(first_15_ret) < min_first_15:
            continue

        # Entry after 09:30 (first bar after 570)
        entry_mask = am['mins'] > 570
        entry_bars = am[entry_mask]
        if len(entry_bars) < 5:
            continue

        entry_idx = entry_bars.index[0]
        entry_price = am.loc[entry_idx, 'close']
        atr = am.loc[entry_idx, 'atr']
        if pd.isna(atr) or atr <= 0:
            continue

        direction = 1 if first_15_ret > 0 else -1
        sl_price = entry_price - direction * 3.0 * atr

        exit_price = entry_price
        exit_reason = 'MAX_HOLD'

        for j in range(entry_bars.index.get_loc(entry_idx) + 1, len(entry_bars)):
            bar = entry_bars.iloc[j]
            if bar['mins'] >= 685:
                exit_price = bar['close']
                exit_reason = 'SESSION'
                break
            if direction == 1 and bar['low'] <= sl_price:
                exit_price = sl_price
                exit_reason = 'SL'
                break
            if direction == -1 and bar['high'] >= sl_price:
                exit_price = sl_price
                exit_reason = 'SL'
                break
        else:
            exit_price = entry_bars.iloc[-1]['close']
            exit_reason = 'SESSION'

        pnl = (exit_price - entry_price) * direction - COST
        trades.append({
            'date': str(date),
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'atr': atr,
            'first_15_ret': first_15_ret,
        })

    return trades


def backtest_ema_trend_am(df):
    """Backtest: AM entry when EMA8 > EMA21 (BUY) or EMA8 < EMA21 (SELL) + ADX > 20."""
    trades = []

    for date in df['date'].unique():
        am_mask = (df['date'] == date) & (df['mins'] >= 570) & (df['mins'] <= 685)
        am = df[am_mask].reset_index(drop=True)
        if len(am) < 20:
            continue

        # Check at 09:30 (after market settles)
        check_bar = am.iloc[0]
        atr = check_bar['atr']
        adx = check_bar['adx']
        ema8 = check_bar['ema8']
        ema21 = check_bar['ema21']

        if pd.isna(atr) or atr <= 0 or pd.isna(adx) or pd.isna(ema8) or pd.isna(ema21):
            continue

        if adx < 20:
            continue

        if abs(ema8 - ema21) < 0.3 * atr:
            continue

        direction = 1 if ema8 > ema21 else -1
        entry_price = check_bar['close']
        sl_price = entry_price - direction * 3.0 * atr

        exit_price = entry_price
        exit_reason = 'MAX_HOLD'

        for j in range(1, len(am)):
            bar = am.iloc[j]
            if bar['mins'] >= 685:
                exit_price = bar['close']
                exit_reason = 'SESSION'
                break
            if direction == 1 and bar['low'] <= sl_price:
                exit_price = sl_price
                exit_reason = 'SL'
                break
            if direction == -1 and bar['high'] >= sl_price:
                exit_price = sl_price
                exit_reason = 'SL'
                break
        else:
            exit_price = am.iloc[-1]['close']
            exit_reason = 'SESSION'

        pnl = (exit_price - entry_price) * direction - COST
        trades.append({
            'date': str(date),
            'direction': 'BUY' if direction == 1 else 'SELL',
            'pnl': pnl,
            'exit_reason': exit_reason,
            'adx': adx,
            'atr': atr,
        })

    return trades


def backtest_breakout_am(df, lookback_mins=15, threshold_mult=0.8):
    """Backtest: AM breakout above/below first N-min range."""
    trades = []

    for date in df['date'].unique():
        am_mask = (df['date'] == date) & (df['mins'] >= 555) & (df['mins'] <= 685)
        am = df[am_mask].reset_index(drop=True)
        if len(am) < 30:
            continue

        # First N-min range (09:15 - 09:30)
        opening = am[am['mins'] <= 555 + lookback_mins]
        if len(opening) < 5:
            continue

        range_high = opening['high'].max()
        range_low = opening['low'].min()
        range_size = range_high - range_low

        atr = opening['atr'].iloc[-1]
        if pd.isna(atr) or atr <= 0:
            continue

        if range_size < threshold_mult * atr:
            continue  # Range too tight, not a valid setup

        # Wait for breakout after opening period
        post_open = am[am['mins'] > 555 + lookback_mins]
        if len(post_open) < 5:
            continue

        traded = False
        for j in range(len(post_open)):
            bar = post_open.iloc[j]
            if traded:
                break

            direction = 0
            if bar['high'] > range_high + 0.1:
                direction = 1
            elif bar['low'] < range_low - 0.1:
                direction = -1

            if direction == 0:
                continue

            traded = True
            entry_price = range_high + 0.1 if direction == 1 else range_low - 0.1
            sl_price = entry_price - direction * 3.0 * atr

            exit_price = entry_price
            for k in range(j + 1, len(post_open)):
                kbar = post_open.iloc[k]
                if kbar['mins'] >= 685:
                    exit_price = kbar['close']
                    break
                if direction == 1 and kbar['low'] <= sl_price:
                    exit_price = sl_price
                    break
                if direction == -1 and kbar['high'] >= sl_price:
                    exit_price = sl_price
                    break
            else:
                exit_price = post_open.iloc[-1]['close']

            pnl = (exit_price - entry_price) * direction - COST
            trades.append({
                'date': str(date),
                'direction': 'BUY' if direction == 1 else 'SELL',
                'pnl': pnl,
                'range_size': range_size,
                'atr': atr,
            })

    return trades


def print_backtest_results(trades, label):
    """Print standard backtest metrics."""
    if not trades:
        print(f'  {label}: No trades')
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
    per_day = total / 682

    print(f'  {label}:')
    print(f'    N={n}, WR={wr:.1f}%, PF={pf:.2f}, Total={total:.1f}pts, /day={per_day:.2f}')

    # By direction
    buy_pnls = [t['pnl'] for t in trades if t['direction'] == 'BUY']
    sell_pnls = [t['pnl'] for t in trades if t['direction'] == 'SELL']
    if buy_pnls:
        buy_wr = len([p for p in buy_pnls if p > 0]) / len(buy_pnls) * 100
        print(f'    BUY: N={len(buy_pnls)}, WR={buy_wr:.1f}%, PnL={sum(buy_pnls):.1f}')
    if sell_pnls:
        sell_wr = len([p for p in sell_pnls if p > 0]) / len(sell_pnls) * 100
        print(f'    SELL: N={len(sell_pnls)}, WR={sell_wr:.1f}%, PnL={sum(sell_pnls):.1f}')


def main():
    df = load_data()

    # 1. Find and analyze AM moves
    moves_df = find_am_moves(df)
    big_moves = analyze_move_patterns(moves_df)

    # 2. Backtest opening momentum
    print()
    print('=' * 70)
    print('  AM STRATEGY BACKTESTS (SL=3.0xATR, session exit, 682d)')
    print('=' * 70)
    print()

    # Opening momentum with different thresholds
    for thresh in [0.5, 1.0, 1.5, 2.0]:
        trades = backtest_opening_momentum(df, moves_df, min_first_15=thresh)
        print_backtest_results(trades, f'Opening Momentum (first_15 > {thresh}pt)')

    print()

    # EMA trend AM
    trades_ema = backtest_ema_trend_am(df)
    print_backtest_results(trades_ema, 'EMA Trend AM (EMA8 vs EMA21, ADX>20)')

    print()

    # Breakout of opening range
    for lb in [15, 20, 30]:
        for thr in [0.5, 0.8, 1.0]:
            trades_bo = backtest_breakout_am(df, lookback_mins=lb, threshold_mult=thr)
            if trades_bo and len(trades_bo) >= 15:
                pnls = [t['pnl'] for t in trades_bo]
                wins = [p for p in pnls if p > 0]
                gl = abs(sum([p for p in pnls if p <= 0])) or 0.001
                pf = sum(wins) / gl if wins else 0
                if pf > 1.0:
                    print_backtest_results(trades_bo, f'Opening Range Breakout ({lb}min, thr={thr}xATR)')

    # Summary: what's the best AM approach?
    print()
    print('=' * 70)
    print('  TIMING ANALYSIS: When do big AM moves start?')
    print('=' * 70)
    print()

    # Find the minute where the big move begins
    timing_data = []
    for _, row in big_moves.iterrows():
        date = pd.Timestamp(row['date']).date()
        am_mask = (df['date'] == date) & (df['mins'] >= 555) & (df['mins'] <= 685)
        am = df[am_mask].reset_index(drop=True)
        if len(am) < 20:
            continue

        prices = am['close'].values
        if row['best_direction'] == 'BUY':
            # Find the low before the big BUY move
            low_idx = np.argmin(prices[:len(prices)//2+10])
            move_start_min = am.iloc[low_idx]['mins']
        else:
            # Find the high before the big SELL move
            high_idx = np.argmax(prices[:len(prices)//2+10])
            move_start_min = am.iloc[high_idx]['mins']

        timing_data.append({
            'date': row['date'],
            'direction': row['best_direction'],
            'move_start_min': move_start_min,
            'move_size': row['best_move'],
        })

    if timing_data:
        tdf = pd.DataFrame(timing_data)
        print(f'Big AM moves (>=5pt) start time distribution:')
        print(f'  Mean: {tdf["move_start_min"].mean():.0f} min ({int(tdf["move_start_min"].mean()//60):02d}:{int(tdf["move_start_min"].mean()%60):02d})')
        print(f'  Median: {tdf["move_start_min"].median():.0f} min ({int(tdf["move_start_min"].median()//60):02d}:{int(tdf["move_start_min"].median()%60):02d})')
        print(f'  25th pct: {tdf["move_start_min"].quantile(0.25):.0f}')
        print(f'  75th pct: {tdf["move_start_min"].quantile(0.75):.0f}')
        print()

        # Bucket by time
        buckets = [(555, 570, '09:15-09:30'), (570, 585, '09:30-09:45'),
                   (585, 600, '09:45-10:00'), (600, 615, '10:00-10:15'),
                   (615, 630, '10:15-10:30'), (630, 645, '10:30-10:45')]
        for start, end, label in buckets:
            n = len(tdf[(tdf['move_start_min'] >= start) & (tdf['move_start_min'] < end)])
            print(f'  {label}: {n} moves ({n/len(tdf)*100:.1f}%)')


if __name__ == '__main__':
    main()
