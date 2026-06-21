"""Research AM drop patterns and backtest entry signals to catch them.

Today (17/06): AM dropped -8.9pts. How to catch these moves early?
Key challenge: 60% of big AM moves start by 09:30 (too early for indicators).

Approaches tested:
1. Gap-and-go: Open below prev close -> SELL
2. First 5-min bearish bar: SELL on close if range > 1.5xATR
3. Opening range breakdown (15min): SELL on break below
4. Early momentum: First 10-min return < -1pt -> SELL
5. Multi-signal: EMA8<EMA21 + DI->DI+ on 1m in first 20min
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import pandas_ta as ta
from src.data_fetcher import DataFetcher

COST = 0.96


def load_1m_data():
    fetcher = DataFetcher()
    df = fetcher.get_futures_ohlcv('VN30F1M', '2026-05-17', '2026-06-17', interval='1m')
    if df is None or len(df) == 0:
        print('No data')
        sys.exit(1)

    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['session'] = df['mins'].apply(lambda m: 'AM' if m < 720 else 'PM')

    # Indicators
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema8'] = ta.ema(df['close'], length=8)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema50'] = ta.ema(df['close'], length=50)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14']
    df['di_plus'] = adx_df['DMP_14']
    df['di_minus'] = adx_df['DMN_14']
    df['rsi'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd_hist'] = macd['MACDh_12_26_9']

    print(f'[DATA] {len(df)} bars, {df["date"].nunique()} days')
    return df


def get_prev_close(df, date):
    """Get previous day's PM session close."""
    dates = sorted(df['date'].unique())
    idx = list(dates).index(date) if date in dates else -1
    if idx <= 0:
        return None
    prev_date = dates[idx - 1]
    prev_pm = df[(df['date'] == prev_date) & (df['session'] == 'PM')]
    if len(prev_pm) == 0:
        prev_am = df[(df['date'] == prev_date) & (df['session'] == 'AM')]
        if len(prev_am) == 0:
            return None
        return prev_am['close'].iloc[-1]
    return prev_pm['close'].iloc[-1]


def report(trades, label):
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
    per_day = total / 23  # 23 trading days

    print(f'  {label}:')
    print(f'    N={n}, WR={wr:.1f}%, PF={pf:.2f}, Total={total:+.1f}pts, /day={per_day:+.2f}')
    if wins:
        print(f'    Avg win={np.mean(wins):.2f}, Avg loss={np.mean(losses):.2f}')

    # Per-trade detail
    print(f'    Trades:')
    for t in trades:
        emoji = '+' if t['pnl'] > 0 else '-'
        print(f'      {t["date"]} {t.get("entry_time","")}: '
              f'entry={t["entry"]:.1f} exit={t["exit"]:.1f} '
              f'({t["exit_reason"]}) {t["pnl"]:+.1f}pts')
    print()


def simulate_exit(df, entry_idx, direction, sl_atr_mult=3.0):
    """Simulate raw exit: SL=3.0xATR, hold to session end."""
    row = df.iloc[entry_idx]
    entry = row['close']
    atr = row['atr']
    if pd.isna(atr) or atr <= 0:
        return None

    sl_price = entry - direction * sl_atr_mult * atr
    session = row['session']
    date = row['date']

    for j in range(entry_idx + 1, min(entry_idx + 200, len(df))):
        bar = df.iloc[j]
        if bar['date'] != date or bar['session'] != session:
            return {'exit': df.iloc[j - 1]['close'], 'reason': 'SESSION'}
        if session == 'AM' and bar['mins'] >= 685:
            return {'exit': bar['close'], 'reason': 'SESSION'}
        if session == 'PM' and bar['mins'] >= 865:
            return {'exit': bar['close'], 'reason': 'SESSION'}
        if direction == 1 and bar['low'] <= sl_price:
            return {'exit': sl_price, 'reason': 'SL'}
        if direction == -1 and bar['high'] >= sl_price:
            return {'exit': sl_price, 'reason': 'SL'}

    return {'exit': df.iloc[min(entry_idx + 199, len(df) - 1)]['close'], 'reason': 'MAX_HOLD'}


def main():
    df = load_1m_data()
    dates = sorted(df['date'].unique())
    print(f'Testing {len(dates)} trading days: {dates[0]} to {dates[-1]}')
    print()

    # First: show AM session characteristics for each day
    print('=' * 75)
    print('  AM SESSION MOVES (last 23 days)')
    print('=' * 75)
    print(f'  {"Date":<12} {"Open":>7} {"Close":>7} {"Low":>7} {"High":>7} {"Net":>6} {"Range":>6} {"Dir":<5}')
    print(f'  {"-"*65}')

    big_sell_days = []
    for date in dates:
        am = df[(df['date'] == date) & (df['session'] == 'AM') & (df['mins'] >= 540)]
        if len(am) < 10:
            continue
        am_open = am['open'].iloc[0]
        am_close = am['close'].iloc[-1]
        am_low = am['low'].min()
        am_high = am['high'].max()
        net = am_close - am_open
        rng = am_high - am_low
        direction = 'SELL' if net < -3 else ('BUY' if net > 3 else 'FLAT')
        marker = ' <--' if net < -5 else ''
        print(f'  {str(date):<12} {am_open:>7.1f} {am_close:>7.1f} {am_low:>7.1f} {am_high:>7.1f} '
              f'{net:>+6.1f} {rng:>6.1f} {direction:<5}{marker}')
        if net < -3:
            big_sell_days.append(date)

    print()
    print(f'  Days with AM drop > 3pts: {len(big_sell_days)}/{len(dates)} '
          f'({len(big_sell_days)/len(dates)*100:.0f}%)')
    print()

    # Strategy 1: Gap-down SELL (open below prev close by > 1pt)
    print('=' * 75)
    print('  STRATEGY BACKTESTS (23 days, raw exit SL=3.0xATR)')
    print('=' * 75)
    print()

    # Strategy 1: Gap-and-go
    trades1 = []
    for date in dates:
        prev_close = get_prev_close(df, date)
        if prev_close is None:
            continue
        am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
        if len(am) < 10:
            continue
        am_open = am['open'].iloc[0]
        gap = am_open - prev_close

        if gap >= -1.0:
            continue  # Only SELL when gap down > 1pt

        # Enter SELL on first bar close
        entry_idx = am.index[0]
        result = simulate_exit(df, entry_idx, -1)
        if result is None:
            continue
        pnl = (df.iloc[entry_idx]['close'] - result['exit']) - COST
        trades1.append({
            'date': str(date), 'entry': df.iloc[entry_idx]['close'],
            'exit': result['exit'], 'exit_reason': result['reason'],
            'pnl': pnl, 'entry_time': df.iloc[entry_idx]['time'].strftime('%H:%M'),
            'gap': gap,
        })

    report(trades1, '1. Gap-Down SELL (open < prev_close - 1pt)')

    # Strategy 2: First 5-min bearish candle
    trades2 = []
    for date in dates:
        am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
        if len(am) < 10:
            continue
        # First 5 bars (09:00-09:04)
        first5 = am.iloc[:5]
        if len(first5) < 5:
            continue
        first5_ret = first5['close'].iloc[-1] - first5['open'].iloc[0]
        first5_range = first5['high'].max() - first5['low'].min()
        atr_at_entry = am['atr'].iloc[5] if len(am) > 5 and not pd.isna(am['atr'].iloc[5]) else 2.0

        if first5_ret >= -1.0:
            continue  # Only SELL when first 5min drops > 1pt

        # Enter at bar 5 (09:05)
        if len(am) <= 5:
            continue
        entry_idx = am.index[5]
        result = simulate_exit(df, entry_idx, -1)
        if result is None:
            continue
        pnl = (df.iloc[entry_idx]['close'] - result['exit']) - COST
        trades2.append({
            'date': str(date), 'entry': df.iloc[entry_idx]['close'],
            'exit': result['exit'], 'exit_reason': result['reason'],
            'pnl': pnl, 'entry_time': df.iloc[entry_idx]['time'].strftime('%H:%M'),
        })

    report(trades2, '2. First 5-min Drop SELL (first 5 bars < -1pt)')

    # Strategy 3: First 10-min momentum
    trades3 = []
    for date in dates:
        am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
        if len(am) < 15:
            continue
        first10 = am.iloc[:10]
        ret10 = first10['close'].iloc[-1] - first10['open'].iloc[0]

        if ret10 >= -1.5:
            continue

        entry_idx = am.index[10]
        result = simulate_exit(df, entry_idx, -1)
        if result is None:
            continue
        pnl = (df.iloc[entry_idx]['close'] - result['exit']) - COST
        trades3.append({
            'date': str(date), 'entry': df.iloc[entry_idx]['close'],
            'exit': result['exit'], 'exit_reason': result['reason'],
            'pnl': pnl, 'entry_time': df.iloc[entry_idx]['time'].strftime('%H:%M'),
        })

    report(trades3, '3. First 10-min Drop SELL (10 bars < -1.5pt)')

    # Strategy 4: Opening range breakdown (15min)
    trades4 = []
    for date in dates:
        am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
        if len(am) < 20:
            continue
        # Opening 15 bars (09:00-09:14)
        opening = am.iloc[:15]
        range_low = opening['low'].min()
        range_high = opening['high'].max()
        opening_range = range_high - range_low

        # Wait for breakdown (break below opening low)
        post = am.iloc[15:]
        if len(post) < 5:
            continue

        traded = False
        for j in range(len(post)):
            if traded:
                break
            bar = post.iloc[j]
            if bar['low'] < range_low - 0.1:
                traded = True
                entry_idx = post.index[j]
                result = simulate_exit(df, entry_idx, -1)
                if result is None:
                    continue
                entry_price = range_low - 0.1
                pnl = (entry_price - result['exit']) - COST
                trades4.append({
                    'date': str(date), 'entry': entry_price,
                    'exit': result['exit'], 'exit_reason': result['reason'],
                    'pnl': pnl, 'entry_time': df.iloc[entry_idx]['time'].strftime('%H:%M'),
                    'opening_range': opening_range,
                })

    report(trades4, '4. Opening Range Breakdown SELL (15min range, break below)')

    # Strategy 5: First 15-min momentum with confirmation (first 15 < -2pt + next bar red)
    trades5 = []
    for date in dates:
        am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
        if len(am) < 20:
            continue
        first15 = am.iloc[:15]
        ret15 = first15['close'].iloc[-1] - first15['open'].iloc[0]

        if ret15 >= -2.0:
            continue

        # Confirmation: next bar (bar 15) must be red
        if len(am) <= 15:
            continue
        confirm_bar = am.iloc[15]
        if confirm_bar['close'] >= confirm_bar['open']:
            continue  # Skip if next bar is green (potential reversal)

        entry_idx = am.index[15]
        result = simulate_exit(df, entry_idx, -1)
        if result is None:
            continue
        pnl = (df.iloc[entry_idx]['close'] - result['exit']) - COST
        trades5.append({
            'date': str(date), 'entry': df.iloc[entry_idx]['close'],
            'exit': result['exit'], 'exit_reason': result['reason'],
            'pnl': pnl, 'entry_time': df.iloc[entry_idx]['time'].strftime('%H:%M'),
        })

    report(trades5, '5. First 15-min Drop + Confirm SELL (15bars<-2pt + red bar)')

    # Strategy 6: EMA-based early (EMA8 crosses below EMA21 before 09:30)
    trades6 = []
    for date in dates:
        am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
        if len(am) < 30:
            continue

        entered = False
        for j in range(14, min(30, len(am))):  # Check bars 14-29 (09:14-09:29)
            if entered:
                break
            bar = am.iloc[j]
            prev_bar = am.iloc[j - 1]

            ema8 = bar['ema8']
            ema21 = bar['ema21']
            prev_ema8 = prev_bar['ema8']
            prev_ema21 = prev_bar['ema21']

            if pd.isna(ema8) or pd.isna(ema21) or pd.isna(prev_ema8) or pd.isna(prev_ema21):
                continue

            # EMA8 crosses below EMA21
            if prev_ema8 >= prev_ema21 and ema8 < ema21:
                entered = True
                entry_idx = am.index[j]
                result = simulate_exit(df, entry_idx, -1)
                if result is None:
                    continue
                pnl = (df.iloc[entry_idx]['close'] - result['exit']) - COST
                trades6.append({
                    'date': str(date), 'entry': df.iloc[entry_idx]['close'],
                    'exit': result['exit'], 'exit_reason': result['reason'],
                    'pnl': pnl, 'entry_time': df.iloc[entry_idx]['time'].strftime('%H:%M'),
                })

    report(trades6, '6. EMA8 Cross Below EMA21 (before 09:30)')

    # Strategy 7: Combined - first 10min drop > 1pt + EMA8 < EMA21 at entry
    trades7 = []
    for date in dates:
        am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
        if len(am) < 15:
            continue
        first10 = am.iloc[:10]
        ret10 = first10['close'].iloc[-1] - first10['open'].iloc[0]

        if ret10 >= -1.0:
            continue

        # Check EMA at bar 10
        bar10 = am.iloc[10] if len(am) > 10 else None
        if bar10 is None:
            continue
        if pd.isna(bar10['ema8']) or pd.isna(bar10['ema21']):
            continue
        if bar10['ema8'] >= bar10['ema21']:
            continue  # EMA must confirm bearish

        entry_idx = am.index[10]
        result = simulate_exit(df, entry_idx, -1)
        if result is None:
            continue
        pnl = (df.iloc[entry_idx]['close'] - result['exit']) - COST
        trades7.append({
            'date': str(date), 'entry': df.iloc[entry_idx]['close'],
            'exit': result['exit'], 'exit_reason': result['reason'],
            'pnl': pnl, 'entry_time': df.iloc[entry_idx]['time'].strftime('%H:%M'),
        })

    report(trades7, '7. First 10min Drop + EMA8<EMA21 (combined)')

    # Strategy 8: AM BUY (for comparison - first 10min rise > 1.5pt)
    print()
    print('  --- AM BUY for comparison ---')
    print()
    trades8 = []
    for date in dates:
        am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
        if len(am) < 15:
            continue
        first10 = am.iloc[:10]
        ret10 = first10['close'].iloc[-1] - first10['open'].iloc[0]

        if ret10 <= 1.5:
            continue

        entry_idx = am.index[10]
        result = simulate_exit(df, entry_idx, 1)
        if result is None:
            continue
        pnl = (result['exit'] - df.iloc[entry_idx]['close']) - COST
        trades8.append({
            'date': str(date), 'entry': df.iloc[entry_idx]['close'],
            'exit': result['exit'], 'exit_reason': result['reason'],
            'pnl': pnl, 'entry_time': df.iloc[entry_idx]['time'].strftime('%H:%M'),
        })

    report(trades8, '8. First 10-min Rise BUY (10 bars > +1.5pt)')


if __name__ == '__main__':
    main()
