"""PM Session Opening Analysis by Day-of-Week.

Analyzes PM opening patterns: UP/DOWN tendency, ORB, AM→PM continuation.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import pandas_ta as ta
from src.data_fetcher import DataFetcher

COST = 0.96
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

fetcher = DataFetcher()
df = fetcher.get_futures_ohlcv('VN30F1M', '2026-05-02', '2026-06-17', interval='1m')
df['date'] = df['time'].dt.date
df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
df['weekday'] = df['time'].dt.weekday
df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
dates = sorted(df['date'].unique())
n_days = len(dates)

print('=' * 80)
print(f'  PM SESSION OPENING ANALYSIS BY WEEKDAY ({n_days} days)')
print('=' * 80)
print()

# Part 1: PM overall direction
print('  PART 1: PM SESSION OVERALL DIRECTION')
print(f'  {"Day":<5} {"N":>3} {"Avg Net":>8} {"UP>2":>5} {"DN>2":>5} {"UP%":>5} {"DN%":>5} {"Avg Rng":>8}')
print(f'  {"-"*50}')

for dow in range(5):
    day_dates = [d for d in dates if pd.Timestamp(d).weekday() == dow]
    nets, rngs = [], []
    ups, downs = 0, 0
    for date in day_dates:
        pm = df[(df['date'] == date) & (df['mins'] >= 780) & (df['mins'] <= 870)]
        if len(pm) < 10:
            continue
        net = pm['close'].iloc[-1] - pm['open'].iloc[0]
        rng = pm['high'].max() - pm['low'].min()
        nets.append(net)
        rngs.append(rng)
        if net > 2:
            ups += 1
        if net < -2:
            downs += 1
    if not nets:
        continue
    n = len(nets)
    print(f'  {DOW[dow]:<5} {n:>3} {np.mean(nets):>+8.2f} {ups:>5} {downs:>5}'
          f' {ups/n*100:>4.0f}% {downs/n*100:>4.0f}% {np.mean(rngs):>8.1f}')

print()

# Part 2: First N minutes direction
print('  PART 2: PM OPENING MOVE (first N min from 13:00)')
print()

for window in [5, 10, 15, 30]:
    print(f'  --- First {window} min (13:00-13:{window:02d}) ---')
    print(f'  {"Day":<5} {"N":>3} {"Avg":>7} {"UP":>4} {"DN":>4} {"UP%":>5} {"DN%":>5} {"Cont%":>6}')
    print(f'  {"-"*45}')

    for dow in range(5):
        day_dates = [d for d in dates if pd.Timestamp(d).weekday() == dow]
        moves = []
        cont_count = 0
        ups, downs = 0, 0

        for date in day_dates:
            pm = df[(df['date'] == date) & (df['mins'] >= 780) & (df['mins'] <= 870)]
            if len(pm) < max(window + 5, 20):
                continue
            opening = pm.iloc[:window]
            rest = pm.iloc[window:]
            move = opening['close'].iloc[-1] - opening['open'].iloc[0]
            moves.append(move)
            if move > 1:
                ups += 1
            if move < -1:
                downs += 1
            rest_move = rest['close'].iloc[-1] - rest['open'].iloc[0]
            if (move > 1 and rest_move > 0) or (move < -1 and rest_move < 0):
                cont_count += 1

        if not moves:
            continue
        n = len(moves)
        total_dir = ups + downs
        cont_pct = cont_count / total_dir * 100 if total_dir > 0 else 0
        print(f'  {DOW[dow]:<5} {n:>3} {np.mean(moves):>+7.2f} {ups:>4} {downs:>4}'
              f' {ups/n*100:>4.0f}% {downs/n*100:>4.0f}% {cont_pct:>5.0f}%')
    print()

# Part 3: PM ORB
print('  PART 3: PM ORB (15min opening range 13:00-13:14)')
print()


def simulate_exit_pm(df, entry_idx, direction, sl_price):
    row = df.iloc[entry_idx]
    date = row['date']
    for j in range(entry_idx + 1, min(entry_idx + 200, len(df))):
        bar = df.iloc[j]
        if bar['date'] != date or bar['mins'] < 780:
            return df.iloc[j - 1]['close'], 'SESSION'
        if bar['mins'] >= 865:
            return bar['close'], 'SESSION'
        if direction == 1 and bar['low'] <= sl_price:
            return sl_price, 'SL'
        if direction == -1 and bar['high'] >= sl_price:
            return sl_price, 'SL'
    return df.iloc[min(entry_idx + 199, len(df) - 1)]['close'], 'MAX'


for dir_name, direction in [('SELL (break below)', -1), ('BUY (break above)', 1)]:
    print(f'  --- PM ORB {dir_name}, SL=3xATR ---')
    print(f'  {"Day":<5} {"N":>3} {"WR":>5} {"PF":>5} {"PnL":>7} {"Trades":<50}')
    print(f'  {"-"*75}')

    for dow in range(5):
        trades = []
        for date in dates:
            if pd.Timestamp(date).weekday() != dow:
                continue
            pm = df[(df['date'] == date) & (df['mins'] >= 780) & (df['mins'] <= 865)]
            if len(pm) < 20:
                continue
            opening = pm.iloc[:15]
            or_high = opening['high'].max()
            or_low = opening['low'].min()
            atr = opening['atr'].iloc[-1]
            if pd.isna(atr) or atr <= 0:
                continue
            post = pm.iloc[15:]
            if len(post) < 5:
                continue
            traded = False
            for j in range(len(post)):
                if traded:
                    break
                bar = post.iloc[j]
                if bar['mins'] > 855:
                    break
                if direction == -1 and bar['close'] < or_low:
                    traded = True
                    sl = or_low + 3.0 * atr
                    exit_p, exit_r = simulate_exit_pm(df, post.index[j], -1, sl)
                    pnl = (or_low - exit_p) - COST
                    trades.append({'date': str(date), 'pnl': pnl})
                elif direction == 1 and bar['close'] > or_high:
                    traded = True
                    sl = or_high - 3.0 * atr
                    exit_p, exit_r = simulate_exit_pm(df, post.index[j], 1, sl)
                    pnl = (exit_p - or_high) - COST
                    trades.append({'date': str(date), 'pnl': pnl})

        if not trades:
            print(f'  {DOW[dow]:<5}   0   n/a   n/a     n/a')
            continue
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        gl = abs(sum([p for p in pnls if p <= 0])) or 0.001
        pf = sum(wins) / gl
        wr = len(wins) / len(pnls) * 100
        detail = ', '.join([f'{t["date"][-5:]}({t["pnl"]:+.0f})' for t in trades])
        print(f'  {DOW[dow]:<5} {len(pnls):>3} {wr:>4.0f}% {pf:>5.2f} {sum(pnls):>+7.1f} {detail}')
    print()

# Part 4: AM → PM relationship
print('  PART 4: AM to PM RELATIONSHIP (continuation vs reversal)')
print(f'  {"Day":<5} {"N":>3} {"Cont":>5} {"Rev":>5} {"Cont%":>6} {"Avg AM":>8} {"Avg PM":>8}')
print(f'  {"-"*50}')

for dow in range(5):
    day_dates = [d for d in dates if pd.Timestamp(d).weekday() == dow]
    cont, rev, total = 0, 0, 0
    am_moves, pm_moves = [], []

    for date in day_dates:
        am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 690)]
        pm = df[(df['date'] == date) & (df['mins'] >= 780) & (df['mins'] <= 870)]
        if len(am) < 10 or len(pm) < 10:
            continue
        am_net = am['close'].iloc[-1] - am['open'].iloc[0]
        pm_net = pm['close'].iloc[-1] - pm['open'].iloc[0]
        am_moves.append(am_net)
        pm_moves.append(pm_net)
        if abs(am_net) < 2:
            continue
        total += 1
        if (am_net > 0 and pm_net > 0) or (am_net < 0 and pm_net < 0):
            cont += 1
        else:
            rev += 1

    if not am_moves:
        continue
    n = len(am_moves)
    cont_pct = cont / total * 100 if total > 0 else 0
    print(f'  {DOW[dow]:<5} {n:>3} {cont:>5} {rev:>5} {cont_pct:>5.0f}%'
          f' {np.mean(am_moves):>+8.2f} {np.mean(pm_moves):>+8.2f}')

# Part 5: Per-day PM detail
print()
print()
print('  PART 5: PM SESSION DETAIL PER DATE')
print(f'  {"Date":<12} {"Day":<5} {"Open":>7} {"Close":>7} {"Hi":>7} {"Lo":>7}'
      f' {"Net":>7} {"Range":>7} {"Dir":<5}')
print(f'  {"-"*68}')

for date in dates:
    pm = df[(df['date'] == date) & (df['mins'] >= 780) & (df['mins'] <= 870)]
    if len(pm) < 10:
        continue
    dow = pd.Timestamp(date).weekday()
    o = pm['open'].iloc[0]
    c = pm['close'].iloc[-1]
    h = pm['high'].max()
    lo = pm['low'].min()
    net = c - o
    rng = h - lo
    d = 'UP' if net > 2 else ('DOWN' if net < -2 else 'FLAT')
    marker = ' ***' if abs(net) > 5 else ''
    print(f'  {str(date):<12} {DOW[dow]:<5} {o:>7.1f} {c:>7.1f} {h:>7.1f} {lo:>7.1f}'
          f' {net:>+7.1f} {rng:>7.1f} {d:<5}{marker}')
