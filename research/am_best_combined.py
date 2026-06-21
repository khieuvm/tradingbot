"""Best combined AM strategies with weekday filter."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import pandas_ta as ta
from src.data_fetcher import DataFetcher

COST = 0.96
DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

fetcher = DataFetcher()
df = fetcher.get_futures_ohlcv('VN30F1M', '2026-05-02', '2026-06-17', interval='1m')
df['date'] = df['time'].dt.date
df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
df['session'] = df['mins'].apply(lambda m: 'AM' if m < 720 else 'PM')
df['weekday'] = df['time'].dt.weekday
df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
dates = sorted(df['date'].unique())
n_days = len(dates)


def daily_ranges(df):
    ranges = {}
    for date in sorted(df['date'].unique()):
        day_df = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 690)]
        if len(day_df) < 10:
            continue
        ranges[date] = day_df['high'].max() - day_df['low'].min()
    return ranges


def simulate_exit(df, entry_idx, direction, sl_price):
    row = df.iloc[entry_idx]
    session = row['session']
    date = row['date']
    for j in range(entry_idx + 1, min(entry_idx + 200, len(df))):
        bar = df.iloc[j]
        if bar['date'] != date or bar['session'] != session:
            return df.iloc[j - 1]['close'], 'SESSION'
        if session == 'AM' and bar['mins'] >= 685:
            return bar['close'], 'SESSION'
        if direction == 1 and bar['low'] <= sl_price:
            return sl_price, 'SL'
        if direction == -1 and bar['high'] >= sl_price:
            return sl_price, 'SL'
    return df.iloc[min(entry_idx + 199, len(df) - 1)]['close'], 'MAX'


d_ranges = daily_ranges(df)
dates_list = sorted(d_ranges.keys())

print('=' * 75)
print(f'  BEST COMBINED STRATEGIES ({n_days} days, 2026-05-04 to 2026-06-17)')
print('=' * 75)
print()

# ══════════════════════════════════════════════════════════════
# BEST 1: Crabel Stretch SELL (SMA5, 50%, SL=open) - ALL days
# ══════════════════════════════════════════════════════════════
print('1. CRABEL STRETCH SELL (SMA5, 50%, stop=open) - ALL DAYS')
print('-' * 60)
trades_all = []
for i, date in enumerate(dates_list):
    if i < 5:
        continue
    recent_ranges = [d_ranges[dates_list[k]] for k in range(i - 5, i)]
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
            sl = am_open
            exit_p, exit_r = simulate_exit(df, am.index[j], -1, sl)
            pnl = (entry_price - exit_p) - COST
            trades_all.append({
                'date': str(date), 'pnl': pnl, 'exit_reason': exit_r,
                'entry': entry_price, 'exit': exit_p,
                'entry_time': am.iloc[j]['time'].strftime('%H:%M'),
                'weekday': pd.Timestamp(date).weekday()
            })

for t in trades_all:
    dow = DOW_NAMES[t['weekday']]
    icon = '+' if t['pnl'] > 0 else '-'
    print(f'  {t["date"]} ({dow}) {t["entry_time"]}: '
          f'entry={t["entry"]:.1f} exit={t["exit"]:.1f} ({t["exit_reason"]}) '
          f'PnL={t["pnl"]:+.1f}')

pnls = [t['pnl'] for t in trades_all]
wins = [p for p in pnls if p > 0]
gl = abs(sum([p for p in pnls if p <= 0])) or 0.001
print(f'\n  TOTAL: N={len(pnls)}, WR={len(wins)/len(pnls)*100:.0f}%, '
      f'PF={sum(wins)/gl:.2f}, PnL={sum(pnls):+.1f}, /day={sum(pnls)/n_days:+.2f}')
print()
print('  By weekday:')
for dow in range(5):
    dow_pnls = [t['pnl'] for t in trades_all if t['weekday'] == dow]
    if dow_pnls:
        w = len([p for p in dow_pnls if p > 0])
        print(f'    {DOW_NAMES[dow]}: N={len(dow_pnls)}, '
              f'WR={w/len(dow_pnls)*100:.0f}%, PnL={sum(dow_pnls):+.1f}')

# ══════════════════════════════════════════════════════════════
# Mon+Wed filter
# ══════════════════════════════════════════════════════════════
print()
print()
print('2. CRABEL STRETCH SELL - Mon+Wed ONLY')
print('-' * 60)
mw_trades = [t for t in trades_all if t['weekday'] in [0, 2]]
for t in mw_trades:
    dow = DOW_NAMES[t['weekday']]
    print(f'  {t["date"]} ({dow}) {t["entry_time"]}: '
          f'entry={t["entry"]:.1f} exit={t["exit"]:.1f} ({t["exit_reason"]}) '
          f'PnL={t["pnl"]:+.1f}')
pnls_mw = [t['pnl'] for t in mw_trades]
if pnls_mw:
    wins_mw = [p for p in pnls_mw if p > 0]
    gl_mw = abs(sum([p for p in pnls_mw if p <= 0])) or 0.001
    print(f'\n  TOTAL: N={len(pnls_mw)}, WR={len(wins_mw)/len(pnls_mw)*100:.0f}%, '
          f'PF={sum(wins_mw)/gl_mw:.2f}, PnL={sum(pnls_mw):+.1f}, /day={sum(pnls_mw)/n_days:+.2f}')

# ══════════════════════════════════════════════════════════════
# BEST 3: ORB Breakdown SELL all days + by weekday
# ══════════════════════════════════════════════════════════════
print()
print()
print('3. OPENING RANGE BREAKDOWN SELL (15min, SL=3xATR) - ALL DAYS')
print('-' * 60)
trades_orb = []
for date in dates:
    am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
    if len(am) < 20:
        continue
    opening = am.iloc[:15]
    or_low = opening['low'].min()
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
        if bar['close'] < or_low:
            traded = True
            entry_price = or_low
            sl = entry_price + 3.0 * atr
            exit_p, exit_r = simulate_exit(df, post.index[j], -1, sl)
            pnl = (entry_price - exit_p) - COST
            trades_orb.append({
                'date': str(date), 'pnl': pnl, 'exit_reason': exit_r,
                'entry': entry_price, 'exit': exit_p,
                'entry_time': post.iloc[j]['time'].strftime('%H:%M'),
                'weekday': bar['weekday']
            })

for t in trades_orb:
    dow = DOW_NAMES[t['weekday']]
    print(f'  {t["date"]} ({dow}) {t["entry_time"]}: '
          f'entry={t["entry"]:.1f} exit={t["exit"]:.1f} ({t["exit_reason"]}) '
          f'PnL={t["pnl"]:+.1f}')

pnls_orb = [t['pnl'] for t in trades_orb]
if pnls_orb:
    wins_orb = [p for p in pnls_orb if p > 0]
    gl_orb = abs(sum([p for p in pnls_orb if p <= 0])) or 0.001
    print(f'\n  TOTAL: N={len(pnls_orb)}, WR={len(wins_orb)/len(pnls_orb)*100:.0f}%, '
          f'PF={sum(wins_orb)/gl_orb:.2f}, PnL={sum(pnls_orb):+.1f}, /day={sum(pnls_orb)/n_days:+.2f}')
    print()
    print('  By weekday:')
    for dow in range(5):
        dow_pnls = [t['pnl'] for t in trades_orb if t['weekday'] == dow]
        if dow_pnls:
            w = len([p for p in dow_pnls if p > 0])
            print(f'    {DOW_NAMES[dow]}: N={len(dow_pnls)}, '
                  f'WR={w/len(dow_pnls)*100:.0f}%, PnL={sum(dow_pnls):+.1f}')

# ══════════════════════════════════════════════════════════════
# ORB Mon+Wed
# ══════════════════════════════════════════════════════════════
print()
print()
print('4. ORB BREAKDOWN SELL - Mon+Wed ONLY')
print('-' * 60)
mw_orb = [t for t in trades_orb if t['weekday'] in [0, 2]]
for t in mw_orb:
    dow = DOW_NAMES[t['weekday']]
    print(f'  {t["date"]} ({dow}) {t["entry_time"]}: '
          f'entry={t["entry"]:.1f} exit={t["exit"]:.1f} ({t["exit_reason"]}) '
          f'PnL={t["pnl"]:+.1f}')
pnls_mw_orb = [t['pnl'] for t in mw_orb]
if pnls_mw_orb:
    wins_mw_orb = [p for p in pnls_mw_orb if p > 0]
    gl_mw_orb = abs(sum([p for p in pnls_mw_orb if p <= 0])) or 0.001
    print(f'\n  TOTAL: N={len(pnls_mw_orb)}, WR={len(wins_mw_orb)/len(pnls_mw_orb)*100:.0f}%, '
          f'PF={sum(wins_mw_orb)/gl_mw_orb:.2f}, PnL={sum(pnls_mw_orb):+.1f}, /day={sum(pnls_mw_orb)/n_days:+.2f}')

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print()
print()
print('=' * 75)
print('  SUMMARY: BEST AM STRATEGIES')
print('=' * 75)
print()
print(f'  {"Strategy":<40} {"N":>3} {"WR":>5} {"PF":>5} {"PnL":>7} {"/day":>6}')
print(f'  {"-"*68}')

configs = [
    ('Crabel Stretch SELL (all days)', trades_all),
    ('Crabel Stretch SELL (Mon+Wed)', mw_trades),
    ('ORB Breakdown SELL (all days)', trades_orb),
    ('ORB Breakdown SELL (Mon+Wed)', mw_orb),
]
for label, tr in configs:
    if not tr:
        continue
    p = [t['pnl'] for t in tr]
    w = [x for x in p if x > 0]
    l_abs = abs(sum([x for x in p if x <= 0])) or 0.001
    print(f'  {label:<40} {len(p):>3} {len(w)/len(p)*100:>4.0f}% '
          f'{sum(w)/l_abs:>5.2f} {sum(p):>+7.1f} {sum(p)/n_days:>+6.2f}')
