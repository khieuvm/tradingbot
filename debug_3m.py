import sys; sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta, date
from src.data_fetcher import DataFetcher

COST = 0.96
CUTOFF = date(2026, 5, 2)

fetcher = DataFetcher()
end = datetime.now().strftime("%Y-%m-%d")
df = fetcher.get_futures_ohlcv("VN30F1M", "2026-05-01", end, interval="3m")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df['date'] = df['time'].dt.date
df['mins'] = df['time'].dt.hour*60 + df['time'].dt.minute
df = df[((df['mins']>=9*60) & (df['mins']<11*60+30)) | ((df['mins']>=13*60) & (df['mins']<14*60+30))]
df = df[df['date'] >= CUTOFF].reset_index(drop=True)

n_days = df['date'].nunique()
bpd = df.groupby('date').size()
print(f"3m data (tu {CUTOFF}): {len(df)} bars, {n_days} days ({df['date'].min()} -> {df['date'].max()})")
print(f"Bars/day: min={bpd.min()} max={bpd.max()} mean={bpd.mean():.0f}")
print()
print("Bars per day:")
for d, n in bpd.items():
    print(f"  {d}: {n} bars")
print()

df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
df['rsi14'] = ta.rsi(df['close'], length=14)
df['session'] = np.where(df['mins']<12*60, 'AM', 'PM')
df['range'] = df['high'] - df['low']


def detect_compression(df, n_bars, threshold=0.7):
    ranges = df['range'].values
    atr_v = df['atr'].values
    comp = np.zeros(len(df), dtype=bool)
    for i in range(n_bars, len(df)):
        if atr_v[i] > 0 and max(ranges[i-n_bars:i]) < threshold * atr_v[i]:
            comp[i] = True
    df['compressed'] = comp
    return df


def dedup(mask, min_bars):
    r = mask.copy()
    last = -999
    for i in range(len(r)):
        if r.iloc[i]:
            if i - last < min_bars:
                r.iloc[i] = False
            else:
                last = i
    return r


def sim(idx, df, sl_mult, trail_activate, trail_mult, max_hold, be_trigger=4.0, be_atr_min=2.5):
    row = df.loc[idx]
    atr = row['atr']
    if pd.isna(atr) or atr <= 0:
        return None
    i_pos = df.index.get_loc(idx)
    if i_pos + 1 >= len(df):
        return None
    nxt = df.iloc[i_pos + 1]
    if nxt['date'] != row['date'] or nxt['session'] != row['session']:
        return None
    direction = 1 if nxt['close'] > row['close'] else -1
    entry = row['close']
    sl = entry - direction * sl_mult * atr
    best = entry
    trail_on = False
    be_done = False
    exit_p = None
    exit_r = None
    bars = 0
    for jp in range(i_pos + 1, min(i_pos + 1 + max_hold, len(df))):
        b = df.iloc[jp]
        if b['date'] != row['date'] or b['session'] != row['session']:
            exit_p = df.iloc[jp - 1]['close']
            exit_r = 'SESSION'
            break
        if (b['session'] == 'PM' and b['mins'] >= 14*60+25) or \
           (b['session'] == 'AM' and b['mins'] >= 11*60+25):
            exit_p = b['close']
            exit_r = 'SESSION'
            break
        bars += 1
        if direction == 1:
            if b['low'] <= sl:
                exit_p = sl
                exit_r = 'BE' if be_done else 'SL'
                break
            if b['high'] > best:
                best = b['high']
            mfe = best - entry
            if not be_done and mfe >= be_trigger and atr >= be_atr_min:
                be_done = True
                sl = max(sl, entry)
            if mfe >= trail_activate:
                trail_on = True
            if trail_on:
                sl = max(sl, best - trail_mult * atr)
                if b['low'] <= sl:
                    exit_p = sl
                    exit_r = 'TRAIL'
                    break
        else:
            if b['high'] >= sl:
                exit_p = sl
                exit_r = 'BE' if be_done else 'SL'
                break
            if b['low'] < best:
                best = b['low']
            mfe = entry - best
            if not be_done and mfe >= be_trigger and atr >= be_atr_min:
                be_done = True
                sl = min(sl, entry)
            if mfe >= trail_activate:
                trail_on = True
            if trail_on:
                sl = min(sl, best + trail_mult * atr)
                if b['high'] >= sl:
                    exit_p = sl
                    exit_r = 'TRAIL'
                    break
    if exit_p is None:
        exit_p = entry
        exit_r = 'MAX_HOLD'
    mfe_f = (best - entry) if direction == 1 else (entry - best)
    pnl = direction * (exit_p - entry) - COST
    return {
        'date': row['date'], 'time': row['time'].strftime('%H:%M'),
        'sess': row['session'], 'dir': 'BUY' if direction == 1 else 'SELL',
        'entry': entry, 'exit': exit_p, 'atr': atr,
        'sl_dist': sl_mult * atr, 'mfe': mfe_f, 'pnl': pnl,
        'reason': exit_r, 'bars': bars
    }


# Quick comparison of n_bars configs
print("So sanh n_bars configs:")
for nb, dedup_b in [(3, 5), (4, 7), (5, 8)]:
    df2 = detect_compression(df.copy(), n_bars=nb, threshold=0.7)
    filt = (df2['atr'] <= 3.0) & (df2['atr'] >= 1.5) & (df2['rsi14'] < 70).fillna(True)
    mAM = dedup(df2['compressed'] & (df2['mins']>=9*60+15) & (df2['mins']<=10*60+45) & filt, dedup_b)
    mPM = dedup(df2['compressed'] & (df2['mins']>=13*60+15) & (df2['mins']<=14*60+15) & filt, dedup_b)
    trades = []
    for idx in df2.index[mAM]:
        t = sim(idx, df2, 1.2, 5.0, 2.0, 40)
        if t:
            trades.append(t)
    for idx in df2.index[mPM]:
        t = sim(idx, df2, 1.0, 4.0, 1.5, 20)
        if t:
            trades.append(t)
    if trades:
        tdf = pd.DataFrame(trades)
        wins = tdf[tdf['pnl'] > 0]
        losses = tdf[tdf['pnl'] <= 0]
        pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 999
        print(f"  {nb}bar: {len(tdf):>3} trades | WR {(tdf['pnl']>0).mean()*100:>5.1f}% | PF {pf:>5.2f} | PnL {tdf['pnl'].sum():>+7.1f} | {tdf['pnl'].sum()/n_days:>+5.2f}/d | AM={mAM.sum()} PM={mPM.sum()}")
    else:
        print(f"  {nb}bar: no trades")

print()

# Detailed: 5-bar (equivalent to 3-bar 5m = 15 min compression)
NB = 5
df2 = detect_compression(df.copy(), n_bars=NB, threshold=0.7)
filt = (df2['atr'] <= 3.0) & (df2['atr'] >= 1.5) & (df2['rsi14'] < 70).fillna(True)
maskAM = dedup(df2['compressed'] & (df2['mins']>=9*60+15) & (df2['mins']<=10*60+45) & filt, 8)
maskPM = dedup(df2['compressed'] & (df2['mins']>=13*60+15) & (df2['mins']<=14*60+15) & filt, 8)

trades = []
for idx in df2.index[maskAM]:
    t = sim(idx, df2, 1.2, 5.0, 2.0, 40)
    if t:
        trades.append(t)
for idx in df2.index[maskPM]:
    t = sim(idx, df2, 1.0, 4.0, 1.5, 20)
    if t:
        trades.append(t)

if not trades:
    print("No trades for 5-bar config")
    exit()

tdf = pd.DataFrame(trades).sort_values(['date', 'time']).reset_index(drop=True)
wins = tdf[tdf['pnl'] > 0]
losses = tdf[tdf['pnl'] <= 0]
sl_trades = tdf[tdf['reason'] == 'SL']
pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 999

print("=" * 88)
print(f"CB 3m 5bar (tu 2/5) | AM:SL1.2/T@5/2x H40 | PM:SL1.0/T@4/1.5x H20")
print("=" * 88)
print(f"\n{'Date':<12}{'Time':<7}{'Sess':<5}{'Dir':<5}{'Entry':>7}{'Exit':>7}{'ATR':>5}{'MFE':>5}{'PnL':>7}{'Bars':>5}  Reason")
print('-' * 78)
for _, t in tdf.iterrows():
    mark = ' <<' if t['pnl'] > 0 else ''
    print(f"{str(t['date']):<12}{t['time']:<7}{t['sess']:<5}{t['dir']:<5}"
          f"{t['entry']:>7.1f}{t['exit']:>7.1f}{t['atr']:>5.2f}"
          f"{t['mfe']:>5.1f}{t['pnl']:>+7.1f}{t['bars']:>5}  {t['reason']}{mark}")

print(f"\n{'=' * 88}")
print(f"TONG KET: {len(tdf)} trades | Win: {len(wins)} | WR: {len(wins)/len(tdf)*100:.1f}% | PF: {pf:.2f}")
print(f"PnL: {tdf['pnl'].sum():+.1f} pts | {tdf['pnl'].sum()/n_days:+.2f}/d")

print("\nExit breakdown:")
for r, cnt in tdf['reason'].value_counts().items():
    sub = tdf[tdf['reason'] == r]
    wr = (sub['pnl'] > 0).mean() * 100
    print(f"  {r:<12}: {cnt:>3} | WR {wr:>5.1f}% | avg PnL {sub['pnl'].mean():>+5.2f} | avg MFE {sub['mfe'].mean():>4.1f}")

print(f"\n{'=' * 88}")
print("PHAN TICH SL")
if len(sl_trades) > 0:
    print(f"\n{len(sl_trades)} SL trades:")
    print(f"  Avg ATR: {sl_trades['atr'].mean():.2f} | Avg SL dist: {sl_trades['sl_dist'].mean():.2f} | Avg MFE: {sl_trades['mfe'].mean():.2f}")
    print(f"  MFE < 1 pts  (reverse ngay):   {(sl_trades['mfe']<1).sum()}/{len(sl_trades)}")
    print(f"  MFE 1-2 pts  (di yeu roi SL):  {((sl_trades['mfe']>=1)&(sl_trades['mfe']<2)).sum()}/{len(sl_trades)}")
    print(f"  MFE 2-4 pts  (di duoc, SL):    {((sl_trades['mfe']>=2)&(sl_trades['mfe']<4)).sum()}/{len(sl_trades)}")
    print(f"  MFE >= 4 pts (gap BE roi SL):  {(sl_trades['mfe']>=4).sum()}/{len(sl_trades)}")
    print(f"\n  SL detail:")
    for _, t in sl_trades.iterrows():
        print(f"    {t['date']} {t['time']} {t['sess']} {t['dir']} ATR={t['atr']:.2f} MFE={t['mfe']:.1f} PnL={t['pnl']:+.1f}")
else:
    print("  Khong co SL!")

print(f"\n{'=' * 88}")
print("MFE DISTRIBUTION")
buckets = [(0, 1, '0-1'), (1, 2, '1-2'), (2, 4, '2-4'), (4, 6, '4-6'), (6, 9, '6-9'), (9, 99, '9+')]
print(f"\n{'MFE':^8} {'N':>4} {'WR':>6} {'AvgPnL':>8}")
print('-' * 40)
for lo, hi, lbl in buckets:
    sub = tdf[(tdf['mfe'] >= lo) & (tdf['mfe'] < hi)]
    if len(sub):
        wr = (sub['pnl'] > 0).mean() * 100
        print(f"  {lbl:<6} {len(sub):>4} {wr:>5.1f}% {sub['pnl'].mean():>+8.2f}")

print(f"\n{'=' * 88}")
print("TP TARGETS")
am_t = tdf[tdf['sess'] == 'AM']
pm_t = tdf[tdf['sess'] == 'PM']
print(f"\nAM ({len(am_t)} trades):")
for pts in [5, 8, 12]:
    n = (am_t['mfe'] >= pts).sum()
    pct = n / len(am_t) * 100 if len(am_t) else 0
    print(f"  MFE >= {pts:>2} pts: {n}/{len(am_t)} = {pct:.1f}%")
if len(pm_t):
    print(f"\nPM ({len(pm_t)} trades):")
    for pts in [4, 8]:
        n = (pm_t['mfe'] >= pts).sum()
        pct = n / len(pm_t) * 100 if len(pm_t) else 0
        print(f"  MFE >= {pts:>2} pts: {n}/{len(pm_t)} = {pct:.1f}%")

trail_w = tdf[tdf['reason'] == 'TRAIL']
if len(trail_w):
    print(f"\nTrail exits: {len(trail_w)} | avg PnL {trail_w['pnl'].mean():+.2f} | avg MFE {trail_w['mfe'].mean():.1f}")

print(f"\nAM vs PM:")
for s in ['AM', 'PM']:
    sub = tdf[tdf['sess'] == s]
    if len(sub):
        w = (sub['pnl'] > 0).sum()
        wsum = sub[sub['pnl'] > 0]['pnl'].sum()
        lsum = abs(sub[sub['pnl'] <= 0]['pnl'].sum())
        spf = wsum / lsum if lsum > 0 else 999
        print(f"  {s}: {len(sub):>3} trades | WR {w/len(sub)*100:>5.1f}% | PF {spf:>5.2f} | PnL {sub['pnl'].sum():>+7.1f}")
