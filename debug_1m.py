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
df = fetcher.get_futures_ohlcv("VN30F1M", "2026-05-01", end, interval="1m")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df['date'] = df['time'].dt.date
df['mins'] = df['time'].dt.hour*60 + df['time'].dt.minute
df = df[((df['mins']>=9*60) & (df['mins']<11*60+30)) | ((df['mins']>=13*60) & (df['mins']<14*60+30))]
df = df[df['date'] >= CUTOFF].reset_index(drop=True)

n_days = df['date'].nunique()
print(f"1m data (tu {CUTOFF}): {len(df)} bars, {n_days} days ({df['date'].min()} -> {df['date'].max()})")
print(f"Bars/day: min={df.groupby('date').size().min()} max={df.groupby('date').size().max()} mean={df.groupby('date').size().mean():.0f}")
print()

df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
df['rsi14'] = ta.rsi(df['close'], length=14)
df['session'] = np.where(df['mins']<12*60, 'AM', 'PM')
df['range'] = df['high'] - df['low']

N_BARS = 10
ranges = df['range'].values; atr_v = df['atr'].values
compressed = np.zeros(len(df), dtype=bool)
for i in range(N_BARS, len(df)):
    if atr_v[i] > 0 and max(ranges[i-N_BARS:i]) < 0.7*atr_v[i]:
        compressed[i] = True
df['compressed'] = compressed

filt = (df['atr'] <= 1.5) & (df['atr'] >= 0.7) & (df['rsi14'] < 70).fillna(True)
maskAM = df['compressed'] & (df['mins']>=9*60+15) & (df['mins']<=10*60+45) & filt
maskPM = df['compressed'] & (df['mins']>=13*60+15) & (df['mins']<=14*60+15) & filt

def dedup(mask, min_bars=15):
    r = mask.copy(); last = -999
    for i in range(len(r)):
        if r.iloc[i]:
            if i-last < min_bars: r.iloc[i] = False
            else: last = i
    return r

maskAM = dedup(maskAM.fillna(False))
maskPM = dedup(maskPM.fillna(False))
print(f"Signals: AM={maskAM.sum()}, PM={maskPM.sum()}, total={maskAM.sum()+maskPM.sum()}")
print()


def sim(idx, sl_mult, trail_activate, trail_mult, max_hold, sess_label):
    row = df.loc[idx]; atr = row['atr']
    if pd.isna(atr) or atr <= 0: return None
    i_pos = df.index.get_loc(idx)
    if i_pos+1 >= len(df): return None
    nxt = df.iloc[i_pos+1]
    if nxt['date'] != row['date'] or nxt['session'] != row['session']: return None
    direction = 1 if nxt['close'] > row['close'] else -1
    entry = row['close']
    sl = entry - direction*sl_mult*atr
    best = entry; trail_on = False; be_done = False
    exit_p = None; exit_r = None; bars = 0
    for jp in range(i_pos+1, min(i_pos+1+max_hold, len(df))):
        b = df.iloc[jp]
        if b['date'] != row['date'] or b['session'] != row['session']:
            exit_p = df.iloc[jp-1]['close']; exit_r = 'SESSION'; break
        if (b['session']=='PM' and b['mins']>=14*60+25) or \
           (b['session']=='AM' and b['mins']>=11*60+25):
            exit_p = b['close']; exit_r = 'SESSION'; break
        bars += 1
        if direction == 1:
            if b['low'] <= sl: exit_p=sl; exit_r='BE' if be_done else 'SL'; break
            if b['high'] > best: best = b['high']
            mfe = best - entry
            if not be_done and mfe >= 4.0 and atr >= 1.0:
                be_done = True; sl = max(sl, entry)
            if mfe >= trail_activate: trail_on = True
            if trail_on:
                sl = max(sl, best - trail_mult*atr)
                if b['low'] <= sl: exit_p=sl; exit_r='TRAIL'; break
        else:
            if b['high'] >= sl: exit_p=sl; exit_r='BE' if be_done else 'SL'; break
            if b['low'] < best: best = b['low']
            mfe = entry - best
            if not be_done and mfe >= 4.0 and atr >= 1.0:
                be_done = True; sl = min(sl, entry)
            if mfe >= trail_activate: trail_on = True
            if trail_on:
                sl = min(sl, best + trail_mult*atr)
                if b['high'] >= sl: exit_p=sl; exit_r='TRAIL'; break
    if exit_p is None: exit_p=entry; exit_r='MAX_HOLD'
    mfe_f = (best-entry) if direction==1 else (entry-best)
    pnl = direction*(exit_p-entry) - COST
    return {'date': row['date'], 'time': row['time'].strftime('%H:%M'), 'sess': sess_label,
            'dir': 'BUY' if direction==1 else 'SELL', 'entry': entry, 'exit': exit_p,
            'atr': atr, 'sl_dist': sl_mult*atr, 'mfe': mfe_f, 'pnl': pnl,
            'reason': exit_r, 'bars': bars}


trades = []
for idx in df.index[maskAM]:
    t = sim(idx, 1.2, 5.0, 2.0, 90, 'AM')
    if t: trades.append(t)
for idx in df.index[maskPM]:
    t = sim(idx, 1.0, 4.0, 1.5, 45, 'PM')
    if t: trades.append(t)

if not trades:
    print("No trades!"); exit()

tdf = pd.DataFrame(trades).sort_values(['date','time']).reset_index(drop=True)
wins = tdf[tdf['pnl'] > 0]
losses = tdf[tdf['pnl'] <= 0]
sl_trades = tdf[tdf['reason'] == 'SL']
pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses)>0 and losses['pnl'].sum()!=0 else 999

print("="*88)
print("CB 1m (tu 2/5) — 10bar/<0.7x ATR0.7-1.5 RSI<70 | AM:SL1.2/T@5/2x H90 | PM:SL1.0/T@4/1.5x H45")
print("="*88)
print(f"\n{'Date':<12}{'Time':<7}{'Sess':<5}{'Dir':<5}{'Entry':>7}{'Exit':>7}{'ATR':>5}{'MFE':>5}{'PnL':>7}{'Bars':>5}  Reason")
print('-'*78)
for _, t in tdf.iterrows():
    mark = ' <<' if t['pnl'] > 0 else ''
    print(f"{str(t['date']):<12}{t['time']:<7}{t['sess']:<5}{t['dir']:<5}"
          f"{t['entry']:>7.1f}{t['exit']:>7.1f}{t['atr']:>5.2f}"
          f"{t['mfe']:>5.1f}{t['pnl']:>+7.1f}{t['bars']:>5}  {t['reason']}{mark}")

print(f"\n{'='*88}")
print(f"TONG KET: {len(tdf)} trades | Win: {len(wins)} | WR: {len(wins)/len(tdf)*100:.1f}% | PF: {pf:.2f}")
print(f"Total PnL: {tdf['pnl'].sum():+.1f} pts | Avg/day: {tdf['pnl'].sum()/n_days:+.2f}/d")

print("\nExit breakdown:")
for r, cnt in tdf['reason'].value_counts().items():
    sub = tdf[tdf['reason']==r]
    wr = (sub['pnl']>0).mean()*100
    print(f"  {r:<12}: {cnt:>3} trades | WR {wr:>5.1f}% | avg PnL {sub['pnl'].mean():>+5.2f} | avg MFE {sub['mfe'].mean():>4.1f}")

print(f"\n{'='*88}")
print("PHAN TICH SL")
print('='*88)
if len(sl_trades) > 0:
    print(f"\n{len(sl_trades)} SL trades:")
    print(f"  Avg ATR khi SL:           {sl_trades['atr'].mean():.2f}")
    print(f"  Avg SL distance:          {sl_trades['sl_dist'].mean():.2f} pts")
    print(f"  Avg MFE truoc khi bi SL:  {sl_trades['mfe'].mean():.2f} pts")
    print(f"  Max MFE truoc SL:         {sl_trades['mfe'].max():.2f} pts")

    mfe_lt1 = (sl_trades['mfe'] < 1).sum()
    mfe_1_2 = ((sl_trades['mfe']>=1) & (sl_trades['mfe']<2)).sum()
    mfe_2_4 = ((sl_trades['mfe']>=2) & (sl_trades['mfe']<4)).sum()
    mfe_ge4 = (sl_trades['mfe'] >= 4).sum()
    print(f"\n  MFE truoc SL (chi tiet):")
    print(f"    < 1 pts (reverse ngay): {mfe_lt1}/{len(sl_trades)} — signal sai huong")
    print(f"    1-2 pts (di yeu roi SL): {mfe_1_2}/{len(sl_trades)}")
    print(f"    2-4 pts (di duoc roi SL): {mfe_2_4}/{len(sl_trades)}")
    print(f"    >= 4 pts (gan BE roi SL): {mfe_ge4}/{len(sl_trades)}")

    print(f"\n  SL theo session:")
    for s in ['AM','PM']:
        sub = sl_trades[sl_trades['sess']==s]
        if len(sub):
            print(f"    {s}: {len(sub)} SLs | avg MFE {sub['mfe'].mean():.2f} | avg ATR {sub['atr'].mean():.2f}")

    print(f"\n  SL detail:")
    for _, t in sl_trades.iterrows():
        print(f"    {str(t['date'])} {t['time']} {t['sess']} {t['dir']} ATR={t['atr']:.2f} SL_dist={t['sl_dist']:.2f} MFE={t['mfe']:.1f} PnL={t['pnl']:+.1f}")
else:
    print("  Khong co SL trades!")

print(f"\n{'='*88}")
print("MFE DISTRIBUTION (Maximum Favorable Excursion)")
print('='*88)
buckets = [(0,1,'0-1'),(1,2,'1-2'),(2,4,'2-4'),(4,6,'4-6'),(6,9,'6-9'),(9,99,'9+')]
print(f"\n{'MFE range':<12} {'N':>4} {'WR':>6} {'AvgPnL':>8}  phan tich")
print('-'*55)
for lo,hi,lbl in buckets:
    sub = tdf[(tdf['mfe']>=lo) & (tdf['mfe']<hi)]
    if len(sub):
        wr = (sub['pnl']>0).mean()*100
        note = ''
        if lo >= 5: note = '(trail active AM)'
        elif lo >= 4: note = '(trail active PM)'
        elif lo >= 0 and hi <= 1: note = '(signal ngu, reverse ngay)'
        print(f"  {lbl:<10} {len(sub):>4} {wr:>5.1f}% {sub['pnl'].mean():>+8.2f}  {note}")

print(f"\n{'='*88}")
print("TP TARGETS")
print('='*88)
am_t = tdf[tdf['sess']=='AM']
pm_t = tdf[tdf['sess']=='PM']
print(f"\nAM session ({len(am_t)} trades):")
print(f"  Dat MFE >= 5 pts (trail activate): {(am_t['mfe']>=5).sum()}/{len(am_t)} = {(am_t['mfe']>=5).mean()*100:.1f}%")
print(f"  Dat MFE >= 8 pts:                  {(am_t['mfe']>=8).sum()}/{len(am_t)} = {(am_t['mfe']>=8).mean()*100:.1f}%")
print(f"  Dat MFE >= 12 pts:                 {(am_t['mfe']>=12).sum()}/{len(am_t)} = {(am_t['mfe']>=12).mean()*100:.1f}%")
print(f"\nPM session ({len(pm_t)} trades):")
print(f"  Dat MFE >= 4 pts (trail activate): {(pm_t['mfe']>=4).sum()}/{len(pm_t)} = {(pm_t['mfe']>=4).mean()*100:.1f}%")
print(f"  Dat MFE >= 8 pts:                  {(pm_t['mfe']>=8).sum()}/{len(pm_t)} = {(pm_t['mfe']>=8).mean()*100:.1f}%")

trail_w = tdf[tdf['reason']=='TRAIL']
if len(trail_w):
    print(f"\nTrailing stop exits: {len(trail_w)} trades | avg PnL {trail_w['pnl'].mean():+.2f} | avg MFE {trail_w['mfe'].mean():.1f}")

# So sanh: neu dung fixed TP = 8
print(f"\nSo sanh Fixed TP = 8 pts vs Trailing stop:")
tp8_trades = tdf[tdf['mfe'] >= 8]
not_tp8 = tdf[tdf['mfe'] < 8]
fixed_tp_pnl = len(tp8_trades)*7.04 + not_tp8[not_tp8['pnl']>0]['pnl'].sum() + not_tp8[not_tp8['pnl']<=0]['pnl'].sum()
print(f"  Fixed TP=8: {len(tp8_trades)} TP hits x +7.04 + others = {fixed_tp_pnl:+.1f} pts est.")
print(f"  Trailing:   {tdf['pnl'].sum():+.1f} pts actual")
print(f"  Trailing winners avg: {wins['pnl'].mean():+.2f} | losers avg: {losses['pnl'].mean():+.2f}")
