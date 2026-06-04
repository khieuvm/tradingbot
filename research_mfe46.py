"""
Research: Filter MFE 4-6 pts zone on CB 5m
- Identify trades reaching 4+ pts MFE
- Compare: 'continue' (MFE 6+) vs 'fail' (MFE 4-6 then reverse)
- Test: ATR, RSI, volume, time, speed-to-4pts, bar pattern
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from src.data_fetcher import DataFetcher

COST = 0.96

fetcher = DataFetcher()
end = datetime.now().strftime("%Y-%m-%d")
start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
df = fetcher.get_futures_ohlcv("VN30F1M", start, end, interval="5m")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df['date'] = df['time'].dt.date
df['mins'] = df['time'].dt.hour*60 + df['time'].dt.minute
df = df[((df['mins']>=9*60)&(df['mins']<11*60+30))|((df['mins']>=13*60)&(df['mins']<14*60+30))]
df = df.reset_index(drop=True)

df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
df['rsi14'] = ta.rsi(df['close'], length=14)
df['ema9'] = ta.ema(df['close'], length=9)
df['ema21'] = ta.ema(df['close'], length=21)
df['session'] = np.where(df['mins']<12*60, 'AM', 'PM')
df['range'] = df['high'] - df['low']
df['body'] = abs(df['close'] - df['open'])
df['vol_sma20'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_sma20'].replace(0, np.nan)

n_days = df['date'].nunique()
print(f"5m data: {len(df)} bars, {n_days} days\n")

# ===================================================================
# Compression detection
# ===================================================================
ranges = df['range'].values; atr_v = df['atr'].values
comp = np.zeros(len(df), dtype=bool)
for i in range(3, len(df)):
    if atr_v[i] > 0 and max(ranges[i-3:i]) < 0.7 * atr_v[i]:
        comp[i] = True
df['comp'] = comp

def dedup(mask, min_bars=5):
    r = mask.copy(); last = -999
    for i in range(len(r)):
        if r.iloc[i]:
            if i-last < min_bars: r.iloc[i] = False
            else: last = i
    return r

filt = (df['atr']<=4.5)&(df['atr']>=2.5)&(df['rsi14']<70).fillna(True)
mAM = dedup(df['comp']&(df['mins']>=9*60+15)&(df['mins']<=10*60+45)&filt)
mPM = dedup(df['comp']&(df['mins']>=13*60+15)&(df['mins']<=14*60+15)&filt)

# ===================================================================
# Simulate with full trade detail (capture bar-by-bar path)
# ===================================================================
def sim_detailed(idx, sl_mult, trail_activate, trail_mult, max_hold, be_atr_min=3.5):
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
    bars_to_4 = None  # how many bars to reach 4 pts MFE

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
            if bars_to_4 is None and mfe >= 4.0: bars_to_4 = bars
            if not be_done and mfe >= 4.0 and atr >= be_atr_min:
                be_done = True; sl = max(sl, entry)
            if mfe >= trail_activate: trail_on = True
            if trail_on:
                sl = max(sl, best - trail_mult*atr)
                if b['low'] <= sl: exit_p=sl; exit_r='TRAIL'; break
        else:
            if b['high'] >= sl: exit_p=sl; exit_r='BE' if be_done else 'SL'; break
            if b['low'] < best: best = b['low']
            mfe = entry - best
            if bars_to_4 is None and mfe >= 4.0: bars_to_4 = bars
            if not be_done and mfe >= 4.0 and atr >= be_atr_min:
                be_done = True; sl = min(sl, entry)
            if mfe >= trail_activate: trail_on = True
            if trail_on:
                sl = min(sl, best + trail_mult*atr)
                if b['high'] >= sl: exit_p=sl; exit_r='TRAIL'; break

    if exit_p is None: exit_p=entry; exit_r='MAX_HOLD'
    mfe_f = (best-entry) if direction==1 else (entry-best)
    pnl = direction*(exit_p-entry) - COST

    # Signal bar characteristics
    sig_bar_range = row['range']
    sig_bar_body = row['body']

    # Pre-signal momentum (last 3 bars)
    pre_bars = df.iloc[max(0, i_pos-3):i_pos]
    pre_move = 0.0
    if len(pre_bars) >= 2:
        pre_move = (pre_bars.iloc[-1]['close'] - pre_bars.iloc[0]['open']) * direction

    # Volume at signal bar
    sig_vol_ratio = row.get('vol_ratio', 1.0)
    if pd.isna(sig_vol_ratio): sig_vol_ratio = 1.0

    # EMA alignment
    ema9 = row.get('ema9', np.nan)
    ema21 = row.get('ema21', np.nan)
    ema_aligned = 0
    if pd.notna(ema9) and pd.notna(ema21):
        if direction == 1 and ema9 > ema21: ema_aligned = 1
        elif direction == -1 and ema9 < ema21: ema_aligned = 1

    return {
        'date': row['date'], 'time': row['time'].strftime('%H:%M'),
        'sess': row['session'], 'mins': row['mins'],
        'dir': direction, 'entry': entry, 'exit': exit_p,
        'atr': atr, 'mfe': mfe_f, 'pnl': pnl, 'reason': exit_r, 'bars': bars,
        # Features for analysis
        'rsi': row['rsi14'], 'vol_ratio': sig_vol_ratio,
        'sig_range_atr': sig_bar_range / atr,
        'sig_body_ratio': sig_bar_body / sig_bar_range if sig_bar_range > 0 else 0,
        'pre_move': pre_move,
        'ema_aligned': ema_aligned,
        'bars_to_4': bars_to_4,
        'atr_regime': 'LOW' if atr < 3.0 else ('HIGH' if atr > 4.0 else 'MID'),
    }

trades = []
for idx in df.index[mAM]:
    t = sim_detailed(idx, 1.2, 5.0, 2.0, 24)
    if t: t['sess_type'] = 'AM'; trades.append(t)
for idx in df.index[mPM]:
    t = sim_detailed(idx, 1.0, 4.0, 1.5, 12)
    if t: t['sess_type'] = 'PM'; trades.append(t)

tdf = pd.DataFrame(trades)
wins = tdf[tdf['pnl']>0]; losses = tdf[tdf['pnl']<=0]
pf_all = wins['pnl'].sum()/abs(losses['pnl'].sum())
print(f"All trades: {len(tdf)} | WR {(tdf['pnl']>0).mean()*100:.1f}% | PF {pf_all:.2f} | +{tdf['pnl'].sum():.1f}pts")

# ===================================================================
# FOCUS: Trades that reached 4+ pts MFE (trail zone)
# ===================================================================
reached_4 = tdf[tdf['mfe'] >= 4.0].copy()
failed = reached_4[reached_4['mfe'] < 6.0]     # reached 4 but stopped < 6
cont = reached_4[reached_4['mfe'] >= 6.0]       # continued beyond 6

print(f"\nTrades dat 4+ pts MFE: {len(reached_4)}/{len(tdf)} ({len(reached_4)/len(tdf)*100:.0f}%)")
print(f"  -> Tiep tuc >= 6 pts (GOOD): {len(cont)} ({len(cont)/len(reached_4)*100:.0f}%)")
print(f"  -> Dung lai 4-6 pts (BAD):   {len(failed)} ({len(failed)/len(reached_4)*100:.0f}%)")

print(f"\n{'='*75}")
print("SO SANH: 'TIEP TUC >= 6pts' vs 'DUNG LAI 4-6pts'")
print('='*75)

features = [
    ('ATR', 'atr'),
    ('RSI', 'rsi'),
    ('Vol ratio', 'vol_ratio'),
    ('Sig bar range/ATR', 'sig_range_atr'),
    ('Sig bar body ratio', 'sig_body_ratio'),
    ('Pre-move (3 bars)', 'pre_move'),
    ('EMA9>EMA21 aligned', 'ema_aligned'),
    ('Bars to 4 pts', 'bars_to_4'),
]

print(f"\n  {'Feature':<25} {'TIEP_TUC(>=6)':>15} {'DUNG_LAI(4-6)':>15} {'Diff':>8}")
print('  ' + '-'*65)
for label, col in features:
    c_val = cont[col].mean() if col in cont.columns else np.nan
    f_val = failed[col].mean() if col in failed.columns else np.nan
    diff = c_val - f_val if pd.notna(c_val) and pd.notna(f_val) else np.nan
    marker = ' **' if abs(diff) > 0.1 * abs(f_val + 0.001) else ''
    print(f"  {label:<25} {c_val:>15.2f} {f_val:>15.2f} {diff:>+8.2f}{marker}")

# ===================================================================
# ATR regime breakdown
# ===================================================================
print(f"\n{'='*75}")
print("BREAKDOWN THEO ATR REGIME (toan bo trades)")
print('='*75)
print(f"\n  {'ATR range':<12} {'T':>4} {'WR':>6} {'PF':>5} {'P/D':>7} | 4-6 fail rate")
print('  '+'-'*55)
for regime, lo, hi in [('LOW <3.0', 2.5, 3.0), ('MID 3-4', 3.0, 4.0), ('HIGH >4', 4.0, 4.5)]:
    sub = tdf[(tdf['atr']>=lo)&(tdf['atr']<hi)]
    if len(sub) == 0: continue
    w = sub[sub['pnl']>0]; l = sub[sub['pnl']<=0]
    spf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else 999
    sub_4 = sub[sub['mfe']>=4.0]
    fail_4_6 = sub_4[(sub_4['mfe']>=4.0)&(sub_4['mfe']<6.0)]
    rate = len(fail_4_6)/len(sub_4)*100 if len(sub_4)>0 else 0
    print(f"  {regime:<12} {len(sub):>4} {(sub['pnl']>0).mean()*100:>5.1f}% {spf:>5.2f} "
          f"{sub['pnl'].sum()/n_days:>+6.2f}/d | {len(fail_4_6)}/{len(sub_4)} = {rate:.0f}%")

# ===================================================================
# RSI breakdown at signal
# ===================================================================
print(f"\n{'='*75}")
print("BREAKDOWN THEO RSI (toan bo trades)")
print('='*75)
print(f"\n  {'RSI range':<15} {'T':>4} {'WR':>6} {'PF':>5} | MFE 4-6 fail")
print('  '+'-'*50)
for label, lo, hi in [('<30',0,30),('30-45',30,45),('45-55',45,55),('55-70',55,70)]:
    sub = tdf[(tdf['rsi']>=lo)&(tdf['rsi']<hi)]
    if len(sub) < 3: continue
    w = sub[sub['pnl']>0]; l = sub[sub['pnl']<=0]
    spf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else 999
    sub_4 = sub[sub['mfe']>=4.0]
    fail = sub_4[(sub_4['mfe']<6.0)]
    rate = len(fail)/len(sub_4)*100 if len(sub_4)>0 else 0
    print(f"  {label:<15} {len(sub):>4} {(sub['pnl']>0).mean()*100:>5.1f}% {spf:>5.2f} | "
          f"{len(fail)}/{len(sub_4)} = {rate:.0f}%")

# ===================================================================
# EMA alignment
# ===================================================================
print(f"\n{'='*75}")
print("BREAKDOWN: EMA9 vs EMA21 alignment")
print('='*75)
for aligned, lbl in [(1,'EMA9 aligned w/ direction'), (0,'EMA9 NOT aligned')]:
    sub = tdf[tdf['ema_aligned']==aligned]
    if len(sub) == 0: continue
    w = sub[sub['pnl']>0]; l = sub[sub['pnl']<=0]
    spf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else 999
    sub_4 = sub[sub['mfe']>=4.0]
    fail = sub_4[sub_4['mfe']<6.0]
    rate = len(fail)/len(sub_4)*100 if len(sub_4)>0 else 0
    print(f"  {lbl:<35}: {len(sub):>4} trades | WR {(sub['pnl']>0).mean()*100:>5.1f}% | PF {spf:>5.2f} | "
          f"fail@4-6: {len(fail)}/{len(sub_4)} = {rate:.0f}%")

# ===================================================================
# Volume ratio
# ===================================================================
print(f"\n{'='*75}")
print("BREAKDOWN: Volume ratio luc signal")
print('='*75)
for label, lo, hi in [('<0.8',0,0.8),('0.8-1.2',0.8,1.2),('>1.2',1.2,99)]:
    sub = tdf[(tdf['vol_ratio']>=lo)&(tdf['vol_ratio']<hi)]
    if len(sub) < 3: continue
    w = sub[sub['pnl']>0]; l = sub[sub['pnl']<=0]
    spf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else 999
    sub_4 = sub[sub['mfe']>=4.0]
    fail = sub_4[sub_4['mfe']<6.0]
    rate = len(fail)/len(sub_4)*100 if len(sub_4)>0 else 0
    print(f"  Vol {label:<10}: {len(sub):>4} | WR {(sub['pnl']>0).mean()*100:>5.1f}% | PF {spf:>5.2f} | "
          f"fail@4-6: {len(fail)}/{len(sub_4)} = {rate:.0f}%")

# ===================================================================
# Speed: bars to reach 4 pts MFE
# ===================================================================
print(f"\n{'='*75}")
print("BREAKDOWN: Toc do dat 4pts MFE (chi trades dat 4+ pts)")
print('='*75)
reached_clean = reached_4[reached_4['bars_to_4'].notna()].copy()
reached_clean['bars_to_4'] = reached_clean['bars_to_4'].astype(int)
for label, lo, hi in [('1-2 bars',1,3),('3-5 bars',3,6),('6-10 bars',6,11),('>10 bars',11,999)]:
    sub = reached_clean[(reached_clean['bars_to_4']>=lo)&(reached_clean['bars_to_4']<hi)]
    if len(sub) < 2: continue
    fail = sub[sub['mfe']<6.0]
    cont_s = sub[sub['mfe']>=6.0]
    rate = len(fail)/len(sub)*100
    avg_final = sub['mfe'].mean()
    print(f"  {label:<12}: {len(sub):>4} | fail@4-6: {len(fail)}/{len(sub)} = {rate:.0f}% | cont>=6: {len(cont_s)} | avg_final_mfe: {avg_final:.1f}")

# ===================================================================
# Pre-signal momentum
# ===================================================================
print(f"\n{'='*75}")
print("BREAKDOWN: Pre-signal momentum (3 bars truoc signal, theo huong trade)")
print('='*75)
for label, lo, hi in [('Against (<0)', -99, 0), ('Neutral 0-1', 0, 1), ('With 1-3', 1, 3), ('Strong >3', 3, 99)]:
    sub = tdf[(tdf['pre_move']>=lo)&(tdf['pre_move']<hi)]
    if len(sub) < 3: continue
    w = sub[sub['pnl']>0]; l = sub[sub['pnl']<=0]
    spf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else 999
    sub_4 = sub[sub['mfe']>=4.0]
    fail = sub_4[sub_4['mfe']<6.0]
    rate = len(fail)/len(sub_4)*100 if len(sub_4)>0 else 0
    print(f"  {label:<18}: {len(sub):>4} | WR {(sub['pnl']>0).mean()*100:>5.1f}% | PF {spf:>5.2f} | "
          f"fail@4-6: {len(fail)}/{len(sub_4)} = {rate:.0f}%")

# ===================================================================
# Test candidate filters
# ===================================================================
print(f"\n{'='*75}")
print("TEST CANDIDATE FILTERS (ap dung len toan bo trades)")
print('='*75)
print(f"\n  {'Filter':<45} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'P/D':>7}")
print('  '+'-'*70)

def test_filter(mask, label):
    sub = tdf[mask]
    if len(sub) < 10:
        print(f"  {label:<45} {len(sub):>4} (too few)"); return
    w = sub[sub['pnl']>0]; l = sub[sub['pnl']<=0]
    spf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else 999
    print(f"  {label:<45} {len(sub):>4} {(sub['pnl']>0).mean()*100:>5.1f}% {spf:>5.2f} "
          f"{sub['pnl'].sum():>+8.1f} {sub['pnl'].sum()/n_days:>+6.2f}/d")

# Baseline
test_filter(pd.Series([True]*len(tdf), index=tdf.index), 'BASELINE (all)')

# ATR filters
test_filter(tdf['atr'] < 3.5, 'ATR < 3.5 (low vol)')
test_filter(tdf['atr'] >= 3.5, 'ATR >= 3.5 (high vol)')

# RSI filters
test_filter(tdf['rsi'] < 50, 'RSI < 50')
test_filter(tdf['rsi'] >= 50, 'RSI >= 50')
test_filter((tdf['rsi']>=30)&(tdf['rsi']<55), 'RSI 30-55')

# EMA alignment
test_filter(tdf['ema_aligned']==1, 'EMA9 aligned with direction')
test_filter(tdf['ema_aligned']==0, 'EMA9 against direction')

# Volume
test_filter(tdf['vol_ratio'] >= 1.0, 'Vol >= 1.0x avg')
test_filter(tdf['vol_ratio'] >= 1.2, 'Vol >= 1.2x avg')

# Pre-move
test_filter(tdf['pre_move'] > 0, 'Pre-move > 0 (momentum with)')
test_filter(tdf['pre_move'] > 1.0, 'Pre-move > 1 pt (strong momentum)')
test_filter(tdf['pre_move'] <= 0, 'Pre-move <= 0 (counter-trend)')

# Combinations
test_filter((tdf['atr']>=3.0)&(tdf['atr']<4.5)&(tdf['ema_aligned']==1), 'ATR 3-4.5 + EMA aligned')
test_filter((tdf['vol_ratio']>=1.0)&(tdf['ema_aligned']==1), 'Vol>=1.0 + EMA aligned')
test_filter((tdf['pre_move']>0)&(tdf['ema_aligned']==1), 'Pre-momentum + EMA aligned')
test_filter((tdf['atr']>=3.0)&(tdf['vol_ratio']>=1.0)&(tdf['pre_move']>0), 'ATR>=3 + Vol>=1.0 + pre-move>0')
test_filter((tdf['sess_type']=='PM'), 'PM only')
test_filter((tdf['sess_type']=='PM')|(tdf['ema_aligned']==1), 'PM + (AM w/ EMA aligned)')
test_filter((tdf['sess_type']=='PM')|(tdf['pre_move']>1.0), 'PM + (AM w/ pre-move>1)')

print(f"\n--- PM only detail ---")
pm_sub = tdf[tdf['sess_type']=='PM']
w = pm_sub[pm_sub['pnl']>0]; l = pm_sub[pm_sub['pnl']<=0]
spf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else 999
pm_4 = pm_sub[pm_sub['mfe']>=4.0]
pm_fail = pm_4[pm_4['mfe']<6.0]
print(f"  PM: {len(pm_sub)} trades | WR {(pm_sub['pnl']>0).mean()*100:.1f}% | PF {spf:.2f} | "
      f"fail@4-6: {len(pm_fail)}/{len(pm_4)} = {len(pm_fail)/len(pm_4)*100 if len(pm_4)>0 else 0:.0f}%")
