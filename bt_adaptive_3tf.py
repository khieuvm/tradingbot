"""
Adaptive Exit Backtest — All 3 TFs (1m, 3m, 5m)
Rule: At 4pts MFE on AM session, if pre_move > threshold*ATR → exit immediately
Threshold sweep + AM/PM breakdown + compare vs baseline
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta, date
from src.data_fetcher import DataFetcher

COST = 0.96
CUTOFF_SHORT = date(2026, 5, 2)

fetcher = DataFetcher()
end = datetime.now().strftime("%Y-%m-%d")


# ===================================================================
def load(tf, days=180, cutoff=None):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = fetcher.get_futures_ohlcv("VN30F1M", start, end, interval=tf)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour*60 + df['time'].dt.minute
    df = df[((df['mins']>=9*60)&(df['mins']<11*60+30))|((df['mins']>=13*60)&(df['mins']<14*60+30))]
    if cutoff:
        df = df[df['date'] >= cutoff]
    df = df.reset_index(drop=True)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20'].replace(0, np.nan)
    df['session'] = np.where(df['mins']<12*60, 'AM', 'PM')
    df['range'] = df['high'] - df['low']
    return df


def detect_comp(df, n_bars, threshold=0.7):
    ranges = df['range'].values; atr_v = df['atr'].values
    comp = np.zeros(len(df), dtype=bool)
    for i in range(n_bars, len(df)):
        if atr_v[i] > 0 and max(ranges[i-n_bars:i]) < threshold*atr_v[i]:
            comp[i] = True
    df['comp'] = comp
    return df


def dedup(mask, min_bars):
    r = mask.copy(); last = -999
    for i in range(len(r)):
        if r.iloc[i]:
            if i-last < min_bars: r.iloc[i] = False
            else: last = i
    return r


def get_masks(df, n_bars, atr_lo, atr_hi, ded_bars, threshold=0.7):
    df = detect_comp(df, n_bars, threshold)
    filt = (df['atr']<=atr_hi)&(df['atr']>=atr_lo)&(df['rsi14']<70).fillna(True)
    mAM = dedup(df['comp']&(df['mins']>=9*60+15)&(df['mins']<=10*60+45)&filt, ded_bars)
    mPM = dedup(df['comp']&(df['mins']>=13*60+15)&(df['mins']<=14*60+15)&filt, ded_bars)
    return df, mAM, mPM


def sim_adaptive(df, idx, sl_mult, trail_activate, trail_mult, max_hold,
                 be_trigger=4.0, be_atr_min=2.5,
                 adapt_premove_ratio=None):  # None=baseline, else exit@4 if pre_move/ATR > ratio
    row = df.loc[idx]; atr = row['atr']
    if pd.isna(atr) or atr <= 0: return None
    i_pos = df.index.get_loc(idx)
    if i_pos+1 >= len(df): return None
    nxt = df.iloc[i_pos+1]
    if nxt['date'] != row['date'] or nxt['session'] != row['session']: return None
    direction = 1 if nxt['close'] > row['close'] else -1
    entry = row['close']

    # Pre-signal momentum (3 bars before signal, in trade direction)
    pre_bars = df.iloc[max(0, i_pos-3):i_pos]
    pre_move = 0.0
    if len(pre_bars) >= 2:
        pre_move = (pre_bars.iloc[-1]['close'] - pre_bars.iloc[0]['open']) * direction
    pre_ratio = pre_move / atr if atr > 0 else 0.0

    # Adaptive: AM only, triggered at 4pts if pre_ratio exceeds threshold
    is_adapt = (adapt_premove_ratio is not None
                and row['session'] == 'AM'
                and pre_ratio > adapt_premove_ratio)

    sl = entry - direction*sl_mult*atr
    best = entry; trail_on = False; be_done = False; adapt_fired = False
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
            # Adaptive exit at 4pts
            if is_adapt and not adapt_fired and mfe >= 4.0:
                adapt_fired = True
                exit_p = b['close']; exit_r = 'ADAPT_EXIT'; break
            # Normal BE
            if not be_done and mfe >= be_trigger and atr >= be_atr_min:
                be_done = True; sl = max(sl, entry)
            if mfe >= trail_activate: trail_on = True
            if trail_on:
                sl = max(sl, best - trail_mult*atr)
                if b['low'] <= sl: exit_p=sl; exit_r='TRAIL'; break
        else:
            if b['high'] >= sl: exit_p=sl; exit_r='BE' if be_done else 'SL'; break
            if b['low'] < best: best = b['low']
            mfe = entry - best
            if is_adapt and not adapt_fired and mfe >= 4.0:
                adapt_fired = True
                exit_p = b['close']; exit_r = 'ADAPT_EXIT'; break
            if not be_done and mfe >= be_trigger and atr >= be_atr_min:
                be_done = True; sl = min(sl, entry)
            if mfe >= trail_activate: trail_on = True
            if trail_on:
                sl = min(sl, best + trail_mult*atr)
                if b['high'] >= sl: exit_p=sl; exit_r='TRAIL'; break

    if exit_p is None: exit_p=entry; exit_r='MAX_HOLD'
    mfe_f = (best-entry) if direction==1 else (entry-best)
    pnl = direction*(exit_p-entry) - COST
    return {'date': row['date'], 'time': row['time'].strftime('%H:%M'),
            'sess': row['session'], 'dir': 'BUY' if direction==1 else 'SELL',
            'entry': entry, 'exit': exit_p, 'atr': atr,
            'mfe': mfe_f, 'pnl': pnl, 'reason': exit_r, 'bars': bars,
            'pre_ratio': pre_ratio, 'is_adapt': is_adapt}


def run_sim(df, mAM, mPM, am_sl, am_trail, am_mult, am_hold,
            pm_sl, pm_trail, pm_mult, pm_hold, be_atr_min, adapt_ratio):
    trades = []
    for idx in df.index[mAM]:
        t = sim_adaptive(df, idx, am_sl, am_trail, am_mult, am_hold,
                         be_atr_min=be_atr_min, adapt_premove_ratio=adapt_ratio)
        if t: trades.append(t)
    for idx in df.index[mPM]:
        t = sim_adaptive(df, idx, pm_sl, pm_trail, pm_mult, pm_hold,
                         be_atr_min=be_atr_min, adapt_premove_ratio=adapt_ratio)
        if t: trades.append(t)
    if not trades: return None
    return pd.DataFrame(trades).sort_values(['date','time']).reset_index(drop=True)


def metrics(tdf):
    if tdf is None or len(tdf) == 0: return None
    w = tdf[tdf['pnl']>0]; l = tdf[tdf['pnl']<=0]
    pf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else 999
    return {'n': len(tdf), 'wr': (tdf['pnl']>0).mean()*100,
            'pf': pf, 'pnl': tdf['pnl'].sum(), 'mfe': tdf['mfe'].mean()}


def print_row(label, tdf, n_days, width=58):
    if tdf is None:
        print(f"  {label:<{width}}   — no trades"); return
    m = metrics(tdf)
    adapt_n = (tdf['reason']=='ADAPT_EXIT').sum()
    print(f"  {label:<{width}} {m['n']:>4} {m['wr']:>5.1f}% {m['pf']:>5.2f} "
          f"{m['pnl']:>+8.1f} {m['pnl']/n_days:>+5.2f}/d  adapt={adapt_n}")


def detail_ampm(tdf, n_days, label):
    if tdf is None: return
    wins = tdf[tdf['pnl']>0]; losses = tdf[tdf['pnl']<=0]
    pf = wins['pnl'].sum()/abs(losses['pnl'].sum()) if len(losses)>0 and losses['pnl'].sum()!=0 else 999
    adapt_n = (tdf['reason']=='ADAPT_EXIT').sum()
    print(f"\n{'='*90}")
    print(f"DETAIL: {label}")
    print(f"  Total: {len(tdf)} | WR {(tdf['pnl']>0).mean()*100:.1f}% | PF {pf:.2f} | "
          f"PnL {tdf['pnl'].sum():+.1f} | {tdf['pnl'].sum()/n_days:+.2f}/d | adapt_exits={adapt_n}")
    print('='*90)

    print(f"\n  {'Sess':<6} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} | Exit reasons")
    print('  '+'-'*60)
    for s in ['AM', 'PM']:
        sub = tdf[tdf['sess']==s]
        if len(sub) == 0: continue
        w = sub[sub['pnl']>0]; l = sub[sub['pnl']<=0]
        spf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else 999
        exits = sub['reason'].value_counts().to_dict()
        print(f"  {s:<6} {len(sub):>4} {(sub['pnl']>0).mean()*100:>5.1f}% {spf:>5.2f} "
              f"{sub['pnl'].sum():>+8.1f}  {exits}")

    # Adapt trades breakdown
    adapt_t = tdf[tdf['is_adapt']==True]
    normal_t = tdf[tdf['is_adapt']==False]
    if len(adapt_t) > 0:
        aw = adapt_t[adapt_t['pnl']>0]; al = adapt_t[adapt_t['pnl']<=0]
        apf = aw['pnl'].sum()/abs(al['pnl'].sum()) if len(al)>0 and al['pnl'].sum()!=0 else 999
        print(f"\n  Adapt targets (AM pre_ratio>thresh): {len(adapt_t)} trades | "
              f"WR {(adapt_t['pnl']>0).mean()*100:.1f}% | PF {apf:.2f} | PnL {adapt_t['pnl'].sum():+.1f}")
        print(f"    Exited@4pts: {adapt_n} | Missed 4pts (SL/early): {len(adapt_t)-adapt_n}")
        if adapt_n > 0:
            ex4 = tdf[tdf['reason']=='ADAPT_EXIT']
            print(f"    Adapt exits: avg pnl {ex4['pnl'].mean():+.2f} | avg MFE {ex4['mfe'].mean():.1f}")
    if len(normal_t) > 0:
        nw = normal_t[normal_t['pnl']>0]; nl = normal_t[normal_t['pnl']<=0]
        npf = nw['pnl'].sum()/abs(nl['pnl'].sum()) if len(nl)>0 and nl['pnl'].sum()!=0 else 999
        print(f"  Normal trades: {len(normal_t)} | WR {(normal_t['pnl']>0).mean()*100:.1f}% | "
              f"PF {npf:.2f} | PnL {normal_t['pnl'].sum():+.1f}")

    # Time window breakdown
    tmins = tdf['time'].apply(lambda x: int(x.split(':')[0])*60+int(x.split(':')[1]))
    print(f"\n  {'Window':<22} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'SL':>4} {'avgMFE':>7}")
    print('  '+'-'*60)
    for wlabel, wlo, whi in [
        ('AM 09:15-10:00', 9*60+15, 10*60),
        ('AM 10:00-10:45', 10*60, 10*60+45),
        ('PM 13:15-13:45', 13*60+15, 13*60+45),
        ('PM 13:45-14:15', 13*60+45, 14*60+15),
    ]:
        sub = tdf[(tmins>=wlo)&(tmins<whi)]
        if len(sub) == 0: continue
        w2 = sub[sub['pnl']>0]; l2 = sub[sub['pnl']<=0]
        spf2 = w2['pnl'].sum()/abs(l2['pnl'].sum()) if len(l2)>0 and l2['pnl'].sum()!=0 else 999
        sl_n = (sub['reason']=='SL').sum()
        print(f"  {wlabel:<22} {len(sub):>4} {(sub['pnl']>0).mean()*100:>5.1f}% {spf2:>5.2f} "
              f"{sub['pnl'].sum():>+8.1f} {sl_n:>4} {sub['mfe'].mean():>7.1f}")

    # MFE distribution
    print(f"\n  MFE distribution:")
    for lo, hi, lbl in [(0,2,'0-2'),(2,4,'2-4'),(4,6,'4-6'),(6,9,'6-9'),(9,99,'9+')]:
        sub = tdf[(tdf['mfe']>=lo)&(tdf['mfe']<hi)]
        if len(sub):
            wr = (sub['pnl']>0).mean()*100
            print(f"    MFE {lbl}: {len(sub):>4} | WR {wr:>5.1f}% | avg PnL {sub['pnl'].mean():>+5.2f}")


# ===================================================================
# LOAD DATA
# ===================================================================
print("Loading data...")
df5 = load('5m', days=180)
df3 = load('3m', days=60, cutoff=CUTOFF_SHORT)
df1 = load('1m', days=60, cutoff=CUTOFF_SHORT)

n5 = df5['date'].nunique()
n3 = df3['date'].nunique()
n1 = df1['date'].nunique()
print(f"  5m: {n5}d  |  3m: {n3}d (from {CUTOFF_SHORT})  |  1m: {n1}d (from {CUTOFF_SHORT})\n")

# Configs per TF
TF_CONFIGS = {
    '5m': dict(df=df5, n_days=n5, n_bars=3, atr_lo=2.5, atr_hi=4.5, ded=5,
               am_sl=1.2, am_trail=5.0, am_mult=2.0, am_hold=24,
               pm_sl=1.0, pm_trail=4.0, pm_mult=1.5, pm_hold=12, be_atr_min=3.5),
    '3m': dict(df=df3, n_days=n3, n_bars=5, atr_lo=1.5, atr_hi=3.0, ded=8,
               am_sl=1.2, am_trail=5.0, am_mult=2.0, am_hold=40,
               pm_sl=1.0, pm_trail=4.0, pm_mult=1.5, pm_hold=20, be_atr_min=2.5),
    '1m': dict(df=df1, n_days=n1, n_bars=10, atr_lo=0.7, atr_hi=1.5, ded=15,
               am_sl=1.2, am_trail=5.0, am_mult=2.0, am_hold=90,
               pm_sl=1.0, pm_trail=4.0, pm_mult=1.5, pm_hold=45, be_atr_min=1.0),
}

# Adaptive thresholds to sweep (pre_move / ATR)
# 5m: pre>3 with ATR~3.5 ≈ ratio 0.86 → test 0.6, 0.8, 1.0
THRESHOLDS = [None, 0.6, 0.8, 1.0, 1.2]

HDR = f"  {'Config':<58} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'P/D':>6}  adapt="

for tf_name, cfg in TF_CONFIGS.items():
    df_tf = cfg['df'].copy()
    df_tf, mAM, mPM = get_masks(df_tf, cfg['n_bars'], cfg['atr_lo'], cfg['atr_hi'], cfg['ded'])
    n_d = cfg['n_days']

    print(f"\n{'='*92}")
    print(f"CB {tf_name} — Adaptive Exit Sweep | pre_move/ATR threshold (AM only)")
    print(f"Signals: AM={mAM.sum()} PM={mPM.sum()} | {n_d} days")
    print('='*92)
    print(HDR)
    print('  '+'-'*86)

    best_pnl = -999; best_tdf = None; best_label = ''
    for thresh in THRESHOLDS:
        label = f'BASELINE' if thresh is None else f'AM exit@4 if pre/ATR > {thresh:.1f}'
        tdf = run_sim(df_tf, mAM, mPM,
                      cfg['am_sl'], cfg['am_trail'], cfg['am_mult'], cfg['am_hold'],
                      cfg['pm_sl'], cfg['pm_trail'], cfg['pm_mult'], cfg['pm_hold'],
                      cfg['be_atr_min'], thresh)
        print_row(label, tdf, n_d)
        if tdf is not None and tdf['pnl'].sum() > best_pnl:
            best_pnl = tdf['pnl'].sum(); best_tdf = tdf; best_label = label

    # Detail for best
    detail_ampm(best_tdf, n_d, f"CB {tf_name} — {best_label}")

print("\n\nDone.")
