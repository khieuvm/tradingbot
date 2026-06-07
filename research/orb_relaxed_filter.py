"""Quick check: 3m and 1m ORB with adapted OR range filters."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pandas_ta as pta
from datetime import datetime, timedelta, date
from src.data_fetcher import DataFetcher

COST = 0.96
CUTOFF = date(2026, 5, 2)
fetcher = DataFetcher()
end_str = datetime.now().strftime("%Y-%m-%d")

AM_SL=1.2; AM_TRAIL=5.0; AM_MULT=2.0
PM_SL=1.0; PM_TRAIL=4.0; PM_MULT=1.5
BE_TRIGGER=4.0

def load_tf(tf, days, cutoff=None):
    start_str = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    df = fetcher.get_futures_ohlcv("VN30F1M", start_str, end_str, interval=tf)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df = df[((df['mins']>=540)&(df['mins']<690))|((df['mins']>=780)&(df['mins']<870))].reset_index(drop=True)
    if cutoff:
        df = df[df['date'] >= cutoff].reset_index(drop=True)
    df['session'] = np.where(df['mins']<720,'AM','PM')
    df['atr'] = pta.atr(df['high'],df['low'],df['close'],length=14)
    return df

def detect_signals(df, n_or_bars, atr_lo, atr_hi, or_lo, or_hi, dir_filter='BOTH'):
    AM_HI=10*60+45; PM_HI=14*60+15
    signals=[]
    for (dt,sess), grp in df.groupby(['date','session'],sort=True):
        if len(grp)<=n_or_bars: continue
        og=grp.iloc[:n_or_bars]; pg=grp.iloc[n_or_bars:]
        if len(pg)==0: continue
        oh=og['high'].max(); ol=og['low'].min(); orng=oh-ol
        atr=og.iloc[-1]['atr']
        if pd.isna(atr) or atr<=0 or atr<atr_lo or atr>atr_hi: continue
        if orng<or_lo*atr or orng>or_hi*atr: continue
        or_end_mins=og.iloc[-1]['mins']
        bd=False; sd=False
        for oi in pg.index:
            row=df.loc[oi]; mins=row['mins']
            if sess=='AM' and mins>AM_HI: break
            if sess=='PM' and mins>PM_HI: break
            if not bd and dir_filter in ('BOTH','LONG') and row['high']>oh+0.1:
                signals.append({'idx':oi,'direction':1,'date':dt,'sess':sess,'mins':mins,
                                 'atr':atr,'or_range':orng,'or_ratio':orng/atr,
                                 'mins_after_or':mins-or_end_mins}); bd=True
            if not sd and dir_filter in ('BOTH','SHORT') and row['low']<ol-0.1:
                signals.append({'idx':oi,'direction':-1,'date':dt,'sess':sess,'mins':mins,
                                 'atr':atr,'or_range':orng,'or_ratio':orng/atr,
                                 'mins_after_or':mins-or_end_mins}); sd=True
            if bd and sd: break
    return signals

def sim_trade(df, sig, sl_m, trail_a, trail_m, max_hold, be_atr_min):
    idx=sig['idx']; d=sig['direction']; atr=sig['atr']
    dt=sig['date']; sess=sig['sess']
    i=int(idx)
    if i+1>=len(df): return None
    entry=df.iloc[i]['close']
    sl=entry-d*sl_m*atr; best=entry; ton=False; be=False; ep=None; er=None
    for jp in range(i+1, min(i+1+max_hold,len(df))):
        b=df.iloc[jp]
        if b['date']!=dt or b['session']!=sess:
            ep=df.iloc[jp-1]['close']; er='SESSION'; break
        if (sess=='PM' and b['mins']>=14*60+25) or (sess=='AM' and b['mins']>=11*60+25):
            ep=b['close']; er='SESSION'; break
        if d==1:
            if b['low']<=sl: ep=sl; er='BE' if be else 'SL'; break
            if b['high']>best: best=b['high']
            mfe=best-entry
            if not be and mfe>=BE_TRIGGER and atr>=be_atr_min: be=True; sl=max(sl,entry)
            if mfe>=trail_a: ton=True
            if ton:
                sl=max(sl,best-trail_m*atr)
                if b['low']<=sl: ep=sl; er='TRAIL'; break
        else:
            if b['high']>=sl: ep=sl; er='BE' if be else 'SL'; break
            if b['low']<best: best=b['low']
            mfe=entry-best
            if not be and mfe>=BE_TRIGGER and atr>=be_atr_min: be=True; sl=min(sl,entry)
            if mfe>=trail_a: ton=True
            if ton:
                sl=min(sl,best+trail_m*atr)
                if b['high']>=sl: ep=sl; er='TRAIL'; break
    if ep is None: ep=entry; er='MAX_HOLD'
    mfe_f=(best-entry) if d==1 else (entry-best)
    pnl=d*(ep-entry)-COST
    return {'sess':sess,'dir':'BUY' if d==1 else 'SELL','mfe':mfe_f,'pnl':pnl,'reason':er}

def run_all(df, n_or, atr_lo, atr_hi, or_lo, or_hi, am_hold, pm_hold, be_atr_min, direction='BOTH'):
    sigs = detect_signals(df,n_or,atr_lo,atr_hi,or_lo,or_hi,direction)
    trades=[]
    for s in sigs:
        if s['sess']=='AM': t=sim_trade(df,s,AM_SL,AM_TRAIL,AM_MULT,am_hold,be_atr_min)
        else: t=sim_trade(df,s,PM_SL,PM_TRAIL,PM_MULT,pm_hold,be_atr_min)
        if t: trades.append(t)
    return pd.DataFrame(trades) if trades else None

def show(label, tdf, n_days):
    if tdf is None or len(tdf)==0:
        print(f"  {label:<55}  0 trades"); return
    wins=tdf[tdf['pnl']>0]; losses=tdf[tdf['pnl']<=0]
    pf=wins['pnl'].sum()/abs(losses['pnl'].sum()) if len(losses)>0 and losses['pnl'].sum()!=0 else 999
    sl=(tdf['reason']=='SL').sum()
    print(f"  {label:<55} {len(tdf):>4} {(tdf['pnl']>0).mean()*100:>5.1f}% {pf:>5.2f} "
          f"{tdf['pnl'].sum():>+8.1f} {tdf['pnl'].sum()/n_days:>+6.2f}/d  SL={sl}")

print("Loading data...")
df3=load_tf('3m',60,CUTOFF)
df1=load_tf('1m',60,CUTOFF)
n3=df3['date'].nunique(); n1=df1['date'].nunique()
print(f"  3m: {len(df3)} bars {n3}d  |  1m: {len(df1)} bars {n1}d")

# Adapted filters based on observed OR/ATR distributions
# 3m: p10=1.61, median=2.34, p90=4.11 -> use [1.5, 4.0] to capture ~80% of sessions
# 1m: p10=3.16, median=4.58, p90=6.87 -> use [2.5, 7.0] to capture ~80% of sessions

print("\n--- 3m ORB with adapted OR range filter [1.5, 4.0]xATR ---")
print(f"  Note: original [0.3,1.5] eliminated 96-100% of sessions")
print(f"  {'Config':<55} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'P/D':>7}")
print('  ' + '-'*75)
for n_or, lbl in [(5,'15min'),(8,'24min'),(10,'30min')]:
    for direction in ['BOTH','LONG','SHORT']:
        tdf = run_all(df3, n_or, 1.5, 3.0, 1.5, 4.0, 40, 20, 2.5, direction)
        show(f"3m ORB {lbl} (N={n_or}) {direction}", tdf, n3)

print("\n--- 1m ORB with adapted OR range filter [2.5, 7.0]xATR ---")
print(f"  Note: original [0.3,1.5] eliminated 100% of sessions")
print(f"  {'Config':<55} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'P/D':>7}")
print('  ' + '-'*75)
for n_or, lbl in [(15,'15min'),(25,'25min'),(30,'30min')]:
    for direction in ['BOTH','LONG','SHORT']:
        tdf = run_all(df1, n_or, 0.7, 1.5, 2.5, 7.0, 120, 60, 1.0, direction)
        show(f"1m ORB {lbl} (N={n_or}) {direction}", tdf, n1)

print("\nDone.")
