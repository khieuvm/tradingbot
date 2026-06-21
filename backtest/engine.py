"""
CB Backtest Engine — Trail Activation Sweep & Performance Analysis.

Usage:
    python -m backtest.engine
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta, date
from pathlib import Path

from src.data_fetcher import DataFetcher
from src.strategy_config import get_combo_config, get_session_params

COST = 0.96
CUTOFF_SHORT = date(2026, 5, 2)
DATA_DIR = Path(__file__).parent.parent / "data"

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
    df['session'] = np.where(df['mins']<12*60, 'AM', 'PM')
    df['range'] = df['high'] - df['low']
    return df


def load_parquet(tf, cutoff=None, start_date=None):
    """Load data from parquet cache (much faster, longer history).

    Args:
        tf: Timeframe ("1m", "3m", "5m")
        cutoff: Only keep data from this date onwards (date object)
        start_date: Alternative to cutoff, string "YYYY-MM-DD"

    Returns:
        DataFrame with columns: time, open, high, low, close, volume, date, mins, atr, rsi14, session, range
    """
    parquet_file = DATA_DIR / f"vn30f1m_{tf}.parquet"
    if not parquet_file.exists():
        print(f"[WARN] Parquet not found: {parquet_file}. Falling back to API.")
        days = 800 if tf == "5m" else 400
        return load(tf, days=days, cutoff=cutoff)

    df = pd.read_parquet(parquet_file)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df = df[((df['mins'] >= 9*60) & (df['mins'] < 11*60+30)) |
            ((df['mins'] >= 13*60) & (df['mins'] < 14*60+30))]

    if cutoff:
        df = df[df['date'] >= cutoff]
    if start_date:
        from datetime import date as _d
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        df = df[df['date'] >= start_date]

    df = df.reset_index(drop=True)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    df['session'] = np.where(df['mins'] < 12*60, 'AM', 'PM')
    df['range'] = df['high'] - df['low']

    days_count = df['date'].nunique()
    print(f"[DATA] Loaded {tf} parquet: {len(df)} bars, {days_count} days "
          f"({df['date'].iloc[0]} to {df['date'].iloc[-1]})")
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


def sim_all(df, maskAM, maskPM,
            am_sl, am_trail, am_mult, am_hold,
            pm_sl, pm_trail, pm_mult, pm_hold,
            be_trigger=3.0, be_atr_min=3.5, be_partial=1.0):
    trades = []

    def run(idx, sl_mult, trail_activate, trail_mult, max_hold):
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
                if not be_done and mfe >= be_trigger and atr >= be_atr_min:
                    be_done = True; sl = max(sl, entry + be_partial)
                if mfe >= trail_activate: trail_on = True
                if trail_on:
                    sl = max(sl, best - trail_mult*atr)
                    if b['low'] <= sl: exit_p=sl; exit_r='TRAIL'; break
            else:
                if b['high'] >= sl: exit_p=sl; exit_r='BE' if be_done else 'SL'; break
                if b['low'] < best: best = b['low']
                mfe = entry - best
                if not be_done and mfe >= be_trigger and atr >= be_atr_min:
                    be_done = True; sl = min(sl, entry - be_partial)
                if mfe >= trail_activate: trail_on = True
                if trail_on:
                    sl = min(sl, best + trail_mult*atr)
                    if b['high'] >= sl: exit_p=sl; exit_r='TRAIL'; break
        if exit_p is None: exit_p=entry; exit_r='MAX_HOLD'
        mfe_f = (best-entry) if direction==1 else (entry-best)
        pnl = direction*(exit_p-entry) - COST
        return {'date':row['date'], 'time':row['time'].strftime('%H:%M'),
                'sess':row['session'], 'mins': row['mins'],
                'dir':'BUY' if direction==1 else 'SELL',
                'entry':entry, 'exit':exit_p, 'atr':atr,
                'mfe':mfe_f, 'pnl':pnl, 'reason':exit_r, 'bars':bars}

    for idx in df.index[maskAM]:
        t = run(idx, am_sl, am_trail, am_mult, am_hold)
        if t: trades.append(t)
    for idx in df.index[maskPM]:
        t = run(idx, pm_sl, pm_trail, pm_mult, pm_hold)
        if t: trades.append(t)

    if not trades: return None
    return pd.DataFrame(trades).sort_values(['date','time']).reset_index(drop=True)


def metrics(tdf):
    if tdf is None or len(tdf) == 0: return None
    wins = tdf[tdf['pnl']>0]; losses = tdf[tdf['pnl']<=0]
    pf = wins['pnl'].sum()/abs(losses['pnl'].sum()) if len(losses)>0 and losses['pnl'].sum()!=0 else 999
    return {'n': len(tdf), 'wr': (tdf['pnl']>0).mean()*100, 'pf': pf,
            'pnl': tdf['pnl'].sum(), 'mfe': tdf['mfe'].mean()}


def run_cb(df, n_bars, threshold, atr_min, atr_max, dedup_b,
           am_sl, am_trail, am_mult, am_hold,
           pm_sl, pm_trail, pm_mult, pm_hold, be_atr_min=3.5, be_trigger=3.0,
           be_partial=1.0, atr_min_pm=None, atr_max_pm=None, dead_zones=None):
    atr_min_pm = atr_min_pm or atr_min
    atr_max_pm = atr_max_pm or atr_max
    df2 = detect_comp(df.copy(), n_bars=n_bars, threshold=threshold)
    filt_am = (df2['atr'] <= atr_max) & (df2['atr'] >= atr_min) & (df2['rsi14']<70).fillna(True)
    filt_pm = (df2['atr'] <= atr_max_pm) & (df2['atr'] >= atr_min_pm) & (df2['rsi14']<70).fillna(True)
    am_time = (df2['mins']>=9*60+15)&(df2['mins']<=10*60+45)
    # Exclude dead zones from AM
    if dead_zones:
        for slot_start, slot_end in dead_zones:
            am_time = am_time & ~((df2['mins']>=slot_start)&(df2['mins']<slot_end))
    mAM = dedup(df2['comp']&am_time&filt_am, dedup_b)
    mPM = dedup(df2['comp']&(df2['mins']>=13*60+15)&(df2['mins']<=14*60+15)&filt_pm, dedup_b)
    return sim_all(df2, mAM, mPM,
                   am_sl, am_trail, am_mult, am_hold,
                   pm_sl, pm_trail, pm_mult, pm_hold,
                   be_trigger=be_trigger, be_atr_min=be_atr_min, be_partial=be_partial)


def print_breakdown(tdf, n_days, label=''):
    if tdf is None or len(tdf) == 0:
        print(f"  {label}: no trades"); return
    wins = tdf[tdf['pnl']>0]; losses = tdf[tdf['pnl']<=0]
    pf = wins['pnl'].sum()/abs(losses['pnl'].sum()) if len(losses)>0 and losses['pnl'].sum()!=0 else 999
    sl = tdf[tdf['reason']=='SL']
    print(f"  {label:<55} {len(tdf):>4} {(tdf['pnl']>0).mean()*100:>5.1f}% {pf:>5.2f} "
          f"{tdf['pnl'].sum():>+8.1f} {tdf['pnl'].sum()/n_days:>+5.2f}/d  SL={len(sl)}")


# ===================================================================
# MAIN — run when called as `python -m backtest.engine`
# ===================================================================
def main():
    print("Loading data...")
    df5 = load('5m', days=180)
    df3 = load('3m', days=60, cutoff=CUTOFF_SHORT)
    df1 = load('1m', days=60, cutoff=CUTOFF_SHORT)
    n5 = df5['date'].nunique()
    n3 = df3['date'].nunique()
    n1 = df1['date'].nunique()
    print(f"  5m: {len(df5)} bars, {n5}d | 3m: {len(df3)} bars, {n3}d | 1m: {len(df1)} bars, {n1}d")

    # Load 5m config from YAML (single source of truth)
    cb_cfg = get_combo_config("CB")
    sp = get_session_params()
    cb_comp = cb_cfg.get("compression", {})
    cb_ef = cb_cfg.get("entry_filter", {})
    cb_5m_nb = cb_comp.get("n_bars", 3)
    cb_5m_thresh = cb_comp.get("threshold", 0.7)
    cb_5m_atr_lo = cb_ef.get("atr_min", 2.5)
    cb_5m_atr_hi = cb_ef.get("atr_max", 4.5)
    cb_5m_atr_lo_pm = cb_ef.get("atr_min_pm", cb_5m_atr_lo)
    cb_5m_atr_hi_pm = cb_ef.get("atr_max_pm", cb_5m_atr_hi)
    cb_5m_ded = cb_cfg.get("dedup_bars", 5)
    am_sp = sp.get("AM", {})
    pm_sp = sp.get("PM", {})

    # Parse dead zones from config
    cb_dead_zones = []
    for slot in cb_cfg.get("dead_zones", []):
        parts = str(slot).split("-")
        if len(parts) == 2:
            def _parse(s):
                p = s.strip().split(":")
                return int(p[0])*60+int(p[1])
            cb_dead_zones.append((_parse(parts[0]), _parse(parts[1])))

    HDR = f"  {'Config':<55} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'P/D':>7}  SL_cnt"

    # ===================================================================
    # TRAIL ACTIVATION SWEEP: 3 / 4 / 5 pts
    # ===================================================================
    trail_vals = [3.0, 4.0, 5.0]

    for tf_name, df_tf, n_d, nb, atr_lo, atr_hi, ded, hold_am, hold_pm, be_min, be_trig, be_part, atr_lo_pm, atr_hi_pm, dz in [
        ('5m', df5, n5, cb_5m_nb, cb_5m_atr_lo, cb_5m_atr_hi, cb_5m_ded,
         int(am_sp.get("max_hold_bars", 24)), int(pm_sp.get("max_hold_bars", 12)),
         float(am_sp.get("be_atr_min", 3.5)), float(am_sp.get("be_trigger_pts", 3.0)),
         float(am_sp.get("be_partial_pts", 1.0)),
         cb_5m_atr_lo_pm, cb_5m_atr_hi_pm, cb_dead_zones),
        ('3m', df3, n3, 5, 1.5, 3.0, 8, 40, 20, 2.5, 3.0, 1.0, None, None, None),
        ('1m', df1, n1, 10, 0.7, 1.5, 15, 90, 45, 1.0, 3.0, 1.0, None, None, None),
    ]:
        print(f'\n{"="*90}')
        print(f'CB {tf_name} — Trail Activation Sweep | SL: AM=1.2xATR PM=1.0xATR | Trail: AM=2.0x PM=1.5x')
        print('='*90)
        print(HDR)
        print('  '+'-'*84)
        for ta_val in trail_vals:
            label = f'{tf_name} CB {nb}bar | Trail@+{ta_val:.0f}pts (AM=2x PM=1.5x) | BE@{be_trig} ATR>={be_min}'
            tdf = run_cb(df_tf, nb, cb_5m_thresh if tf_name == '5m' else 0.7, atr_lo, atr_hi, ded,
                         am_sl=1.2, am_trail=ta_val, am_mult=2.0, am_hold=hold_am,
                         pm_sl=1.0, pm_trail=max(ta_val-1, 2.0), pm_mult=1.5, pm_hold=hold_pm,
                         be_atr_min=be_min, be_trigger=be_trig, be_partial=be_part,
                         atr_min_pm=atr_lo_pm, atr_max_pm=atr_hi_pm,
                         dead_zones=dz)
            print_breakdown(tdf, n_d, label)

    # ===================================================================
    # BEST CONFIG DETAIL: AM/PM + TIME WINDOW BREAKDOWN
    # ===================================================================
    best_configs = {
        '5m': (df5, n5, cb_5m_nb, cb_5m_atr_lo, cb_5m_atr_hi, cb_5m_ded,
               am_sp.get("sl_atr_mult", 1.5), am_sp.get("trail_activate_pts", 7.0),
               am_sp.get("trail_atr_mult", 2.5), int(am_sp.get("max_hold_bars", 30)),
               pm_sp.get("sl_atr_mult", 1.2), pm_sp.get("trail_activate_pts", 6.0),
               pm_sp.get("trail_atr_mult", 1.0), int(pm_sp.get("max_hold_bars", 8)),
               float(am_sp.get("be_atr_min", 3.5)), float(am_sp.get("be_trigger_pts", 3.0)),
               float(am_sp.get("be_partial_pts", 1.0)),
               cb_5m_atr_lo_pm, cb_5m_atr_hi_pm, cb_dead_zones),
        '3m': (df3, n3, 5, 1.5, 3.0, 8, 1.2, 4.0, 2.0, 40, 1.0, 3.0, 1.5, 20, 2.5, 3.0, 1.0, None, None, None),
        '1m': (df1, n1, 10, 0.7, 1.5, 15, 1.2, 4.0, 2.0, 90, 1.0, 3.0, 1.5, 45, 1.0, 3.0, 1.0, None, None, None),
    }

    for tf_name, vals in best_configs.items():
        df_tf, n_d, nb, atr_lo, atr_hi, ded, am_sl, am_trail, am_mult, am_hold, pm_sl, pm_trail, pm_mult, pm_hold, be_min, be_trig, be_part, atr_lo_pm, atr_hi_pm, dz = vals
        tdf = run_cb(df_tf, nb, cb_5m_thresh if tf_name == '5m' else 0.7, atr_lo, atr_hi, ded,
                     am_sl, am_trail, am_mult, am_hold,
                     pm_sl, pm_trail, pm_mult, pm_hold,
                     be_atr_min=be_min, be_trigger=be_trig, be_partial=be_part,
                     atr_min_pm=atr_lo_pm, atr_max_pm=atr_hi_pm, dead_zones=dz)
        if tdf is None: continue

        wins = tdf[tdf['pnl']>0]; losses = tdf[tdf['pnl']<=0]
        pf = wins['pnl'].sum()/abs(losses['pnl'].sum()) if len(losses)>0 and losses['pnl'].sum()!=0 else 999

        print(f'\n{"="*90}')
        print(f'DETAIL: {tf_name} CB {nb}bar | Trail@+{am_trail}pts | {n_d}d | {len(tdf)} trades | WR {(tdf["pnl"]>0).mean()*100:.1f}% | PF {pf:.2f} | {tdf["pnl"].sum():+.1f}pts | {tdf["pnl"].sum()/n_d:+.2f}/d')
        print('='*90)

        # AM/PM breakdown
        print(f"\n--- AM vs PM ---")
        print(f"  {'Session':<15} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'SL':>4} {'TRAIL':>6} {'avgMFE':>7}")
        print('  '+'-'*60)
        for s in ['AM', 'PM']:
            sub = tdf[tdf['sess']==s]
            if len(sub) == 0: continue
            w = sub[sub['pnl']>0]; l = sub[sub['pnl']<=0]
            spf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else 999
            sl_n = (sub['reason']=='SL').sum()
            tr_n = (sub['reason']=='TRAIL').sum()
            print(f"  {s:<15} {len(sub):>4} {(sub['pnl']>0).mean()*100:>5.1f}% {spf:>5.2f} "
                  f"{sub['pnl'].sum():>+8.1f} {sl_n:>4} {tr_n:>6} {sub['mfe'].mean():>7.1f}")

        # Time window breakdown
        print(f"\n--- Theo khung gio ---")
        print(f"  {'Window':<20} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'SL':>4} {'avgMFE':>7}")
        print('  '+'-'*60)
        windows = [
            ('AM 09:15-10:00', 9*60+15, 10*60),
            ('AM 10:00-10:45', 10*60, 10*60+45),
            ('PM 13:15-13:45', 13*60+15, 13*60+45),
            ('PM 13:45-14:15', 13*60+45, 14*60+15),
        ]
        for wlabel, wlo, whi in windows:
            sub = tdf[(tdf['mins']>=wlo)&(tdf['mins']<whi)]
            if len(sub) == 0: continue
            w = sub[sub['pnl']>0]; l = sub[sub['pnl']<=0]
            spf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else 999
            sl_n = (sub['reason']=='SL').sum()
            print(f"  {wlabel:<20} {len(sub):>4} {(sub['pnl']>0).mean()*100:>5.1f}% {spf:>5.2f} "
                  f"{sub['pnl'].sum():>+8.1f} {sl_n:>4} {sub['mfe'].mean():>7.1f}")

        # Exit breakdown
        print(f"\n--- Exit reasons ---")
        for r, cnt in tdf['reason'].value_counts().items():
            sub = tdf[tdf['reason']==r]
            wr = (sub['pnl']>0).mean()*100
            print(f"  {r:<12}: {cnt:>4} | WR {wr:>5.1f}% | avg PnL {sub['pnl'].mean():>+5.2f} | avg MFE {sub['mfe'].mean():>5.1f}")

        # MFE distribution
        print(f"\n--- MFE distribution ---")
        buckets = [(0,1,'0-1'),(1,2,'1-2'),(2,4,'2-4'),(4,6,'4-6'),(6,9,'6-9'),(9,99,'9+')]
        for lo,hi,lbl in buckets:
            sub = tdf[(tdf['mfe']>=lo)&(tdf['mfe']<hi)]
            if len(sub):
                wr = (sub['pnl']>0).mean()*100
                print(f"  MFE {lbl:>5}: {len(sub):>4} trades | WR {wr:>5.1f}% | avg PnL {sub['pnl'].mean():>+5.2f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
