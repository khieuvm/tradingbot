"""
Adaptive Exit at 4pts MFE:
- At 4pts MFE, check signal conditions
- If reversal risk HIGH -> tighten trail or exit
- If continuation likely -> keep normal trail
Goal: capture more profit from good trades, protect from 4-6 reversals
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

n_days = df['date'].nunique()
print(f"Data: {len(df)} bars, {n_days} days\n")


# ===================================================================
# Simulation with adaptive exit at 4pts
# ===================================================================
def sim_adaptive(idx, sl_mult, trail_activate, trail_mult, max_hold,
                 be_atr_min=3.5,
                 # Adaptive params
                 at4_tight_mult=None,   # if not None: switch trail to this at 4pts (for reversal signal)
                 at4_exit=False,        # if True: exit immediately at 4pts (for worst reversal signals)
                 reversal_fn=None):     # function(row, pre_move, vol_ratio) -> bool: True=reversal risk
    row = df.loc[idx]; atr = row['atr']
    if pd.isna(atr) or atr <= 0: return None
    i_pos = df.index.get_loc(idx)
    if i_pos+1 >= len(df): return None
    nxt = df.iloc[i_pos+1]
    if nxt['date'] != row['date'] or nxt['session'] != row['session']: return None
    direction = 1 if nxt['close'] > row['close'] else -1
    entry = row['close']

    # Pre-signal features
    pre_bars = df.iloc[max(0, i_pos-3):i_pos]
    pre_move = 0.0
    if len(pre_bars) >= 2:
        pre_move = (pre_bars.iloc[-1]['close'] - pre_bars.iloc[0]['open']) * direction
    vol_ratio = row.get('vol_ratio', 1.0)
    if pd.isna(vol_ratio): vol_ratio = 1.0

    # Determine reversal risk at signal time
    is_reversal_risk = reversal_fn(row, pre_move, vol_ratio) if reversal_fn else False

    sl = entry - direction*sl_mult*atr
    best = entry; trail_on = False; be_done = False
    adaptive_triggered = False  # has the 4pt adaptive logic fired
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
            # === ADAPTIVE LOGIC AT 4pts ===
            if not adaptive_triggered and mfe >= 4.0 and is_reversal_risk:
                adaptive_triggered = True
                if at4_exit:
                    # Exit now at current best (simulate next bar open = current bar close)
                    exit_p = b['close']; exit_r = 'ADAPT_EXIT'; break
                elif at4_tight_mult is not None:
                    # Tighten trail immediately
                    sl = max(sl, best - at4_tight_mult*atr)
                    trail_on = True
            # Normal BE
            if not be_done and mfe >= 4.0 and atr >= be_atr_min:
                be_done = True; sl = max(sl, entry)
            if mfe >= trail_activate: trail_on = True
            if trail_on:
                # Use tight mult if adaptive triggered, else normal mult
                m = at4_tight_mult if (adaptive_triggered and at4_tight_mult) else trail_mult
                sl = max(sl, best - m*atr)
                if b['low'] <= sl: exit_p=sl; exit_r='TRAIL'; break
        else:
            if b['high'] >= sl: exit_p=sl; exit_r='BE' if be_done else 'SL'; break
            if b['low'] < best: best = b['low']
            mfe = entry - best
            if not adaptive_triggered and mfe >= 4.0 and is_reversal_risk:
                adaptive_triggered = True
                if at4_exit:
                    exit_p = b['close']; exit_r = 'ADAPT_EXIT'; break
                elif at4_tight_mult is not None:
                    sl = min(sl, best + at4_tight_mult*atr)
                    trail_on = True
            if not be_done and mfe >= 4.0 and atr >= be_atr_min:
                be_done = True; sl = min(sl, entry)
            if mfe >= trail_activate: trail_on = True
            if trail_on:
                m = at4_tight_mult if (adaptive_triggered and at4_tight_mult) else trail_mult
                sl = min(sl, best + m*atr)
                if b['high'] >= sl: exit_p=sl; exit_r='TRAIL'; break

    if exit_p is None: exit_p=entry; exit_r='MAX_HOLD'
    mfe_f = (best-entry) if direction==1 else (entry-best)
    pnl = direction*(exit_p-entry) - COST
    return {'date': row['date'], 'time': row['time'].strftime('%H:%M'),
            'sess': row['session'], 'dir': direction,
            'entry': entry, 'exit': exit_p, 'atr': atr,
            'mfe': mfe_f, 'pnl': pnl, 'reason': exit_r,
            'pre_move': pre_move, 'vol_ratio': vol_ratio,
            'is_reversal_risk': is_reversal_risk}


def run_config(reversal_fn, at4_tight_mult=None, at4_exit=False):
    trades = []
    for idx in df.index[mAM]:
        t = sim_adaptive(idx, 1.2, 5.0, 2.0, 24,
                         reversal_fn=reversal_fn,
                         at4_tight_mult=at4_tight_mult, at4_exit=at4_exit)
        if t: t['sess_type'] = 'AM'; trades.append(t)
    for idx in df.index[mPM]:
        t = sim_adaptive(idx, 1.0, 4.0, 1.5, 12,
                         reversal_fn=reversal_fn,
                         at4_tight_mult=at4_tight_mult, at4_exit=at4_exit)
        if t: t['sess_type'] = 'PM'; trades.append(t)
    if not trades: return None
    return pd.DataFrame(trades)


def show(tdf, label):
    if tdf is None or len(tdf) == 0:
        print(f"  {label}: no trades"); return
    wins = tdf[tdf['pnl']>0]; losses = tdf[tdf['pnl']<=0]
    pf = wins['pnl'].sum()/abs(losses['pnl'].sum()) if len(losses)>0 and losses['pnl'].sum()!=0 else 999
    adapt = (tdf['reason']=='ADAPT_EXIT').sum()
    print(f"  {label:<58} {len(tdf):>4} {(tdf['pnl']>0).mean()*100:>5.1f}% {pf:>5.2f} "
          f"{tdf['pnl'].sum():>+8.1f} {tdf['pnl'].sum()/n_days:>+5.2f}/d  adapt={adapt}")


# ===================================================================
# Define reversal conditions (at signal bar)
# ===================================================================
def no_filter(row, pre, vol): return False                        # baseline

# Key finding: pre_move > 3 = exhaustion (40% fail@4-6)
def cond_premove3(row, pre, vol): return pre > 3.0
def cond_premove2(row, pre, vol): return pre > 2.0

# Vol < 0.8 = low participation (32% fail but weak signal)
def cond_lowvol(row, pre, vol): return vol < 0.8

# Combined
def cond_premove3_or_lowvol(row, pre, vol): return (pre > 3.0) or (vol < 0.8)
def cond_premove2_or_lowvol(row, pre, vol): return (pre > 2.0) or (vol < 0.8)

# Strongest: premove > 3 AND vol < 1.0 (both bad)
def cond_both_bad(row, pre, vol): return (pre > 3.0) and (vol < 1.0)

# AM only (PM is strong, don't touch)
def cond_am_premove3(row, pre, vol):
    return (row['session'] == 'AM') and (pre > 3.0)

def cond_am_premove2(row, pre, vol):
    return (row['session'] == 'AM') and (pre > 2.0)

def cond_am_premove3_or_lowvol(row, pre, vol):
    return (row['session'] == 'AM') and ((pre > 3.0) or (vol < 0.8))


print(f"{'='*100}")
print(f"ADAPTIVE EXIT TEST — Baseline vs Tighten Trail / Exit at 4pts")
print(f"Format: n_trades | WR | PF | PnL | PnL/d | adapt_exits")
print(f"{'='*100}")
print(f"\n  {'Config':<58} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'P/D':>7}  adapt=")
print('  '+'-'*90)

# === BASELINE ===
bl = run_config(no_filter)
show(bl, 'BASELINE (no adaptive)')

print()

# === OPTION A: TIGHT TRAIL at 4pts for reversal signals ===
print("  -- Option A: Switch to TIGHT trail (0.8x ATR) at 4pts if reversal signal --")
for label, fn in [
    ('  A1: pre_move>3 → trail 0.8x', cond_premove3),
    ('  A2: pre_move>2 → trail 0.8x', cond_premove2),
    ('  A3: low_vol<0.8 → trail 0.8x', cond_lowvol),
    ('  A4: (pre>3 OR vol<0.8) → trail 0.8x', cond_premove3_or_lowvol),
    ('  A5: (pre>2 OR vol<0.8) → trail 0.8x', cond_premove2_or_lowvol),
    ('  A6: AM only: pre>3 → trail 0.8x', cond_am_premove3),
    ('  A7: AM only: pre>2 → trail 0.8x', cond_am_premove2),
    ('  A8: AM only: (pre>3 OR vol<0.8) → 0.8x', cond_am_premove3_or_lowvol),
]:
    tdf = run_config(fn, at4_tight_mult=0.8)
    show(tdf, label)

print()
print("  -- Option B: Tighter trail (1.0x ATR) at 4pts --")
for label, fn in [
    ('  B1: pre_move>3 → trail 1.0x', cond_premove3),
    ('  B2: (pre>3 OR vol<0.8) → trail 1.0x', cond_premove3_or_lowvol),
    ('  B3: AM only: pre>3 → trail 1.0x', cond_am_premove3),
    ('  B4: AM only: (pre>3 OR vol<0.8) → 1.0x', cond_am_premove3_or_lowvol),
]:
    tdf = run_config(fn, at4_tight_mult=1.0)
    show(tdf, label)

print()
print("  -- Option C: EXIT immediately at 4pts for reversal signals --")
for label, fn in [
    ('  C1: pre_move>3 → exit@4', cond_premove3),
    ('  C2: (pre>3 OR vol<0.8) → exit@4', cond_premove3_or_lowvol),
    ('  C3: AM only: pre>3 → exit@4', cond_am_premove3),
    ('  C4: AM only: (pre>3 OR vol<0.8) → exit@4', cond_am_premove3_or_lowvol),
    ('  C5: both_bad (pre>3 AND vol<1) → exit@4', cond_both_bad),
]:
    tdf = run_config(fn, at4_exit=True)
    show(tdf, label)

# ===================================================================
# BEST CONFIG DETAIL
# ===================================================================
print(f"\n{'='*100}")
print("DETAIL: Best configs vs Baseline")
print('='*100)

configs_to_detail = [
    ('BASELINE', no_filter, None, False),
    ('A6: AM pre>3 → tight 0.8x', cond_am_premove3, 0.8, False),
    ('A8: AM (pre>3|vol<0.8) → 0.8x', cond_am_premove3_or_lowvol, 0.8, False),
    ('C3: AM pre>3 → exit@4', cond_am_premove3, None, True),
    ('C4: AM (pre>3|vol<0.8) → exit@4', cond_am_premove3_or_lowvol, None, True),
]

for name, fn, mult, do_exit in configs_to_detail:
    tdf = run_config(fn, at4_tight_mult=mult, at4_exit=do_exit)
    if tdf is None: continue
    wins = tdf[tdf['pnl']>0]; losses = tdf[tdf['pnl']<=0]
    pf = wins['pnl'].sum()/abs(losses['pnl'].sum()) if len(losses)>0 and losses['pnl'].sum()!=0 else 999

    print(f"\n--- {name} ---")
    print(f"  Total: {len(tdf)} | WR {(tdf['pnl']>0).mean()*100:.1f}% | PF {pf:.2f} | "
          f"PnL {tdf['pnl'].sum():+.1f} | {tdf['pnl'].sum()/n_days:+.2f}/d")

    print(f"  {'Session':<8} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} | Exit breakdown")
    for s in ['AM', 'PM']:
        sub = tdf[tdf['sess']==s]
        if len(sub) == 0: continue
        w = sub[sub['pnl']>0]; l = sub[sub['pnl']<=0]
        spf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l)>0 and l['pnl'].sum()!=0 else 999
        exits = sub['reason'].value_counts().to_dict()
        print(f"  {s:<8} {len(sub):>4} {(sub['pnl']>0).mean()*100:>5.1f}% {spf:>5.2f} "
              f"{sub['pnl'].sum():>+8.1f}  {exits}")

    # Reversal risk trades: how they fared
    risk_t = tdf[tdf['is_reversal_risk']==True]
    safe_t = tdf[tdf['is_reversal_risk']==False]
    if len(risk_t) > 0:
        rw = risk_t[risk_t['pnl']>0]; rl = risk_t[risk_t['pnl']<=0]
        rpf = rw['pnl'].sum()/abs(rl['pnl'].sum()) if len(rl)>0 and rl['pnl'].sum()!=0 else 999
        print(f"  Reversal-risk trades: {len(risk_t)} | WR {(risk_t['pnl']>0).mean()*100:.1f}% | "
              f"PF {rpf:.2f} | PnL {risk_t['pnl'].sum():+.1f}")
    if len(safe_t) > 0:
        sw = safe_t[safe_t['pnl']>0]; sl2 = safe_t[safe_t['pnl']<=0]
        spf2 = sw['pnl'].sum()/abs(sl2['pnl'].sum()) if len(sl2)>0 and sl2['pnl'].sum()!=0 else 999
        print(f"  Normal trades:        {len(safe_t)} | WR {(safe_t['pnl']>0).mean()*100:.1f}% | "
              f"PF {spf2:.2f} | PnL {safe_t['pnl'].sum():+.1f}")

    # MFE distribution comparison
    print(f"  MFE distribution:")
    for lo, hi, lbl in [(0,2,'0-2'),(2,4,'2-4'),(4,6,'4-6'),(6,9,'6-9'),(9,99,'9+')]:
        sub = tdf[(tdf['mfe']>=lo)&(tdf['mfe']<hi)]
        if len(sub):
            wr = (sub['pnl']>0).mean()*100
            print(f"    MFE {lbl}: {len(sub):>4} | WR {wr:>5.1f}% | avg PnL {sub['pnl'].mean():>+5.2f}")

print("\nDone.")
