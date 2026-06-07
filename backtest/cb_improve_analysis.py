"""
backtest/cb_improve_analysis.py — CB 5m Performance Decay Diagnosis + Improvement Tests

Investigates WHY CB is decaying in May-June 2026 and tests 8 independent filters.
Then combines the best filters and reports monthly breakdown.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta

from src.data_fetcher import DataFetcher

COST = 0.96

# ─── 1. DATA LOAD ──────────────────────────────────────────────────────────────
fetcher = DataFetcher()
end   = datetime.now().strftime("%Y-%m-%d")
start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
print(f"Fetching 5m data {start} -> {end} ...")

df = fetcher.get_futures_ohlcv("VN30F1M", start, end, interval="5m")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df['date'] = df['time'].dt.date
df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
df = df[
    ((df['mins'] >= 9*60) & (df['mins'] < 11*60+30)) |
    ((df['mins'] >= 13*60) & (df['mins'] < 14*60+30))
].reset_index(drop=True)

# ─── 2. INDICATORS ─────────────────────────────────────────────────────────────
df['atr']    = ta.atr(df['high'], df['low'], df['close'], length=14)
df['rsi14']  = ta.rsi(df['close'], length=14)
df['ema20']  = ta.ema(df['close'], length=20)

adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
if adx_df is not None:
    adx_cols = [c for c in adx_df.columns if c.startswith('ADX')]
    df['adx14'] = adx_df[adx_cols[0]] if adx_cols else np.nan
else:
    df['adx14'] = np.nan

df['atr_sma50']  = df['atr'].rolling(50, min_periods=20).mean()
df['atr_ratio']  = df['atr'] / df['atr_sma50']
df['vol_sma20']  = df['volume'].rolling(20, min_periods=5).mean()
rng              = (df['high'] - df['low']).replace(0, np.nan)
df['body_ratio'] = (df['close'] - df['open']).abs() / rng
df['session']    = np.where(df['mins'] < 12*60, 'AM', 'PM')
df['range']      = df['high'] - df['low']
df['month']      = pd.to_datetime(df['date'].astype(str)).dt.to_period('M')

n_days = df['date'].nunique()
print(f"Loaded {len(df)} bars | {n_days}d | {df['date'].min()} -> {df['date'].max()}")

# ─── 3. NUMPY ARRAYS ───────────────────────────────────────────────────────────
highs          = df['high'].values.astype(float)
lows           = df['low'].values.astype(float)
opens          = df['open'].values.astype(float)
closes         = df['close'].values.astype(float)
atrs_arr       = df['atr'].values.astype(float)
rsi_arr        = df['rsi14'].values.astype(float)
ema_arr        = df['ema20'].values.astype(float)
adx_arr        = df['adx14'].values.astype(float)
atr_ratio_arr  = df['atr_ratio'].values.astype(float)
vol_arr        = df['volume'].values.astype(float)
vol_sma_arr    = df['vol_sma20'].values.astype(float)
body_ratio_arr = df['body_ratio'].values.astype(float)
ranges         = df['range'].values.astype(float)
dates          = df['date'].values
sessions       = df['session'].values
mins_arr       = df['mins'].values.astype(int)


# ─── 4. HELPERS ────────────────────────────────────────────────────────────────
def dedup_arr(arr, min_bars=5):
    out = arr.copy().astype(bool)
    last = -999
    for i in range(len(out)):
        if out[i]:
            if i - last < min_bars:
                out[i] = False
            else:
                last = i
    return out


def pf_calc(sub):
    wins   = sub[sub['pnl'] > 0]['pnl'].sum()
    losses = abs(sub[sub['pnl'] <= 0]['pnl'].sum())
    return wins / losses if losses > 0 else 999.0


def summarise(tdf, n_d):
    if tdf is None or len(tdf) == 0:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'pnl': 0.0, 'ppd': 0.0}
    n   = len(tdf)
    wr  = (tdf['pnl'] > 0).mean() * 100
    pf  = pf_calc(tdf)
    pnl = tdf['pnl'].sum()
    ppd = pnl / n_d
    return {'n': n, 'wr': wr, 'pf': pf, 'pnl': pnl, 'ppd': ppd}


def monthly_table(tdf, label=''):
    if tdf is None or len(tdf) == 0:
        print("  (no trades)")
        return
    if label:
        print(f"\n  Monthly breakdown — {label}")
    df_m = df.copy()
    df_m['mstr'] = df_m['month'].astype(str)
    tdf2 = tdf.copy()
    tdf2['mstr'] = tdf2['month'].astype(str)
    all_months = sorted(tdf2['mstr'].unique())
    print(f"  {'Month':<8} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'Days':>5} {'P/D':>7}")
    print("  " + "-"*50)
    for m in all_months:
        sub    = tdf2[tdf2['mstr'] == m]
        m_days = df_m[df_m['mstr'] == m]['date'].nunique()
        wr_m   = (sub['pnl'] > 0).mean() * 100
        pf_m   = pf_calc(sub)
        pnl_m  = sub['pnl'].sum()
        ppd_m  = pnl_m / m_days if m_days > 0 else 0.0
        print(f"  {m:<8} {len(sub):>4} {wr_m:>5.1f}% {pf_m:>5.2f} "
              f"{pnl_m:>+8.1f} {m_days:>5} {ppd_m:>+6.2f}")


# ─── 5. SIGNAL DETECTION ───────────────────────────────────────────────────────
def detect_signals(threshold=0.7, n_comp=3,
                   atr_min=2.5, atr_max=4.5,
                   pm_atr_min=None, pm_atr_max=None,
                   adx_max=None, atr_ratio_max=None,
                   vol_filter=False, body_filter=False,
                   exclude_slots=None, dedup_bars=5):
    """Returns (am_mask, pm_mask) as numpy bool arrays."""
    # Compression
    comp = np.zeros(len(df), dtype=bool)
    for i in range(n_comp, len(df)):
        a = atrs_arr[i]
        if np.isnan(a) or a <= 0:
            continue
        if np.nanmax(ranges[i - n_comp: i]) < threshold * a:
            comp[i] = True

    # Base ATR + RSI
    rsi_ok     = np.where(np.isnan(rsi_arr), True, rsi_arr < 70)
    atr_ok_am  = (~np.isnan(atrs_arr)) & (atrs_arr >= atr_min) & (atrs_arr <= atr_max)
    pm_lo      = pm_atr_min if pm_atr_min is not None else atr_min
    pm_hi      = pm_atr_max if pm_atr_max is not None else atr_max
    atr_ok_pm  = (~np.isnan(atrs_arr)) & (atrs_arr >= pm_lo) & (atrs_arr <= pm_hi)

    base_am = comp & atr_ok_am & rsi_ok
    base_pm = comp & atr_ok_pm & rsi_ok

    # Optional filters
    if adx_max is not None:
        adx_ok  = np.where(np.isnan(adx_arr), True, adx_arr < adx_max)
        base_am = base_am & adx_ok
        base_pm = base_pm & adx_ok

    if atr_ratio_max is not None:
        ratio_ok = np.where(np.isnan(atr_ratio_arr), True, atr_ratio_arr < atr_ratio_max)
        base_am  = base_am & ratio_ok
        base_pm  = base_pm & ratio_ok

    if vol_filter:
        vol_ok  = (~np.isnan(vol_sma_arr)) & (vol_arr > 0.5 * vol_sma_arr)
        base_am = base_am & vol_ok
        base_pm = base_pm & vol_ok

    if body_filter:
        body_ok = np.where(np.isnan(body_ratio_arr), True, body_ratio_arr < 0.3)
        base_am = base_am & body_ok
        base_pm = base_pm & body_ok

    # Time windows
    am_time = (mins_arr >= 9*60+15) & (mins_arr <= 10*60+45)
    pm_time = (mins_arr >= 13*60+15) & (mins_arr <= 14*60+15)

    if exclude_slots:
        for s_lo, s_hi in exclude_slots:
            am_time = am_time & ~((mins_arr >= s_lo) & (mins_arr < s_hi))

    am_mask = dedup_arr(base_am & am_time, dedup_bars)
    pm_mask = dedup_arr(base_pm & pm_time, dedup_bars)
    return am_mask, pm_mask


# ─── 6. BREAKOUT DIRECTION + ENTRY ─────────────────────────────────────────────
def get_breakout(sig_loc):
    """H+0.1 / L-0.1 trigger with 2-bar expiry."""
    buy_trig  = highs[sig_loc] + 0.1
    sell_trig = lows[sig_loc]  - 0.1
    sig_date  = dates[sig_loc]
    sig_sess  = sessions[sig_loc]
    for delta in (1, 2):
        j = sig_loc + delta
        if j >= len(df):
            break
        if dates[j] != sig_date or sessions[j] != sig_sess:
            break
        buy_hit  = highs[j] > buy_trig
        sell_hit = lows[j]  < sell_trig
        if buy_hit and sell_hit:
            if opens[j] > buy_trig:
                return 1, buy_trig, j
            elif opens[j] < sell_trig:
                return -1, sell_trig, j
            else:
                return (1, buy_trig, j) if closes[j] >= opens[j] else (-1, sell_trig, j)
        elif buy_hit:
            return 1, buy_trig, j
        elif sell_hit:
            return -1, sell_trig, j
    return None, None, None


# ─── 7. SIMULATION ─────────────────────────────────────────────────────────────
_PARAMS = {
    'AM': dict(sl_mult=1.2, trail_act=5.0, trail_mult=2.0, max_hold=24, exit_mins=11*60+25),
    'PM': dict(sl_mult=1.0, trail_act=4.0, trail_mult=1.5, max_hold=12, exit_mins=14*60+25),
}
BE_TRIGGER   = 4.0;  BE_ATR_MIN     = 3.5
TIGHTEN_MFE  = 8.0;  TIGHTEN_MULT   = 1.2;  TIGHTEN_ATR_MIN = 3.5
ADAPT_PMR    = 0.8;  ADAPT_MFE      = 4.0;  ADAPT_PNL       = 3.2


def simulate(sig_loc, entry_bar_loc, direction, entry_price, use_ema_filter=False):
    atr     = float(atrs_arr[sig_loc])
    session = sessions[sig_loc]
    p       = _PARAMS[session]

    # EMA context filter: only allow direction aligned with EMA20
    if use_ema_filter:
        ev = ema_arr[sig_loc]
        if not np.isnan(ev):
            if direction == 1 and closes[sig_loc] < ev:
                return None   # BUY signal but price below EMA → skip
            if direction == -1 and closes[sig_loc] > ev:
                return None   # SELL signal but price above EMA → skip

    pre_move_ratio = (
        float((closes[sig_loc] - opens[sig_loc - 2]) * direction / atr)
        if sig_loc >= 2 and atr > 0 else 0.0
    )

    sl            = entry_price - direction * p['sl_mult'] * atr
    best          = entry_price
    mfe           = 0.0
    trail_on      = False
    be_done       = False
    adapt_checked = False
    exit_p        = None
    exit_r        = None
    bars_held     = 0
    sig_date      = dates[sig_loc]
    end_loc       = min(entry_bar_loc + p['max_hold'], len(df))

    for k in range(entry_bar_loc, end_loc):
        if dates[k] != sig_date or sessions[k] != session:
            exit_p = float(closes[k - 1]) if k > entry_bar_loc else entry_price
            exit_r = 'SESSION'
            break
        if mins_arr[k] >= p['exit_mins']:
            exit_p = float(closes[k])
            exit_r = 'SESSION'
            break

        bars_held += 1
        h = float(highs[k]); l = float(lows[k]); c = float(closes[k])

        if direction == 1:
            best = max(best, h);  mfe = max(mfe, best - entry_price)
        else:
            best = min(best, l);  mfe = max(mfe, entry_price - best)

        # Adaptive exit (AM only)
        if session == 'AM' and pre_move_ratio > ADAPT_PMR and mfe >= ADAPT_MFE and not adapt_checked:
            adapt_checked = True
            pnl_pts = direction * (c - entry_price)
            if pnl_pts >= ADAPT_PNL:
                exit_p = c; exit_r = 'ADAPT_EXIT'; break
            else:
                sl = max(sl, entry_price + 2.0) if direction == 1 else min(sl, entry_price - 2.0)

        # Breakeven
        if not be_done and mfe >= BE_TRIGGER and atr >= BE_ATR_MIN:
            be_done = True
            sl = max(sl, entry_price) if direction == 1 else min(sl, entry_price)

        # Trail
        if mfe >= p['trail_act']:
            trail_on = True
        if trail_on:
            t_mult   = TIGHTEN_MULT if (mfe >= TIGHTEN_MFE and atr >= TIGHTEN_ATR_MIN) else p['trail_mult']
            trail_sl = best - t_mult * atr if direction == 1 else best + t_mult * atr
            sl = max(sl, trail_sl) if direction == 1 else min(sl, trail_sl)

        # SL hit
        sl_hit = (direction == 1 and l <= sl) or (direction == -1 and h >= sl)
        if sl_hit:
            exit_p = sl
            exit_r = 'TRAIL' if trail_on else ('BE' if be_done else 'SL')
            break

    if exit_p is None:
        exit_p = float(closes[min(end_loc - 1, len(df) - 1)])
        exit_r = 'MAX_HOLD'

    pnl = direction * (exit_p - entry_price) - COST
    return {
        'date'     : dates[sig_loc],
        'time'     : str(df.iloc[sig_loc]['time'])[11:16],
        'session'  : session,
        'mins'     : int(mins_arr[sig_loc]),
        'direction': 'BUY' if direction == 1 else 'SELL',
        'entry'    : float(entry_price),
        'exit'     : float(exit_p),
        'atr'      : float(atr),
        'adx'      : float(adx_arr[sig_loc]),
        'atr_ratio': float(atr_ratio_arr[sig_loc]),
        'mfe'      : float(mfe),
        'pnl'      : float(pnl),
        'reason'   : exit_r,
        'bars'     : bars_held,
        'pre_move_ratio': float(pre_move_ratio),
    }


def run_backtest(am_mask, pm_mask, use_ema_filter=False):
    sig_locs = np.where(am_mask | pm_mask)[0]
    trades = []; expired = 0
    for sl in sig_locs:
        direction, entry_price, entry_bar_loc = get_breakout(int(sl))
        if direction is None:
            expired += 1; continue
        t = simulate(int(sl), entry_bar_loc, direction, entry_price, use_ema_filter)
        if t is not None:
            trades.append(t)
        else:
            expired += 1
    if not trades:
        return None
    tdf = pd.DataFrame(trades).sort_values(['date', 'time']).reset_index(drop=True)
    tdf['month'] = pd.to_datetime(tdf['date'].astype(str)).dt.to_period('M')
    return tdf


def test_filter(label, use_ema=False, **signal_kwargs):
    am_m, pm_m = detect_signals(**signal_kwargs)
    tdf  = run_backtest(am_m, pm_m, use_ema_filter=use_ema)
    s    = summarise(tdf, n_days)
    return s, tdf, am_m.sum() + pm_m.sum()


SEP  = "=" * 72
SEP2 = "-" * 60


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CURRENT BASELINE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 1 — CURRENT BASELINE")
print(SEP)

am_base, pm_base = detect_signals()
print(f"Signals detected: AM={am_base.sum()}, PM={pm_base.sum()}")
tdf_base = run_backtest(am_base, pm_base)
base_s   = summarise(tdf_base, n_days)

print(f"\nTOTAL: T={base_s['n']}, WR={base_s['wr']:.1f}%, PF={base_s['pf']:.2f}, "
      f"PnL={base_s['pnl']:+.1f}, P/D={base_s['ppd']:+.2f}")

if tdf_base is not None:
    for sess in ('AM', 'PM'):
        sub = tdf_base[tdf_base['session'] == sess]
        if len(sub) == 0:
            print(f"{sess}: 0 trades")
            continue
        print(f"{sess}: T={len(sub)}, WR={(sub['pnl']>0).mean()*100:.1f}%, "
              f"PF={pf_calc(sub):.2f}, P/D={sub['pnl'].sum()/n_days:+.2f}")

    monthly_table(tdf_base, "BASELINE")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: WHY DID PM SIGNALS DISAPPEAR IN MAY-JUNE?
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 2 — WHY PM SIGNALS DISAPPEARED IN MAY-JUNE")
print(SEP)

# Build per-bar analysis of PM slot
comp_global = np.zeros(len(df), dtype=bool)
for i in range(3, len(df)):
    a = atrs_arr[i]
    if np.isnan(a) or a <= 0: continue
    if np.nanmax(ranges[i-3:i]) < 0.7 * a:
        comp_global[i] = True

pm_slot_mask = (mins_arr >= 13*60+15) & (mins_arr <= 14*60+15)
pm_idx       = np.where(pm_slot_mask)[0]

pm_atr       = atrs_arr[pm_idx]
pm_comp      = comp_global[pm_idx]
pm_atr_ok    = (~np.isnan(pm_atr)) & (pm_atr >= 2.5) & (pm_atr <= 4.5)
pm_both      = pm_atr_ok & pm_comp
pm_dates     = dates[pm_idx]

df_pm_diag = pd.DataFrame({
    'date'    : pm_dates,
    'atr'     : pm_atr,
    'comp'    : pm_comp,
    'atr_ok'  : pm_atr_ok,
    'both'    : pm_both,
})
df_pm_diag['month'] = pd.to_datetime(df_pm_diag['date'].astype(str)).dt.to_period('M')
df_pm_diag['mstr']  = df_pm_diag['month'].astype(str)

print(f"\nPM bar analysis by month (13:15–14:15 slot):")
print(f"  {'Month':<8} {'Bars':>5} {'ATR_ok':>7}  {'ATR%':>5}  {'Comp':>5}  {'Both':>5}  {'ATR mean':>8}  {'ATR range':>12}")
print("  " + "-"*65)
for m in sorted(df_pm_diag['mstr'].unique()):
    sub    = df_pm_diag[df_pm_diag['mstr'] == m]
    n_b    = len(sub)
    n_aok  = sub['atr_ok'].sum()
    n_comp = sub['comp'].sum()
    n_both = sub['both'].sum()
    a_vals = sub['atr'].dropna()
    a_mean = a_vals.mean() if len(a_vals) else 0
    a_rng  = f"{a_vals.min():.1f}-{a_vals.max():.1f}" if len(a_vals) else "N/A"
    print(f"  {m:<8} {n_b:>5} {n_aok:>7}  {n_aok/n_b*100:>4.0f}%  "
          f"{n_comp:>5}  {n_both:>5}  {a_mean:>8.2f}  [{a_rng}]")

print(f"\nPM ATR distribution — Jan-Feb vs May-Jun 2026:")
for mlist, label in [(['2026-01','2026-02'], 'Jan-Feb'), (['2026-05','2026-06'], 'May-Jun')]:
    sub = df_pm_diag[df_pm_diag['mstr'].isin(mlist)]
    a   = sub['atr'].dropna()
    if len(a) == 0:
        print(f"  {label}: no data"); continue
    in_range = ((a >= 2.5) & (a <= 4.5)).sum()
    too_low  = (a < 2.5).sum()
    too_high = (a > 4.5).sum()
    print(f"  {label}: n={len(a)}, mean={a.mean():.2f}, median={a.median():.2f}, "
          f"in_range={in_range} ({in_range/len(a)*100:.0f}%), "
          f"<2.5={too_low}, >4.5={too_high}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: WHY IS AM LOSING IN MAY-JUNE?
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 3 — WHY AM IS LOSING IN MAY-JUNE")
print(SEP)

if tdf_base is not None:
    tdf_am = tdf_base[tdf_base['session'] == 'AM'].copy()
    tdf_am['mstr'] = tdf_am['month'].astype(str)

    print(f"\n3a. AM Performance by Period:")
    periods = [
        ('Jan-Feb', ['2026-01','2026-02']),
        ('Mar-Apr', ['2026-03','2026-04']),
        ('May-Jun', ['2026-05','2026-06']),
    ]
    for label, mlist in periods:
        sub = tdf_am[tdf_am['mstr'].isin(mlist)]
        if len(sub) == 0: continue
        buy  = sub[sub['direction']=='BUY']
        sell = sub[sub['direction']=='SELL']
        print(f"  {label}: T={len(sub)}, WR={(sub['pnl']>0).mean()*100:.1f}%, "
              f"PF={pf_calc(sub):.2f}, P/D={sub['pnl'].sum()/n_days:+.2f}")
        if len(buy) > 0:
            print(f"    BUY : T={len(buy)},  WR={(buy['pnl']>0).mean()*100:.1f}%, "
                  f"avg pnl={buy['pnl'].mean():+.2f}")
        if len(sell) > 0:
            print(f"    SELL: T={len(sell)}, WR={(sell['pnl']>0).mean()*100:.1f}%, "
                  f"avg pnl={sell['pnl'].mean():+.2f}")

    print(f"\n3b. ADX distribution (AM trades):")
    for label, mlist in periods:
        sub  = tdf_am[tdf_am['mstr'].isin(mlist)]
        if len(sub) == 0: continue
        adxv = sub['adx'].dropna()
        if len(adxv) == 0: continue
        print(f"  {label}: mean={adxv.mean():.1f}, median={adxv.median():.1f}, "
              f">20={( adxv>20).sum()}, >25={(adxv>25).sum()}, "
              f">30={(adxv>30).sum()} / n={len(adxv)}")

    print(f"\n3c. ATR ratio (AM trades):")
    for label, mlist in periods:
        sub  = tdf_am[tdf_am['mstr'].isin(mlist)]
        if len(sub) == 0: continue
        arv = sub['atr_ratio'].dropna()
        if len(arv) == 0: continue
        print(f"  {label}: mean={arv.mean():.2f}, >1.3={(arv>1.3).sum()}, "
              f">1.5={(arv>1.5).sum()} / n={len(arv)}")

    print(f"\n3d. MFE distribution — Jan-Feb vs May-Jun:")
    jf_sub = tdf_am[tdf_am['mstr'].isin(['2026-01','2026-02'])]
    mj_sub = tdf_am[tdf_am['mstr'].isin(['2026-05','2026-06'])]
    buckets = [(0,2,'0-2'),(2,4,'2-4'),(4,6,'4-6'),(6,9,'6-9'),(9,999,'9+')]
    print(f"  {'Bucket':>6}  {'Jan-Feb (n / WR)':>20}  {'May-Jun (n / WR)':>20}")
    print("  " + "-"*50)
    for lo, hi, lbl in buckets:
        jf = jf_sub[(jf_sub['mfe']>=lo) & (jf_sub['mfe']<hi)]
        mj = mj_sub[(mj_sub['mfe']>=lo) & (mj_sub['mfe']<hi)]
        jf_str = f"{len(jf):>3} / {(jf['pnl']>0).mean()*100:>4.0f}%" if len(jf) > 0 else "  0 /   N/A"
        mj_str = f"{len(mj):>3} / {(mj['pnl']>0).mean()*100:>4.0f}%" if len(mj) > 0 else "  0 /   N/A"
        print(f"  MFE {lbl:>5}:  {jf_str:>20}  {mj_str:>20}")

    print(f"\n3e. AM time slot breakdown — May-Jun only:")
    slots = [
        ('09:15-09:30', 9*60+15, 9*60+30),
        ('09:30-09:45', 9*60+30, 9*60+45),
        ('09:45-10:00', 9*60+45, 10*60),
        ('10:00-10:15', 10*60,   10*60+15),
        ('10:15-10:30', 10*60+15,10*60+30),
        ('10:30-10:45', 10*60+30,10*60+45),
    ]
    for slabel, lo, hi in slots:
        sub = mj_sub[(mj_sub['mins']>=lo) & (mj_sub['mins']<hi)]
        if len(sub) == 0: continue
        print(f"  {slabel}: T={len(sub)}, "
              f"WR={(sub['pnl']>0).mean()*100:.1f}%, "
              f"PnL={sub['pnl'].sum():+.1f}, "
              f"avgMFE={sub['mfe'].mean():.1f}")

    print(f"\n3f. Compression quality — Jan-Feb vs May-Jun (AM signal bars only):")
    # Measure actual compression ratio = max_range_3bar / ATR at signal bars
    am_sig_locs = np.where(am_base)[0]
    comp_ratios = []
    for sl in am_sig_locs:
        a = atrs_arr[sl]
        if np.isnan(a) or a <= 0: continue
        mr = np.nanmax(ranges[sl-3:sl])
        comp_ratios.append({'loc': sl, 'date': dates[sl], 'ratio': mr / a,
                            'mstr': str(pd.Period(str(dates[sl])[:7], 'M'))})
    df_cr = pd.DataFrame(comp_ratios)
    if len(df_cr) > 0:
        for mlist, label in [(['2026-01','2026-02'], 'Jan-Feb'), (['2026-05','2026-06'], 'May-Jun')]:
            sub = df_cr[df_cr['mstr'].isin(mlist)]
            if len(sub) == 0: continue
            print(f"  {label}: mean_comp_ratio={sub['ratio'].mean():.3f}, "
                  f"median={sub['ratio'].median():.3f}, "
                  f"<0.5={(sub['ratio']<0.5).sum()}, "
                  f"0.5-0.6={(((sub['ratio']>=0.5)&(sub['ratio']<0.6)).sum())}, "
                  f"0.6-0.7={(((sub['ratio']>=0.6)&(sub['ratio']<0.7)).sum())} / n={len(sub)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: IMPROVEMENT TESTS (each independent)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 4 — IMPROVEMENT TESTS (each independent from baseline)")
print(SEP)
print(f"\nBASELINE: T={base_s['n']}, WR={base_s['wr']:.1f}%, "
      f"PF={base_s['pf']:.2f}, P/D={base_s['ppd']:+.2f}\n")

all_results = []

def record_test(label, use_ema=False, **signal_kwargs):
    s, tdf, n_sig = test_filter(label, use_ema=use_ema, **signal_kwargs)
    delta = s['ppd'] - base_s['ppd']
    all_results.append({
        'label': label, 'T': s['n'], 'WR': s['wr'], 'PF': s['pf'],
        'PPD': s['ppd'], 'delta': delta, 'n_sig': n_sig,
    })
    return s, tdf


print("[a] Compression threshold (current=0.7 → tighter):")
for thresh in [0.50, 0.55, 0.60]:
    s, _ = record_test(f"Thresh={thresh}", threshold=thresh)
    print(f"  thresh={thresh}: T={s['n']}, WR={s['wr']:.1f}%, "
          f"PF={s['pf']:.2f}, P/D={s['ppd']:+.2f}")

print("\n[b] EMA20 context filter (BUY only close>EMA, SELL only close<EMA):")
s, _ = record_test("EMA20_context", use_ema=True)
print(f"  EMA20 filter: T={s['n']}, WR={s['wr']:.1f}%, "
      f"PF={s['pf']:.2f}, P/D={s['ppd']:+.2f}")

print("\n[c] ADX filter (trade only when ADX < threshold = ranging):")
for adx_lim in [20, 25, 30]:
    s, _ = record_test(f"ADX<{adx_lim}", adx_max=adx_lim)
    print(f"  ADX<{adx_lim}: T={s['n']}, WR={s['wr']:.1f}%, "
          f"PF={s['pf']:.2f}, P/D={s['ppd']:+.2f}")

print("\n[d] ATR ratio regime filter (tighter than current 1.5 VOLATILE threshold):")
for ratio_lim in [1.1, 1.2, 1.3]:
    s, _ = record_test(f"ATR_ratio<{ratio_lim}", atr_ratio_max=ratio_lim)
    print(f"  ATR_ratio<{ratio_lim}: T={s['n']}, WR={s['wr']:.1f}%, "
          f"PF={s['pf']:.2f}, P/D={s['ppd']:+.2f}")

print("\n[e] Exclude weak AM time slots (09:45-10:00 and 10:15-10:30):")
s, _ = record_test("Excl_weak_slots",
                   exclude_slots=[(9*60+45, 10*60), (10*60+15, 10*60+30)])
print(f"  Excl weak slots: T={s['n']}, WR={s['wr']:.1f}%, "
      f"PF={s['pf']:.2f}, P/D={s['ppd']:+.2f}")

print("\n[f] Wider ATR range for PM only (2.0-5.0 vs current 2.5-4.5):")
s, tdf_f = record_test("PM_wide_ATR", pm_atr_min=2.0, pm_atr_max=5.0)
pm_f = tdf_f[tdf_f['session']=='PM'] if tdf_f is not None else pd.DataFrame()
print(f"  PM wide ATR: T={s['n']}, WR={s['wr']:.1f}%, "
      f"PF={s['pf']:.2f}, P/D={s['ppd']:+.2f}  "
      f"[PM trades: {len(pm_f)} vs baseline {(tdf_base['session']=='PM').sum() if tdf_base is not None else 0}]")

print("\n[g] Volume filter (volume > 50% of 20-bar SMA):")
s, _ = record_test("Vol_gt50pct", vol_filter=True)
print(f"  Vol filter: T={s['n']}, WR={s['wr']:.1f}%, "
      f"PF={s['pf']:.2f}, P/D={s['ppd']:+.2f}")

print("\n[h] Candle body ratio < 30% of range (doji/inside-bar compression):")
s, _ = record_test("Body_lt30pct", body_filter=True)
print(f"  Body filter: T={s['n']}, WR={s['wr']:.1f}%, "
      f"PF={s['pf']:.2f}, P/D={s['ppd']:+.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: COMBINE BEST FILTERS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 5 — COMBINE BEST FILTERS")
print(SEP)

sorted_results = sorted(all_results, key=lambda x: x['delta'], reverse=True)
print("\nTop individual filters by delta P/D:")
for r in sorted_results[:5]:
    print(f"  {r['label']:<30} T={r['T']:>3}, WR={r['WR']:>5.1f}%, "
          f"PF={r['PF']:>5.2f}, P/D={r['PPD']:>+5.2f}, delta={r['delta']:>+5.2f}")

print("\nTesting combinations...")

# Build candidate combos (systematic)
combo_tests = [
    # (label, use_ema, signal_kwargs)
    ("ADX<25 + Thresh=0.60",      False, dict(threshold=0.60, adx_max=25)),
    ("ADX<25 + Thresh=0.55",      False, dict(threshold=0.55, adx_max=25)),
    ("ADX<20 + Thresh=0.60",      False, dict(threshold=0.60, adx_max=20)),
    ("EMA20 + Thresh=0.60",       True,  dict(threshold=0.60)),
    ("EMA20 + ADX<25",            True,  dict(adx_max=25)),
    ("EMA20 + ADX<25 + T=0.60",   True,  dict(threshold=0.60, adx_max=25)),
    ("EMA20 + ATR_ratio<1.3",     True,  dict(atr_ratio_max=1.3)),
    ("ATR_ratio<1.3 + T=0.60",    False, dict(threshold=0.60, atr_ratio_max=1.3)),
    ("Excl_slots + EMA20",        True,  dict(exclude_slots=[(9*60+45,10*60),(10*60+15,10*60+30)])),
    ("Excl_slots + ADX<25",       False, dict(adx_max=25,
                                               exclude_slots=[(9*60+45,10*60),(10*60+15,10*60+30)])),
    ("Excl_slots + T=0.60 + ADX<25", False, dict(threshold=0.60, adx_max=25,
                                                   exclude_slots=[(9*60+45,10*60),(10*60+15,10*60+30)])),
    ("Vol + EMA20 + ADX<25",      True,  dict(vol_filter=True, adx_max=25)),
    ("ADX<25 + PM_wide_ATR",      False, dict(adx_max=25, pm_atr_min=2.0, pm_atr_max=5.0)),
    ("EMA20 + ADX<25 + PM_wide",  True,  dict(adx_max=25, pm_atr_min=2.0, pm_atr_max=5.0)),
    ("T=0.55 + EMA20",            True,  dict(threshold=0.55)),
    ("T=0.55 + ADX<25 + PM_wide", False, dict(threshold=0.55, adx_max=25,
                                               pm_atr_min=2.0, pm_atr_max=5.0)),
]

combo_results = []
print(f"\n  {'Combo':<40} {'T':>4} {'WR':>6} {'PF':>5} {'P/D':>7} {'Delta':>7}")
print("  " + "-"*70)
for clabel, use_ema, skwargs in combo_tests:
    s, ctdf, n_sig = test_filter(clabel, use_ema=use_ema, **skwargs)
    delta = s['ppd'] - base_s['ppd']
    combo_results.append((clabel, s, ctdf, delta, use_ema, skwargs))
    print(f"  {clabel:<40} {s['n']:>4} {s['wr']:>5.1f}% {s['pf']:>5.2f} "
          f"{s['ppd']:>+6.2f}  {delta:>+6.2f}")

best_combo = max(combo_results, key=lambda x: x[1]['ppd'])
bc_label, bc_s, bc_tdf, bc_delta, bc_ema, bc_kwargs = best_combo

print(f"\n{'='*72}")
print(f"BEST COMBINED FILTER: {bc_label}")
print(f"  T={bc_s['n']}, WR={bc_s['wr']:.1f}%, PF={bc_s['pf']:.2f}, "
      f"PnL={bc_s['pnl']:+.1f}, P/D={bc_s['ppd']:+.2f}  (delta={bc_delta:+.2f})")
print(f"{'='*72}")

if bc_tdf is not None:
    for sess in ('AM', 'PM'):
        sub = bc_tdf[bc_tdf['session'] == sess]
        if len(sub) == 0:
            print(f"  {sess}: 0 trades"); continue
        print(f"  {sess}: T={len(sub)}, WR={(sub['pnl']>0).mean()*100:.1f}%, "
              f"PF={pf_calc(sub):.2f}, P/D={sub['pnl'].sum()/n_days:+.2f}")

    monthly_table(bc_tdf, bc_label)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: FINAL REPORT IN REQUESTED FORMAT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SECTION 6 — FINAL SUMMARY (REQUESTED FORMAT)")
print(SEP)

print(f"""
CURRENT BASELINE ({n_days}d):
  T={base_s['n']}, WR={base_s['wr']:.1f}%, PF={base_s['pf']:.2f}, P/D={base_s['ppd']:+.2f}
""")

print("IMPROVEMENT TESTS (each independent):")
print(f"  {'Filter':<35} {'T':>4} {'WR':>6} {'PF':>5} {'P/D':>7} {'Delta P/D':>9}")
print("  " + "-"*67)
print(f"  {'BASELINE':<35} {base_s['n']:>4} {base_s['wr']:>5.1f}% "
      f"{base_s['pf']:>5.2f} {base_s['ppd']:>+6.2f}        ---")
for r in sorted(all_results, key=lambda x: x['delta'], reverse=True):
    print(f"  {r['label']:<35} {r['T']:>4} {r['WR']:>5.1f}% {r['PF']:>5.2f} "
          f"{r['PPD']:>+6.2f}  {r['delta']:>+8.2f}")

print(f"""
BEST COMBINED: {bc_label}
  T={bc_s['n']}, WR={bc_s['wr']:.1f}%, PF={bc_s['pf']:.2f}, P/D={bc_s['ppd']:+.2f}
""")

print(f"\n{SEP}")
print("DIAGNOSIS NOTES (data-driven — read results above)")
print(SEP)
print("""
STRUCTURAL DECAY DRIVERS (investigate from Section 2-3 output):

  PM Disappearance:
    - If ATR consistently < 2.5 or > 4.5 in May-Jun PM, the ATR gate is blocking signals
    - If comp count drops while ATR_ok stays same, market entered sustained trend (no squeezes)
    - Both conditions can coexist in a trending/news-driven market

  AM Decay:
    - Higher ADX in May-Jun = trending regime, CB breakouts whipsaw in trends
    - BUY/SELL direction imbalance = directional bias the compression doesn't filter
    - Lower MFE distribution = market not running after breakout (faster reversals)
    - 09:45-10:00, 10:15-10:30 as weakest slots = likely opening range fade after initial move

  Best Fix Direction (from filter test results):
    - ADX filter is the most explainable fix (regime-aware)
    - EMA20 adds direction alignment
    - Tighter threshold 0.55-0.60 requires stronger compression = higher conviction signals
    - Combination of ADX<25 + threshold should address both trending failure modes
""")

print(f"\n{SEP}")
print("DONE")
print(SEP)
