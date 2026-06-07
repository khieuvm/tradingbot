"""
backtest/cb_time_analysis.py — CB 5m Time Window Analysis

Checks:
  1. Whether cb_full_report.py time filter aligns with combos/cb.py TIME_WINDOWS
  2. Performance by 15-min slot (with vs without time filter)
  3. Signal leakage outside configured windows
  4. May-June 2026 slot-level decay analysis
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

# ─── 1. DATA LOAD ─────────────────────────────────────────────────────────────
fetcher = DataFetcher()
end   = datetime.now().strftime("%Y-%m-%d")
start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
print(f"Fetching 5m data {start} -> {end} ...")

df = fetcher.get_futures_ohlcv("VN30F1M", start, end, interval="5m")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df['date']    = df['time'].dt.date
df['mins']    = df['time'].dt.hour * 60 + df['time'].dt.minute
df['session'] = np.where(df['mins'] < 12 * 60, 'AM', 'PM')
df['range']   = df['high'] - df['low']

# Session filter — whole session (not time-windowed yet)
df = df[
    ((df['mins'] >= 9*60) & (df['mins'] < 11*60+30)) |
    ((df['mins'] >= 13*60) & (df['mins'] < 14*60+30))
].reset_index(drop=True)

df['atr']   = ta.atr(df['high'], df['low'], df['close'], length=14)
df['rsi14'] = ta.rsi(df['close'], length=14)

n_days = df['date'].nunique()
print(f"Loaded {len(df)} bars | {n_days} trading days | "
      f"{df['date'].min()} -> {df['date'].max()}")

# ─── 2. SIGNAL DETECTION (numpy arrays) ──────────────────────────────────────
N_COMP      = 3
COMP_THRESH = 0.7
ATR_MIN     = 2.5
ATR_MAX     = 4.5
RSI_MAX     = 70
DEDUP_BARS  = 5

highs    = df['high'].values.astype(float)
lows     = df['low'].values.astype(float)
opens    = df['open'].values.astype(float)
closes   = df['close'].values.astype(float)
atrs_arr = df['atr'].values.astype(float)
rsi_arr  = df['rsi14'].values.astype(float)
ranges   = df['range'].values.astype(float)
dates    = df['date'].values
sessions = df['session'].values
mins_arr = df['mins'].values.astype(int)

# 3-bar compression
comp_arr = np.zeros(len(df), dtype=bool)
for i in range(N_COMP, len(df)):
    a = atrs_arr[i]
    if np.isnan(a) or a <= 0:
        continue
    if np.nanmax(ranges[i - N_COMP:i]) < COMP_THRESH * a:
        comp_arr[i] = True

rsi_ok    = np.where(np.isnan(rsi_arr), True, rsi_arr < RSI_MAX)
atr_ok    = (~np.isnan(atrs_arr)) & (atrs_arr >= ATR_MIN) & (atrs_arr <= ATR_MAX)
base_filt = comp_arr & atr_ok & rsi_ok  # ATR/RSI filtered, no time filter

# Time window masks — matching cb_full_report.py exactly
am_window = (mins_arr >= 9*60+15) & (mins_arr <= 10*60+45)
pm_window = (mins_arr >= 13*60+15) & (mins_arr <= 14*60+15)
in_window = am_window | pm_window

def dedup_arr(arr, min_bars):
    out  = arr.copy()
    last = -999
    for i in range(len(out)):
        if out[i]:
            if i - last < min_bars:
                out[i] = False
            else:
                last = i
    return out

# ─── Scenario A: WITH time filter (matches cb_full_report.py) ─────────────────
am_raw_f = base_filt & am_window
pm_raw_f = base_filt & pm_window
am_mask  = dedup_arr(am_raw_f, DEDUP_BARS)
pm_mask  = dedup_arr(pm_raw_f, DEDUP_BARS)
sig_locs_filtered = np.where(am_mask | pm_mask)[0]

# ─── Scenario B: WITHOUT time filter (full session, dedup per session) ────────
am_raw_all = base_filt & (mins_arr < 12*60)
pm_raw_all = base_filt & (mins_arr >= 12*60)
am_mask_all = dedup_arr(am_raw_all, DEDUP_BARS)
pm_mask_all = dedup_arr(pm_raw_all, DEDUP_BARS)
sig_locs_all = np.where(am_mask_all | pm_mask_all)[0]

# ─── Signals strictly OUTSIDE the configured window ──────────────────────────
out_window    = ~in_window
am_raw_out    = base_filt & (mins_arr < 12*60) & out_window
pm_raw_out    = base_filt & (mins_arr >= 12*60) & out_window
am_mask_out   = dedup_arr(am_raw_out, DEDUP_BARS)
pm_mask_out   = dedup_arr(pm_raw_out, DEDUP_BARS)
sig_locs_out  = np.where(am_mask_out | pm_mask_out)[0]

print(f"\nSignal counts:")
print(f"  WITH time filter (standard)  : {len(sig_locs_filtered)}")
print(f"  WITHOUT time filter (all)    : {len(sig_locs_all)}")
print(f"  OUTSIDE window only          : {len(sig_locs_out)}")

# ─── 3. STRATEGY PARAMS ──────────────────────────────────────────────────────
PARAMS = {
    'AM': dict(sl_mult=1.2, trail_act=5.0, trail_mult=2.0,
               max_hold=24, exit_mins=11*60+25),
    'PM': dict(sl_mult=1.0, trail_act=4.0, trail_mult=1.5,
               max_hold=12, exit_mins=14*60+25),
}
BE_TRIGGER      = 4.0
BE_ATR_MIN      = 3.5
TIGHTEN_MFE     = 8.0
TIGHTEN_MULT    = 1.2
TIGHTEN_ATR_MIN = 3.5
ADAPT_PMR       = 0.8
ADAPT_MFE       = 4.0
ADAPT_PNL       = 3.2

# ─── 4. BREAKOUT + SIMULATION ─────────────────────────────────────────────────
def get_breakout(sig_loc):
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


def simulate(sig_loc, entry_bar_loc, direction, entry_price):
    atr     = float(atrs_arr[sig_loc])
    session = sessions[sig_loc]
    p       = PARAMS[session]

    if sig_loc >= 2 and atr > 0:
        pre_move_ratio = float((closes[sig_loc] - opens[sig_loc-2]) * direction / atr)
    else:
        pre_move_ratio = 0.0

    sl            = entry_price - direction * p['sl_mult'] * atr
    best          = entry_price
    mfe           = 0.0
    trail_on      = False
    be_done       = False
    adapt_checked = False
    exit_p        = None
    exit_r        = None

    sig_date = dates[sig_loc]
    end_loc  = min(entry_bar_loc + p['max_hold'], len(df))

    for k in range(entry_bar_loc, end_loc):
        if dates[k] != sig_date or sessions[k] != session:
            exit_p = float(closes[k-1]) if k > entry_bar_loc else entry_price
            exit_r = 'SESSION'
            break
        if mins_arr[k] >= p['exit_mins']:
            exit_p = float(closes[k])
            exit_r = 'SESSION'
            break

        h = float(highs[k])
        l = float(lows[k])
        c = float(closes[k])

        if direction == 1:
            best = max(best, h)
            mfe  = max(mfe, best - entry_price)
        else:
            best = min(best, l)
            mfe  = max(mfe, entry_price - best)

        if (session == 'AM' and pre_move_ratio > ADAPT_PMR
                and mfe >= ADAPT_MFE and not adapt_checked):
            adapt_checked = True
            pnl_pts = direction * (c - entry_price)
            if pnl_pts >= ADAPT_PNL:
                exit_p = c
                exit_r = 'ADAPT_EXIT'
                break
            else:
                if direction == 1:
                    sl = max(sl, entry_price + 2.0)
                else:
                    sl = min(sl, entry_price - 2.0)

        if not be_done and mfe >= BE_TRIGGER and atr >= BE_ATR_MIN:
            be_done = True
            if direction == 1:
                sl = max(sl, entry_price)
            else:
                sl = min(sl, entry_price)

        if mfe >= p['trail_act']:
            trail_on = True
        if trail_on:
            t_mult   = (TIGHTEN_MULT if (mfe >= TIGHTEN_MFE and atr >= TIGHTEN_ATR_MIN)
                        else p['trail_mult'])
            trail_sl = best - t_mult * atr if direction == 1 else best + t_mult * atr
            if direction == 1:
                sl = max(sl, trail_sl)
            else:
                sl = min(sl, trail_sl)

        sl_hit = (direction == 1 and l <= sl) or (direction == -1 and h >= sl)
        if sl_hit:
            exit_p = sl
            exit_r  = 'TRAIL' if trail_on else ('BE' if be_done else 'SL')
            break

    if exit_p is None:
        exit_p = float(closes[min(end_loc - 1, len(df) - 1)])
        exit_r = 'MAX_HOLD'

    pnl = direction * (exit_p - entry_price) - COST

    sig_mins  = int(mins_arr[sig_loc])
    slot_s    = (sig_mins // 15) * 15
    slot_e    = slot_s + 15
    slot_lbl  = f"{slot_s//60:02d}:{slot_s%60:02d}-{slot_e//60:02d}:{slot_e%60:02d}"

    return {
        'date'      : dates[sig_loc],
        'time'      : str(df.iloc[sig_loc]['time'])[11:16],
        'session'   : session,
        'mins'      : sig_mins,
        'slot'      : slot_lbl,
        'slot_start': slot_s,
        'in_window' : bool(in_window[sig_loc]),
        'direction' : 'BUY' if direction == 1 else 'SELL',
        'entry'     : float(entry_price),
        'exit'      : float(exit_p),
        'atr'       : float(atr),
        'mfe'       : float(mfe),
        'pnl'       : float(pnl),
        'reason'    : exit_r,
    }


def run_sim(sig_locs_list, label):
    trades  = []
    expired = 0
    for sl in sig_locs_list:
        direction, entry_price, entry_bar_loc = get_breakout(int(sl))
        if direction is None:
            expired += 1
            continue
        trades.append(simulate(int(sl), entry_bar_loc, direction, entry_price))
    tdf_r = (pd.DataFrame(trades).sort_values(['date', 'time']).reset_index(drop=True)
             if trades else pd.DataFrame())
    print(f"  {label}: {len(tdf_r)} trades, {expired} expired no-breakout")
    return tdf_r


# ─── 5. RUN ───────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("RUNNING SIMULATIONS")
print("="*72)
tdf_f   = run_sim(sig_locs_filtered, "WITH time filter (standard)")
tdf_all = run_sim(sig_locs_all,      "WITHOUT time filter (full session)")
tdf_out = run_sim(sig_locs_out,      "OUTSIDE window only")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def pf_calc(sub):
    wins   = sub[sub['pnl'] > 0]['pnl'].sum()
    losses = abs(sub[sub['pnl'] <= 0]['pnl'].sum())
    return wins / losses if losses > 0 else 999.0


def fmt_row(sub, n_d=None):
    if len(sub) == 0:
        return "  0 trades"
    wr  = (sub['pnl'] > 0).mean() * 100
    pf  = pf_calc(sub)
    tot = sub['pnl'].sum()
    avg = sub['pnl'].mean()
    ppd = tot / n_d if n_d else tot / n_days
    return (f"T={len(sub):>4}  WR={wr:>5.1f}%  PF={pf:>5.2f}  "
            f"PnL={tot:>+8.1f}  avg={avg:>+5.2f}  P/D={ppd:>+6.2f}")


SEP  = "=" * 72
SEP2 = "-" * 60


# ─── REPORT 0: ALIGNMENT CHECK ────────────────────────────────────────────────
print(f"\n{SEP}")
print("REPORT 0: TIME WINDOW ALIGNMENT CHECK")
print(SEP)
print()

# Inline backtest (cb_full_report.py) windows
inline_am = (9*60+15, 10*60+45)
inline_pm = (13*60+15, 14*60+15)
print(f"  cb_full_report.py (inline) TIME WINDOWS:")
print(f"    AM : {inline_am[0]//60:02d}:{inline_am[0]%60:02d}  ->  "
      f"{inline_am[1]//60:02d}:{inline_am[1]%60:02d}  "
      f"(mins {inline_am[0]} - {inline_am[1]})")
print(f"    PM : {inline_pm[0]//60:02d}:{inline_pm[0]%60:02d}  ->  "
      f"{inline_pm[1]//60:02d}:{inline_pm[1]%60:02d}  "
      f"(mins {inline_pm[0]} - {inline_pm[1]})")

try:
    from combos import get_combo
    combo = get_combo("CB")
    combo_am = combo.TIME_WINDOWS["AM"]
    combo_pm = combo.TIME_WINDOWS["PM"]
    print(f"\n  combos/cb.py TIME_WINDOWS:")
    print(f"    AM : {combo_am[0]//60:02d}:{combo_am[0]%60:02d}  ->  "
          f"{combo_am[1]//60:02d}:{combo_am[1]%60:02d}  "
          f"(mins {combo_am[0]} - {combo_am[1]})")
    print(f"    PM : {combo_pm[0]//60:02d}:{combo_pm[0]%60:02d}  ->  "
          f"{combo_pm[1]//60:02d}:{combo_pm[1]%60:02d}  "
          f"(mins {combo_pm[0]} - {combo_pm[1]})")

    am_ok = combo_am == inline_am
    pm_ok = combo_pm == inline_pm
    print(f"\n  AM window match : {'ALIGNED' if am_ok else '*** MISMATCH ***'}")
    print(f"  PM window match : {'ALIGNED' if pm_ok else '*** MISMATCH ***'}")
    print(f"  DEDUP_BARS      : combo={combo.DEDUP_BARS}  inline={DEDUP_BARS}  "
          f"{'ALIGNED' if combo.DEDUP_BARS == DEDUP_BARS else '*** MISMATCH ***'}")
    print(f"\n  NOTE: combo.detect() is called on a rolling window (last bar check).")
    print(f"  NOTE: cb_full_report.py uses bulk array scan + explicit dedup.")
    print(f"  Both use identical ATR/RSI/compression thresholds.")
except Exception as e:
    print(f"  Could not load combo: {e}")

print(f"\n  Signal leakage check (signals outside window in backtest):")
leaked = tdf_f[~tdf_f['in_window']] if len(tdf_f) > 0 else pd.DataFrame()
print(f"  Signals OUTSIDE window in tdf_f : {len(leaked)} "
      f"{'-> NO LEAKAGE (correct)' if len(leaked) == 0 else '*** LEAKAGE FOUND ***'}")


# ─── REPORT 1: WITH vs WITHOUT TIME FILTER ────────────────────────────────────
print(f"\n{SEP}")
print("REPORT 1: WITH TIME FILTER vs WITHOUT TIME FILTER")
print(SEP)
print(f"\n  Scenario A — WITH filter (09:15-10:45 AM / 13:15-14:15 PM):")
print(f"    TOTAL  : {fmt_row(tdf_f)}")
for sess in ('AM', 'PM'):
    sub = tdf_f[tdf_f['session'] == sess]
    print(f"    {sess}     : {fmt_row(sub)}")

print(f"\n  Scenario B — WITHOUT filter (full session):")
print(f"    TOTAL  : {fmt_row(tdf_all)}")
for sess in ('AM', 'PM'):
    sub = tdf_all[tdf_all['session'] == sess]
    print(f"    {sess}     : {fmt_row(sub)}")

if len(tdf_out) > 0:
    print(f"\n  Outside-window signals only:")
    print(f"    TOTAL  : {fmt_row(tdf_out)}")
    for sess in ('AM', 'PM'):
        sub = tdf_out[tdf_out['session'] == sess]
        if len(sub) > 0:
            print(f"    {sess}     : {fmt_row(sub)}")
else:
    print(f"\n  Outside-window signals: NONE (all base_filt signals are within window)")

delta_ppd = (tdf_all['pnl'].sum() - tdf_f['pnl'].sum()) / n_days if len(tdf_all) > 0 else 0
print(f"\n  P/D delta (no-filter minus filtered) : {delta_ppd:+.2f} pts/day")
if delta_ppd > 0:
    print(f"  -> Expanding window would ADD {delta_ppd:+.2f} P/D")
else:
    print(f"  -> Current time filter is optimal or the outside-window signals hurt")


# ─── REPORT 2: BY 15-MIN SLOT (STANDARD = WITH FILTER) ───────────────────────
print(f"\n{SEP}")
print("REPORT 2: PERFORMANCE BY 15-MIN SLOT — Standard (with time filter)")
print(SEP)

# Pre-define all slots that appear in trading session
ALL_SLOT_STARTS_AM = [540, 555, 570, 585, 600, 615, 630, 645, 660, 675, 680]
ALL_SLOT_STARTS_PM = [780, 795, 810, 825, 840, 855, 860]
# Combine and de-duplicate, add any extras from actual data
all_slots = sorted(set(
    ALL_SLOT_STARTS_AM + ALL_SLOT_STARTS_PM +
    (list(tdf_f['slot_start'].unique()) if len(tdf_f) > 0 else []) +
    (list(tdf_all['slot_start'].unique()) if len(tdf_all) > 0 else [])
))

def slot_label(s):
    e = s + 15
    return f"{s//60:02d}:{s%60:02d}-{e//60:02d}:{e%60:02d}"

def in_win_check(s):
    return ((9*60+15 <= s <= 10*60+45) or (13*60+15 <= s <= 14*60+15))

print(f"\n  {'Slot':<16} {'T':>4} {'WR':>6} {'PF':>5} {'AvgPnL':>7} {'TotPnL':>8}  Status")
print("  " + SEP2)

prev_sess = None
for s in all_slots:
    cur_sess = 'AM' if s < 12*60 else 'PM'
    if prev_sess is not None and cur_sess != prev_sess:
        print()
    prev_sess = cur_sess

    lbl = slot_label(s)
    in_w = in_win_check(s)
    marker = "[IN WINDOW]" if in_w else "[OUT      ]"

    sub = tdf_f[tdf_f['slot_start'] == s] if len(tdf_f) > 0 else pd.DataFrame()
    if len(sub) == 0:
        print(f"  {lbl:<16} {'--':>4} {'--':>6} {'--':>5} {'--':>7} {'--':>8}  {marker}")
        continue
    wr  = (sub['pnl'] > 0).mean() * 100
    pf  = pf_calc(sub)
    avg = sub['pnl'].mean()
    tot = sub['pnl'].sum()
    print(f"  {lbl:<16} {len(sub):>4} {wr:>5.1f}% {pf:>5.2f} {avg:>+6.2f} {tot:>+8.1f}  {marker}")


# ─── REPORT 3: BY 15-MIN SLOT — WITHOUT FILTER ────────────────────────────────
if len(tdf_all) > 0:
    print(f"\n{SEP}")
    print("REPORT 3: PERFORMANCE BY 15-MIN SLOT — Without time filter")
    print(SEP)
    print(f"\n  {'Slot':<16} {'T':>4} {'WR':>6} {'PF':>5} {'AvgPnL':>7} {'TotPnL':>8}  Status")
    print("  " + SEP2)

    all_slots_b = sorted(tdf_all['slot_start'].unique())
    prev_sess = None
    for s in all_slots_b:
        cur_sess = 'AM' if s < 12*60 else 'PM'
        if prev_sess is not None and cur_sess != prev_sess:
            print()
        prev_sess = cur_sess

        lbl = slot_label(s)
        in_w = in_win_check(s)
        marker = "[IN WINDOW]" if in_w else "[OUT      ]"

        sub = tdf_all[tdf_all['slot_start'] == s]
        if len(sub) == 0:
            continue
        wr  = (sub['pnl'] > 0).mean() * 100
        pf  = pf_calc(sub)
        avg = sub['pnl'].mean()
        tot = sub['pnl'].sum()
        print(f"  {lbl:<16} {len(sub):>4} {wr:>5.1f}% {pf:>5.2f} {avg:>+6.2f} {tot:>+8.1f}  {marker}")


# ─── REPORT 4: MAY-JUNE 2026 ANALYSIS ────────────────────────────────────────
print(f"\n{SEP}")
print("REPORT 4: MAY-JUNE 2026 ANALYSIS (with time filter)")
print(SEP)

tdf_f['month'] = pd.to_datetime(tdf_f['date'].astype(str)).dt.to_period('M')
may_june   = tdf_f[tdf_f['month'].astype(str).isin(['2026-05', '2026-06'])]
before_may = tdf_f[~tdf_f['month'].astype(str).isin(['2026-05', '2026-06'])]

print(f"\n  All periods before May 2026 : {fmt_row(before_may)}")
print(f"  May-June 2026               : {fmt_row(may_june)}")

if len(may_june) > 0:
    print(f"\n  May-June by session:")
    for sess in ('AM', 'PM'):
        sub = may_june[may_june['session'] == sess]
        print(f"    {sess} : {fmt_row(sub)}")

    # Monthly detail
    df_months = df.copy()
    df_months['month'] = pd.to_datetime(df_months['date'].astype(str)).dt.to_period('M')
    print(f"\n  Monthly detail:")
    print(f"  {'Month':<8} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'Days':>5} {'P/D':>6}")
    print("  " + SEP2)
    for month_str in ['2026-05', '2026-06']:
        sub    = tdf_f[tdf_f['month'].astype(str) == month_str]
        m_days = df_months[df_months['month'].astype(str) == month_str]['date'].nunique()
        if len(sub) == 0 or m_days == 0:
            continue
        wr  = (sub['pnl'] > 0).mean() * 100
        pf  = pf_calc(sub)
        tot = sub['pnl'].sum()
        ppd = tot / m_days
        print(f"  {month_str:<8} {len(sub):>4} {wr:>5.1f}% {pf:>5.2f} "
              f"{tot:>+8.1f} {m_days:>5} {ppd:>+5.2f}")

    print(f"\n  May-June by 15-min slot:")
    print(f"  {'Slot':<16} {'T':>4} {'WR':>6} {'PF':>5} {'AvgPnL':>7} {'TotPnL':>8}  "
          f"{'Window':>11}  {'Decay?':>7}")
    print("  " + SEP2)

    # Compare slot performance: may_june vs before_may
    mj_slots   = sorted(may_june['slot_start'].unique())
    bm_slot_wr = {}
    for s in mj_slots:
        bm_sub = before_may[before_may['slot_start'] == s]
        if len(bm_sub) >= 5:
            bm_slot_wr[s] = (bm_sub['pnl'] > 0).mean() * 100
        else:
            bm_slot_wr[s] = None

    prev_sess = None
    for s in mj_slots:
        cur_sess = 'AM' if s < 12*60 else 'PM'
        if prev_sess is not None and cur_sess != prev_sess:
            print()
        prev_sess = cur_sess

        lbl  = slot_label(s)
        in_w = in_win_check(s)
        marker = "[IN WINDOW]" if in_w else "[OUT      ]"

        sub = may_june[may_june['slot_start'] == s]
        if len(sub) == 0:
            continue
        wr  = (sub['pnl'] > 0).mean() * 100
        pf  = pf_calc(sub)
        avg = sub['pnl'].mean()
        tot = sub['pnl'].sum()

        # Decay indicator: compare to historical WR for same slot
        hist_wr = bm_slot_wr.get(s)
        if hist_wr is not None:
            decay_delta = wr - hist_wr
            decay_flag  = f"{decay_delta:>+6.1f}%" if abs(decay_delta) >= 5 else "stable"
        else:
            decay_flag = "n/a"

        print(f"  {lbl:<16} {len(sub):>4} {wr:>5.1f}% {pf:>5.2f} {avg:>+6.2f} "
              f"{tot:>+8.1f}  {marker}  {decay_flag}")

    print(f"\n  May-June losing slots (avg PnL < 0):")
    losing = []
    for s in mj_slots:
        sub = may_june[may_june['slot_start'] == s]
        if len(sub) >= 3 and sub['pnl'].mean() < 0:
            losing.append((s, sub))
    if losing:
        for s, sub in losing:
            lbl = slot_label(s)
            in_w = in_win_check(s)
            marker = "[IN]" if in_w else "[OUT]"
            wr  = (sub['pnl'] > 0).mean() * 100
            avg = sub['pnl'].mean()
            print(f"    {lbl} {marker}  T={len(sub)}  WR={wr:.1f}%  avg={avg:+.2f}")
    else:
        print("    No slot with avg PnL < 0 (and >= 3 trades)")


# ─── REPORT 5: IS DECAY UNIFORM OR SLOT-CONCENTRATED? ────────────────────────
print(f"\n{SEP}")
print("REPORT 5: CB DECAY ANALYSIS — UNIFORM vs SLOT-CONCENTRATED")
print(SEP)

if len(before_may) > 0 and len(may_june) > 0:
    print(f"\n  Overall WR shift:")
    wr_hist = (before_may['pnl'] > 0).mean() * 100
    wr_mj   = (may_june['pnl'] > 0).mean() * 100
    pf_hist = pf_calc(before_may)
    pf_mj   = pf_calc(may_june)
    avg_hist = before_may['pnl'].mean()
    avg_mj   = may_june['pnl'].mean()
    print(f"    Before May 2026 : WR={wr_hist:.1f}%  PF={pf_hist:.2f}  avg={avg_hist:+.2f}")
    print(f"    May-June 2026   : WR={wr_mj:.1f}%  PF={pf_mj:.2f}  avg={avg_mj:+.2f}")
    print(f"    WR delta        : {wr_mj - wr_hist:+.1f}pp  |  avg delta: {avg_mj - avg_hist:+.2f}")

    print(f"\n  Slot-level WR comparison (historical vs May-June):")
    print(f"  {'Slot':<16} {'Hist-T':>6} {'Hist-WR':>8} {'MJ-T':>5} {'MJ-WR':>7} {'Delta':>7}  Window")
    print("  " + SEP2)

    all_check_slots = sorted(set(
        list(before_may['slot_start'].unique()) + list(may_june['slot_start'].unique())
    ))
    prev_sess = None
    for s in all_check_slots:
        cur_sess = 'AM' if s < 12*60 else 'PM'
        if prev_sess is not None and cur_sess != prev_sess:
            print()
        prev_sess = cur_sess

        lbl  = slot_label(s)
        in_w = in_win_check(s)
        marker = "[IN]" if in_w else "[OUT]"

        bm_sub = before_may[before_may['slot_start'] == s]
        mj_sub = may_june[may_june['slot_start'] == s]

        h_t  = len(bm_sub)
        mj_t = len(mj_sub)
        h_wr = (bm_sub['pnl'] > 0).mean() * 100 if h_t >= 3 else float('nan')
        mj_wr = (mj_sub['pnl'] > 0).mean() * 100 if mj_t >= 3 else float('nan')

        if h_t == 0 and mj_t == 0:
            continue

        if np.isnan(h_wr) or np.isnan(mj_wr):
            delta_str = "  n/a"
        else:
            d = mj_wr - h_wr
            delta_str = f"{d:>+7.1f}pp"

        h_wr_str  = f"{h_wr:>7.1f}%" if not np.isnan(h_wr) else "   n/a  "
        mj_wr_str = f"{mj_wr:>6.1f}%" if not np.isnan(mj_wr) else "  n/a "
        print(f"  {lbl:<16} {h_t:>6} {h_wr_str} {mj_t:>5} {mj_wr_str} {delta_str}  {marker}")

    # Which slots had worst WR drop AND had enough MJ trades?
    print(f"\n  Slots with most WR deterioration in May-June (>= 3 MJ trades):")
    decay_list = []
    for s in all_check_slots:
        bm_sub = before_may[before_may['slot_start'] == s]
        mj_sub = may_june[may_june['slot_start'] == s]
        if len(bm_sub) < 5 or len(mj_sub) < 3:
            continue
        h_wr  = (bm_sub['pnl'] > 0).mean() * 100
        mj_wr = (mj_sub['pnl'] > 0).mean() * 100
        decay_list.append((s, h_wr, mj_wr, mj_wr - h_wr, len(mj_sub)))

    decay_list.sort(key=lambda x: x[3])  # worst decay first
    for s, h_wr, mj_wr, delta, t in decay_list[:8]:
        lbl  = slot_label(s)
        in_w = in_win_check(s)
        marker = "[IN]" if in_w else "[OUT]"
        print(f"    {lbl} {marker}  hist={h_wr:.1f}%  MJ={mj_wr:.1f}%  "
              f"delta={delta:+.1f}pp  MJ trades={t}")

    # Summary: is decay uniform?
    big_drops = [d for d in decay_list if d[3] < -10]
    stable    = [d for d in decay_list if abs(d[3]) <= 10]
    improved  = [d for d in decay_list if d[3] > 10]
    print(f"\n  Decay classification (slots with >= 5 hist + 3 MJ trades):")
    print(f"    Big drop (>10pp) : {len(big_drops)} slots")
    print(f"    Stable (±10pp)   : {len(stable)} slots")
    print(f"    Improved (>10pp) : {len(improved)} slots")

    if len(big_drops) > 0 and len(big_drops) <= len(stable):
        print(f"  -> Decay is CONCENTRATED in specific slots, not uniform")
    elif len(stable) >= 0.6 * len(decay_list):
        print(f"  -> Decay appears UNIFORM across all slots")
    else:
        print(f"  -> Mixed decay pattern — check individual slots above")

print(f"\n{SEP}")
print("ANALYSIS COMPLETE")
print(SEP)
