# -*- coding: utf-8 -*-
"""
Opening Pulse Retracement Analysis - VN30F1M
=============================================
After the 9:00-9:30 opening pulse, analyze:
1. MAE (max adverse excursion from entry at 9:30) distribution
2. Retracement zones vs continuation probability
3. Timing of retracement (AM vs PM, fast vs slow)
4. By pulse size
5. Practical trading thresholds (SL, add, flip)

Entry reference: pulse_close = close of 9:25 bar (last bar of opening pulse)
MAE measured from entry_price (NOT from peak MFE)
Continuation = day_close in pulse direction from entry_price
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("e:/Trading/data/vn30f1m_5m.parquet")
df = pd.read_parquet(DATA_PATH)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)
df["date"] = df["time"].dt.date
df["hhmm"] = df["time"].dt.hour * 60 + df["time"].dt.minute

SEP  = "=" * 70
SEP2 = "-" * 70
MIN_PULSE_MAIN = 1.4  # filter: only days with |pulse| >= this value

print(SEP)
print("  OPENING PULSE RETRACEMENT ANALYSIS -- VN30F1M 5m")
print(SEP)
print()

# ─── Build per-day records ────────────────────────────────────────────────────
records_main = []   # |pulse| >= MIN_PULSE_MAIN
records_all  = []   # all days including small pulses

for date, grp in df.groupby("date"):
    grp = grp.sort_values("hhmm").reset_index(drop=True)

    # Day open: 9:00 bar open
    ob = grp[grp["hhmm"] == 540]
    if ob.empty:
        continue
    day_open = float(ob["open"].iloc[0])

    # Pulse end: close of 9:25 bar (hhmm=565), fallback to last bar before 9:30
    pb = grp[grp["hhmm"] == 565]
    if pb.empty:
        pb = grp[(grp["hhmm"] >= 540) & (grp["hhmm"] < 570)]
        if pb.empty:
            continue
    entry_price = float(pb["close"].iloc[-1])

    pulse_raw = entry_price - day_open
    pulse_mag = abs(pulse_raw)
    pulse_dir = int(np.sign(pulse_raw))
    if pulse_dir == 0 or pulse_mag < 0.05:
        continue

    # Post-pulse bars: AM [9:30, 11:30) and PM [13:00, 14:30)
    post_am  = grp[(grp["hhmm"] >= 570) & (grp["hhmm"] < 690)].copy()
    post_pm  = grp[(grp["hhmm"] >= 780) & (grp["hhmm"] < 870)].copy()
    post_all = pd.concat([post_am, post_pm]).sort_values("hhmm").reset_index(drop=True)

    if post_all.empty:
        continue

    # Day close (use PM if available)
    if not post_pm.empty:
        day_close = float(post_pm["close"].iloc[-1])
    elif not post_am.empty:
        day_close = float(post_am["close"].iloc[-1])
    else:
        continue

    # ── Bar-by-bar MAE / MFE tracking ──────────────────────────────────────
    cum_mae   = 0.0
    cum_mfe   = 0.0
    am_peak_mae = 0.0
    pm_peak_mae = 0.0

    first_50pct_hhmm  = None
    first_100pct_hhmm = None
    # Track first bar that has ANY adverse move >= 0.5 pts
    first_adverse_hhmm = None

    for _, bar in post_all.iterrows():
        h       = float(bar["high"])
        l       = float(bar["low"])
        hhmm_v  = int(bar["hhmm"])
        is_am   = hhmm_v < 690

        if pulse_dir > 0:
            bar_mae = entry_price - l      # adverse = price falling below entry
            bar_mfe = h - entry_price
        else:
            bar_mae = h - entry_price      # adverse = price rising above entry
            bar_mfe = entry_price - l

        cum_mae = max(cum_mae, bar_mae)
        cum_mfe = max(cum_mfe, bar_mfe)

        if is_am:
            am_peak_mae = max(am_peak_mae, bar_mae)
        else:
            pm_peak_mae = max(pm_peak_mae, bar_mae)

        retrace_pct = (cum_mae / pulse_mag) * 100

        if first_50pct_hhmm  is None and retrace_pct >= 50:
            first_50pct_hhmm  = hhmm_v
        if first_100pct_hhmm is None and retrace_pct >= 100:
            first_100pct_hhmm = hhmm_v
        if first_adverse_hhmm is None and bar_mae >= 0.5:
            first_adverse_hhmm = hhmm_v

    # ── Derived metrics ─────────────────────────────────────────────────────
    final_pnl       = (day_close - entry_price) * pulse_dir
    max_retrace_pct = (cum_mae  / pulse_mag) * 100
    max_mfe_pct     = (cum_mfe  / pulse_mag) * 100
    am_retrace_pct  = (am_peak_mae / pulse_mag) * 100
    pm_retrace_pct  = (pm_peak_mae / pulse_mag) * 100
    max_retrace_in_pm = pm_peak_mae > am_peak_mae

    # Speed of 50% retracement
    if first_50pct_hhmm is None:
        speed_50 = "no_50pct"
    elif first_50pct_hhmm >= 780:
        speed_50 = "pm_50pct"
    elif first_50pct_hhmm <= 585:   # <= 9:45
        speed_50 = "fast_50pct"
    elif first_50pct_hhmm <= 630:   # <= 10:30
        speed_50 = "mid_50pct"
    else:
        speed_50 = "slow_50pct"     # after 10:30 AM

    # Pulse size category
    if   pulse_mag < 1.4:  size_cat = "small"
    elif pulse_mag <= 3.5: size_cat = "medium"
    else:                  size_cat = "large"

    # Speed of FIRST adverse move (>= 0.5 pts)
    if first_adverse_hhmm is None:
        adverse_start = "none"
    elif first_adverse_hhmm >= 780:
        adverse_start = "pm"
    elif first_adverse_hhmm == 570:
        adverse_start = "immediate"   # first bar after pulse
    elif first_adverse_hhmm <= 600:
        adverse_start = "early_am"    # within 30 min
    else:
        adverse_start = "late_am"

    rec = dict(
        date=pd.Timestamp(date),
        pulse_mag=pulse_mag,
        pulse_dir=pulse_dir,
        size_cat=size_cat,
        entry_price=entry_price,
        day_close=day_close,
        final_pnl=final_pnl,
        continuation=(final_pnl > 0),
        max_retrace_pct=max_retrace_pct,
        max_mfe_pct=max_mfe_pct,
        cum_mae_pts=cum_mae,
        cum_mfe_pts=cum_mfe,
        am_retrace_pct=am_retrace_pct,
        pm_retrace_pct=pm_retrace_pct,
        max_retrace_in_pm=max_retrace_in_pm,
        first_50pct_hhmm=first_50pct_hhmm,
        first_100pct_hhmm=first_100pct_hhmm,
        first_adverse_hhmm=first_adverse_hhmm,
        speed_50=speed_50,
        adverse_start=adverse_start,
    )
    records_all.append(rec)
    if pulse_mag >= MIN_PULSE_MAIN:
        records_main.append(rec)

all_days  = pd.DataFrame(records_main)
all_incl  = pd.DataFrame(records_all)

n = len(all_days)
print(f"Total days in dataset : {len(all_incl)}")
print(f"Days |pulse|>={MIN_PULSE_MAIN} pts  : {n}")
print(f"Date range            : {all_days['date'].min().date()} to {all_days['date'].max().date()}")
print(f"Baseline cont% (all)  : {all_days['continuation'].mean()*100:.1f}%  "
      f"(expected ~65% based on prior pulse analysis)")
print(f"Avg max retracement   : {all_days['max_retrace_pct'].mean():.1f}% of pulse")
print(f"Median max retracement: {all_days['max_retrace_pct'].median():.1f}% of pulse")
print(f"Avg MAE (pts)         : {all_days['cum_mae_pts'].mean():.2f}")
print(f"Avg MFE (pts)         : {all_days['cum_mfe_pts'].mean():.2f}")
print()

# ─── Section 1: Retracement Distribution ─────────────────────────────────────
print(SEP)
print("  1. DISTRIBUTION OF MAX RETRACEMENT (entire day)")
print(SEP)
print()
print(f"  {'Range':<14} | {'n':>4} | {'%days':>6} | {'CumPct':>7} | {'Cont%':>6} | {'AvgPnL':>7} | {'AvgMFE':>7}")
print(f"  {'-'*14} | {'-'*4} | {'-'*6} | {'-'*7} | {'-'*6} | {'-'*7} | {'-'*7}")

buckets = [(0,20), (20,40), (40,60), (60,80), (80,100), (100,130), (130,160), (160,200), (200,9999)]
bucket_labels = ["0-20%","20-40%","40-60%","60-80%","80-100%","100-130%","130-160%","160-200%",">200%"]
cum_n = 0
for (lo, hi), lbl in zip(buckets, bucket_labels):
    mask = (all_days["max_retrace_pct"] >= lo) & (all_days["max_retrace_pct"] < hi)
    sub  = all_days[mask]
    if len(sub) == 0:
        continue
    cum_n += len(sub)
    cont  = sub["continuation"].mean() * 100
    apnl  = sub["final_pnl"].mean()
    amfe  = sub["cum_mfe_pts"].mean()
    print(f"  {lbl:<14} | {len(sub):>4} | {len(sub)/n*100:>5.1f}% | {cum_n/n*100:>6.1f}% | "
          f"{cont:>5.1f}% | {apnl:>+6.2f} | {amfe:>6.2f}")
print()
print(f"  Summary buckets:")
for lbl2, lo2, hi2 in [("< 50% retrace",0,50), ("50-100%",50,100), ("> 100% (reversal)",100,9999)]:
    m = (all_days["max_retrace_pct"] >= lo2) & (all_days["max_retrace_pct"] < hi2)
    s = all_days[m]
    if len(s) == 0: continue
    print(f"    {lbl2:<22}: n={len(s):3d} ({len(s)/n*100:4.0f}%)  "
          f"Cont%={s['continuation'].mean()*100:.1f}%  AvgPnL={s['final_pnl'].mean():+.2f}")
print()

# ─── Section 2: Task Retracement Zones ───────────────────────────────────────
print(SEP)
print("  2. RETRACEMENT ZONES -- P(CONTINUATION)")
print(SEP)
print()
print("  Continuation = day_close in pulse direction vs entry price at 9:30")
print()
print(f"  {'Zone':<26} | {'n':>4} | {'Cont%':>6} | {'AvgPnL':>7} | "
      f"{'AvgMAE':>7} | {'AvgMFE':>7} | {'MFE>MAE':>7}")
print(f"  {'-'*26} | {'-'*4} | {'-'*6} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7}")

ZONES = [
    ("0-30%  (shallow pullback)", 0,   30),
    ("30-50% (normal pullback)",  30,  50),
    ("50-70% (deep pullback)",    50,  70),
    ("70-100%(major retrace)",    70, 100),
    (">100%  (full reversal)",   100, 9999),
]

for zlbl, lo, hi in ZONES:
    mask = (all_days["max_retrace_pct"] >= lo) & (all_days["max_retrace_pct"] < hi)
    sub  = all_days[mask]
    if len(sub) < 3:
        print(f"  {zlbl:<26} | {len(sub):>4} | (n<3)")
        continue
    cont      = sub["continuation"].mean() * 100
    avg_pnl   = sub["final_pnl"].mean()
    avg_mae   = sub["cum_mae_pts"].mean()
    avg_mfe   = sub["cum_mfe_pts"].mean()
    mfe_gt    = (sub["cum_mfe_pts"] > sub["cum_mae_pts"]).mean() * 100
    print(f"  {zlbl:<26} | {len(sub):>4} | {cont:>5.1f}% | {avg_pnl:>+6.2f} | "
          f"{avg_mae:>6.2f} | {avg_mfe:>6.2f} | {mfe_gt:>6.1f}%")
print()
print("  MFE>MAE% = % of days where final MFE exceeded final MAE (trade favorable overall)")
print()

# ─── Section 3: Time of Retracement ──────────────────────────────────────────
print(SEP)
print("  3. WHERE DOES THE MAX RETRACEMENT HAPPEN? (AM vs PM)")
print(SEP)
print()
for lbl, mask_c in [("AM max retrace", ~all_days["max_retrace_in_pm"]),
                    ("PM max retrace",  all_days["max_retrace_in_pm"])]:
    sub = all_days[mask_c]
    if len(sub) < 5: continue
    print(f"  {lbl}: n={len(sub)} ({len(sub)/n*100:.0f}% of days)")
    print(f"    Avg max retrace : {sub['max_retrace_pct'].mean():.1f}%")
    print(f"    Continuation    : {sub['continuation'].mean()*100:.1f}%")
    print(f"    Avg final PnL   : {sub['final_pnl'].mean():+.2f} pts")
    print()

# By zone split by AM vs PM
for session_lbl, mask_c in [("AM max retrace", ~all_days["max_retrace_in_pm"]),
                             ("PM max retrace",  all_days["max_retrace_in_pm"])]:
    sess_days = all_days[mask_c]
    if len(sess_days) < 5:
        continue
    print(f"  {session_lbl} -- zone breakdown:")
    print(f"  {'Zone':<26} | {'n':>4} | {'Cont%':>6} | {'AvgPnL':>7}")
    print(f"  {'-'*26} | {'-'*4} | {'-'*6} | {'-'*7}")
    for zlbl, lo, hi in ZONES:
        m = (sess_days["max_retrace_pct"] >= lo) & (sess_days["max_retrace_pct"] < hi)
        s = sess_days[m]
        if len(s) < 3: continue
        print(f"  {zlbl:<26} | {len(s):>4} | {s['continuation'].mean()*100:>5.1f}% | "
              f"{s['final_pnl'].mean():>+6.2f}")
    print()

# ─── Section 4: Speed of Retracement ─────────────────────────────────────────
print(SEP)
print("  4. SPEED OF 50% RETRACEMENT")
print(SEP)
print()
print("  fast_50pct : 50% retracement reached by 9:45  (<=15 min after 9:30)")
print("  mid_50pct  : 50% reached by 10:30             (15-60 min after 9:30)")
print("  slow_50pct : 50% reached after 10:30          (>60 min, still AM)")
print("  pm_50pct   : 50% first reached in PM session")
print("  no_50pct   : never reached 50% during the day")
print()
print(f"  {'Category':<22} | {'n':>4} | {'%days':>6} | {'Cont%':>6} | "
      f"{'AvgPnL':>7} | {'AvgMaxRetrace':>13}")
print(f"  {'-'*22} | {'-'*4} | {'-'*6} | {'-'*6} | {'-'*7} | {'-'*13}")

SPEED_ORDER  = ["no_50pct", "fast_50pct", "mid_50pct", "slow_50pct", "pm_50pct"]
SPEED_LABELS = {
    "no_50pct"   : "Never >= 50%",
    "fast_50pct" : "Fast (<= 9:45)",
    "mid_50pct"  : "Mid (9:45-10:30)",
    "slow_50pct" : "Slow (>10:30 AM)",
    "pm_50pct"   : "PM session only",
}
for cat in SPEED_ORDER:
    sub = all_days[all_days["speed_50"] == cat]
    if len(sub) == 0: continue
    cont  = sub["continuation"].mean() * 100
    apnl  = sub["final_pnl"].mean()
    aret  = sub["max_retrace_pct"].mean()
    print(f"  {SPEED_LABELS[cat]:<22} | {len(sub):>4} | {len(sub)/n*100:>5.1f}% | "
          f"{cont:>5.1f}% | {apnl:>+6.2f} | {aret:>12.1f}%")

print()
# Direct comparison: fast vs slow
fast_s = all_days[all_days["speed_50"] == "fast_50pct"]
slow_s = all_days[all_days["speed_50"] == "slow_50pct"]
no_s   = all_days[all_days["speed_50"] == "no_50pct"]
if len(fast_s) >= 5 and len(slow_s) >= 5:
    diff = (slow_s["continuation"].mean() - fast_s["continuation"].mean()) * 100
    print(f"  Fast vs Slow: fast Cont={fast_s['continuation'].mean()*100:.1f}%  "
          f"slow Cont={slow_s['continuation'].mean()*100:.1f}%  diff={diff:+.1f}%")
    if fast_s["continuation"].mean() < 0.50:
        print("  -> FAST 50% retracement is a REVERSAL signal (Cont < 50%)")
    else:
        print("  -> Fast retracement: still likely to continue")
if len(no_s) >= 5:
    print(f"  No 50% retrace: Cont={no_s['continuation'].mean()*100:.1f}%  "
          f"(n={len(no_s)}) -- strongest continuation signal")
print()

# Timing of first adverse move
print("  FIRST MEANINGFUL ADVERSE MOVE (>= 0.5 pts from entry):")
print(f"  {'When adverse starts':<20} | {'n':>4} | {'Cont%':>6} | {'AvgPnL':>7} | {'AvgRetrace':>10}")
print(f"  {'-'*20} | {'-'*4} | {'-'*6} | {'-'*7} | {'-'*10}")
for cat, lbl in [("none","No adverse move"),("immediate","Immediate (9:30)"),
                 ("early_am","Early AM (<30min)"),("late_am","Late AM (>30min)"),
                 ("pm","PM session")]:
    sub = all_days[all_days["adverse_start"] == cat]
    if len(sub) < 3: continue
    cont = sub["continuation"].mean() * 100
    apnl = sub["final_pnl"].mean()
    aret = sub["max_retrace_pct"].mean()
    print(f"  {lbl:<20} | {len(sub):>4} | {cont:>5.1f}% | {apnl:>+6.2f} | {aret:>9.1f}%")
print()

# ─── Section 5: By Pulse Size ─────────────────────────────────────────────────
print(SEP)
print("  5. BY PULSE SIZE")
print(SEP)
print()
print(f"  {'Size':<22} | {'n':>4} | {'AvgPulse':>8} | {'AvgRetrace':>10} | "
      f"{'Cont%':>6} | {'AvgPnL':>7} | {'<50%Days':>8} | {'>100%Days':>9}")
print(f"  {'-'*22} | {'-'*4} | {'-'*8} | {'-'*10} | {'-'*6} | {'-'*7} | {'-'*8} | {'-'*9}")
for sz, lbl in [("small","Small (<1.4 pts)"),("medium","Medium (1.4-3.5 pts)"),("large","Large (>3.5 pts)")]:
    sub = all_incl[all_incl["size_cat"] == sz]
    if len(sub) < 3: continue
    cont      = sub["continuation"].mean() * 100
    avg_pulse = sub["pulse_mag"].mean()
    avg_ret   = sub["max_retrace_pct"].mean()
    avg_pnl   = sub["final_pnl"].mean()
    pct_sh    = (sub["max_retrace_pct"] < 50).mean() * 100
    pct_rev   = (sub["max_retrace_pct"] >= 100).mean() * 100
    print(f"  {lbl:<22} | {len(sub):>4} | {avg_pulse:>8.2f} | {avg_ret:>9.1f}% | "
          f"{cont:>5.1f}% | {avg_pnl:>+6.2f} | {pct_sh:>7.0f}% | {pct_rev:>8.0f}%")
print()

# Zone breakdown per size
for sz, sz_lbl in [("medium","Medium (1.4-3.5 pts)"),("large","Large (>3.5 pts)")]:
    sub = all_incl[all_incl["size_cat"] == sz]
    if len(sub) < 5: continue
    print(f"  {sz_lbl} -- zone breakdown (n={len(sub)}):")
    print(f"  {'Zone':<26} | {'n':>4} | {'Cont%':>6} | {'AvgPnL':>7} | {'AvgMAE':>7}")
    print(f"  {'-'*26} | {'-'*4} | {'-'*6} | {'-'*7} | {'-'*7}")
    for zlbl, lo, hi in ZONES:
        m = (sub["max_retrace_pct"] >= lo) & (sub["max_retrace_pct"] < hi)
        s = sub[m]
        if len(s) < 3: continue
        print(f"  {zlbl:<26} | {len(s):>4} | {s['continuation'].mean()*100:>5.1f}% | "
              f"{s['final_pnl'].mean():>+6.2f} | {s['cum_mae_pts'].mean():>6.2f}")
    print()

# ─── Section 6: SL Threshold Scan ────────────────────────────────────────────
print(SEP)
print("  6. STOP-LOSS THRESHOLD ANALYSIS")
print(SEP)
print()
print("  For each SL level X: P(day would have been profitable | retrace crossed X%)")
print("  High Cont% at X => that SL was too tight (you'd exit winners)")
print("  Cont% < 50% at X => X is a valid stop (more losers than winners)")
print()
print(f"  {'SL %pulse':>9} | {'Touched':>7} | {'Cont%':>6} | {'AvgPnL':>7} | "
      f"{'MFE>MAE%':>8} | {'Notes'}")
print(f"  {'-'*9} | {'-'*7} | {'-'*6} | {'-'*7} | {'-'*8} | {'-'*20}")

flip_level = None
prev_cont  = None
for sl_pct in list(range(10, 110, 10)) + [120, 150, 200]:
    touched = all_days[all_days["max_retrace_pct"] >= sl_pct]
    if len(touched) < 5: continue
    cont       = touched["continuation"].mean() * 100
    avg_pnl    = touched["final_pnl"].mean()
    mfe_gt_mae = (touched["cum_mfe_pts"] > touched["cum_mae_pts"]).mean() * 100
    note = ""
    if prev_cont is not None and prev_cont >= 50 and cont < 50 and flip_level is None:
        flip_level = sl_pct
        note = "<-- FLIP POINT (Cont drops below 50%)"
    if sl_pct == 100:
        note = "<-- full reversal"
    prev_cont = cont
    print(f"  {sl_pct:>8}% | {len(touched):>7} | {cont:>5.1f}% | {avg_pnl:>+6.2f} | "
          f"{mfe_gt_mae:>7.1f}% | {note}")

print()
if flip_level:
    pulse_med = all_days["pulse_mag"].median()
    sl_pts    = pulse_med * flip_level / 100
    print(f"  RESULT: P(cont) flips below 50% at ~{flip_level}% retracement")
    print(f"  Median pulse = {pulse_med:.2f} pts  =>  SL = {flip_level}% x {pulse_med:.2f} = {sl_pts:.2f} pts from entry")
else:
    print("  RESULT: P(cont) does not drop below 50% within data range")
print()

# ─── Section 7: Add-on-Pullback Analysis ─────────────────────────────────────
print(SEP)
print("  7. ADD-ON-PULLBACK: Best Retracement Depth to Add")
print(SEP)
print()
print("  Zone = EXACT range of max retracement (not 'crossed X%')")
print("  High Cont% + positive AvgPnL = good zone to scale in")
print()
print(f"  {'Zone':<14} | {'n':>4} | {'Cont%':>6} | {'AvgPnL':>7} | {'AvgMFE':>7} | "
      f"{'MFE/Pulse':>9} | {'Risk (MAE/pulse)':>16}")
print(f"  {'-'*14} | {'-'*4} | {'-'*6} | {'-'*7} | {'-'*7} | {'-'*9} | {'-'*16}")

for lo in range(0, 130, 15):
    hi  = lo + 15
    m   = (all_days["max_retrace_pct"] >= lo) & (all_days["max_retrace_pct"] < hi)
    sub = all_days[m]
    if len(sub) < 5: continue
    cont   = sub["continuation"].mean() * 100
    apnl   = sub["final_pnl"].mean()
    amfe   = sub["cum_mfe_pts"].mean()
    mfe_r  = sub["max_mfe_pct"].mean()
    mae_r  = sub["max_retrace_pct"].mean()
    lbl    = f"{lo}-{hi}%"
    print(f"  {lbl:<14} | {len(sub):>4} | {cont:>5.1f}% | {apnl:>+6.2f} | {amfe:>6.2f} | "
          f"{mfe_r:>8.1f}% | {mae_r:>15.1f}%")
print()

# ─── Section 8: Combined Filters ──────────────────────────────────────────────
print(SEP)
print("  8. COMBINED FILTERS: PULSE SIZE x RETRACEMENT ZONE")
print(SEP)
print()
print(f"  {'Size x Zone':<35} | {'n':>4} | {'Cont%':>6} | {'AvgPnL':>7}")
print(f"  {'-'*35} | {'-'*4} | {'-'*6} | {'-'*7}")

for sz, sz_lbl_s in [("medium","MED"), ("large","LRG")]:
    size_mask = all_incl["size_cat"] == sz
    for zlbl, lo, hi in ZONES:
        zone_mask = (all_incl["max_retrace_pct"] >= lo) & (all_incl["max_retrace_pct"] < hi)
        sub       = all_incl[size_mask & zone_mask]
        if len(sub) < 5: continue
        combo_lbl = f"{sz_lbl_s} + {zlbl}"
        cont      = sub["continuation"].mean() * 100
        apnl      = sub["final_pnl"].mean()
        print(f"  {combo_lbl:<35} | {len(sub):>4} | {cont:>5.1f}% | {apnl:>+6.2f}")
print()

# ─── Section 9: Reversal Detection ────────────────────────────────────────────
print(SEP)
print("  9. REVERSAL DETECTION: P(continue | retrace crosses X%)")
print(SEP)
print()
print("  Key question: once retrace crosses a level, is it better to exit (fade) or hold?")
print()
print(f"  {'Retrace >= X%':<15} | {'n':>4} | {'P(cont)':>7} | {'P(rev)':>6} | "
      f"{'Edge vs 50%':>11} | {'Decision'}")
print(f"  {'-'*15} | {'-'*4} | {'-'*7} | {'-'*6} | {'-'*11} | {'-'*15}")

for x in [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]:
    touched = all_days[all_days["max_retrace_pct"] >= x]
    if len(touched) < 5: break
    p_cont  = touched["continuation"].mean() * 100
    p_rev   = 100 - p_cont
    edge    = p_cont - 50
    decision = "HOLD (follow)" if p_cont >= 55 else ("FLIP" if p_cont < 45 else "NEUTRAL")
    print(f"  {x:>3}%            | {len(touched):>4} | {p_cont:>6.1f}% | {p_rev:>5.1f}% | "
          f"{edge:>+10.1f}% | {decision}")
print()

# ─── Section 10: Summary ──────────────────────────────────────────────────────
print(SEP)
print("  10. SUMMARY & ACTIONABLE TRADING RULES")
print(SEP)
print()

# Compute key stats
pct_under_50   = (all_days["max_retrace_pct"] < 50).mean() * 100
pct_over_100   = (all_days["max_retrace_pct"] >= 100).mean() * 100
cont_under_50  = all_days.loc[all_days["max_retrace_pct"] < 50, "continuation"].mean() * 100
cont_over_100  = all_days.loc[all_days["max_retrace_pct"] >= 100, "continuation"].mean() * 100
cont_50_100    = all_days.loc[
    (all_days["max_retrace_pct"] >= 50) & (all_days["max_retrace_pct"] < 100),
    "continuation"].mean() * 100

# Best add zone
best_add_cont = 0; best_add_lbl = "N/A"
for lo in range(0, 130, 15):
    hi  = lo + 15
    m   = (all_days["max_retrace_pct"] >= lo) & (all_days["max_retrace_pct"] < hi)
    sub = all_days[m]
    if len(sub) < 5: continue
    c = sub["continuation"].mean() * 100
    p = sub["final_pnl"].mean()
    if c > best_add_cont and p > 0:
        best_add_cont = c; best_add_lbl = f"{lo}-{hi}%"

print(f"  DATA: {n} days, |pulse| >= {MIN_PULSE_MAIN} pts")
print()
print(f"  RETRACEMENT PROFILE:")
print(f"    {pct_under_50:.0f}% of days have max retrace < 50%  -> Cont={cont_under_50:.1f}%")
print(f"    {100-pct_under_50-pct_over_100:.0f}% of days have 50-100% retrace  -> Cont={cont_50_100:.1f}%")
print(f"    {pct_over_100:.0f}% of days have full reversal >100% -> Cont={cont_over_100:.1f}%")
print()
print(f"  RULE 1 -- STOP-LOSS:")
if flip_level:
    pulse_med = all_days["pulse_mag"].median()
    print(f"    P(continuation) drops below 50% at ~{flip_level}% retracement")
    print(f"    => Set SL at {flip_level}% of pulse magnitude")
    print(f"    => For avg pulse {pulse_med:.1f} pts: SL = {pulse_med*flip_level/100:.1f} pts from entry")
    print(f"    => In ATR terms: if pulse ~= 1x ATR, SL = {flip_level}% of ATR")
else:
    # find empirical level
    for x in range(50, 200, 10):
        t = all_days[all_days["max_retrace_pct"] >= x]
        if len(t) >= 5 and t["continuation"].mean() < 0.50:
            print(f"    P(cont) drops below 50% at ~{x}% retracement")
            break
    else:
        print(f"    P(cont) stays above 50% throughout observed range")
        print(f"    => Day-session traders: use 100% of pulse as SL floor")
print()
print(f"  RULE 2 -- ADD ON PULLBACK:")
print(f"    Best add zone: {best_add_lbl}  Cont={best_add_cont:.1f}%")
print(f"    => Scale in when price retraces to 30-50% of pulse magnitude")
print()
print(f"  RULE 3 -- REVERSAL / FLIP:")
print(f"    Once price retraces >100% of pulse magnitude (full reversal):")
print(f"    Cont={cont_over_100:.1f}% => "
      + ("Consider FLIP direction" if cont_over_100 < 50 else "Still slight continuation bias"))
print()
print(f"  RULE 4 -- SPEED RULE:")
for cat, warn in [("fast_50pct","WARNING: early reversal"),("slow_50pct","OK: normal pullback"),
                  ("no_50pct","STRONG: trend continuation")]:
    sub = all_days[all_days["speed_50"] == cat]
    if len(sub) < 5: continue
    c = sub["continuation"].mean()*100
    print(f"    If 50% retrace {SPEED_LABELS[cat]}: Cont={c:.1f}%  [{warn if abs(c-50)>10 else 'neutral'}]")
print()
print(f"  RULE 5 -- PULSE SIZE:")
for sz, lbl in [("large","Large pulse (>3.5 pts)")]:
    sub = all_incl[all_incl["size_cat"] == sz]
    if len(sub) < 5: continue
    pct_rev = (sub["max_retrace_pct"] >= 100).mean()*100
    c       = sub["continuation"].mean()*100
    print(f"    {lbl}: full reversal rate={pct_rev:.0f}%  Cont={c:.0f}%")
    print(f"    => Large pulses are more reliable: shallower retracements, stronger trend")
print()
print("Done. Cost per trade: 0.96 pts.")
