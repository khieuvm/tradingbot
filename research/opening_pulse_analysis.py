# -*- coding: utf-8 -*-
"""
Opening Pulse Analysis - VN30F1M
=================================
Phan tich tuong quan giua "pulse" dau phien (9:00-9:30) va
toan bo dien bien trong ngay.

- Opening pulse : open[9:00] -> close[9:25] (6 bars 5m)
- AM session    : open[9:00] -> close[last AM bar <= 11:25]
- Full day      : open[9:00] -> close[last PM bar <= 14:25]
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = Path("e:/Trading/data/vn30f1m_5m.parquet")
df = pd.read_parquet(DATA_PATH)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# Add helper columns
df["date"] = df["time"].dt.date
df["hhmm"] = df["time"].dt.hour * 60 + df["time"].dt.minute  # minutes since midnight
df["dow"]  = df["time"].dt.dayofweek   # 0=Mon .. 4=Fri
DOW_NAME   = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build per-day metrics
# ─────────────────────────────────────────────────────────────────────────────
# 5m bars (left-label): 9:00 bar covers 9:00-9:05, labeled hhmm=540
#   Opening pulse  -> bars hhmm in {540,545,550,555,560,565}  -> need close of hhmm=565
#   AM session     -> bars hhmm in [540, 690)                  -> need close of last bar < 690
#   Full day PM    -> bars hhmm in [780, 870)                  -> need close of last bar < 870

records = []

for date, grp in df.groupby("date"):
    grp = grp.sort_values("hhmm")

    # Opening bar (9:00)
    open_bar = grp[grp["hhmm"] == 540]
    if open_bar.empty:
        continue
    day_open = float(open_bar["open"].iloc[0])

    # Opening pulse: close of 9:25 bar (hhmm=565)
    pulse_end_bar = grp[grp["hhmm"] == 565]
    if pulse_end_bar.empty:
        # fallback: last bar before 9:30 (hhmm < 570)
        pulse_end_bar = grp[grp["hhmm"] < 570]
        if pulse_end_bar.empty:
            continue
    pulse_close = float(pulse_end_bar["close"].iloc[-1])

    # AM session: last bar with hhmm in [540, 690)
    am_bars = grp[(grp["hhmm"] >= 540) & (grp["hhmm"] < 690)]
    if am_bars.empty:
        continue
    am_close = float(am_bars["close"].iloc[-1])
    am_high   = float(am_bars["high"].max())
    am_low    = float(am_bars["low"].min())

    # Full day: last bar with hhmm in [780, 870)
    pm_bars = grp[(grp["hhmm"] >= 780) & (grp["hhmm"] < 870)]
    if pm_bars.empty:
        day_close = am_close
        day_high  = am_high
        day_low   = am_low
    else:
        day_close = float(pm_bars["close"].iloc[-1])
        day_high  = float(pd.concat([am_bars, pm_bars])["high"].max())
        day_low   = float(pd.concat([am_bars, pm_bars])["low"].min())

    dow = int(grp["dow"].iloc[0])

    records.append({
        "date":          pd.Timestamp(date),
        "dow":           dow,
        "dow_name":      DOW_NAME[dow],
        "day_open":      day_open,
        "pulse_close":   pulse_close,
        "am_close":      am_close,
        "day_close":     day_close,
        "am_high":       am_high,
        "am_low":        am_low,
        "day_high":      day_high,
        "day_low":       day_low,
    })

daily = pd.DataFrame(records)
daily = daily.sort_values("date").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Compute moves
# ─────────────────────────────────────────────────────────────────────────────
daily["pulse_chg"] = daily["pulse_close"] - daily["day_open"]
daily["am_chg"]    = daily["am_close"]    - daily["day_open"]
daily["day_chg"]   = daily["day_close"]   - daily["day_open"]
daily["pulse_dir"] = np.sign(daily["pulse_chg"])
daily["am_dir"]    = np.sign(daily["am_chg"])
daily["day_dir"]   = np.sign(daily["day_chg"])
daily["day_range"] = daily["day_high"] - daily["day_low"]
daily["am_range"]  = daily["am_high"]  - daily["am_low"]

print(f"Total trading days loaded: {len(daily)}")
print(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def hit_rate(pulse_dir, target_dir):
    """% days where pulse direction == target direction (ignoring zeros)."""
    mask = (pulse_dir != 0) & (target_dir != 0)
    if mask.sum() == 0:
        return np.nan
    return (pulse_dir[mask] == target_dir[mask]).mean() * 100

def pearson(x, y):
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return r, p

def cond_prob(df_sub, cond_col, cond_val, tgt_col):
    """P(tgt_col == cond_val | cond_col == cond_val)"""
    mask = (df_sub[cond_col] == cond_val) & (df_sub[cond_col] != 0) & (df_sub[tgt_col] != 0)
    if mask.sum() == 0:
        return np.nan, 0
    p = (df_sub.loc[mask, tgt_col] == cond_val).mean() * 100
    return p, int(mask.sum())

SEP = "=" * 72

# ─────────────────────────────────────────────────────────────────────────────
# 5. Overall correlation stats
# ─────────────────────────────────────────────────────────────────────────────
n = len(daily)
r_pulse_am,  p_pulse_am  = pearson(daily["pulse_chg"], daily["am_chg"])
r_pulse_day, p_pulse_day = pearson(daily["pulse_chg"], daily["day_chg"])
hr_am  = hit_rate(daily["pulse_dir"], daily["am_dir"])
hr_day = hit_rate(daily["pulse_dir"], daily["day_dir"])

prob_up_am_up,    n_up_am   = cond_prob(daily, "pulse_dir",  1, "am_dir")
prob_dn_am_dn,    n_dn_am   = cond_prob(daily, "pulse_dir", -1, "am_dir")
prob_up_day_up,   n_up_day  = cond_prob(daily, "pulse_dir",  1, "day_dir")
prob_dn_day_dn,   n_dn_day  = cond_prob(daily, "pulse_dir", -1, "day_dir")

print(SEP)
print("  OPENING PULSE ANALYSIS -- VN30F1M 5m")
print(SEP)
print()
print("TONG QUAN:")
print(f"  So ngay phan tich : {n}")
print(f"  Pulse trung binh  : {daily['pulse_chg'].mean():+.2f} pts  |  std {daily['pulse_chg'].std():.2f} pts")
print(f"  AM move TB        : {daily['am_chg'].mean():+.2f} pts   |  std {daily['am_chg'].std():.2f} pts")
print(f"  Day move TB       : {daily['day_chg'].mean():+.2f} pts   |  std {daily['day_chg'].std():.2f} pts")
print(f"  Day range TB      : {daily['day_range'].mean():.2f} pts")
print()
print("TUONG QUAN PEARSON (magnitude):")
sig_am  = "***" if p_pulse_am  < 0.001 else "**" if p_pulse_am  < 0.01 else "*" if p_pulse_am  < 0.05 else ""
sig_day = "***" if p_pulse_day < 0.001 else "**" if p_pulse_day < 0.01 else "*" if p_pulse_day < 0.05 else ""
print(f"  Pulse_chg <-> AM_chg  : r = {r_pulse_am:+.3f}   p = {p_pulse_am:.4f}  {sig_am}")
print(f"  Pulse_chg <-> Day_chg : r = {r_pulse_day:+.3f}   p = {p_pulse_day:.4f}  {sig_day}")
print()
print("HIT RATE (huong pulse == huong ket thuc):")
print(f"  Pulse -> AM direction  : {hr_am:.1f}%   (random baseline = 50%)")
print(f"  Pulse -> Day direction : {hr_day:.1f}%   (random baseline = 50%)")
print(f"  Edge tren random       : AM +{hr_am-50:.1f}%   Day +{hr_day-50:.1f}%")
print()
print("XAC SUAT CO DIEU KIEN:")
print(f"  Pulse UP   -> AM  UP   : {prob_up_am_up:.1f}%   (n={n_up_am})")
print(f"  Pulse DOWN -> AM  DOWN : {prob_dn_am_dn:.1f}%   (n={n_dn_am})")
print(f"  Pulse UP   -> Day UP   : {prob_up_day_up:.1f}%   (n={n_up_day})")
print(f"  Pulse DOWN -> Day DOWN : {prob_dn_day_dn:.1f}%   (n={n_dn_day})")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Per day-of-week breakdown
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  BREAKDOWN THEO THU (Day of Week)")
print(SEP)
print()

dow_rows = []
for dow in range(5):
    sub = daily[daily["dow"] == dow].copy()
    if len(sub) < 10:
        continue
    name = DOW_NAME[dow]
    n_sub = len(sub)

    r_am,  p_am  = pearson(sub["pulse_chg"], sub["am_chg"])
    r_day, p_day = pearson(sub["pulse_chg"], sub["day_chg"])

    hr_am_d  = hit_rate(sub["pulse_dir"], sub["am_dir"])
    hr_day_d = hit_rate(sub["pulse_dir"], sub["day_dir"])

    p_up_up_am,  _  = cond_prob(sub, "pulse_dir",  1, "am_dir")
    p_dn_dn_am,  _  = cond_prob(sub, "pulse_dir", -1, "am_dir")
    p_up_up_day, _  = cond_prob(sub, "pulse_dir",  1, "day_dir")
    p_dn_dn_day, _  = cond_prob(sub, "pulse_dir", -1, "day_dir")

    dow_rows.append({
        "DOW":          name,
        "n":            n_sub,
        "PulseTB":      round(sub["pulse_chg"].mean(), 2),
        "AM_TB":        round(sub["am_chg"].mean(), 2),
        "Day_TB":       round(sub["day_chg"].mean(), 2),
        "r_P_AM":       round(r_am, 3),
        "p_P_AM":       round(p_am, 3),
        "r_P_Day":      round(r_day, 3),
        "p_P_Day":      round(p_day, 3),
        "HR_AM_pct":    round(hr_am_d, 1),
        "HR_Day_pct":   round(hr_day_d, 1),
        "P_up_AM_pct":  round(p_up_up_am, 1),
        "P_dn_AM_pct":  round(p_dn_dn_am, 1),
        "P_up_Day_pct": round(p_up_up_day, 1),
        "P_dn_Day_pct": round(p_dn_dn_day, 1),
    })

dow_df = pd.DataFrame(dow_rows)

# Pretty print table
hdr = (f"{'DOW':<5} | {'n':>4} | {'PulseTB':>8} | {'AM_TB':>6} | {'Day_TB':>7} | "
       f"{'r_P_AM':>7} | {'p_AM':>6} | {'r_P_Day':>8} | {'p_Day':>6} | "
       f"{'HR_AM%':>7} | {'HR_Day%':>8} | {'P(up->AU)%':>11} | {'P(dn->AD)%':>11} | "
       f"{'P(up->Du)%':>11} | {'P(dn->Dd)%':>11}")
print(hdr)
print("-" * len(hdr))
for _, row in dow_df.iterrows():
    sig_am  = "*" if row["p_P_AM"]  < 0.05 else " "
    sig_day = "*" if row["p_P_Day"] < 0.05 else " "
    print(f"  {row['DOW']:<3} | {row['n']:>4} | {row['PulseTB']:>+8.2f} | {row['AM_TB']:>+6.2f} | {row['Day_TB']:>+7.2f} | "
          f"{row['r_P_AM']:>+7.3f} | {row['p_P_AM']:>5.3f}{sig_am} | {row['r_P_Day']:>+8.3f} | {row['p_P_Day']:>5.3f}{sig_day} | "
          f"{row['HR_AM_pct']:>7.1f} | {row['HR_Day_pct']:>8.1f} | {row['P_up_AM_pct']:>11.1f} | {row['P_dn_AM_pct']:>11.1f} | "
          f"{row['P_up_Day_pct']:>11.1f} | {row['P_dn_Day_pct']:>11.1f}")
print()
print("Legend: r_P_AM = Pearson(pulse_chg, am_chg) | HR_AM% = hit rate pulse->AM dir | * = p<0.05")
print("        P(up->Du)% = P(day UP | pulse UP)    | P(dn->Dd)% = P(day DOWN | pulse DOWN)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Magnitude analysis
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  PHAN TICH THEO MAGNITUDE CUA OPENING PULSE")
print(SEP)

p33 = daily["pulse_chg"].abs().quantile(0.33)
p67 = daily["pulse_chg"].abs().quantile(0.67)
print(f"  Percentile breakpoints: 33rd = {p33:.2f} pts, 67th = {p67:.2f} pts")
print()

for label, lo, hi in [
    (f"Small  (|pulse| < {p33:.2f})",  0,    p33),
    (f"Medium ({p33:.2f} - {p67:.2f})", p33,  p67),
    (f"Large  (|pulse| > {p67:.2f})",  p67, 99999),
]:
    mask = (daily["pulse_chg"].abs() >= lo) & (daily["pulse_chg"].abs() < hi)
    sub  = daily[mask]
    if len(sub) < 10:
        continue
    hr_d = hit_rate(sub["pulse_dir"], sub["day_dir"])
    hr_a = hit_rate(sub["pulse_dir"], sub["am_dir"])
    r_d, p_d = pearson(sub["pulse_chg"], sub["day_chg"])
    avg_mag = sub["pulse_chg"].abs().mean()
    print(f"  {label}   n={len(sub):3d}  avg|pulse|={avg_mag:.2f}")
    print(f"    HR->AM={hr_a:.1f}%  HR->Day={hr_d:.1f}%  r(P,Day)={r_d:+.3f}  p={p_d:.3f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 8. Pulse UP vs DOWN summary
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  TONG KET: PULSE UP vs PULSE DOWN")
print(SEP)

for pdir, label in [(1, "Pulse UP"), (-1, "Pulse DOWN")]:
    sub = daily[daily["pulse_dir"] == pdir]
    if len(sub) == 0:
        continue
    n_sub      = len(sub)
    pct_day_same = (sub["day_dir"] == pdir).mean() * 100
    pct_am_same  = (sub["am_dir"]  == pdir).mean() * 100
    pct_reversal = (sub["day_dir"] == -pdir).mean() * 100
    avg_day_chg  = sub["day_chg"].mean()
    avg_am_chg   = sub["am_chg"].mean()
    avg_pulse_mag = sub["pulse_chg"].abs().mean()

    print(f"  {label}  (n={n_sub}, avg|pulse|={avg_pulse_mag:.2f} pts)")
    print(f"    AM ket thuc cung chieu pulse  : {pct_am_same:.1f}%  (avg AM chg {avg_am_chg:+.2f} pts)")
    print(f"    Day ket thuc cung chieu pulse : {pct_day_same:.1f}%  (avg day chg {avg_day_chg:+.2f} pts)")
    print(f"    Day ket thuc NGUOC chieu pulse: {pct_reversal:.1f}%")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 9. Sub-segment: early (9:00-9:15) vs late (9:15-9:30)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  PULSE SUB-SEGMENT: 9:00-9:15 vs 9:15-9:30")
print(SEP)

seg_records = []
for date, grp in df.groupby("date"):
    grp = grp.sort_values("hhmm")
    open_bar = grp[grp["hhmm"] == 540]
    if open_bar.empty:
        continue
    day_open = float(open_bar["open"].iloc[0])

    # Early sub-pulse: 9:00 open -> 9:15 close (bars 540,545,550 -> close of hhmm=550)
    early_bar = grp[grp["hhmm"] < 555]
    if early_bar.empty:
        continue
    early_close = float(early_bar["close"].iloc[-1])

    # Late sub-pulse: 9:15 open -> 9:30 close (bars 555,560,565)
    late_bar = grp[(grp["hhmm"] >= 555) & (grp["hhmm"] < 570)]
    if late_bar.empty:
        continue
    late_open  = float(late_bar["open"].iloc[0])
    late_close = float(late_bar["close"].iloc[-1])

    # Full day close
    full_bars = grp[(grp["hhmm"] >= 780) & (grp["hhmm"] < 870)]
    if full_bars.empty:
        am_bars = grp[(grp["hhmm"] >= 540) & (grp["hhmm"] < 690)]
        if am_bars.empty:
            continue
        day_close = float(am_bars["close"].iloc[-1])
    else:
        day_close = float(full_bars["close"].iloc[-1])

    seg_records.append({
        "date":        pd.Timestamp(date),
        "day_open":    day_open,
        "early_close": early_close,
        "late_close":  late_close,
        "day_close":   day_close,
        "early_chg":   early_close - day_open,
        "late_chg":    late_close  - late_open,
        "full_pulse":  late_close  - day_open,
        "day_chg":     day_close   - day_open,
    })

seg = pd.DataFrame(seg_records)
seg["early_dir"]      = np.sign(seg["early_chg"])
seg["late_dir"]       = np.sign(seg["late_chg"])
seg["full_pulse_dir"] = np.sign(seg["full_pulse"])
seg["day_dir"]        = np.sign(seg["day_chg"])

r_early, p_early = pearson(seg["early_chg"], seg["day_chg"])
r_late,  p_late  = pearson(seg["late_chg"],  seg["day_chg"])
r_full,  p_full  = pearson(seg["full_pulse"], seg["day_chg"])

hr_early = hit_rate(seg["early_dir"], seg["day_dir"])
hr_late  = hit_rate(seg["late_dir"],  seg["day_dir"])
hr_full  = hit_rate(seg["full_pulse_dir"], seg["day_dir"])

print()
print(f"  {'Segment':<20} | {'r(chg,day)':>11} | {'p-value':>8} | {'HR->Day%':>9}")
print(f"  {'-'*20} | {'-'*11} | {'-'*8} | {'-'*9}")
print(f"  {'Early (9:00-9:15)':<20} | {r_early:>+11.3f} | {p_early:>8.4f} | {hr_early:>9.1f}%")
print(f"  {'Late  (9:15-9:30)':<20} | {r_late:>+11.3f} | {p_late:>8.4f} | {hr_late:>9.1f}%")
print(f"  {'Full  (9:00-9:30)':<20} | {r_full:>+11.3f} | {p_full:>8.4f} | {hr_full:>9.1f}%")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 10. Fade vs Follow analysis
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  FADE vs FOLLOW STRATEGY COMPARISON")
print(SEP)
print("  Follow: trade same direction as 9:00-9:30 pulse after 9:30")
print("  Fade:   trade OPPOSITE direction after 9:30")
print()

for pdir, label in [(1, "Pulse UP"), (-1, "Pulse DOWN")]:
    sub = daily[daily["pulse_dir"] == pdir]
    if len(sub) == 0:
        continue
    pct_follow = (sub["day_dir"] == pdir).mean() * 100
    pct_fade   = (sub["day_dir"] == -pdir).mean() * 100
    pct_flat   = 100 - pct_follow - pct_fade
    best = "Follow" if pct_follow > pct_fade else "Fade"
    print(f"  {label} (n={len(sub)}):")
    print(f"    Follow wins (day same dir)  : {pct_follow:.1f}%")
    print(f"    Fade wins   (day opp dir)   : {pct_fade:.1f}%")
    print(f"    Flat/uncertain              : {pct_flat:.1f}%")
    print(f"    -> {best} is better")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 11. Notable patterns summary
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  NHAN XET TONG HOP")
print(SEP)

best_idx  = dow_df["HR_Day_pct"].idxmax()
worst_idx = dow_df["HR_Day_pct"].idxmin()
best_dow_row  = dow_df.loc[best_idx]
worst_dow_row = dow_df.loc[worst_idx]

print(f"  1. Pearson r(pulse->day)  = {r_pulse_day:+.3f}  (p={p_pulse_day:.4f})")
if p_pulse_day < 0.05:
    print(f"     -> Tuong quan co y nghia thong ke (p<0.05)")
else:
    print(f"     -> Khong co y nghia thong ke (p>={0.05})")

print(f"  2. Pearson r(pulse->AM)   = {r_pulse_am:+.3f}  (p={p_pulse_am:.4f})")
if p_pulse_am < 0.05:
    print(f"     -> Tuong quan co y nghia thong ke (p<0.05)")
else:
    print(f"     -> Khong co y nghia thong ke")

print(f"  3. Overall hit rate:")
print(f"     Pulse -> AM direction  : {hr_am:.1f}%  (edge = +{hr_am-50:.1f}% vs random)")
print(f"     Pulse -> Day direction : {hr_day:.1f}%  (edge = +{hr_day-50:.1f}% vs random)")

print(f"  4. Best DOW  (pulse -> day predictive): {best_dow_row['DOW']}  HR={best_dow_row['HR_Day_pct']}%")
print(f"     Worst DOW (pulse -> day predictive): {worst_dow_row['DOW']}  HR={worst_dow_row['HR_Day_pct']}%")

print(f"  5. Conditional probabilities:")
print(f"     Pulse UP   -> Day UP   : {prob_up_day_up:.1f}%  (n={n_up_day})")
print(f"     Pulse DOWN -> Day DOWN : {prob_dn_day_dn:.1f}%  (n={n_dn_day})")

print(f"  6. Large pulses are more predictive than small ones:")
large_mask = daily["pulse_chg"].abs() >= p67
hr_large = hit_rate(daily.loc[large_mask, "pulse_dir"], daily.loc[large_mask, "day_dir"])
small_mask = daily["pulse_chg"].abs() < p33
hr_small = hit_rate(daily.loc[small_mask, "pulse_dir"], daily.loc[small_mask, "day_dir"])
print(f"     Large pulse HR->Day : {hr_large:.1f}%")
print(f"     Small pulse HR->Day : {hr_small:.1f}%")

print()
if hr_day > 55:
    print("  VERDICT: Opening pulse has PREDICTIVE VALUE for day direction.")
    print("           Follow strategy (+{:.1f}% edge) is viable if cost < 2x 0.96 = 1.92 pts".format(hr_day-50))
elif hr_day < 45:
    print("  VERDICT: Opening pulse tends to REVERSE. Fade strategy may have edge.")
else:
    print("  VERDICT: Opening pulse direction is near-random for full day prediction.")

print()
print("Done. Cost per trade: 0.96 pts.")
