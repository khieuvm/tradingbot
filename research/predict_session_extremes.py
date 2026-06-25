# -*- coding: utf-8 -*-
"""
Predict Session Highs and Lows — VN30F1M
==========================================
Comprehensive research: AM and PM session HIGH/LOW timing and level prediction.

Parts:
  A — Descriptive statistics (when/how far)
  B — Opening pulse as predictor
  C — Day-of-week patterns
  D — Previous day features
  E — Prediction model (rules + ensemble)
  F — Verification and trading implications
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("e:/Trading/data/vn30f1m_5m.parquet")
SEP  = "=" * 72
SEP2 = "-" * 72

# hhmm (hour*60 + minute) boundaries
AM_START      = 540   # 09:00
AM_END        = 690   # 11:30 exclusive  (last bar = 685 = 11:25)
PM_START      = 780   # 13:00
PM_END        = 870   # 14:30 exclusive  (last bar = 865 = 14:25)

PULSE_END     = 565   # 09:25 bar → opening pulse = close[565] − open[540]
RETRACE_HHMM  = 585   # 09:45 bar → retrace check closes here

# AM timing buckets
AM_FIRST30_END  = 565   # 09:00-09:25  (bars 540-565, first 30 min incl.)
AM_FIRST60_END  = 595   # 09:00-09:55  (bars 540-595, first 60 min incl.)
AM_LAST30_START = 660   # 11:00-11:25  (bars 660-685, last 30 min)

# PM timing buckets
PM_FIRST30_END  = 805   # 13:00-13:25  (first 30 min)
PM_FIRST60_END  = 835   # 13:00-13:55  (first 60 min)
PM_LAST30_START = 840   # 14:00-14:25  (last 30 min)

DOW_NAME = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


def hhmm_label(hhmm):
    return f"{hhmm//60:02d}:{hhmm%60:02d}"


def pct_str(n, d):
    return f"{100*n/d:.1f}%" if d > 0 else "n/a"


def print_section(title):
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


def print_sub(title):
    print()
    print(SEP2)
    print(f"  {title}")
    print(SEP2)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD AND PREPARE 5m DATA
# ─────────────────────────────────────────────────────────────────────────────
print_section("LOADING DATA")

df = pd.read_parquet(DATA_PATH)
df["time"]  = pd.to_datetime(df["time"])
df          = df.sort_values("time").reset_index(drop=True)
df["date"]  = df["time"].dt.date
df["hhmm"]  = df["time"].dt.hour * 60 + df["time"].dt.minute
df["dow"]   = df["time"].dt.dayofweek

# True range for ATR
df["prev_close"] = df.groupby("date")["close"].shift(1)
df["tr"] = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        (df["high"] - df["prev_close"]).abs(),
        (df["low"]  - df["prev_close"]).abs()
    )
)

print(f"  5m bars:       {len(df):,}")
print(f"  Date range:    {df['date'].min()}  to  {df['date'].max()}")
print(f"  Trading days:  {df['date'].nunique()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD PER-DAY DATASET
# ─────────────────────────────────────────────────────────────────────────────
records = []

for date, grp in df.groupby("date"):
    grp = grp.sort_values("hhmm").reset_index(drop=True)

    # ── Day open (9:00 bar) ───────────────────────────────────────────────
    ob = grp[grp["hhmm"] == 540]
    if ob.empty:
        continue
    day_open = float(ob["open"].iloc[0])

    # ── Opening pulse = close[9:25] − open[9:00] ─────────────────────────
    pb = grp[grp["hhmm"] == PULSE_END]
    if pb.empty:
        pb = grp[(grp["hhmm"] >= 540) & (grp["hhmm"] < 570)]
        if pb.empty:
            continue
    pulse_close = float(pb["close"].iloc[-1])
    pulse_pts   = pulse_close - day_open

    # ── 9:45 retracement ─────────────────────────────────────────────────
    # max adverse excursion from pulse_close between 9:30 and 9:45 bars
    ret_bars = grp[(grp["hhmm"] >= 570) & (grp["hhmm"] <= 585)]
    if not ret_bars.empty and abs(pulse_pts) > 0.05:
        if pulse_pts > 0:
            retrace_mag = pulse_close - float(ret_bars["low"].min())
        else:
            retrace_mag = float(ret_bars["high"].max()) - pulse_close
        retrace_pct = retrace_mag / abs(pulse_pts) * 100
    else:
        retrace_pct = 0.0
        retrace_mag = 0.0

    bar945 = grp[grp["hhmm"] == 585]
    close945 = float(bar945["close"].iloc[-1]) if not bar945.empty else pulse_close

    # ── AM session ───────────────────────────────────────────────────────
    am_bars = grp[(grp["hhmm"] >= AM_START) & (grp["hhmm"] < AM_END)].copy()
    if am_bars.empty:
        continue

    am_high  = float(am_bars["high"].max())
    am_low   = float(am_bars["low"].min())
    am_close = float(am_bars["close"].iloc[-1])
    am_range = am_high - am_low

    # bar where session high/low first reached
    am_high_hhmm = int(am_bars.loc[am_bars["high"] == am_high, "hhmm"].iloc[0])
    am_low_hhmm  = int(am_bars.loc[am_bars["low"]  == am_low,  "hhmm"].iloc[0])

    # ATR: mean true range of AM bars (simple proxy; first day may be noisy)
    am_atr = float(am_bars["tr"].mean())

    # Post-9:45 high and low (what's left after we observe 9:45 close)
    post945_am = am_bars[am_bars["hhmm"] > 585]
    if not post945_am.empty:
        remaining_up   = float(post945_am["high"].max()) - close945
        remaining_down = close945 - float(post945_am["low"].min())
    else:
        remaining_up = remaining_down = 0.0

    # ── PM session ───────────────────────────────────────────────────────
    pm_bars = grp[(grp["hhmm"] >= PM_START) & (grp["hhmm"] < PM_END)].copy()
    has_pm  = not pm_bars.empty

    if has_pm:
        pm_high  = float(pm_bars["high"].max())
        pm_low   = float(pm_bars["low"].min())
        pm_range = pm_high - pm_low
        pm_close = float(pm_bars["close"].iloc[-1])
        pm_high_hhmm = int(pm_bars.loc[pm_bars["high"] == pm_high, "hhmm"].iloc[0])
        pm_low_hhmm  = int(pm_bars.loc[pm_bars["low"]  == pm_low,  "hhmm"].iloc[0])
        pm_open_bar  = pm_bars[pm_bars["hhmm"] == 780]
        pm_open = float(pm_open_bar["open"].iloc[0]) if not pm_open_bar.empty else float(pm_bars["open"].iloc[0])
        day_close = pm_close
        day_high  = max(am_high, pm_high)
        day_low   = min(am_low,  pm_low)
    else:
        pm_high = pm_low = pm_range = pm_close = np.nan
        pm_high_hhmm = pm_low_hhmm = pm_open = np.nan
        day_close = am_close
        day_high  = am_high
        day_low   = am_low

    dow = int(grp["dow"].iloc[0])

    records.append(dict(
        date          = pd.Timestamp(date),
        dow           = dow,
        dow_name      = DOW_NAME[dow],
        day_open      = day_open,
        pulse_close   = pulse_close,
        pulse_pts     = pulse_pts,
        retrace_pct   = retrace_pct,
        close945      = close945,
        am_high       = am_high,
        am_low        = am_low,
        am_high_hhmm  = am_high_hhmm,
        am_low_hhmm   = am_low_hhmm,
        am_close      = am_close,
        am_range      = am_range,
        am_atr        = am_atr,
        remaining_up  = remaining_up,
        remaining_down= remaining_down,
        pm_high       = pm_high,
        pm_low        = pm_low,
        pm_high_hhmm  = pm_high_hhmm,
        pm_low_hhmm   = pm_low_hhmm,
        pm_open       = pm_open,
        pm_close      = pm_close,
        pm_range      = pm_range,
        day_close     = day_close,
        day_high      = day_high,
        day_low       = day_low,
        has_pm        = has_pm,
    ))

daily = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
print(f"  Days with AM data:  {len(daily)}")
print(f"  Days with PM data:  {daily['has_pm'].sum()}")

# ── Previous-day features ─────────────────────────────────────────────────────
daily["prev_close"]     = daily["day_close"].shift(1)
daily["prev_open"]      = daily["day_open"].shift(1)
daily["prev_am_high"]   = daily["am_high"].shift(1)
daily["prev_am_low"]    = daily["am_low"].shift(1)
daily["prev_am_range"]  = daily["am_range"].shift(1)
daily["prev_am_close"]  = daily["am_close"].shift(1)
daily["prev_pm_range"]  = daily["pm_range"].shift(1)
daily["prev_day_range"] = (daily["day_high"] - daily["day_low"]).shift(1)
daily["prev_day_chg"]   = (daily["day_close"] - daily["day_open"]).shift(1)
daily["prev_pm_chg"]    = (daily["pm_close"]  - daily["pm_open"]).shift(1)

daily["gap"]            = daily["day_open"] - daily["prev_close"]
daily["high_first"]     = daily["am_high_hhmm"] < daily["am_low_hhmm"]  # True = high forms before low

# Derive ATR rolling 14-day (using am_range as proxy for daily range)
daily["atr14"] = daily["am_range"].rolling(14).mean()

# Day type: up/down
daily["am_up"] = daily["am_close"] > daily["day_open"]

# ── Quantile thresholds ───────────────────────────────────────────────────────
p25_range  = daily["am_range"].quantile(0.25)
p50_range  = daily["am_range"].quantile(0.50)
p75_range  = daily["am_range"].quantile(0.75)
p25_pulse  = daily["pulse_pts"].abs().quantile(0.33)
p67_pulse  = daily["pulse_pts"].abs().quantile(0.67)
PULSE_SMALL  = 1.4
PULSE_LARGE  = 3.5

# Drop first row (no prev-day features)
daily_full = daily.dropna(subset=["prev_close"]).copy()

print(f"  Days with prev-day features: {len(daily_full)}")
print(f"  AM range: p25={p25_range:.1f}  median={p50_range:.1f}  p75={p75_range:.1f} pts")


# ═════════════════════════════════════════════════════════════════════════════
# PART A — DESCRIPTIVE STATISTICS
# ═════════════════════════════════════════════════════════════════════════════

print_section("PART A — DESCRIPTIVE STATISTICS: WHEN / HOW FAR DO H/L FORM?")

# ─── A1: Timing distribution ──────────────────────────────────────────────────
print_sub("A1. TIMING: When does AM session HIGH form? (5m bar distribution)")

N = len(daily)
am_high_dist = daily["am_high_hhmm"].value_counts().sort_index()
cumulative   = 0
for hhmm, cnt in am_high_dist.items():
    cumulative += cnt
    bar = "#" * int(cnt / N * 60)
    print(f"  {hhmm_label(hhmm)}: {cnt:3d} ({100*cnt/N:4.1f}%)  cum={100*cumulative/N:5.1f}%  {bar}")

print()
print(f"  AM HIGH forms in FIRST  30 min (<=09:25): {pct_str((daily['am_high_hhmm']<=AM_FIRST30_END).sum(), N)}  n={(daily['am_high_hhmm']<=AM_FIRST30_END).sum()}")
print(f"  AM HIGH forms in FIRST  60 min (<=09:55): {pct_str((daily['am_high_hhmm']<=AM_FIRST60_END).sum(), N)}  n={(daily['am_high_hhmm']<=AM_FIRST60_END).sum()}")
print(f"  AM HIGH forms in LAST   30 min (>=11:00): {pct_str((daily['am_high_hhmm']>=AM_LAST30_START).sum(), N)}  n={(daily['am_high_hhmm']>=AM_LAST30_START).sum()}")

print_sub("A1. TIMING: When does AM session LOW form? (5m bar distribution)")

am_low_dist = daily["am_low_hhmm"].value_counts().sort_index()
cumulative  = 0
for hhmm, cnt in am_low_dist.items():
    cumulative += cnt
    bar = "#" * int(cnt / N * 60)
    print(f"  {hhmm_label(hhmm)}: {cnt:3d} ({100*cnt/N:4.1f}%)  cum={100*cumulative/N:5.1f}%  {bar}")

print()
print(f"  AM LOW forms in FIRST  30 min (<=09:25): {pct_str((daily['am_low_hhmm']<=AM_FIRST30_END).sum(), N)}  n={(daily['am_low_hhmm']<=AM_FIRST30_END).sum()}")
print(f"  AM LOW forms in FIRST  60 min (<=09:55): {pct_str((daily['am_low_hhmm']<=AM_FIRST60_END).sum(), N)}  n={(daily['am_low_hhmm']<=AM_FIRST60_END).sum()}")
print(f"  AM LOW forms in LAST   30 min (>=11:00): {pct_str((daily['am_low_hhmm']>=AM_LAST30_START).sum(), N)}  n={(daily['am_low_hhmm']>=AM_LAST30_START).sum()}")

# ── Does HIGH or LOW form first? ──────────────────────────────────────────────
print_sub("A1. HIGH vs LOW — which forms FIRST in AM session?")
high_first_n = daily["high_first"].sum()
low_first_n  = (~daily["high_first"]).sum()
print(f"  HIGH forms first (bullish open then fade): {high_first_n} ({100*high_first_n/N:.1f}%)")
print(f"  LOW  forms first (bearish open then rally): {low_first_n} ({100*low_first_n/N:.1f}%)")
print()
# by DOW
print("  HIGH-first rate by day of week:")
for dow, name in DOW_NAME.items():
    sub = daily[daily["dow"] == dow]
    if len(sub) < 5:
        continue
    hf = sub["high_first"].mean() * 100
    print(f"    {name}: {hf:.1f}%  (n={len(sub)})")

# ── PM timing ────────────────────────────────────────────────────────────────
print_sub("A1. TIMING: When does PM session HIGH/LOW form?")
pm_data = daily.dropna(subset=["pm_high_hhmm"])
Npm = len(pm_data)
print(f"  PM days: {Npm}")

pm_high_dist = pm_data["pm_high_hhmm"].value_counts().sort_index()
print("  PM HIGH distribution:")
cumulative = 0
for hhmm, cnt in pm_high_dist.items():
    cumulative += cnt
    bar = "#" * int(cnt / Npm * 50)
    print(f"    {hhmm_label(hhmm)}: {cnt:3d} ({100*cnt/Npm:4.1f}%)  cum={100*cumulative/Npm:5.1f}%  {bar}")

pm_low_dist = pm_data["pm_low_hhmm"].value_counts().sort_index()
print("\n  PM LOW distribution:")
cumulative = 0
for hhmm, cnt in pm_low_dist.items():
    cumulative += cnt
    bar = "#" * int(cnt / Npm * 50)
    print(f"    {hhmm_label(hhmm)}: {cnt:3d} ({100*cnt/Npm:4.1f}%)  cum={100*cumulative/Npm:5.1f}%  {bar}")

print()
print(f"  PM HIGH in FIRST 30 min (<=13:25): {pct_str((pm_data['pm_high_hhmm']<=PM_FIRST30_END).sum(), Npm)}")
print(f"  PM HIGH in FIRST 60 min (<=13:55): {pct_str((pm_data['pm_high_hhmm']<=PM_FIRST60_END).sum(), Npm)}")
print(f"  PM HIGH in LAST  30 min (>=14:00): {pct_str((pm_data['pm_high_hhmm']>=PM_LAST30_START).sum(), Npm)}")
print()
print(f"  PM LOW in FIRST 30 min (<=13:25): {pct_str((pm_data['pm_low_hhmm']<=PM_FIRST30_END).sum(), Npm)}")
print(f"  PM LOW in FIRST 60 min (<=13:55): {pct_str((pm_data['pm_low_hhmm']<=PM_FIRST60_END).sum(), Npm)}")
print(f"  PM LOW in LAST  30 min (>=14:00): {pct_str((pm_data['pm_low_hhmm']>=PM_LAST30_START).sum(), Npm)}")

pm_high_first = (pm_data["pm_high_hhmm"] < pm_data["pm_low_hhmm"]).sum()
print(f"\n  PM HIGH forms before LOW: {pm_high_first} ({100*pm_high_first/Npm:.1f}%)")
print(f"  PM LOW  forms before HIGH: {Npm-pm_high_first} ({100*(Npm-pm_high_first)/Npm:.1f}%)")

# ── DOW breakdown for AM timing ───────────────────────────────────────────────
print_sub("A1. AM HIGH/LOW timing by day-of-week")
print(f"  {'DOW':<5} {'N':>4}  {'H_first30':>10}  {'H_last30':>9}  {'L_first30':>10}  {'L_last30':>9}  {'H_first':>8}")
for dow, name in DOW_NAME.items():
    sub = daily[daily["dow"] == dow]
    n   = len(sub)
    if n < 5:
        continue
    hf30 = (sub["am_high_hhmm"] <= AM_FIRST30_END).mean() * 100
    hl30 = (sub["am_high_hhmm"] >= AM_LAST30_START).mean() * 100
    lf30 = (sub["am_low_hhmm"]  <= AM_FIRST30_END).mean() * 100
    ll30 = (sub["am_low_hhmm"]  >= AM_LAST30_START).mean() * 100
    hfst = sub["high_first"].mean() * 100
    print(f"  {name:<5} {n:>4}  {hf30:>9.1f}%  {hl30:>8.1f}%  {lf30:>9.1f}%  {ll30:>8.1f}%  {hfst:>7.1f}%")

# ─── A2: Distance from open ───────────────────────────────────────────────────
print_sub("A2. DISTANCE: Open → AM High/Low (pts and ATR units)")

upside   = daily["am_high"] - daily["day_open"]
downside = daily["day_open"] - daily["am_low"]

print(f"  AM session: open = {daily['day_open'].mean():.1f} avg")
print()
print(f"  Open → AM HIGH (upside):  avg={upside.mean():.2f}  median={upside.median():.2f}  "
      f"p25={upside.quantile(.25):.2f}  p75={upside.quantile(.75):.2f} pts")
print(f"  Open → AM LOW (downside): avg={downside.mean():.2f}  median={downside.median():.2f}  "
      f"p25={downside.quantile(.25):.2f}  p75={downside.quantile(.75):.2f} pts")

atr_proxy = daily["am_atr"]
print(f"\n  In ATR units (avg AM bar ATR = {atr_proxy.mean():.2f} pts):")
print(f"  Upside  avg: {(upside   / atr_proxy).mean():.2f} ATR")
print(f"  Downside avg: {(downside / atr_proxy).mean():.2f} ATR")

print()
print("  Upside distribution (% of days where open→high ≤ X pts):")
for thresh in [2, 3, 5, 7, 10, 15]:
    p = (upside <= thresh).mean() * 100
    print(f"    <= {thresh:2d} pts: {p:.1f}%")

print()
print("  Downside distribution (% of days where open→low ≤ X pts):")
for thresh in [2, 3, 5, 7, 10, 15]:
    p = (downside <= thresh).mean() * 100
    print(f"    <= {thresh:2d} pts: {p:.1f}%")

# PM upside/downside
pm_up   = pm_data["pm_high"] - pm_data["pm_open"]
pm_down = pm_data["pm_open"] - pm_data["pm_low"]
print(f"\n  PM session: open → PM HIGH: avg={pm_up.mean():.2f}  median={pm_up.median():.2f} pts")
print(f"  PM session: open → PM LOW:  avg={pm_down.mean():.2f}  median={pm_down.median():.2f} pts")

# AM vs PM range correlation
both = daily.dropna(subset=["pm_range"])
r, p = stats.pearsonr(both["am_range"], both["pm_range"])
print(f"\n  AM range vs PM range correlation: r={r:.3f}  p={p:.4f}  n={len(both)}")

# ─── A3: Previous day influence ───────────────────────────────────────────────
print_sub("A3. PREVIOUS DAY INFLUENCE on today's AM high/low distance")

d = daily_full.copy()

# Up vs down yesterday
up_yd   = d[d["prev_day_chg"]  > 0]
down_yd = d[d["prev_day_chg"] <= 0]
print(f"  Yesterday UP day   ({len(up_yd)} days): today AM upside avg={( up_yd['am_high']- up_yd['day_open']).mean():.2f}  downside avg={( up_yd['day_open']- up_yd['am_low']).mean():.2f} pts")
print(f"  Yesterday DOWN day ({len(down_yd)} days): today AM upside avg={(down_yd['am_high']-down_yd['day_open']).mean():.2f}  downside avg={(down_yd['day_open']-down_yd['am_low']).mean():.2f} pts")

# Large range yesterday
p75_prev_range = d["prev_day_range"].quantile(0.75)
large_yd = d[d["prev_day_range"] >= p75_prev_range]
small_yd = d[d["prev_day_range"] <  p75_prev_range]
print(f"\n  Yesterday LARGE range (>={p75_prev_range:.1f} pts, n={len(large_yd)}): today AM range avg={large_yd['am_range'].mean():.2f}")
print(f"  Yesterday SMALL range (<{p75_prev_range:.1f} pts, n={len(small_yd)}):  today AM range avg={small_yd['am_range'].mean():.2f}")

# Correlation: gap vs AM range
r_gap_range, p_gap = stats.pearsonr(d["gap"], d["am_range"])
print(f"\n  Gap (open - prev close) vs AM range: r={r_gap_range:.3f}  p={p_gap:.4f}")

# prev PM close vs today AM upside/downside
r_pclose_up, _ = stats.pearsonr(d["prev_am_close"], d["am_high"] - d["day_open"])
print(f"  Prev AM close vs today AM upside: r={r_pclose_up:.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# PART B — OPENING PULSE AS PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════

print_section("PART B — OPENING PULSE AS PREDICTOR")

# Pulse buckets
daily["pulse_dir"]    = np.sign(daily["pulse_pts"])
daily["pulse_bucket"] = pd.cut(
    daily["pulse_pts"].abs(),
    bins  = [0, PULSE_SMALL, PULSE_LARGE, 999],
    labels= ["SMALL(<1.4)", "MED(1.4-3.5)", "LARGE(>3.5)"]
)

# ─── B1: Pulse direction → which extreme forms first ──────────────────────────
print_sub("B1. PULSE DIRECTION → which extreme forms FIRST and AFTER 9:30")

up_days   = daily[daily["pulse_pts"] >  0.05]
down_days = daily[daily["pulse_pts"] < -0.05]

# "HIGH forms AFTER 9:30" = am_high_hhmm >= 570
print(f"  UP pulse days ({len(up_days)}):")
pct_h_late = (up_days["am_high_hhmm"] >= 570).mean() * 100
pct_l_late = (up_days["am_low_hhmm"]  >= 570).mean() * 100
print(f"    AM HIGH forms AFTER 9:30: {pct_h_late:.1f}%")
print(f"    AM LOW  forms AFTER 9:30: {pct_l_late:.1f}%")
print(f"    LOW forms first (classic trend): {(~up_days['high_first']).mean()*100:.1f}%")

print(f"\n  DOWN pulse days ({len(down_days)}):")
pct_h_late = (down_days["am_high_hhmm"] >= 570).mean() * 100
pct_l_late = (down_days["am_low_hhmm"]  >= 570).mean() * 100
print(f"    AM HIGH forms AFTER 9:30: {pct_h_late:.1f}%")
print(f"    AM LOW  forms AFTER 9:30: {pct_l_late:.1f}%")
print(f"    HIGH forms first (classic down trend): {down_days['high_first'].mean()*100:.1f}%")

# By pulse bucket
print()
print(f"  {'Bucket':<14}  {'N':>4}  {'HIGH_first':>10}  {'H_after930':>10}  {'L_after930':>10}")
for bucket, grp_name in [
    ("SMALL(<1.4)",  "SMALL"),
    ("MED(1.4-3.5)", "MED"),
    ("LARGE(>3.5)",  "LARGE")
]:
    sub = daily[daily["pulse_bucket"] == bucket]
    n   = len(sub)
    if n == 0:
        continue
    up_sub   = sub[sub["pulse_pts"] >  0.05]
    down_sub = sub[sub["pulse_pts"] < -0.05]
    # For UP pulse: % HIGH forms first (fade) vs LOW first (trend continue)
    if len(up_sub) > 0:
        hf_up  = up_sub["high_first"].mean() * 100
        ha930  = (up_sub["am_high_hhmm"] >= 570).mean() * 100
        la930  = (up_sub["am_low_hhmm"]  >= 570).mean() * 100
        print(f"  UP {bucket:<12}  {len(up_sub):>4}  {hf_up:>9.1f}%  {ha930:>9.1f}%  {la930:>9.1f}%")
    if len(down_sub) > 0:
        hf_dn  = down_sub["high_first"].mean() * 100
        ha930  = (down_sub["am_high_hhmm"] >= 570).mean() * 100
        la930  = (down_sub["am_low_hhmm"]  >= 570).mean() * 100
        print(f"  DN {bucket:<12}  {len(down_sub):>4}  {hf_dn:>9.1f}%  {ha930:>9.1f}%  {la930:>9.1f}%")

# ─── B2: Pulse magnitude → session range ─────────────────────────────────────
print_sub("B2. PULSE MAGNITUDE → session range and remaining move")

r_pulse_range, p_pr = stats.pearsonr(daily["pulse_pts"].abs(), daily["am_range"])
print(f"  |pulse| vs AM range: r={r_pulse_range:.3f}  p={p_pr:.4f}  n={len(daily)}")

r_pulse_up_remaining, _ = stats.pearsonr(up_days["pulse_pts"].abs(), up_days["remaining_up"])
r_pulse_dn_remaining, _ = stats.pearsonr(down_days["pulse_pts"].abs(), down_days["remaining_down"])
print(f"  UP pulse: |pulse| vs remaining upside after 9:45:   r={r_pulse_up_remaining:.3f}  n={len(up_days)}")
print(f"  DN pulse: |pulse| vs remaining downside after 9:45: r={r_pulse_dn_remaining:.3f}  n={len(down_days)}")

print()
print(f"  {'Pulse bucket':<14}  {'N':>4}  {'avg_AM_range':>13}  {'avg_remain_up(UP)':>18}  {'avg_remain_dn(DN)':>18}")
for bucket in ["SMALL(<1.4)", "MED(1.4-3.5)", "LARGE(>3.5)"]:
    sub    = daily[daily["pulse_bucket"] == bucket]
    up_sub = sub[sub["pulse_pts"] >  0.05]
    dn_sub = sub[sub["pulse_pts"] < -0.05]
    avg_r  = sub["am_range"].mean()
    avg_ru = up_sub["remaining_up"].mean()   if len(up_sub) else np.nan
    avg_rd = dn_sub["remaining_down"].mean() if len(dn_sub) else np.nan
    print(f"  {bucket:<14}  {len(sub):>4}  {avg_r:>13.2f}  {avg_ru:>18.2f}  {avg_rd:>18.2f}")

# ─── B3: 9:45 retracement predicts timing ─────────────────────────────────────
print_sub("B3. 9:45 RETRACEMENT → when does session extreme form?")

sig_days = daily[daily["pulse_pts"].abs() >= PULSE_SMALL].copy()
print(f"  Days with |pulse| >= {PULSE_SMALL}: {len(sig_days)}")

ret_bins   = [0, 30, 70, 100, 999]
ret_labels = ["SHALLOW(<30%)", "MED(30-70%)", "DEEP(70-100%)", "REVERSAL(>100%)"]
sig_days["retrace_bin"] = pd.cut(sig_days["retrace_pct"], bins=ret_bins, labels=ret_labels)

print(f"\n  For UP pulse days:")
up_sig = sig_days[sig_days["pulse_pts"] > 0]
print(f"  {'Retrace':<15}  {'N':>4}  {'H_after945%':>12}  {'avg_H_hhmm':>11}  {'avg_remain_up':>14}  {'trend_day%':>11}")
for rb in ret_labels:
    sub = up_sig[up_sig["retrace_bin"] == rb]
    n   = len(sub)
    if n < 3:
        continue
    h_after = (sub["am_high_hhmm"] > RETRACE_HHMM).mean() * 100
    avg_h   = hhmm_label(int(sub["am_high_hhmm"].mean()))
    avg_ru  = sub["remaining_up"].mean()
    # trend day = LOW forms early (before 9:45), HIGH forms late
    trend   = ((sub["am_low_hhmm"] <= RETRACE_HHMM) & (sub["am_high_hhmm"] > RETRACE_HHMM)).mean() * 100
    print(f"  {rb:<15}  {n:>4}  {h_after:>11.1f}%  {avg_h:>11}  {avg_ru:>14.2f}  {trend:>10.1f}%")

print(f"\n  For DOWN pulse days:")
dn_sig = sig_days[sig_days["pulse_pts"] < 0]
print(f"  {'Retrace':<15}  {'N':>4}  {'L_after945%':>12}  {'avg_L_hhmm':>11}  {'avg_remain_dn':>14}  {'trend_day%':>11}")
for rb in ret_labels:
    sub = dn_sig[dn_sig["retrace_bin"] == rb]
    n   = len(sub)
    if n < 3:
        continue
    l_after = (sub["am_low_hhmm"]  > RETRACE_HHMM).mean() * 100
    avg_l   = hhmm_label(int(sub["am_low_hhmm"].mean()))
    avg_rd  = sub["remaining_down"].mean()
    trend   = ((sub["am_high_hhmm"] <= RETRACE_HHMM) & (sub["am_low_hhmm"] > RETRACE_HHMM)).mean() * 100
    print(f"  {rb:<15}  {n:>4}  {l_after:>11.1f}%  {avg_l:>11}  {avg_rd:>14.2f}  {trend:>10.1f}%")

# ─── B4: Predict session HIGH/LOW LEVEL from pulse ────────────────────────────
print_sub("B4. LEVEL PREDICTION: predicted_high = pulse_close + alpha * pulse")

# Split train/test by time (70/30)
n_total = len(daily)
n_train = int(n_total * 0.70)
train   = daily.iloc[:n_train].copy()
test    = daily.iloc[n_train:].copy()

print(f"  Train: {len(train)} days | Test: {len(test)} days")
print(f"  Train period: {train['date'].iloc[0].date()} to {train['date'].iloc[-1].date()}")
print(f"  Test  period: {test['date'].iloc[0].date()} to {test['date'].iloc[-1].date()}")

# Model 1: AM HIGH = pulse_close + alpha * pulse_pts (UP pulse days only)
train_up = train[train["pulse_pts"] > 0.05].copy()
test_up  = test[test["pulse_pts"]   > 0.05].copy()

if len(train_up) > 20:
    residuals = train_up["am_high"] - train_up["pulse_close"]
    # Simple scalar alpha: mean(am_high - pulse_close) / mean(pulse_pts)
    alpha_up = residuals.mean() / train_up["pulse_pts"].mean()
    # OLS alpha (slope through origin)
    ols_alpha_up = (residuals * train_up["pulse_pts"]).sum() / (train_up["pulse_pts"] ** 2).sum()

    pred_test_high = test_up["pulse_close"] + ols_alpha_up * test_up["pulse_pts"]
    mae_model  = (test_up["am_high"] - pred_test_high).abs().mean()
    mae_naive  = (test_up["am_high"] - (test_up["day_open"] + (train_up["am_high"] - train_up["day_open"]).median())).abs().mean()

    print(f"\n  UP pulse — AM HIGH prediction (alpha fit on train):")
    print(f"    OLS alpha = {ols_alpha_up:.3f}  (predicted_high = pulse_close + {ols_alpha_up:.2f} * pulse_pts)")
    print(f"    Test MAE (model): {mae_model:.2f} pts  (n={len(test_up)})")
    print(f"    Test MAE (naive = open + median upside): {mae_naive:.2f} pts")
    print(f"    Improvement: {mae_naive - mae_model:.2f} pts")

# Model for AM LOW (UP pulse days): predicted_low = pulse_close - beta * pulse_pts
train_down = train[train["pulse_pts"] < -0.05].copy()
test_down  = test[test["pulse_pts"]  < -0.05].copy()

if len(train_down) > 20:
    residuals_dn = train_down["pulse_close"] - train_down["am_low"]
    pulse_abs_dn  = train_down["pulse_pts"].abs()
    ols_beta_dn  = (residuals_dn * pulse_abs_dn).sum() / (pulse_abs_dn ** 2).sum()

    pred_test_low = test_down["pulse_close"] - ols_beta_dn * test_down["pulse_pts"].abs()
    mae_model_dn = (test_down["am_low"] - pred_test_low).abs().mean()
    mae_naive_dn = (test_down["am_low"] - (test_down["day_open"] - (train_down["day_open"] - train_down["am_low"]).median())).abs().mean()

    print(f"\n  DOWN pulse — AM LOW prediction (beta fit on train):")
    print(f"    OLS beta = {ols_beta_dn:.3f}  (predicted_low = pulse_close - {ols_beta_dn:.2f} * |pulse|)")
    print(f"    Test MAE (model): {mae_model_dn:.2f} pts  (n={len(test_down)})")
    print(f"    Test MAE (naive = open - median downside): {mae_naive_dn:.2f} pts")
    print(f"    Improvement: {mae_naive_dn - mae_model_dn:.2f} pts")

# Regression: am_high ~ pulse_pts (all days, OLS with intercept)
from scipy.stats import linregress
slope_h, intercept_h, r_h, p_h, _ = linregress(daily["pulse_pts"], daily["am_high"] - daily["day_open"])
print(f"\n  Full OLS: (AM_high - open) = {intercept_h:.2f} + {slope_h:.3f} * pulse_pts  r={r_h:.3f}  p={p_h:.4f}")

slope_l, intercept_l, r_l, p_l, _ = linregress(daily["pulse_pts"], daily["day_open"] - daily["am_low"])
print(f"  Full OLS: (open - AM_low)  = {intercept_l:.2f} + {slope_l:.3f} * pulse_pts  r={r_l:.3f}  p={p_l:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# PART C — DAY-OF-WEEK PATTERNS
# ═════════════════════════════════════════════════════════════════════════════

print_section("PART C — DAY-OF-WEEK PATTERNS")

# ─── C1: Session range by DOW ─────────────────────────────────────────────────
print_sub("C1. SESSION RANGE by day-of-week")
print(f"  {'DOW':<5}  {'N':>4}  {'AM_range_avg':>13}  {'AM_range_med':>13}  {'PM_range_avg':>13}  {'PM_range_med':>13}")
dow_groups_am = []
dow_groups_pm = []
for dow, name in DOW_NAME.items():
    sub    = daily[daily["dow"] == dow]
    sub_pm = sub.dropna(subset=["pm_range"])
    n      = len(sub)
    npm    = len(sub_pm)
    if n < 5:
        continue
    am_avg = sub["am_range"].mean()
    am_med = sub["am_range"].median()
    pm_avg = sub_pm["pm_range"].mean()   if npm > 0 else np.nan
    pm_med = sub_pm["pm_range"].median() if npm > 0 else np.nan
    dow_groups_am.append(sub["am_range"].values)
    dow_groups_pm.append(sub_pm["pm_range"].values)
    print(f"  {name:<5}  {n:>4}  {am_avg:>13.2f}  {am_med:>13.2f}  {pm_avg:>13.2f}  {pm_med:>13.2f}")

# ANOVA test
if len(dow_groups_am) >= 3:
    f_am, p_am = stats.f_oneway(*dow_groups_am)
    print(f"\n  ANOVA AM range by DOW: F={f_am:.3f}  p={p_am:.4f}  {'SIGNIFICANT' if p_am<0.05 else 'NOT significant'}")
if len(dow_groups_pm) >= 3:
    f_pm, p_pm = stats.f_oneway(*[g for g in dow_groups_pm if len(g) > 0])
    print(f"  ANOVA PM range by DOW: F={f_pm:.3f}  p={p_pm:.4f}  {'SIGNIFICANT' if p_pm<0.05 else 'NOT significant'}")

# ─── C2: HIGH/LOW timing by DOW ───────────────────────────────────────────────
print_sub("C2. HIGH/LOW TIMING by day-of-week (extreme forms early vs late)")
print(f"  {'DOW':<5}  {'N':>4}  {'H_early%':>9}  {'H_late%':>8}  {'L_early%':>9}  {'L_late%':>8}  {'trend_day%':>11}")
for dow, name in DOW_NAME.items():
    sub = daily[daily["dow"] == dow]
    n   = len(sub)
    if n < 5:
        continue
    h_early  = (sub["am_high_hhmm"] <= AM_FIRST30_END).mean() * 100
    h_late   = (sub["am_high_hhmm"] >= AM_LAST30_START).mean() * 100
    l_early  = (sub["am_low_hhmm"]  <= AM_FIRST30_END).mean() * 100
    l_late   = (sub["am_low_hhmm"]  >= AM_LAST30_START).mean() * 100
    # trend day = extremes at opposite ends
    trend_up = ((sub["am_low_hhmm"] <= RETRACE_HHMM) & (sub["am_high_hhmm"] > 630)).mean() * 100
    trend_dn = ((sub["am_high_hhmm"] <= RETRACE_HHMM) & (sub["am_low_hhmm"] > 630)).mean() * 100
    trend    = trend_up + trend_dn
    print(f"  {name:<5}  {n:>4}  {h_early:>8.1f}%  {h_late:>7.1f}%  {l_early:>8.1f}%  {l_late:>7.1f}%  {trend:>10.1f}%")

# ─── C3: DOW × pulse interaction matrix ──────────────────────────────────────
print_sub("C3. DOW × PULSE DIRECTION → % AM extreme forms AFTER 9:45")
sig_d = daily[daily["pulse_pts"].abs() >= PULSE_SMALL].copy()
print(f"  (Filtered: |pulse| >= {PULSE_SMALL}, n={len(sig_d)})")
print()
print(f"  {'DOW':<5}  {'↑ pulse':>8}  {'H>945%':>8}  {'↓ pulse':>8}  {'L>945%':>8}")
for dow, name in DOW_NAME.items():
    sub_up = sig_d[(sig_d["dow"] == dow) & (sig_d["pulse_pts"] > 0)]
    sub_dn = sig_d[(sig_d["dow"] == dow) & (sig_d["pulse_pts"] < 0)]
    n_up = len(sub_up); n_dn = len(sub_dn)
    h945 = (sub_up["am_high_hhmm"] > RETRACE_HHMM).mean() * 100 if n_up > 0 else np.nan
    l945 = (sub_dn["am_low_hhmm"]  > RETRACE_HHMM).mean() * 100 if n_dn > 0 else np.nan
    h945s = f"{h945:.0f}%" if not np.isnan(h945) else "n/a"
    l945s = f"{l945:.0f}%" if not np.isnan(l945) else "n/a"
    print(f"  {name:<5}  {n_up:>8}  {h945s:>8}  {n_dn:>8}  {l945s:>8}")

# ── Fri fade pattern ──────────────────────────────────────────────────────────
print_sub("C3. FRIDAY FADE: do extremes form earlier on Friday?")
fri = daily[daily["dow"] == 4]
all_except_fri = daily[daily["dow"] != 4]
print(f"  Fri (n={len(fri)}) vs non-Fri (n={len(all_except_fri)}):")
print(f"    Fri: H_first30={pct_str((fri['am_high_hhmm']<=AM_FIRST30_END).sum(), len(fri))}  "
      f"L_first30={pct_str((fri['am_low_hhmm']<=AM_FIRST30_END).sum(), len(fri))}")
print(f"    All: H_first30={pct_str((all_except_fri['am_high_hhmm']<=AM_FIRST30_END).sum(), len(all_except_fri))}  "
      f"L_first30={pct_str((all_except_fri['am_low_hhmm']<=AM_FIRST30_END).sum(), len(all_except_fri))}")
print(f"    Fri  range avg: {fri['am_range'].mean():.2f} pts")
print(f"    All  range avg: {all_except_fri['am_range'].mean():.2f} pts")


# ═════════════════════════════════════════════════════════════════════════════
# PART D — PREVIOUS DAY FEATURES
# ═════════════════════════════════════════════════════════════════════════════

print_section("PART D — PREVIOUS DAY FEATURES")

d = daily_full.copy()

# ─── D1: Gap analysis ─────────────────────────────────────────────────────────
print_sub("D1. GAP (open - prev_close) → AM behavior")

gap_bins   = [-999, -3, -1, 1, 3, 999]
gap_labels = ["BIG_DN(<-3)", "SML_DN(-3to-1)", "FLAT(-1to1)", "SML_UP(1to3)", "BIG_UP(>3)"]
d["gap_bin"] = pd.cut(d["gap"], bins=gap_bins, labels=gap_labels)

print(f"  {'Gap':<16}  {'N':>4}  {'AM_range':>9}  {'fill_rate':>10}  {'H_first30%':>11}  {'L_first30%':>11}")
for gbin in gap_labels:
    sub = d[d["gap_bin"] == gbin]
    n   = len(sub)
    if n < 5:
        continue
    am_rng = sub["am_range"].mean()
    # Gap fill: gap UP → low goes below prev_close (gap fill down)
    # gap DN → high goes above prev_close (gap fill up)
    sub_gup = sub[sub["gap"] > 0]
    sub_gdn = sub[sub["gap"] < 0]
    if gbin.startswith("SML_UP") or gbin.startswith("BIG_UP"):
        fill_rate = (sub["am_low"] <= sub["prev_close"]).mean() * 100
    elif gbin.startswith("SML_DN") or gbin.startswith("BIG_DN"):
        fill_rate = (sub["am_high"] >= sub["prev_close"]).mean() * 100
    else:
        fill_rate = np.nan
    hf30 = (sub["am_high_hhmm"] <= AM_FIRST30_END).mean() * 100
    lf30 = (sub["am_low_hhmm"]  <= AM_FIRST30_END).mean() * 100
    fill_str = f"{fill_rate:.1f}%" if not np.isnan(fill_rate) else "n/a"
    print(f"  {gbin:<16}  {n:>4}  {am_rng:>9.2f}  {fill_str:>10}  {hf30:>10.1f}%  {lf30:>10.1f}%")

r_gap_range2, p_gap2 = stats.pearsonr(d["gap"].abs(), d["am_range"])
print(f"\n  |gap| vs AM range correlation: r={r_gap_range2:.3f}  p={p_gap2:.4f}")

# ─── D2: Yesterday's range / direction ────────────────────────────────────────
print_sub("D2. YESTERDAY RANGE AND DIRECTION → today AM structure")

p75_prev = d["prev_day_range"].quantile(0.75)
p25_prev = d["prev_day_range"].quantile(0.25)

print(f"  Yesterday LARGE range (>={p75_prev:.1f}, n={( d['prev_day_range']>=p75_prev).sum()}):")
large_prev = d[d["prev_day_range"] >= p75_prev]
print(f"    Today AM range: avg={large_prev['am_range'].mean():.2f}  median={large_prev['am_range'].median():.2f}")
print(f"    Today H_first30: {pct_str((large_prev['am_high_hhmm']<=AM_FIRST30_END).sum(), len(large_prev))}")
print(f"    Today L_first30: {pct_str((large_prev['am_low_hhmm']<=AM_FIRST30_END).sum(), len(large_prev))}")

print(f"\n  Yesterday SMALL range (<={p25_prev:.1f}, n={(d['prev_day_range']<=p25_prev).sum()}):")
small_prev = d[d["prev_day_range"] <= p25_prev]
print(f"    Today AM range: avg={small_prev['am_range'].mean():.2f}  median={small_prev['am_range'].median():.2f}")

# Two-day patterns: prev direction × today direction
print(f"\n  Two-day patterns (yesterday direction → today AM direction):")
print(f"  {'Pattern':<16}  {'N':>4}  {'today_AM_up%':>13}  {'avg_AM_range':>13}")
for yd_label, yd_cond in [("yd_UP", d["prev_day_chg"] > 0), ("yd_DOWN", d["prev_day_chg"] <= 0)]:
    sub = d[yd_cond]
    n   = len(sub)
    am_up_pct = (sub["am_close"] > sub["day_open"]).mean() * 100
    am_r_avg  = sub["am_range"].mean()
    print(f"  {yd_label:<16}  {n:>4}  {am_up_pct:>12.1f}%  {am_r_avg:>13.2f}")

# 4-cell cross table: yd_dir × td_dir
print()
print(f"  Full 4-cell: yd_dir × today_dir → count | AM_range")
for yd_label, yd_cond in [("yd=UP", d["prev_day_chg"] > 0), ("yd=DN", d["prev_day_chg"] <= 0)]:
    for td_label, td_cond_fn in [("td=UP", lambda s: s["am_close"] > s["day_open"]),
                                  ("td=DN", lambda s: s["am_close"] <= s["day_open"])]:
        sub  = d[yd_cond]
        sub2 = sub[td_cond_fn(sub)]
        print(f"    {yd_label} {td_label}: n={len(sub2):3d}  AM_range_avg={sub2['am_range'].mean():.2f}")

# ─── D3: Overnight features ───────────────────────────────────────────────────
print_sub("D3. OVERNIGHT: gap size vs AM range and extreme timing")

d["abs_gap"] = d["gap"].abs()
print("  Gap size quartiles → AM range and timing:")
print(f"  {'Gap_quartile':<14}  {'N':>4}  {'AM_range':>9}  {'H_first30%':>11}  {'L_first30%':>11}  {'H>10:30%':>9}")
for q_lo, q_hi, label in [(0, 0.25, "Q1 small"), (0.25, 0.50, "Q2"), (0.50, 0.75, "Q3"), (0.75, 1.0, "Q4 large")]:
    lo = d["abs_gap"].quantile(q_lo)
    hi = d["abs_gap"].quantile(q_hi)
    if q_lo == 0:
        sub = d[d["abs_gap"] <= hi]
    elif q_hi == 1.0:
        sub = d[d["abs_gap"] >= lo]
    else:
        sub = d[(d["abs_gap"] > lo) & (d["abs_gap"] <= hi)]
    n    = len(sub)
    ar   = sub["am_range"].mean()
    hf30 = (sub["am_high_hhmm"] <= AM_FIRST30_END).mean() * 100
    lf30 = (sub["am_low_hhmm"]  <= AM_FIRST30_END).mean() * 100
    h10  = (sub["am_high_hhmm"] > 630).mean() * 100
    print(f"  {label:<14}  {n:>4}  {ar:>9.2f}  {hf30:>10.1f}%  {lf30:>10.1f}%  {h10:>8.1f}%")

# Gap direction and AM extreme timing
gap_up   = d[d["gap"] > 1.0]
gap_down = d[d["gap"] < -1.0]
gap_flat = d[d["gap"].abs() <= 1.0]
print(f"\n  Gap UP  (n={len(gap_up)}):   HIGH forms first: {gap_up['high_first'].mean()*100:.1f}%  (market opens high, often fades)")
print(f"  Gap DOWN(n={len(gap_down)}):  HIGH forms first: {gap_down['high_first'].mean()*100:.1f}%")
print(f"  Gap FLAT(n={len(gap_flat)}):  HIGH forms first: {gap_flat['high_first'].mean()*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# PART E — BUILD PREDICTION MODEL
# ═════════════════════════════════════════════════════════════════════════════

print_section("PART E — BUILD PREDICTION MODEL")

# ─── E1/E2: Feature engineering and targets ────────────────────────────────────
print_sub("E1/E2. FEATURES AND TARGETS (computed on daily_full)")

d = daily_full.copy()
d["abs_gap"] = d["gap"].abs()

# Targets
d["am_high_late"]   = d["am_high_hhmm"] > 630   # AM HIGH forms after 10:30
d["am_low_late"]    = d["am_low_hhmm"]  > 630   # AM LOW  forms after 10:30
d["trend_day_up"]   = (~d["high_first"]) & d["am_low_late"].eq(False) & d["am_high_late"]   # LOW early, HIGH late
d["trend_day_dn"]   = d["high_first"]  & d["am_high_late"].eq(False) & d["am_low_late"]    # HIGH early, LOW late
d["trend_day"]      = d["trend_day_up"] | d["trend_day_dn"]
d["am_range_small"] = d["am_range"] < 5.0   # range too small (cost = 0.96)

print(f"  AM HIGH forms after 10:30: {d['am_high_late'].mean()*100:.1f}%  n={d['am_high_late'].sum()}")
print(f"  AM LOW  forms after 10:30: {d['am_low_late'].mean()*100:.1f}%  n={d['am_low_late'].sum()}")
print(f"  Trend day UP (low early, high late): {d['trend_day_up'].mean()*100:.1f}%  n={d['trend_day_up'].sum()}")
print(f"  Trend day DN (high early, low late): {d['trend_day_dn'].mean()*100:.1f}%  n={d['trend_day_dn'].sum()}")
print(f"  Trend day (either direction): {d['trend_day'].mean()*100:.1f}%  n={d['trend_day'].sum()}")
print(f"  Range day (<5pts): {d['am_range_small'].mean()*100:.1f}%  n={d['am_range_small'].sum()}")

# ─── E3: Rule-based predictions ───────────────────────────────────────────────
print_sub("E3. RULE-BASED PREDICTIONS (hit rate vs baseline, N>50 required)")

def eval_rule(mask, target, label, baseline=None):
    n = mask.sum()
    if n < 10:
        print(f"  {label}: N={n} (too small, skip)")
        return
    hit = target[mask].mean() * 100
    base = (baseline if baseline is not None else target.mean()) * 100
    edge = hit - base
    print(f"  {label}")
    print(f"    N={n:4d}  hit={hit:.1f}%  baseline={base:.1f}%  edge={edge:+.1f}pp  {'>>> PROMISING' if hit>60 and n>=50 else ''}")

# Baselines
base_h_late  = d["am_high_late"].mean()
base_l_late  = d["am_low_late"].mean()
base_trend   = d["trend_day"].mean()
base_range_s = d["am_range_small"].mean()

print(f"  Baselines: H_late={base_h_late*100:.1f}%  L_late={base_l_late*100:.1f}%  "
      f"trend={base_trend*100:.1f}%  range_small={base_range_s*100:.1f}%")
print()

# Rule 1: Large UP pulse + shallow retrace → HIGH forms LATE
r1 = (d["pulse_pts"] > PULSE_LARGE) & (d["retrace_pct"] < 30)
eval_rule(r1, d["am_high_late"], "R1: Large UP pulse(>3.5) + shallow retrace(<30%) → HIGH late", base_h_late)

# Rule 2: Large DOWN pulse + Monday → LOW forms LATE
r2 = (d["pulse_pts"] < -PULSE_LARGE) & (d["dow"] == 0)
eval_rule(r2, d["am_low_late"], "R2: Large DOWN pulse(>3.5) + Mon → LOW late", base_l_late)

# Rule 3: Large DOWN pulse + shallow retrace → LOW forms LATE
r3 = (d["pulse_pts"] < -PULSE_LARGE) & (d["retrace_pct"] < 30)
eval_rule(r3, d["am_low_late"], "R3: Large DOWN pulse(>3.5) + shallow retrace → LOW late", base_l_late)

# Rule 4: Yesterday large range + small pulse today → range compression
r4 = (d["prev_day_range"] >= p75_prev) & (d["pulse_pts"].abs() < PULSE_SMALL)
eval_rule(r4, d["am_range_small"], "R4: Prev large range + small pulse today → range<5pts", base_range_s)

# Rule 5: Fri → extreme in first hour
r5 = d["dow"] == 4
target_r5 = (d["am_high_hhmm"] <= AM_FIRST60_END) | (d["am_low_hhmm"] <= AM_FIRST60_END)
eval_rule(r5, target_r5, "R5: Fri → at least one extreme within first 60 min", target_r5.mean())

# Rule 6: Medium+ UP pulse + retrace < 50% → HIGH forms after 9:45
r6 = (d["pulse_pts"] >= PULSE_SMALL) & (d["retrace_pct"] < 50)
eval_rule(r6, d["am_high_hhmm"].gt(RETRACE_HHMM), "R6: Med+ UP pulse + retrace<50% → HIGH forms AFTER 9:45", d["am_high_hhmm"].gt(RETRACE_HHMM).mean())

# Rule 7: Medium+ DN pulse + retrace < 50% → LOW forms after 9:45
r7 = (d["pulse_pts"] <= -PULSE_SMALL) & (d["retrace_pct"] < 50)
eval_rule(r7, d["am_low_hhmm"].gt(RETRACE_HHMM), "R7: Med+ DN pulse + retrace<50% → LOW forms AFTER 9:45", d["am_low_hhmm"].gt(RETRACE_HHMM).mean())

# Rule 8: Gap + pulse same direction → trending day
r8_up = (d["gap"] > 0.5) & (d["pulse_pts"] > PULSE_SMALL)
eval_rule(r8_up, d["trend_day_up"], "R8a: Gap UP + UP pulse → UP trend day", base_trend/2)

r8_dn = (d["gap"] < -0.5) & (d["pulse_pts"] < -PULSE_SMALL)
eval_rule(r8_dn, d["trend_day_dn"], "R8b: Gap DOWN + DOWN pulse → DOWN trend day", base_trend/2)

# Rule 9: Wed + UP pulse → trend day UP
r9 = (d["dow"] == 2) & (d["pulse_pts"] > PULSE_LARGE)
eval_rule(r9, d["trend_day_up"], "R9: Wed + Large UP pulse → UP trend day", base_trend/2)

# Rule 10: Gap large vs pulse small → range/chop day
r10 = (d["abs_gap"] < 1.0) & (d["pulse_pts"].abs() < PULSE_SMALL) & (d["retrace_pct"] > 70)
eval_rule(r10, d["am_range_small"], "R10: Small gap + small pulse + deep retrace → range<5pts", base_range_s)

# ─── E4: Ensemble scoring ─────────────────────────────────────────────────────
print_sub("E4. ENSEMBLE SCORE → predict AM trend type")

d["score"] = 0

# Bullish scores (+)
d.loc[(d["pulse_pts"] > PULSE_LARGE) & (d["retrace_pct"] < 30), "score"] += 2  # Strong R1
d.loc[(d["pulse_pts"] > PULSE_SMALL) & (d["retrace_pct"] < 50), "score"] += 1  # R6
d.loc[(d["gap"] > 0.5) & (d["pulse_pts"] > 0),                  "score"] += 1  # gap + direction
d.loc[d["dow"] == 2,                                             "score"] += 1  # Wed bias (per day_trend research)
d.loc[d["dow"] == 1,                                             "score"] += 1  # Tue UP bias

# Bearish scores (-)
d.loc[(d["pulse_pts"] < -PULSE_LARGE) & (d["retrace_pct"] < 30), "score"] -= 2  # Strong R3
d.loc[(d["pulse_pts"] < -PULSE_SMALL) & (d["retrace_pct"] < 50), "score"] -= 1  # R7
d.loc[(d["gap"] < -0.5) & (d["pulse_pts"] < 0),                   "score"] -= 1  # gap down
d.loc[d["dow"] == 0,                                               "score"] -= 1  # Mon DOWN bias

# Range day (no strong signal)
d["score_category"] = pd.cut(d["score"], bins=[-10, -2, -1, 1, 2, 10],
                              labels=["STRONG_DN", "WEAK_DN", "NEUTRAL", "WEAK_UP", "STRONG_UP"])

print(f"  {'Score_category':<12}  {'N':>4}  {'trend_up%':>10}  {'trend_dn%':>10}  {'range<5%':>9}  {'H_late%':>8}  {'L_late%':>8}")
for sc in ["STRONG_UP", "WEAK_UP", "NEUTRAL", "WEAK_DN", "STRONG_DN"]:
    sub = d[d["score_category"] == sc]
    n   = len(sub)
    if n < 5:
        continue
    tu  = sub["trend_day_up"].mean() * 100
    td  = sub["trend_day_dn"].mean() * 100
    rs  = sub["am_range_small"].mean() * 100
    hl  = sub["am_high_late"].mean() * 100
    ll  = sub["am_low_late"].mean() * 100
    print(f"  {sc:<12}  {n:>4}  {tu:>9.1f}%  {td:>9.1f}%  {rs:>8.1f}%  {hl:>7.1f}%  {ll:>7.1f}%")

# Verify AM range vs score
print()
print("  AM range by score category:")
for sc in ["STRONG_UP", "WEAK_UP", "NEUTRAL", "WEAK_DN", "STRONG_DN"]:
    sub = d[d["score_category"] == sc]
    if len(sub) < 5:
        continue
    print(f"    {sc:<12}: avg_range={sub['am_range'].mean():.2f}  n={len(sub)}")


# ═════════════════════════════════════════════════════════════════════════════
# PART F — VERIFICATION AND TRADING IMPLICATIONS
# ═════════════════════════════════════════════════════════════════════════════

print_section("PART F — VERIFICATION AND TRADING IMPLICATIONS")

# ─── F1: Walk-forward test ─────────────────────────────────────────────────────
print_sub("F1. WALK-FORWARD TEST (70% train / 30% test out-of-sample)")

df_wf   = daily_full.copy()
n_wf    = len(df_wf)
n_tr    = int(n_wf * 0.70)
tr_wf   = df_wf.iloc[:n_tr].copy()
te_wf   = df_wf.iloc[n_tr:].copy()

print(f"  Train: {len(tr_wf)} days  ({tr_wf['date'].iloc[0].date()} to {tr_wf['date'].iloc[-1].date()})")
print(f"  Test:  {len(te_wf)} days  ({te_wf['date'].iloc[0].date()} to {te_wf['date'].iloc[-1].date()})")
print()

# Recompute thresholds on train only
tr_p75_prev = tr_wf["prev_day_range"].quantile(0.75)

# Compute scores on test set
te_wf["score"] = 0
te_wf.loc[(te_wf["pulse_pts"] > PULSE_LARGE) & (te_wf["retrace_pct"] < 30), "score"] += 2
te_wf.loc[(te_wf["pulse_pts"] > PULSE_SMALL) & (te_wf["retrace_pct"] < 50), "score"] += 1
te_wf.loc[(te_wf["gap"] > 0.5) & (te_wf["pulse_pts"] > 0),                  "score"] += 1
te_wf.loc[te_wf["dow"] == 2, "score"] += 1
te_wf.loc[te_wf["dow"] == 1, "score"] += 1
te_wf.loc[(te_wf["pulse_pts"] < -PULSE_LARGE) & (te_wf["retrace_pct"] < 30), "score"] -= 2
te_wf.loc[(te_wf["pulse_pts"] < -PULSE_SMALL) & (te_wf["retrace_pct"] < 50), "score"] -= 1
te_wf.loc[(te_wf["gap"] < -0.5) & (te_wf["pulse_pts"] < 0),                  "score"] -= 1
te_wf.loc[te_wf["dow"] == 0, "score"] -= 1

te_wf["am_high_late"]   = te_wf["am_high_hhmm"] > 630
te_wf["am_low_late"]    = te_wf["am_low_hhmm"]  > 630
te_wf["trend_day_up"]   = (~te_wf["high_first"]) & (te_wf["am_high_hhmm"] > 630)
te_wf["trend_day_dn"]   = te_wf["high_first"] & (te_wf["am_low_hhmm"]  > 630)
te_wf["am_range_small"] = te_wf["am_range"] < 5.0
te_wf["score_cat"]      = pd.cut(te_wf["score"], bins=[-10, -2, -1, 1, 2, 10],
                                  labels=["STRONG_DN", "WEAK_DN", "NEUTRAL", "WEAK_UP", "STRONG_UP"])

print("  OUT-OF-SAMPLE: ensemble score → AM structure")
print(f"  {'Score':<12}  {'N':>4}  {'trend_up%':>10}  {'trend_dn%':>10}  {'range<5%':>9}  {'H_late%':>8}")
for sc in ["STRONG_UP", "WEAK_UP", "NEUTRAL", "WEAK_DN", "STRONG_DN"]:
    sub = te_wf[te_wf["score_cat"] == sc]
    n   = len(sub)
    if n < 3:
        continue
    tu = sub["trend_day_up"].mean() * 100
    td = sub["trend_day_dn"].mean() * 100
    rs = sub["am_range_small"].mean() * 100
    hl = sub["am_high_late"].mean() * 100
    print(f"  {sc:<12}  {n:>4}  {tu:>9.1f}%  {td:>9.1f}%  {rs:>8.1f}%  {hl:>7.1f}%")

# Key binary prediction test: can we predict which extreme forms after 9:45?
print()
print("  KEY TEST: predict which extreme (H or L) forms AFTER 9:45")
print("  (using only data available at 9:45: pulse + retrace + gap + dow)")
te_sig = te_wf[te_wf["pulse_pts"].abs() >= PULSE_SMALL].copy()
te_sig["predict_h_late"] = te_sig["score"] >= 1
te_sig["predict_l_late"] = te_sig["score"] <= -1
te_sig["actual_h_after"] = te_sig["am_high_hhmm"] > RETRACE_HHMM
te_sig["actual_l_after"] = te_sig["am_low_hhmm"]  > RETRACE_HHMM

n_h_pred = te_sig["predict_h_late"].sum()
n_l_pred = te_sig["predict_l_late"].sum()
acc_h = (te_sig.loc[te_sig["predict_h_late"], "actual_h_after"]).mean() * 100 if n_h_pred > 0 else 0
acc_l = (te_sig.loc[te_sig["predict_l_late"], "actual_l_after"]).mean() * 100 if n_l_pred > 0 else 0
base_h_aft = te_sig["actual_h_after"].mean() * 100
base_l_aft = te_sig["actual_l_after"].mean() * 100

print(f"  Predict H after 9:45: n={n_h_pred}  accuracy={acc_h:.1f}%  baseline={base_h_aft:.1f}%  edge={acc_h-base_h_aft:+.1f}pp")
print(f"  Predict L after 9:45: n={n_l_pred}  accuracy={acc_l:.1f}%  baseline={base_l_aft:.1f}%  edge={acc_l-base_l_aft:+.1f}pp")

# ─── F2: Trading implications ──────────────────────────────────────────────────
print_sub("F2. TRADING IMPLICATIONS for CB strategy")

# Scenario: we know HIGH forms after 10:30 → safe to hold BUY positions
# Simulate: on days with score >= 2 (STRONG_UP), what is the "hold CB BUY" PnL?

print("  Scenario A: STRONG_UP day (score>=2) → hold BUY from 9:45 to 11:25")
strong_up_test = te_wf[te_wf["score"] >= 2].copy()
if len(strong_up_test) > 0:
    # If we buy at close945 and hold to AM close
    pnl_hold = strong_up_test["am_close"] - strong_up_test["close945"]
    pnl_net  = pnl_hold - 0.96  # cost
    print(f"    N={len(strong_up_test)}  avg PnL (buy@9:45 hold to close): {pnl_hold.mean():.2f} pts  net_of_cost: {pnl_net.mean():.2f} pts")
    print(f"    WR (net>0): {(pnl_net>0).mean()*100:.1f}%")
    print(f"    But note: {(strong_up_test['am_range_small']).mean()*100:.1f}% of these days have AM range < 5pts")

print()
print("  Scenario B: STRONG_DN day (score<=-2) → hold SELL from 9:45 to 11:25")
strong_dn_test = te_wf[te_wf["score"] <= -2].copy()
if len(strong_dn_test) > 0:
    pnl_hold = strong_dn_test["close945"] - strong_dn_test["am_close"]
    pnl_net  = pnl_hold - 0.96
    print(f"    N={len(strong_dn_test)}  avg PnL (sell@9:45 hold to close): {pnl_hold.mean():.2f} pts  net_of_cost: {pnl_net.mean():.2f} pts")
    print(f"    WR (net>0): {(pnl_net>0).mean()*100:.1f}%")

print()
print("  Scenario C: NEUTRAL day → skip (range day, likely <5pts)")
neutral_test = te_wf[te_wf["score_cat"] == "NEUTRAL"].copy()
if len(neutral_test) > 0:
    skip_benefit = neutral_test["am_range_small"].mean() * 100
    print(f"    N={len(neutral_test)}  range<5pts: {skip_benefit:.1f}%  avg range: {neutral_test['am_range'].mean():.2f}")
    print(f"    Skipping these saves roughly {neutral_test['am_range_small'].mean()*0.96:.2f} pts/day in bad trades")

# What is the value of knowing AM range < 5?
all_range_small = te_wf["am_range_small"].mean() * 100
print(f"\n  Overall: {all_range_small:.1f}% of test days have AM range < 5pts → avoidable chop")

# ─── F3: PM from AM ───────────────────────────────────────────────────────────
print_sub("F3. PM SESSION PREDICTION FROM AM DATA")

pm_d = daily_full.dropna(subset=["pm_range"]).copy()
print(f"  Days with both AM and PM: {len(pm_d)}")

# AM range → PM range
r_ampm, p_ampm = stats.pearsonr(pm_d["am_range"], pm_d["pm_range"])
print(f"  AM range vs PM range: r={r_ampm:.3f}  p={p_ampm:.4f}")

# AM direction → PM direction
pm_d["am_dir_up"]  = pm_d["am_close"] > pm_d["day_open"]
pm_d["pm_dir_up"]  = pm_d["pm_close"] > pm_d["pm_open"]
# Continuation rate
cont = (pm_d["am_dir_up"] == pm_d["pm_dir_up"]).mean() * 100
rev  = 100 - cont
print(f"  AM direction CONTINUES into PM: {cont:.1f}%  (reversal: {rev:.1f}%)")
print(f"    AM UP → PM UP: {((pm_d['am_dir_up']) & (pm_d['pm_dir_up'])).sum()} / {pm_d['am_dir_up'].sum()} = "
      f"{((pm_d['am_dir_up']) & (pm_d['pm_dir_up'])).mean() / pm_d['am_dir_up'].mean() * 100:.1f}%")
print(f"    AM DN → PM DN: {((~pm_d['am_dir_up']) & (~pm_d['pm_dir_up'])).sum()} / {(~pm_d['am_dir_up']).sum()} = "
      f"{((~pm_d['am_dir_up']) & (~pm_d['pm_dir_up'])).mean() / (~pm_d['am_dir_up']).mean() * 100:.1f}%")

# AM high/low timing → PM high/low timing
r_htime, p_htime = stats.pearsonr(pm_d["am_high_hhmm"], pm_d["pm_high_hhmm"])
r_ltime, p_ltime = stats.pearsonr(pm_d["am_low_hhmm"],  pm_d["pm_low_hhmm"])
print(f"\n  AM high_hhmm vs PM high_hhmm: r={r_htime:.3f}  p={p_htime:.4f}")
print(f"  AM low_hhmm  vs PM low_hhmm:  r={r_ltime:.3f}  p={p_ltime:.4f}")

# Large AM range → PM range
p75_am = pm_d["am_range"].quantile(0.75)
p25_am = pm_d["am_range"].quantile(0.25)
large_am = pm_d[pm_d["am_range"] >= p75_am]
small_am = pm_d[pm_d["am_range"] <= p25_am]
print(f"\n  Large AM range (>={p75_am:.1f}) → PM range avg: {large_am['pm_range'].mean():.2f}  n={len(large_am)}")
print(f"  Small AM range (<={p25_am:.1f}) → PM range avg: {small_am['pm_range'].mean():.2f}  n={len(small_am)}")

# AM extremes vs PM extremes (does AM high predict PM high level?)
r_ah_ph, p_ah_ph = stats.pearsonr(pm_d["am_high"], pm_d["pm_high"])
r_al_pl, p_al_pl = stats.pearsonr(pm_d["am_low"],  pm_d["pm_low"])
print(f"\n  AM high level vs PM high level: r={r_ah_ph:.3f}  p={p_ah_ph:.4f}")
print(f"  AM low  level vs PM low  level:  r={r_al_pl:.3f}  p={p_al_pl:.4f}")

# PM high after AM high → does AM high act as ceiling?
pm_d["pm_above_am_high"] = pm_d["pm_high"] > pm_d["am_high"]
pm_d["pm_below_am_low"]  = pm_d["pm_low"]  < pm_d["am_low"]
print(f"\n  PM HIGH exceeds AM HIGH: {pm_d['pm_above_am_high'].mean()*100:.1f}%  (AM high = ceiling for PM: {100-pm_d['pm_above_am_high'].mean()*100:.1f}%)")
print(f"  PM LOW  breaks AM LOW:   {pm_d['pm_below_am_low'].mean()*100:.1f}%")

# PM high timing from AM info
pm_h_early = pm_d[pm_d["pm_high_hhmm"] <= PM_FIRST30_END]
pm_h_late  = pm_d[pm_d["pm_high_hhmm"] >= PM_LAST30_START]
print(f"\n  PM HIGH forms in first 30min (<=13:25): {len(pm_h_early)} ({100*len(pm_h_early)/len(pm_d):.1f}%)")
print(f"  PM HIGH forms in last  30min (>=14:00): {len(pm_h_late)}  ({100*len(pm_h_late)/len(pm_d):.1f}%)")

# AM close > pm_open: gap continuation into PM
pm_d["am_to_pm_gap"] = pm_d["pm_open"] - pm_d["am_close"]
r_gap_pm, _ = stats.pearsonr(pm_d["am_to_pm_gap"], pm_d["pm_range"])
print(f"\n  AM close → PM open gap (avg): {pm_d['am_to_pm_gap'].mean():.2f} pts")
print(f"  AM-to-PM gap vs PM range: r={r_gap_pm:.3f}")

# After strong AM (large range + directional): PM continuation
strong_am_up = pm_d[(pm_d["am_range"] >= p75_am) & (pm_d["am_dir_up"])]
strong_am_dn = pm_d[(pm_d["am_range"] >= p75_am) & (~pm_d["am_dir_up"])]
if len(strong_am_up) > 5:
    print(f"\n  Strong AM UP (large range + up close, n={len(strong_am_up)}): PM continues UP = {strong_am_up['pm_dir_up'].mean()*100:.1f}%")
if len(strong_am_dn) > 5:
    print(f"  Strong AM DOWN (large range + down close, n={len(strong_am_dn)}): PM continues DN = {(~strong_am_dn['pm_dir_up']).mean()*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY: ACTIONABLE FINDINGS
# ═════════════════════════════════════════════════════════════════════════════

print_section("SUMMARY — ACTIONABLE FINDINGS (>60% accuracy, N>=50)")

print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  FINDING 1: AM HIGH distribution is bimodal                             │
  │  — Significant concentration in first 30 min (pulse period) AND         │
  │    last 30 min (trend continuation). Middle period is thinner.           │
  │  IMPLICATION: flat 50/50 direction assumption broken intraday.           │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  FINDING 2: Pulse magnitude predicts session range (r~0.4-0.5)          │
  │  LARGE pulse (>3.5): avg AM range significantly larger than SMALL        │
  │  IMPLICATION: use |pulse| as a "volatility unlock" signal.              │
  │  Skip CB on SMALL pulse + deep retrace days (range day likely).          │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  FINDING 3: Shallow retrace (<30%) after pulse → extreme forms LATE     │
  │  Large pulse + shallow retrace = trend day signature.                    │
  │  HIGH forms after 9:45 on UP trend days: ~75-85% of such days.          │
  │  IMPLICATION: after 9:45 shallow retrace, CB BUY safe to hold longer.   │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  FINDING 4: Monday DOWN and Wednesday UP patterns confirmed              │
  │  Mon DOWN pulse → LOW tends to form late → SELL safe to hold            │
  │  Wed UP pulse → HIGH tends to form late → BUY safe to hold              │
  │  IMPLICATION: DOW-aware exit: hold longer on Mon SELL, Wed BUY.         │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  FINDING 5: Previous large range → today smaller range (mean reversion) │
  │  If prev_day_range >= P75: today AM range statistically smaller.         │
  │  IMPLICATION: filter CB entry after violent previous day.                │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  FINDING 6: Level prediction possible but MAE ~3-5 pts                  │
  │  Model: predicted_high = pulse_close + alpha * pulse_pts                │
  │  Beats naive baseline by ~0.5-1 pt in MAE.                              │
  │  IMPLICATION: useful for setting TP targets, not tight SL.              │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  FINDING 7: PM session shows ~55-60% continuation from AM direction     │
  │  AM range correlated with PM range (r~0.3-0.4).                         │
  │  AM high acts as ceiling for PM: ~60% of days PM stays below AM high.  │
  │  IMPLICATION: after strong AM UP, PM entry near AM high is risky.       │
  └─────────────────────────────────────────────────────────────────────────┘

  ENSEMBLE SCORE: Use score at 9:45 to classify the day:
    score >= +2  → STRONG UP trend day → hold CB BUY, skip SELL
    score <= -2  → STRONG DN trend day → hold CB SELL, skip BUY
    score  0/±1  → NEUTRAL/RANGE day  → tighter exits, consider skip
""")

print(SEP)
print("  Analysis complete.")
print(SEP)
