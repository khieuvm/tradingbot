"""
NR4 SHORT-ONLY Backtest — Full Report
======================================
Signal: NR4 (Narrowest Range in 4 bars) + range < 0.7×ATR
Direction: SHORT ONLY (SELL breakout: next_bar low < signal_bar low - 0.1)
Data: VN30F1M 5m, ~144 days

Exit params (matching CB session params):
  AM: SL=1.2×ATR, trail@5pts/2.0×ATR, max_hold=24 bars, exit 11:25
  PM: SL=1.0×ATR, trail@4pts/1.5×ATR, max_hold=12 bars, exit 14:25
  BE: MFE >= 4pts when entry ATR >= 3.5 → SL to entry
  Trail tighten: MFE >= 8pts when ATR >= 3.5 → trail 1.2×ATR
  Adaptive exit (AM): pre_move/ATR > 0.8 + MFE >= 4pts → close if PnL >= 3.2, else SL→entry-2

Baseline (CB 5m + adaptive): WR 67.7%, PF 4.87, +6.34/d (223 trades / 129d)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta, date

from src.data_fetcher import DataFetcher

COST = 0.96
DIRECTION = -1        # SHORT only
BE_ATR_MIN = 3.5
TRAIL_TIGHTEN_MFE = 8.0

# ============================================================
# 1. DATA LOADING
# ============================================================
fetcher = DataFetcher()
end = datetime.now().strftime("%Y-%m-%d")
start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
print(f"Loading 5m data {start} → {end} ...")

df = fetcher.get_futures_ohlcv("VN30F1M", start, end, interval="5m")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df['date'] = df['time'].dt.date
df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute

# Session filter: AM 09:00-11:30, PM 13:00-14:30
df = df[((df['mins'] >= 9*60) & (df['mins'] < 11*60+30)) |
        ((df['mins'] >= 13*60) & (df['mins'] < 14*60+30))]
df = df.reset_index(drop=True)

df['atr']     = ta.atr(df['high'], df['low'], df['close'], length=14)
df['session'] = np.where(df['mins'] < 12*60, 'AM', 'PM')
df['range']   = df['high'] - df['low']
n_days = df['date'].nunique()
print(f"  {len(df):,} bars | {n_days} trading days")

# ============================================================
# 2. NR4 SIGNAL DETECTION (bulk)
# ============================================================
# NR4: current bar range is strictly narrowest in last 4 bars
# AND range < 0.7×ATR
# ATR filter: 2.5 <= ATR <= 4.5 (no RSI filter)
# Time window: AM 09:15-10:45, PM 13:15-14:15
# Dedup: 5 bars (25 min)

def detect_nr4_bulk(df):
    n = len(df)
    nr4 = np.zeros(n, dtype=bool)
    ranges   = df['range'].values
    atr_vals = df['atr'].values

    for i in range(3, n):
        atr_v = atr_vals[i]
        if pd.isna(atr_v) or atr_v <= 0:
            continue
        cur_range = ranges[i]
        prev_min  = min(ranges[i-3], ranges[i-2], ranges[i-1])
        if cur_range < prev_min and cur_range < 0.7 * atr_v:
            nr4[i] = True
    return nr4


def dedup(mask, min_bars):
    r    = mask.copy()
    last = -999
    for i in range(len(r)):
        if r.iloc[i]:
            if i - last < min_bars:
                r.iloc[i] = False
            else:
                last = i
    return r


df['nr4'] = detect_nr4_bulk(df)

filt      = (df['atr'] >= 2.5) & (df['atr'] <= 4.5)
in_window = (((df['mins'] >= 9*60+15) & (df['mins'] <= 10*60+45)) |
             ((df['mins'] >= 13*60+15) & (df['mins'] <= 14*60+15)))

sig_raw  = df['nr4'] & filt & in_window
sig_mask = dedup(sig_raw, 5)

print(f"\nNR4 signals: {sig_raw.sum()} raw  →  {sig_mask.sum()} after dedup(5 bars)")

# ============================================================
# 3. SIMULATION — SHORT ONLY
# ============================================================

def simulate_trades(df, sig_mask):
    trades = []

    for sig_idx in df.index[sig_mask]:
        i_pos       = df.index.get_loc(sig_idx)
        signal_bar  = df.iloc[i_pos]
        atr         = signal_bar['atr']
        session     = signal_bar['session']

        if pd.isna(atr) or atr <= 0:
            continue

        # Session params
        if session == 'AM':
            sl_mult       = 1.2
            trail_activate = 5.0
            trail_mult    = 2.0
            max_hold      = 24
            exit_mins     = 11*60 + 25
        else:
            sl_mult       = 1.0
            trail_activate = 4.0
            trail_mult    = 1.5
            max_hold      = 12
            exit_mins     = 14*60 + 25

        # --- SHORT trigger: check up to 2 bars after signal ---
        entry_trigger  = signal_bar['low'] - 0.1
        entry          = None
        entry_bar_pos  = None

        for k in range(i_pos + 1, min(i_pos + 3, len(df))):
            tb = df.iloc[k]
            if tb['date'] != signal_bar['date'] or tb['session'] != signal_bar['session']:
                break
            if tb['low'] <= entry_trigger:
                entry         = entry_trigger
                entry_bar_pos = k
                break

        if entry is None:
            continue   # signal expired without trigger

        # --- Pre-move ratio (matches scanner: bars i-1 close / i-3 open) ---
        if i_pos >= 3:
            pre_close       = df.iloc[i_pos - 1]['close']
            pre_open        = df.iloc[i_pos - 3]['open']
            pre_move_ratio  = (pre_close - pre_open) * DIRECTION / atr
            # DIRECTION=-1: positive when price dropped before signal (bearish context)
        else:
            pre_move_ratio = 0.0

        # --- Initialise position ---
        sl         = entry + sl_mult * atr   # SL ABOVE entry for SHORT
        best       = entry                    # lowest price seen (MIN tracker)
        trail_on   = False
        be_done    = False
        adapt_done = False
        exit_p     = None
        exit_r     = None
        bars       = 0

        end_pos = min(entry_bar_pos + max_hold, len(df))

        for jp in range(entry_bar_pos, end_pos):
            b = df.iloc[jp]

            # ---- Session boundary (date or session change) ----
            if b['date'] != signal_bar['date'] or b['session'] != signal_bar['session']:
                prev   = df.iloc[jp - 1]
                exit_p = prev['close']
                exit_r = 'SESSION'
                break

            # ---- Hard session time exit ----
            if b['mins'] >= exit_mins:
                exit_p = b['close']
                exit_r = 'SESSION'
                break

            bars += 1

            # Step 1 — SL hit check (uses SL value from end of PREVIOUS bar)
            if b['high'] >= sl:
                exit_p = sl
                exit_r = 'TRAIL' if trail_on else ('BE' if be_done else 'SL')
                break

            # Step 2 — Update best (min low for SHORT)
            if b['low'] < best:
                best = b['low']
            mfe = entry - best   # positive when price drops below entry

            # Step 3 — Adaptive exit (AM + high pre_move + MFE ≥ 4)
            if (session == 'AM'
                    and pre_move_ratio > 0.8
                    and mfe >= 4.0
                    and not adapt_done):
                pnl_pts = entry - b['close']
                if pnl_pts >= 3.2:
                    exit_p = b['close']
                    exit_r = 'ADAPT_EXIT'
                    break
                else:
                    adapt_done = True
                    # Lock in +2 pts: SL → entry - 2  (SL moves BELOW entry)
                    sl = min(sl, entry - 2.0)

            # Step 4 — Break-even (MFE ≥ 4 when ATR ≥ 3.5)
            if not be_done and mfe >= 4.0 and atr >= BE_ATR_MIN:
                be_done = True
                sl      = min(sl, entry)   # SL cannot increase past entry

            # Step 5 — Trail activation
            if mfe >= trail_activate:
                trail_on = True

            # Step 6 — Trail SL update + check
            if trail_on:
                eff_mult = (1.2 if (mfe >= TRAIL_TIGHTEN_MFE and atr >= BE_ATR_MIN)
                            else trail_mult)
                trail_sl = best + eff_mult * atr
                sl       = min(sl, trail_sl)
                if b['high'] >= sl:
                    exit_p = sl
                    exit_r = 'TRAIL'
                    break

        # Exhausted max_hold bars → exit at last bar close
        if exit_p is None:
            last_bar = df.iloc[min(entry_bar_pos + max_hold - 1, len(df) - 1)]
            exit_p   = last_bar['close']
            exit_r   = 'MAX_HOLD'

        mfe_f = entry - best
        pnl   = entry - exit_p - COST   # SHORT: profit when exit < entry

        trades.append({
            'date'           : signal_bar['date'],
            'time'           : signal_bar['time'].strftime('%H:%M'),
            'sess'           : session,
            'mins'           : signal_bar['mins'],
            'entry'          : entry,
            'exit'           : exit_p,
            'atr'            : atr,
            'sl_size'        : sl_mult * atr,
            'mfe'            : mfe_f,
            'pnl'            : pnl,
            'reason'         : exit_r,
            'bars'           : bars,
            'pre_move_ratio' : pre_move_ratio,
        })

    if not trades:
        return None
    return (pd.DataFrame(trades)
              .sort_values(['date', 'time'])
              .reset_index(drop=True))


print("Running simulation ...")
tdf = simulate_trades(df, sig_mask)

if tdf is None or len(tdf) == 0:
    print("No trades generated!")
    sys.exit(1)

# ============================================================
# 4. REPORTING HELPERS
# ============================================================

def profit_factor(sub):
    wins   = sub[sub['pnl'] > 0]
    losses = sub[sub['pnl'] <= 0]
    if len(losses) == 0 or losses['pnl'].sum() == 0:
        return 999.0
    return wins['pnl'].sum() / abs(losses['pnl'].sum())


# ============================================================
# 5. SECTION 1 — TOTAL STATS
# ============================================================
n         = len(tdf)
wr        = (tdf['pnl'] > 0).mean() * 100
pf_val    = profit_factor(tdf)
total_pnl = tdf['pnl'].sum()
ppd       = total_pnl / n_days

print()
print("=" * 70)
print("NR4 SHORT-ONLY BACKTEST   |   VN30F1M 5m")
print("=" * 70)
print(f"\n--- 1. TOTAL ({n_days} days) ---")
print(f"  Trades    : {n}")
print(f"  Win Rate  : {wr:.1f}%")
print(f"  Prof. Fac : {pf_val:.2f}")
print(f"  Total PnL : {total_pnl:+.1f} pts")
print(f"  P/D       : {ppd:+.2f} pts/day")
print(f"  Avg PnL   : {tdf['pnl'].mean():+.2f} pts/trade")
print(f"  Avg MFE   : {tdf['mfe'].mean():.2f} pts")
print(f"  Signals   : {sig_mask.sum()} → triggered {n} ({100*n/sig_mask.sum():.0f}%)")

# ============================================================
# 6. SECTION 2 — AM vs PM
# ============================================================
print(f"\n--- 2. AM vs PM BREAKDOWN ---")
print(f"  {'Session':<8} {'T':>4}  {'WR':>6}  {'PF':>5}  {'PnL':>8}  {'P/D':>7}  {'AvgMFE':>7}  {'Exit@SL':>8}")
print("  " + "-" * 63)
for s in ['AM', 'PM']:
    sub = tdf[tdf['sess'] == s]
    if len(sub) == 0:
        print(f"  {s:<8}   (no trades)")
        continue
    wr_s   = (sub['pnl'] > 0).mean() * 100
    pf_s   = profit_factor(sub)
    pnl_s  = sub['pnl'].sum()
    ppd_s  = pnl_s / n_days
    mfe_s  = sub['mfe'].mean()
    sl_cnt = (sub['reason'] == 'SL').sum()
    print(f"  {s:<8} {len(sub):>4}  {wr_s:>5.1f}%  {pf_s:>5.2f}  {pnl_s:>+8.1f}  {ppd_s:>+7.2f}  {mfe_s:>7.2f}  {sl_cnt:>8}")

# ============================================================
# 7. SECTION 3 — EXIT REASON BREAKDOWN
# ============================================================
print(f"\n--- 3. EXIT REASON BREAKDOWN ---")
print(f"  {'Reason':<12}  {'Cnt':>4}  {'WR':>6}  {'AvgPnL':>8}  {'TotPnL':>9}  {'AvgMFE':>7}  {'AvgBars':>7}")
print("  " + "-" * 62)
for r in ['SL', 'BE', 'TRAIL', 'SESSION', 'MAX_HOLD', 'ADAPT_EXIT']:
    sub = tdf[tdf['reason'] == r]
    if len(sub) == 0:
        continue
    wr_r    = (sub['pnl'] > 0).mean() * 100
    avg_pnl = sub['pnl'].mean()
    tot_pnl = sub['pnl'].sum()
    avg_mfe = sub['mfe'].mean()
    avg_bar = sub['bars'].mean()
    print(f"  {r:<12}  {len(sub):>4}  {wr_r:>5.1f}%  {avg_pnl:>+8.2f}  {tot_pnl:>+9.1f}  {avg_mfe:>7.2f}  {avg_bar:>7.1f}")

# ============================================================
# 8. SECTION 4 — SL ANALYSIS
# ============================================================
sl_trades = tdf[tdf['reason'] == 'SL']
print(f"\n--- 4. SL ANALYSIS ---")
print(f"  Avg initial SL size    : {tdf['sl_size'].mean():.2f} pts")
print(f"  Trades hitting SL      : {len(sl_trades)} / {n} = {100*len(sl_trades)/n:.1f}%")
if len(sl_trades) > 0:
    print(f"  Avg loss on SL trades  : {sl_trades['pnl'].mean():+.2f} pts")
    print(f"  Median bars before SL  : {sl_trades['bars'].median():.0f} bars")
    print(f"  Min loss / Max loss    : {sl_trades['pnl'].min():+.2f} / {sl_trades['pnl'].max():+.2f} pts")
else:
    print("  (no SL hits)")

# All losing trades (SL + BE exits)
losing = tdf[tdf['pnl'] <= 0]
print(f"  Total losing trades    : {len(losing)} ({100*len(losing)/n:.1f}%)")
if len(losing) > 0:
    print(f"  Avg losing trade PnL   : {losing['pnl'].mean():+.2f} pts")

# ============================================================
# 9. SECTION 5 — MFE DISTRIBUTION
# ============================================================
print(f"\n--- 5. MFE DISTRIBUTION ---")
print(f"  {'Bucket':>6}  {'Cnt':>4}  {'WR':>6}  {'AvgPnL':>8}  {'AvgMFE':>8}  {'% of total':>10}")
print("  " + "-" * 50)
buckets = [(0, 2, '0-2'), (2, 4, '2-4'), (4, 6, '4-6'), (6, 9, '6-9'), (9, 999, '9+')]
for lo, hi, lbl in buckets:
    sub = tdf[(tdf['mfe'] >= lo) & (tdf['mfe'] < hi)]
    if len(sub) == 0:
        continue
    wr_b    = (sub['pnl'] > 0).mean() * 100
    avg_pnl = sub['pnl'].mean()
    avg_mfe = sub['mfe'].mean()
    pct     = 100 * len(sub) / n
    print(f"  {lbl:>6}  {len(sub):>4}  {wr_b:>5.1f}%  {avg_pnl:>+8.2f}  {avg_mfe:>8.2f}  {pct:>10.1f}%")

# ============================================================
# 10. SECTION 6 — DRAWDOWN
# ============================================================
print(f"\n--- 6. DRAWDOWN ---")

# Max consecutive losses
max_streak = 0
cur_streak = 0
for p in tdf['pnl']:
    if p <= 0:
        cur_streak += 1
        max_streak  = max(max_streak, cur_streak)
    else:
        cur_streak = 0
print(f"  Max consecutive losses : {max_streak}")

# Max drawdown in pts
cum_pnl     = tdf['pnl'].cumsum()
running_max = cum_pnl.cummax()
drawdown    = running_max - cum_pnl
max_dd      = drawdown.max()
print(f"  Max drawdown (pts)     : {max_dd:.1f}")

# Drawdown in trades
dd_idx     = drawdown.idxmax()
peak_idx   = cum_pnl[:dd_idx+1].idxmax()
dd_trades  = dd_idx - peak_idx
print(f"  DD duration (trades)   : {dd_trades} trades")

# ============================================================
# 11. SECTION 7 — MONTHLY P/D (last 3 months)
# ============================================================
print(f"\n--- 7. MONTHLY P/D (last 3 months) ---")
tdf['month'] = pd.to_datetime(tdf['date']).dt.to_period('M')
df['month_p'] = pd.to_datetime(df['date']).dt.to_period('M')

months_all = sorted(tdf['month'].unique())
months_show = months_all[-3:]

print(f"  {'Month':<10}  {'T':>4}  {'WR':>6}  {'PnL':>8}  {'TradeDays':>9}  {'P/D':>7}")
print("  " + "-" * 52)
for m in months_show:
    sub    = tdf[tdf['month'] == m]
    m_days = df[df['month_p'] == m]['date'].nunique()
    wr_m   = (sub['pnl'] > 0).mean() * 100
    pnl_m  = sub['pnl'].sum()
    ppd_m  = pnl_m / m_days if m_days > 0 else 0
    print(f"  {str(m):<10}  {len(sub):>4}  {wr_m:>5.1f}%  {pnl_m:>+8.1f}  {m_days:>9}  {ppd_m:>+7.2f}")

# ============================================================
# 12. COMPARISON vs BASELINE
# ============================================================
print()
print("=" * 70)
print("VERDICT vs CB BASELINE")
print("=" * 70)
print(f"  CB baseline  : WR 67.7% | PF 4.87 | +6.34/d  (223 trades / 129d)")
print(f"  NR4 SHORT    : WR {wr:.1f}% | PF {pf_val:.2f} | {ppd:+.2f}/d  ({n} trades / {n_days}d)")
delta_ppd = ppd - 6.34
verdict = "BETTER" if delta_ppd > 0 else "WORSE"
print(f"  Delta P/D    : {delta_ppd:+.2f}  [{verdict}]")
print()
