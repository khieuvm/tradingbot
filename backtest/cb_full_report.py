"""
backtest/cb_full_report.py — CB 5m Comprehensive Backtest

Signal: 3-bar compression < 0.7xATR -> breakout
Direction: BUY if next_bar H > signal H+0.1, SELL if L < signal L-0.1
Entry: at trigger price (H+0.1 or L-0.1)
2-bar expiry: skip signal if neither trigger fires within 2 bars

Exit params (per strategy_config.yaml):
  AM: SL=1.2xATR, trail@5pts/2.0xATR, max_hold=24, exit 11:25
  PM: SL=1.0xATR, trail@4pts/1.5xATR, max_hold=12, exit 14:25
  BE: MFE>=4 when ATR>=3.5 -> SL to entry
  Trail tighten: MFE>=8 when ATR>=3.5 -> trail 1.2xATR
  Adaptive: AM + pre_move/ATR>0.8 + MFE>=4 -> exit if PnL>=3.2, else SL->entry+2
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
end = datetime.now().strftime("%Y-%m-%d")
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

df['atr']     = ta.atr(df['high'], df['low'], df['close'], length=14)
df['rsi14']   = ta.rsi(df['close'], length=14)
df['session'] = np.where(df['mins'] < 12*60, 'AM', 'PM')
df['range']   = df['high'] - df['low']

n_days = df['date'].nunique()
print(f"Loaded {len(df)} bars | {n_days} trading days | "
      f"{df['date'].min()} -> {df['date'].max()}")

# ─── 2. SIGNAL DETECTION ─────────────────────────────────────────────────────
N_COMP       = 3
COMP_THRESH  = 0.7
ATR_MIN      = 2.5
ATR_MAX      = 4.5
RSI_MAX      = 70
DEDUP_BARS   = 5

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
    if np.nanmax(ranges[i-N_COMP:i]) < COMP_THRESH * a:
        comp_arr[i] = True

# ATR + RSI filter
rsi_ok = np.where(np.isnan(rsi_arr), True, rsi_arr < RSI_MAX)
atr_ok = (~np.isnan(atrs_arr)) & (atrs_arr >= ATR_MIN) & (atrs_arr <= ATR_MAX)
filt   = comp_arr & atr_ok & rsi_ok

# Time windows (separate dedup per session)
am_raw = filt & (mins_arr >= 9*60+15) & (mins_arr <= 10*60+45)
pm_raw = filt & (mins_arr >= 13*60+15) & (mins_arr <= 14*60+15)

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

am_mask  = dedup_arr(am_raw, DEDUP_BARS)
pm_mask  = dedup_arr(pm_raw, DEDUP_BARS)
sig_locs = np.where(am_mask | pm_mask)[0]
print(f"CB signals: {len(sig_locs)} "
      f"(AM={int(am_mask.sum())}, PM={int(pm_mask.sum())})")

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

# ─── 4. BREAKOUT DIRECTION + ENTRY ───────────────────────────────────────────
def get_breakout(sig_loc):
    """
    Returns (direction, entry_price, bar_loc) or (None, None, None).
    Checks 2 bars after signal for H+0.1 / L-0.1 trigger.
    If both trigger: open direction > buy_trig => BUY,
                     open < sell_trig => SELL,
                     else use close vs open.
    """
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
                if closes[j] >= opens[j]:
                    return 1, buy_trig, j
                else:
                    return -1, sell_trig, j
        elif buy_hit:
            return 1, buy_trig, j
        elif sell_hit:
            return -1, sell_trig, j

    return None, None, None

# ─── 5. SIMULATION ───────────────────────────────────────────────────────────
def simulate(sig_loc, entry_bar_loc, direction, entry_price):
    atr     = float(atrs_arr[sig_loc])
    session = sessions[sig_loc]
    p       = PARAMS[session]

    # pre_move_ratio: (close[sig] - open[sig-2]) * dir / ATR
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
    bars_held     = 0

    sig_date = dates[sig_loc]
    end_loc  = min(entry_bar_loc + p['max_hold'], len(df))

    for k in range(entry_bar_loc, end_loc):
        # Session boundary
        if dates[k] != sig_date or sessions[k] != session:
            exit_p = float(closes[k-1]) if k > entry_bar_loc else entry_price
            exit_r = 'SESSION'
            break

        # Session time cutoff
        if mins_arr[k] >= p['exit_mins']:
            exit_p = float(closes[k])
            exit_r = 'SESSION'
            break

        bars_held += 1
        h = float(highs[k])
        l = float(lows[k])
        c = float(closes[k])

        # Update best price and MFE
        if direction == 1:
            best = max(best, h)
            mfe  = max(mfe, best - entry_price)
        else:
            best = min(best, l)
            mfe  = max(mfe, entry_price - best)

        # Adaptive exit (AM only, once per trade)
        if (session == 'AM' and pre_move_ratio > ADAPT_PMR
                and mfe >= ADAPT_MFE and not adapt_checked):
            adapt_checked = True
            pnl_pts = direction * (c - entry_price)
            if pnl_pts >= ADAPT_PNL:
                exit_p = c
                exit_r = 'ADAPT_EXIT'
                break
            else:
                # Lock SL at entry+2
                if direction == 1:
                    sl = max(sl, entry_price + 2.0)
                else:
                    sl = min(sl, entry_price - 2.0)

        # Breakeven
        if not be_done and mfe >= BE_TRIGGER and atr >= BE_ATR_MIN:
            be_done = True
            if direction == 1:
                sl = max(sl, entry_price)
            else:
                sl = min(sl, entry_price)

        # Trail activation
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

        # SL check
        sl_hit = (direction == 1 and l <= sl) or (direction == -1 and h >= sl)
        if sl_hit:
            exit_p = sl
            if trail_on:
                exit_r = 'TRAIL'
            elif be_done:
                exit_r = 'BE'
            else:
                exit_r = 'SL'
            break

    # Max hold fallback
    if exit_p is None:
        exit_p = float(closes[min(end_loc - 1, len(df) - 1)])
        exit_r = 'MAX_HOLD'

    pnl = direction * (exit_p - entry_price) - COST

    return {
        'date'           : dates[sig_loc],
        'time'           : str(df.iloc[sig_loc]['time'])[11:16],
        'session'        : session,
        'mins'           : int(mins_arr[sig_loc]),
        'direction'      : 'BUY' if direction == 1 else 'SELL',
        'entry'          : float(entry_price),
        'exit'           : float(exit_p),
        'atr'            : float(atr),
        'mfe'            : float(mfe),
        'pnl'            : float(pnl),
        'reason'         : exit_r,
        'bars'           : bars_held,
        'pre_move_ratio' : float(pre_move_ratio),
        'sl_size'        : float(p['sl_mult'] * atr),
    }

# ─── Run ─────────────────────────────────────────────────────────────────────
trades   = []
expired  = 0
for sl in sig_locs:
    direction, entry_price, entry_bar_loc = get_breakout(int(sl))
    if direction is None:
        expired += 1
        continue
    t = simulate(int(sl), entry_bar_loc, direction, entry_price)
    trades.append(t)

tdf = (pd.DataFrame(trades)
       .sort_values(['date', 'time'])
       .reset_index(drop=True))

print(f"Trades taken: {len(tdf)} | Expired (no breakout): {expired}")

# ─── 6. REPORTING ────────────────────────────────────────────────────────────
def pf_calc(sub):
    wins   = sub[sub['pnl'] > 0]['pnl'].sum()
    losses = abs(sub[sub['pnl'] <= 0]['pnl'].sum())
    return wins / losses if losses > 0 else 999.0

SEP  = "=" * 72
SEP2 = "-" * 60

print(f"\n{SEP}")
print(f"CB 5m COMPREHENSIVE BACKTEST REPORT")
print(f"Period : {tdf['date'].min()} -> {tdf['date'].max()}  ({n_days} trading days)")
print(f"Signals: {len(sig_locs)} detected | {len(tdf)} taken | {expired} expired")
print(SEP)

# ── 1. TOTAL ──────────────────────────────────────────────────────────────────
print("\n[1] TOTAL SUMMARY")
n_t      = len(tdf)
wr_t     = (tdf['pnl'] > 0).mean() * 100
pf_t     = pf_calc(tdf)
pnl_t    = tdf['pnl'].sum()
ppd_t    = pnl_t / n_days
avg_win  = tdf[tdf['pnl'] > 0]['pnl'].mean()
avg_loss = tdf[tdf['pnl'] <= 0]['pnl'].mean()
print(f"  Trades : {n_t} | WR : {wr_t:.1f}% | PF : {pf_t:.2f} | "
      f"PnL : {pnl_t:+.1f} pts | P/D : {ppd_t:+.2f}")
print(f"  Avg PnL: {tdf['pnl'].mean():+.2f} | "
      f"Avg win: {avg_win:+.2f} | Avg loss: {avg_loss:+.2f}")

# ── 2. AM / PM ────────────────────────────────────────────────────────────────
print(f"\n[2] AM vs PM BREAKDOWN")
hdr = f"  {'Sess':<5} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'P/D':>6} {'avgMFE':>7}"
print(hdr)
print("  " + SEP2)
for sess in ('AM', 'PM'):
    sub = tdf[tdf['session'] == sess]
    if len(sub) == 0:
        continue
    wr_s   = (sub['pnl'] > 0).mean() * 100
    pf_s   = pf_calc(sub)
    pnl_s  = sub['pnl'].sum()
    ppd_s  = pnl_s / n_days
    mfe_s  = sub['mfe'].mean()
    print(f"  {sess:<5} {len(sub):>4} {wr_s:>5.1f}% {pf_s:>5.2f} "
          f"{pnl_s:>+8.1f} {ppd_s:>+6.2f} {mfe_s:>7.1f}")
    # exit reason sub-breakdown per session
    for reason in ('SL', 'BE', 'TRAIL', 'SESSION', 'MAX_HOLD', 'ADAPT_EXIT'):
        sr = sub[sub['reason'] == reason]
        if len(sr) == 0:
            continue
        wr_r   = (sr['pnl'] > 0).mean() * 100
        avg_r  = sr['pnl'].mean()
        print(f"    {reason:<12}: {len(sr):>3} | WR {wr_r:>5.1f}% | "
              f"avg PnL {avg_r:>+6.2f} | avg MFE {sr['mfe'].mean():>5.1f}")

# ── 3. EXIT REASON BREAKDOWN ──────────────────────────────────────────────────
print(f"\n[3] EXIT REASON BREAKDOWN")
print(f"  {'Reason':<12} {'Cnt':>5} {'%Tot':>5} {'WR':>6} "
      f"{'AvgPnL':>7} {'TotPnL':>8} {'AvgMFE':>7}")
print("  " + SEP2)
for reason in ('SL', 'BE', 'TRAIL', 'SESSION', 'MAX_HOLD', 'ADAPT_EXIT'):
    sub = tdf[tdf['reason'] == reason]
    if len(sub) == 0:
        continue
    cnt    = len(sub)
    pct    = cnt / n_t * 100
    wr_r   = (sub['pnl'] > 0).mean() * 100
    avg_r  = sub['pnl'].mean()
    tot_r  = sub['pnl'].sum()
    mfe_r  = sub['mfe'].mean()
    print(f"  {reason:<12} {cnt:>5} {pct:>4.1f}% {wr_r:>5.1f}% "
          f"{avg_r:>+6.2f} {tot_r:>+8.1f} {mfe_r:>7.1f}")

# ── 4. SL ANALYSIS ────────────────────────────────────────────────────────────
print(f"\n[4] SL ANALYSIS")
sl_trades = tdf[tdf['reason'] == 'SL']
print(f"  Initial SL trades   : {len(sl_trades)} ({len(sl_trades)/n_t*100:.1f}% of trades)")
print(f"  Avg SL size (pts)   : AM={tdf[tdf['session']=='AM']['sl_size'].mean():.2f}  "
      f"PM={tdf[tdf['session']=='PM']['sl_size'].mean():.2f}  "
      f"All={tdf['sl_size'].mean():.2f}")
if len(sl_trades) > 0:
    print(f"  Avg loss on SL      : {sl_trades['pnl'].mean():+.2f} pts")
    print(f"  Max loss on SL      : {sl_trades['pnl'].min():+.2f} pts")
    print(f"  Median bars to SL   : {sl_trades['bars'].median():.1f}")
    print(f"  Avg MFE before SL   : {sl_trades['mfe'].mean():.2f} pts")

# All SL-type exits (SL + BE + TRAIL) — total stopped-out count
all_stops = tdf[tdf['reason'].isin(['SL', 'BE', 'TRAIL'])]
print(f"  All stop exits (SL+BE+TRAIL): {len(all_stops)} "
      f"({len(all_stops)/n_t*100:.1f}%)")

# ── 5. MFE DISTRIBUTION ───────────────────────────────────────────────────────
print(f"\n[5] MFE DISTRIBUTION")
print(f"  {'Bucket':>7} {'Count':>6} {'WR':>6} {'AvgPnL':>7} {'AvgWin':>7} {'AvgLoss':>8}")
print("  " + SEP2)
mfe_buckets = [(0, 2, '0-2'), (2, 4, '2-4'), (4, 6, '4-6'),
               (6, 9, '6-9'), (9, 999, '9+')]
for lo, hi, lbl in mfe_buckets:
    sub = tdf[(tdf['mfe'] >= lo) & (tdf['mfe'] < hi)]
    if len(sub) == 0:
        continue
    wr_b   = (sub['pnl'] > 0).mean() * 100
    avg_b  = sub['pnl'].mean()
    wins_b = sub[sub['pnl'] > 0]['pnl'].mean() if (sub['pnl'] > 0).any() else 0.0
    loss_b = sub[sub['pnl'] <= 0]['pnl'].mean() if (sub['pnl'] <= 0).any() else 0.0
    print(f"  MFE {lbl:>5}: {len(sub):>6} {wr_b:>5.1f}% {avg_b:>+6.2f} "
          f"{wins_b:>+6.2f} {loss_b:>+7.2f}")

# ── 6. DRAWDOWN ───────────────────────────────────────────────────────────────
print(f"\n[6] DRAWDOWN ANALYSIS")
cum_pnl  = tdf['pnl'].cumsum()
peak     = cum_pnl.cummax()
max_dd   = (cum_pnl - peak).min()

streak = 0
max_streak = 0
for v in tdf['pnl']:
    if v <= 0:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        streak = 0

daily_pnl = tdf.groupby('date')['pnl'].sum()
pos_days  = (daily_pnl > 0).sum()
neg_days  = (daily_pnl <= 0).sum()
zero_days = n_days - len(daily_pnl)

print(f"  Max drawdown        : {max_dd:+.1f} pts")
print(f"  Max consecutive losses: {max_streak}")
print(f"  Positive days       : {pos_days} | Negative: {neg_days} | "
      f"No-trade: {zero_days}")
print(f"  Best day            : {daily_pnl.max():+.1f} pts | "
      f"Worst day: {daily_pnl.min():+.1f} pts")
print(f"  Avg P/D (trade days): {daily_pnl.mean():+.2f} pts")

# ── 7. MONTHLY P/D ────────────────────────────────────────────────────────────
print(f"\n[7] MONTHLY P/D BREAKDOWN")
tdf['month']    = pd.to_datetime(tdf['date'].astype(str)).dt.to_period('M')
df_m            = df.copy()
df_m['month']   = pd.to_datetime(df_m['date'].astype(str)).dt.to_period('M')
print(f"  {'Month':<8} {'T':>4} {'WR':>6} {'PF':>5} {'PnL':>8} {'Days':>5} {'P/D':>6}")
print("  " + SEP2)
all_months = sorted(tdf['month'].unique())
for month in all_months:
    sub    = tdf[tdf['month'] == month]
    m_days = df_m[df_m['month'] == month]['date'].nunique()
    wr_m   = (sub['pnl'] > 0).mean() * 100
    pf_m   = pf_calc(sub)
    pnl_m  = sub['pnl'].sum()
    ppd_m  = pnl_m / m_days if m_days > 0 else 0.0
    print(f"  {str(month):<8} {len(sub):>4} {wr_m:>5.1f}% {pf_m:>5.2f} "
          f"{pnl_m:>+8.1f} {m_days:>5} {ppd_m:>+5.2f}")

print(f"\n{SEP}")
print("DONE")
print(SEP)
