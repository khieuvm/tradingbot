"""
NR4 SHORT Time Analysis & Filter Diagnosis
==========================================
Analyzes time-slot performance, entry method differences, and filters
to diagnose why NR4 SHORT is failing in recent months.

Baseline context:
  Overall:  WR 43.4%, PF 1.26, P/D +0.41 / 130d
  May-2026: WR 36.6%, P/D -0.48
  Jun-2026: WR 30.8%, P/D -2.91
  Prior claim: WR 56.6% — suspected look-ahead via close-direction entry
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

COST      = 0.96
BE_ATR_MIN = 3.5

# ============================================================
# 1. DATA LOADING
# ============================================================
fetcher = DataFetcher()
end   = datetime.now().strftime("%Y-%m-%d")
start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
print(f"Loading 5m data {start} -> {end} ...")

df = fetcher.get_futures_ohlcv("VN30F1M", start, end, interval="5m")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df['date'] = df['time'].dt.date
df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute

# Session filter: AM 09:00-11:30, PM 13:00-14:30
df = df[((df['mins'] >= 9*60)  & (df['mins'] < 11*60+30)) |
        ((df['mins'] >= 13*60) & (df['mins'] < 14*60+30))]
df = df.reset_index(drop=True)  # critical: iloc == index after this

df['atr']      = ta.atr(df['high'], df['low'], df['close'], length=14)
df['rsi14']    = ta.rsi(df['close'], length=14)
df['ema20']    = ta.ema(df['close'], length=20)
df['session']  = np.where(df['mins'] < 12*60, 'AM', 'PM')
df['range']    = df['high'] - df['low']
df['is_green'] = (df['close'] > df['open'])
df['ema20_slope'] = df['ema20'].diff(3)  # 15-min slope
df['net3']     = df['close'] - df['close'].shift(3)  # 3-bar net move
df['prior_green'] = df['is_green'].shift(1).fillna(False).astype(bool)

n_days = df['date'].nunique()
print(f"  {len(df):,} bars | {n_days} trading days")

# ============================================================
# 2. NR4 SIGNAL DETECTION (bulk vectorised)
# ============================================================
def detect_nr4_bulk(df):
    n       = len(df)
    nr4     = np.zeros(n, dtype=bool)
    ranges  = df['range'].values
    atr_v   = df['atr'].values
    for i in range(3, n):
        av = atr_v[i]
        if np.isnan(av) or av <= 0:
            continue
        cur = ranges[i]
        prev_min = min(ranges[i-3], ranges[i-2], ranges[i-1])
        if cur < prev_min and cur < 0.7 * av:
            nr4[i] = True
    return nr4


def dedup_mask(mask, min_bars):
    r    = mask.copy()
    last = -999
    for i in range(len(r)):
        if r.iloc[i]:
            if i - last < min_bars:
                r.iloc[i] = False
            else:
                last = i
    return r


df['nr4']    = detect_nr4_bulk(df)
filt_base    = (df['atr'] >= 2.5) & (df['atr'] <= 4.5)
in_window    = (((df['mins'] >= 9*60+15)  & (df['mins'] <= 10*60+45)) |
                ((df['mins'] >= 13*60+15) & (df['mins'] <= 14*60+15)))

sig_raw  = df['nr4'] & filt_base & in_window
sig_mask = dedup_mask(sig_raw, 5)

print(f"\nNR4 signals: {int(sig_raw.sum())} raw  ->  {int(sig_mask.sum())} after dedup(5 bars)")


# ============================================================
# 3. SIMULATION CORE
# ============================================================
def simulate_short(df, positions, use_trigger=True):
    """
    positions: list of int iloc positions of signal bars.
    use_trigger=True  → entry only if next 2 bars break low-0.1 (live-deployable).
    use_trigger=False → always enter SHORT at signal bar close (no trigger needed).
    """
    trades = []

    for pos in positions:
        sig   = df.iloc[pos]
        atr   = sig['atr']
        sess  = sig['session']

        if np.isnan(atr) or atr <= 0:
            continue

        # Session params
        if sess == 'AM':
            sl_mult      = 1.2
            trail_act    = 5.0
            trail_mult   = 2.0
            max_hold     = 24
            exit_mins_h  = 11*60 + 25
        else:
            sl_mult      = 1.0
            trail_act    = 4.0
            trail_mult   = 1.5
            max_hold     = 12
            exit_mins_h  = 14*60 + 25

        # ----- entry resolution -----
        if use_trigger:
            trigger = sig['low'] - 0.1
            entry   = None
            ebar    = None
            for k in range(pos + 1, min(pos + 3, len(df))):
                tb = df.iloc[k]
                if tb['date'] != sig['date'] or tb['session'] != sig['session']:
                    break
                if tb['low'] <= trigger:
                    entry = trigger
                    ebar  = k
                    break
            if entry is None:
                continue  # trigger not hit
        else:
            # close-direction: always SHORT from signal bar close
            if pos + 1 >= len(df):
                continue
            nxt = df.iloc[pos + 1]
            if nxt['date'] != sig['date'] or nxt['session'] != sig['session']:
                continue
            entry = sig['close']
            ebar  = pos + 1

        # pre-move ratio (bearish context indicator)
        if pos >= 3:
            pre_move_ratio = (df.iloc[pos-1]['close'] - df.iloc[pos-3]['open']) * (-1) / atr
        else:
            pre_move_ratio = 0.0

        # ----- position management -----
        sl         = entry + sl_mult * atr   # SL above entry for SHORT
        best       = entry                   # tracks lowest price (MFE for SHORT)
        trail_on   = False
        be_done    = False
        adapt_done = False
        exit_p     = None
        exit_r     = None
        bars       = 0
        end_pos    = min(ebar + max_hold, len(df))

        for jp in range(ebar, end_pos):
            b = df.iloc[jp]

            # session boundary
            if b['date'] != sig['date'] or b['session'] != sig['session']:
                exit_p = df.iloc[jp-1]['close']
                exit_r = 'SESSION'
                break

            # hard time exit
            if b['mins'] >= exit_mins_h:
                exit_p = b['close']
                exit_r = 'SESSION'
                break

            bars += 1

            # SL check
            if b['high'] >= sl:
                exit_p = sl
                exit_r = 'TRAIL' if trail_on else ('BE' if be_done else 'SL')
                break

            # update MFE
            if b['low'] < best:
                best = b['low']
            mfe = entry - best   # positive when price dropped

            # adaptive exit (AM + bearish pre-move + MFE >= 4)
            if (sess == 'AM' and pre_move_ratio > 0.8
                    and mfe >= 4.0 and not adapt_done):
                pnl_pts = entry - b['close']
                if pnl_pts >= 3.2:
                    exit_p = b['close']
                    exit_r = 'ADAPT_EXIT'
                    break
                else:
                    adapt_done = True
                    sl = min(sl, entry - 2.0)  # lock +2 pts

            # BE
            if not be_done and mfe >= 4.0 and atr >= BE_ATR_MIN:
                be_done = True
                sl      = min(sl, entry)

            # trail activation
            if mfe >= trail_act:
                trail_on = True

            # trail SL
            if trail_on:
                eff_mult = 1.2 if (mfe >= 8.0 and atr >= BE_ATR_MIN) else trail_mult
                t_sl     = best + eff_mult * atr
                sl       = min(sl, t_sl)
                if b['high'] >= sl:
                    exit_p = sl
                    exit_r = 'TRAIL'
                    break

        if exit_p is None:
            last = df.iloc[min(ebar + max_hold - 1, len(df)-1)]
            exit_p = last['close']
            exit_r = 'MAX_HOLD'

        mfe_f = entry - best
        pnl   = entry - exit_p - COST   # SHORT profit formula

        trades.append({
            'date' : sig['date'],
            'time' : sig['time'].strftime('%H:%M'),
            'sess' : sess,
            'mins' : int(sig['mins']),
            'entry': entry,
            'exit' : exit_p,
            'atr'  : atr,
            'mfe'  : mfe_f,
            'pnl'  : pnl,
            'reason': exit_r,
            'bars' : bars,
        })

    if not trades:
        return None
    return (pd.DataFrame(trades)
              .sort_values(['date', 'time'])
              .reset_index(drop=True))


# ============================================================
# 4. HELPERS
# ============================================================
def pf(sub):
    wins = sub[sub['pnl'] > 0]['pnl'].sum()
    loss = abs(sub[sub['pnl'] <= 0]['pnl'].sum())
    return wins / loss if loss > 0 else 999.0


def row_stats(label, tdf, n_d, indent='  '):
    if tdf is None or len(tdf) == 0:
        print(f"{indent}{label:<50} | NO TRADES")
        return
    n   = len(tdf)
    wr  = (tdf['pnl'] > 0).mean() * 100
    pf_ = pf(tdf)
    tot = tdf['pnl'].sum()
    ppd = tot / n_d
    sl_ = int((tdf['reason'] == 'SL').sum())
    print(f"{indent}{label:<50} | T={n:>3} WR={wr:>5.1f}% PF={pf_:>5.2f} "
          f"PnL={tot:>+7.1f} P/D={ppd:>+5.2f} SL={sl_}")


def get_pos(mask):
    """Convert boolean mask to list of iloc positions (works because df is reset_index)."""
    return list(df.index[mask])


# ============================================================
# 5.  BASELINE SIMULATION
# ============================================================
pos_base = get_pos(sig_mask)
tdf_base = simulate_short(df, pos_base, use_trigger=True)

print()
print("=" * 80)
print("NR4 SHORT TIME ANALYSIS  |  VN30F1M 5m  |  SHORT-ONLY TRIGGER-BASED")
print("=" * 80)

if tdf_base is not None:
    n_t   = len(tdf_base)
    wr_t  = (tdf_base['pnl'] > 0).mean() * 100
    pf_t  = pf(tdf_base)
    tot_t = tdf_base['pnl'].sum()
    ppd_t = tot_t / n_days
    print(f"\nBASELINE: {sig_mask.sum()} signals -> {n_t} triggered "
          f"({100*n_t/sig_mask.sum():.0f}%) | WR {wr_t:.1f}% | PF {pf_t:.2f} | "
          f"P/D {ppd_t:+.2f} | {n_days}d")


# ============================================================
# 6.  SECTION 1 — 15-MIN TIME SLOT BREAKDOWN
# ============================================================
print(f"\n{'='*80}")
print("SECTION 1 — 15-MIN TIME SLOT BREAKDOWN")
print(f"{'='*80}")
print(f"  {'Slot':<24} {'T':>4} {'WR':>6} {'PF':>6} {'AvgPnL':>8} {'TotPnL':>9} {'AvgMFE':>8}")
print("  " + "-" * 68)

slots = [
    ('AM 09:15-09:30',  9*60+15,  9*60+30),
    ('AM 09:30-09:45',  9*60+30,  9*60+45),
    ('AM 09:45-10:00',  9*60+45, 10*60+0),
    ('AM 10:00-10:15', 10*60+0,  10*60+15),
    ('AM 10:15-10:30', 10*60+15, 10*60+30),
    ('AM 10:30-10:45', 10*60+30, 10*60+45),
    ('PM 13:15-13:30', 13*60+15, 13*60+30),
    ('PM 13:30-13:45', 13*60+30, 13*60+45),
    ('PM 13:45-14:00', 13*60+45, 14*60+0),
    ('PM 14:00-14:15', 14*60+0,  14*60+15),
]

if tdf_base is not None:
    for lbl, lo, hi in slots:
        sub = tdf_base[(tdf_base['mins'] >= lo) & (tdf_base['mins'] < hi)]
        if len(sub) == 0:
            print(f"  {lbl:<24}    0   ---    ---      ---       ---      ---")
            continue
        wr_s  = (sub['pnl'] > 0).mean() * 100
        pf_s  = pf(sub)
        avg_p = sub['pnl'].mean()
        tot_p = sub['pnl'].sum()
        mfe_  = sub['mfe'].mean()
        print(f"  {lbl:<24} {len(sub):>4} {wr_s:>5.1f}% {pf_s:>6.2f} "
              f"{avg_p:>+8.2f} {tot_p:>+9.1f} {mfe_:>8.2f}")

# Also aggregate AM / PM from slot table
if tdf_base is not None:
    print()
    for sess, lo_win, hi_win in [('AM', 9*60+15, 10*60+45), ('PM', 13*60+15, 14*60+15)]:
        sub = tdf_base[(tdf_base['mins'] >= lo_win) & (tdf_base['mins'] <= hi_win)
                       & (tdf_base['sess'] == sess)]
        if len(sub) == 0: continue
        wr_s = (sub['pnl'] > 0).mean() * 100
        pf_s = pf(sub)
        avg_p = sub['pnl'].mean()
        tot_p = sub['pnl'].sum()
        print(f"  TOTAL {sess:<3}: {len(sub):>4} trades | WR {wr_s:.1f}% | PF {pf_s:.2f} | "
              f"AvgPnL {avg_p:+.2f} | TotPnL {tot_p:+.1f}")


# ============================================================
# 7.  SECTION 2 — ENTRY METHOD: TRIGGER vs CLOSE-DIRECTION
# ============================================================
print(f"\n{'='*80}")
print("SECTION 2 — ENTRY METHOD COMPARISON")
print(f"{'='*80}")
print("  NOTE: close-direction enters at signal-bar close without waiting for breakout.")
print("        If it shows higher WR, it confirms look-ahead bias in prior research.\n")

# Close-direction: always SHORT from close (no trigger)
tdf_close = simulate_short(df, pos_base, use_trigger=False)

row_stats("A) Trigger-based  (low-0.1, expiry 2 bars)    [LIVE]", tdf_base,  n_days)
row_stats("B) Close-direction (always SHORT from close)   [BIASED?]", tdf_close, n_days)

# Also test: close-direction BUT only when next bar closes DOWN (pure look-ahead)
tdf_la_trades = []
for pos in pos_base:
    sig = df.iloc[pos]
    if pos + 1 >= len(df): continue
    nxt = df.iloc[pos + 1]
    if nxt['date'] != sig['date'] or nxt['session'] != sig['session']: continue
    if nxt['close'] >= sig['close']: continue  # skip if next closes UP (look-ahead filter)
    tdf_la_trades.append(pos)

tdf_la = simulate_short(df, tdf_la_trades, use_trigger=False)
row_stats("C) Look-ahead SHORT (next close < signal close) [BIASED]", tdf_la, n_days)

if tdf_base is not None and tdf_close is not None:
    n_tri = len(tdf_base)
    n_cls = len(tdf_close) if tdf_close is not None else 0
    n_la  = len(tdf_la) if tdf_la is not None else 0
    print(f"\n  Signal trigger rate : {n_tri} / {sig_mask.sum()} = {100*n_tri/sig_mask.sum():.0f}%")
    print(f"  Look-ahead filter   : {n_la} / {sig_mask.sum()} = {100*n_la/sig_mask.sum():.0f}% "
          f"(next-bar closed DOWN)")
    print(f"  Skipped (up close)  : {sig_mask.sum()-n_la} / {sig_mask.sum()} = "
          f"{100*(sig_mask.sum()-n_la)/sig_mask.sum():.0f}%")


# ============================================================
# 8.  SECTION 3 — ATR SUB-RANGE FILTER
# ============================================================
print(f"\n{'='*80}")
print("SECTION 3 — ATR SUB-RANGE FILTER (trigger-based)")
print(f"{'='*80}")

atr_ranges = [
    ('ATR 2.5-3.0', 2.5, 3.0),
    ('ATR 3.0-3.5', 3.0, 3.5),
    ('ATR 3.5-4.0', 3.5, 4.0),
    ('ATR 4.0-4.5', 4.0, 4.5),
    ('ATR 2.5-3.5 (low-vol)', 2.5, 3.5),
    ('ATR 3.5-4.5 (high-vol)', 3.5, 4.5),
    ('ATR 3.0-4.0 (mid)',     3.0, 4.0),
]

for lbl, alo, ahi in atr_ranges:
    filt_ = (df['atr'] >= alo) & (df['atr'] < ahi)
    m_ = dedup_mask(df['nr4'] & filt_ & in_window, 5)
    t_ = simulate_short(df, get_pos(m_), use_trigger=True)
    row_stats(f"SHORT trigger {lbl}", t_, n_days)


# ============================================================
# 9.  SECTION 4 — SESSION FILTER
# ============================================================
print(f"\n{'='*80}")
print("SECTION 4 — SESSION FILTER (trigger-based)")
print(f"{'='*80}")

in_am   = (df['mins'] >= 9*60+15)  & (df['mins'] <= 10*60+45)
in_pm   = (df['mins'] >= 13*60+15) & (df['mins'] <= 14*60+15)
in_eam  = (df['mins'] >= 9*60+15)  & (df['mins'] < 10*60)
in_lam  = (df['mins'] >= 10*60)    & (df['mins'] <= 10*60+45)

for lbl, win in [
    ('AM only (09:15-10:45)',   in_am),
    ('PM only (13:15-14:15)',   in_pm),
    ('Early AM (09:15-10:00)',  in_eam),
    ('Late  AM (10:00-10:45)',  in_lam),
    ('Full window (AM+PM)',     in_window),
]:
    m_ = dedup_mask(df['nr4'] & filt_base & win, 5)
    t_ = simulate_short(df, get_pos(m_), use_trigger=True)
    row_stats(lbl, t_, n_days)


# ============================================================
# 10. SECTION 5 — RSI FILTER (RSI > threshold for SHORT)
# ============================================================
print(f"\n{'='*80}")
print("SECTION 5 — RSI FILTER (trigger-based, RSI > threshold = overbought)")
print(f"{'='*80}")

for rsi_th in [50, 55, 60, 65, 70]:
    m_ = dedup_mask(df['nr4'] & filt_base & in_window & (df['rsi14'] > rsi_th), 5)
    t_ = simulate_short(df, get_pos(m_), use_trigger=True)
    row_stats(f"SHORT + RSI > {rsi_th}", t_, n_days)

# No RSI filter
m_no_rsi = dedup_mask(df['nr4'] & filt_base & in_window, 5)
row_stats("SHORT + RSI = any (no filter)", simulate_short(df, get_pos(m_no_rsi), use_trigger=True), n_days)

# RSI < 40 (oversold — counter-intuitive SHORT)
m_rsi_low = dedup_mask(df['nr4'] & filt_base & in_window & (df['rsi14'] < 40), 5)
row_stats("SHORT + RSI < 40 (oversold)",  simulate_short(df, get_pos(m_rsi_low), use_trigger=True), n_days)


# ============================================================
# 11. SECTION 6 — PRIOR BAR GREEN / RED FILTER
# ============================================================
print(f"\n{'='*80}")
print("SECTION 6 — CANDLE COLOR FILTER (trigger-based)")
print(f"{'='*80}")
print("  Logic: NR4 after GREEN bar = compression after up-move = bearish reversal setup")
print("         NR4 after RED  bar = compression after down-move = continuation short\n")

# Prior bar color
m_pg = dedup_mask(df['nr4'] & filt_base & in_window & df['prior_green'], 5)
m_pr = dedup_mask(df['nr4'] & filt_base & in_window & ~df['prior_green'], 5)

row_stats("SHORT + prior bar GREEN (bearish reversal)", simulate_short(df, get_pos(m_pg), use_trigger=True), n_days)
row_stats("SHORT + prior bar RED   (continuation)",     simulate_short(df, get_pos(m_pr), use_trigger=True), n_days)

# NR4 bar itself
m_bg = dedup_mask(df['nr4'] & filt_base & in_window & df['is_green'],  5)
m_br = dedup_mask(df['nr4'] & filt_base & in_window & ~df['is_green'], 5)

row_stats("SHORT + NR4 bar GREEN (doji-up, possible reversal)", simulate_short(df, get_pos(m_bg), use_trigger=True), n_days)
row_stats("SHORT + NR4 bar RED   (doji-down, continuation?)",   simulate_short(df, get_pos(m_br), use_trigger=True), n_days)

# Combined: prior GREEN + NR4 bar GREEN (double-up setup)
m_both_g = dedup_mask(df['nr4'] & filt_base & in_window & df['prior_green'] & df['is_green'], 5)
row_stats("SHORT + prior GREEN + NR4 bar GREEN",
          simulate_short(df, get_pos(m_both_g), use_trigger=True), n_days)


# ============================================================
# 12. SECTION 7 — EMA / TREND CONTEXT FILTERS
# ============================================================
print(f"\n{'='*80}")
print("SECTION 7 — TREND / CONTEXT FILTERS (trigger-based)")
print(f"{'='*80}")
print("  Hypothesis: NR4 SHORT should work better in down-trend context\n")

# A: EMA20 slope negative (bearish momentum)
m_ema_dn = dedup_mask(df['nr4'] & filt_base & in_window & (df['ema20_slope'] < 0), 5)
row_stats("SHORT + EMA20 slope < 0 (bearish trend)",    simulate_short(df, get_pos(m_ema_dn), use_trigger=True), n_days)

# B: close < EMA20 (price below trend)
m_below  = dedup_mask(df['nr4'] & filt_base & in_window & (df['close'] < df['ema20']), 5)
row_stats("SHORT + close < EMA20 (below trend)",         simulate_short(df, get_pos(m_below),  use_trigger=True), n_days)

# C: close > EMA20 (SHORT after rally into resistance)
m_above  = dedup_mask(df['nr4'] & filt_base & in_window & (df['close'] > df['ema20']), 5)
row_stats("SHORT + close > EMA20 (above trend = rally)",  simulate_short(df, get_pos(m_above),  use_trigger=True), n_days)

# D: prior 3-bar net negative (recent drop context)
m_net_neg = dedup_mask(df['nr4'] & filt_base & in_window & (df['net3'] < 0), 5)
row_stats("SHORT + prior 3 bars net negative",            simulate_short(df, get_pos(m_net_neg), use_trigger=True), n_days)

# E: prior 3-bar net positive (recent rally, SHORT after bounce)
m_net_pos = dedup_mask(df['nr4'] & filt_base & in_window & (df['net3'] > 0), 5)
row_stats("SHORT + prior 3 bars net positive (after bounce)", simulate_short(df, get_pos(m_net_pos), use_trigger=True), n_days)

# F: EMA slope down + prior GREEN (bearish context + overbought candle)
m_combo_f = dedup_mask(df['nr4'] & filt_base & in_window
                        & (df['ema20_slope'] < 0) & df['prior_green'], 5)
row_stats("SHORT + EMA slope down + prior GREEN",
          simulate_short(df, get_pos(m_combo_f), use_trigger=True), n_days)

# G: above EMA20 + prior GREEN (rally setup)
m_combo_g = dedup_mask(df['nr4'] & filt_base & in_window
                        & (df['close'] > df['ema20']) & df['prior_green'], 5)
row_stats("SHORT + above EMA20 + prior GREEN",
          simulate_short(df, get_pos(m_combo_g), use_trigger=True), n_days)

# H: above EMA20 + RSI > 60 (overbought short squeeze)
m_combo_h = dedup_mask(df['nr4'] & filt_base & in_window
                        & (df['close'] > df['ema20']) & (df['rsi14'] > 60), 5)
row_stats("SHORT + above EMA20 + RSI > 60",
          simulate_short(df, get_pos(m_combo_h), use_trigger=True), n_days)


# ============================================================
# 13. SECTION 8 — BEST VARIANT RANKING
# ============================================================
print(f"\n{'='*80}")
print("SECTION 8 — ALL VARIANTS RANKED BY P/D  (min 20 trades)")
print(f"{'='*80}")

all_variants = {}

def reg(name, mask_fn, trigger=True):
    m = mask_fn()
    t = simulate_short(df, get_pos(m), use_trigger=trigger)
    if t is not None and len(t) >= 20:
        all_variants[name] = {
            'tdf': t,
            'n'  : len(t),
            'wr' : (t['pnl'] > 0).mean() * 100,
            'pf' : pf(t),
            'ppd': t['pnl'].sum() / n_days,
        }

# Register all variants
reg("Baseline trigger (AM+PM)",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window, 5))
reg("Close-direction (no trigger)",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window, 5), trigger=False)
reg("AM only",
    lambda: dedup_mask(df['nr4'] & filt_base & in_am, 5))
reg("PM only",
    lambda: dedup_mask(df['nr4'] & filt_base & in_pm, 5))
reg("Early AM (09:15-10:00)",
    lambda: dedup_mask(df['nr4'] & filt_base & in_eam, 5))
reg("Late AM (10:00-10:45)",
    lambda: dedup_mask(df['nr4'] & filt_base & in_lam, 5))
reg("RSI > 55",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window & (df['rsi14'] > 55), 5))
reg("RSI > 60",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window & (df['rsi14'] > 60), 5))
reg("Prior GREEN",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window & df['prior_green'], 5))
reg("Prior RED",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window & ~df['prior_green'], 5))
reg("EMA slope down",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window & (df['ema20_slope'] < 0), 5))
reg("Close < EMA20",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window & (df['close'] < df['ema20']), 5))
reg("Close > EMA20",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window & (df['close'] > df['ema20']), 5))
reg("Net3 negative",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window & (df['net3'] < 0), 5))
reg("Net3 positive",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window & (df['net3'] > 0), 5))
reg("EMA down + prior GREEN",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window
                       & (df['ema20_slope'] < 0) & df['prior_green'], 5))
reg("Above EMA20 + prior GREEN",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window
                       & (df['close'] > df['ema20']) & df['prior_green'], 5))
reg("Above EMA20 + RSI > 60",
    lambda: dedup_mask(df['nr4'] & filt_base & in_window
                       & (df['close'] > df['ema20']) & (df['rsi14'] > 60), 5))
reg("ATR 2.5-3.5",
    lambda: dedup_mask(df['nr4'] & (df['atr'] >= 2.5) & (df['atr'] < 3.5) & in_window, 5))
reg("ATR 3.5-4.5",
    lambda: dedup_mask(df['nr4'] & (df['atr'] >= 3.5) & (df['atr'] < 4.5) & in_window, 5))
reg("ATR 3.0-4.0",
    lambda: dedup_mask(df['nr4'] & (df['atr'] >= 3.0) & (df['atr'] < 4.0) & in_window, 5))
reg("AM + RSI > 60",
    lambda: dedup_mask(df['nr4'] & filt_base & in_am & (df['rsi14'] > 60), 5))
reg("AM + prior GREEN",
    lambda: dedup_mask(df['nr4'] & filt_base & in_am & df['prior_green'], 5))
reg("AM + above EMA20",
    lambda: dedup_mask(df['nr4'] & filt_base & in_am & (df['close'] > df['ema20']), 5))
reg("AM + above EMA20 + RSI > 60",
    lambda: dedup_mask(df['nr4'] & filt_base & in_am
                       & (df['close'] > df['ema20']) & (df['rsi14'] > 60), 5))
reg("AM + EMA slope down + prior GREEN",
    lambda: dedup_mask(df['nr4'] & filt_base & in_am
                       & (df['ema20_slope'] < 0) & df['prior_green'], 5))

# Sort and print
sorted_v = sorted(all_variants.items(), key=lambda x: x[1]['ppd'], reverse=True)
print(f"  {'Variant':<44} {'T':>4} {'WR':>6} {'PF':>6} {'P/D':>7}")
print("  " + "-" * 72)
for nm, v in sorted_v:
    print(f"  {nm:<44} {v['n']:>4} {v['wr']:>5.1f}% {v['pf']:>6.2f} {v['ppd']:>+7.2f}")


# ============================================================
# 14. SECTION 9 — MONTHLY BREAKDOWN FOR BEST VARIANT
# ============================================================
best_nm, best_v = sorted_v[0]
best_tdf        = best_v['tdf']
print(f"\n{'='*80}")
print(f"SECTION 9 — MONTHLY BREAKDOWN: Best = '{best_nm}'")
print(f"{'='*80}")
print(f"  Overall: {best_v['n']} trades | WR {best_v['wr']:.1f}% | "
      f"PF {best_v['pf']:.2f} | P/D {best_v['ppd']:+.2f}\n")

best_tdf['month'] = pd.to_datetime(best_tdf['date']).dt.to_period('M')
df['month_p']     = pd.to_datetime(df['date']).dt.to_period('M')

print(f"  {'Month':<10} {'T':>4} {'WR':>6} {'PF':>6} {'PnL':>8} {'Days':>5} {'P/D':>7}")
print("  " + "-" * 55)
for m in sorted(best_tdf['month'].unique()):
    sub    = best_tdf[best_tdf['month'] == m]
    m_days = df[df['month_p'] == m]['date'].nunique()
    wr_m   = (sub['pnl'] > 0).mean() * 100
    pf_m   = pf(sub)
    pnl_m  = sub['pnl'].sum()
    ppd_m  = pnl_m / m_days if m_days > 0 else 0
    print(f"  {str(m):<10} {len(sub):>4} {wr_m:>5.1f}% {pf_m:>6.2f} "
          f"{pnl_m:>+8.1f} {m_days:>5} {ppd_m:>+7.2f}")

# Exit breakdown for best
print(f"\n  --- Exit breakdown ---")
print(f"  {'Reason':<12} {'Cnt':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>9} {'AvgMFE':>8}")
print("  " + "-" * 52)
for r in ['SL', 'BE', 'TRAIL', 'SESSION', 'MAX_HOLD', 'ADAPT_EXIT']:
    sub = best_tdf[best_tdf['reason'] == r]
    if len(sub) == 0:
        continue
    wr_r = (sub['pnl'] > 0).mean() * 100
    print(f"  {r:<12} {len(sub):>4} {wr_r:>5.1f}% "
          f"{sub['pnl'].mean():>+8.2f} {sub['pnl'].sum():>+9.1f} "
          f"{sub['mfe'].mean():>8.2f}")

# AM/PM breakdown for best
print(f"\n  --- AM vs PM ---")
print(f"  {'Sess':<6} {'T':>4} {'WR':>6} {'PF':>6} {'PnL':>8} {'P/D':>7}")
print("  " + "-" * 42)
for s in ['AM', 'PM']:
    sub = best_tdf[best_tdf['sess'] == s]
    if len(sub) == 0: continue
    wr_s  = (sub['pnl'] > 0).mean() * 100
    pf_s  = pf(sub)
    pnl_s = sub['pnl'].sum()
    ppd_s = pnl_s / n_days
    print(f"  {s:<6} {len(sub):>4} {wr_s:>5.1f}% {pf_s:>6.2f} {pnl_s:>+8.1f} {ppd_s:>+7.2f}")

# MFE distribution for best
print(f"\n  --- MFE distribution ---")
print(f"  {'Bucket':>6} {'Cnt':>4} {'WR':>6} {'AvgPnL':>8} {'AvgMFE':>8}")
print("  " + "-" * 40)
for lo_, hi_, lbl_ in [(0,2,'0-2'),(2,4,'2-4'),(4,6,'4-6'),(6,9,'6-9'),(9,999,'9+')]:
    sub = best_tdf[(best_tdf['mfe'] >= lo_) & (best_tdf['mfe'] < hi_)]
    if len(sub) == 0: continue
    wr_b  = (sub['pnl'] > 0).mean() * 100
    print(f"  {lbl_:>6} {len(sub):>4} {wr_b:>5.1f}% "
          f"{sub['pnl'].mean():>+8.2f} {sub['mfe'].mean():>8.2f}")


# ============================================================
# 15. ALSO SHOW MONTHLY FOR BASELINE (to confirm degradation)
# ============================================================
print(f"\n{'='*80}")
print("SECTION 10 — MONTHLY BREAKDOWN: BASELINE (trigger, no filter)")
print(f"{'='*80}")
if tdf_base is not None:
    tdf_base['month'] = pd.to_datetime(tdf_base['date']).dt.to_period('M')
    print(f"  {'Month':<10} {'T':>4} {'WR':>6} {'PF':>6} {'PnL':>8} {'Days':>5} {'P/D':>7}")
    print("  " + "-" * 55)
    for m in sorted(tdf_base['month'].unique()):
        sub    = tdf_base[tdf_base['month'] == m]
        m_days = df[df['month_p'] == m]['date'].nunique()
        wr_m   = (sub['pnl'] > 0).mean() * 100
        pf_m   = pf(sub)
        pnl_m  = sub['pnl'].sum()
        ppd_m  = pnl_m / m_days if m_days > 0 else 0
        print(f"  {str(m):<10} {len(sub):>4} {wr_m:>5.1f}% {pf_m:>6.2f} "
              f"{pnl_m:>+8.1f} {m_days:>5} {ppd_m:>+7.2f}")


# ============================================================
# 16. FINAL VERDICT
# ============================================================
print(f"\n{'='*80}")
print("FINAL VERDICT")
print(f"{'='*80}")
print(f"  CB baseline   : WR 67.7% | PF 4.87 | +6.34/d  (223 trades / 129d)")
if tdf_base is not None:
    ppd_b_ = tdf_base['pnl'].sum() / n_days
    wr_b_  = (tdf_base['pnl'] > 0).mean() * 100
    print(f"  NR4 baseline  : WR {wr_b_:.1f}% | PF {pf(tdf_base):.2f} | "
          f"{ppd_b_:+.2f}/d  ({len(tdf_base)} trades / {n_days}d)")
if sorted_v:
    best_nm, best_v = sorted_v[0]
    print(f"  NR4 BEST var  : WR {best_v['wr']:.1f}% | PF {best_v['pf']:.2f} | "
          f"{best_v['ppd']:+.2f}/d  ({best_v['n']} trades / {n_days}d) [{best_nm}]")

    second_best = [x for x in sorted_v if x[1]['ppd'] > 0]
    print(f"\n  Positive P/D variants  : {len(second_best)} / {len(sorted_v)}")
    print(f"\n  RECOMMENDATION:")
    if best_v['ppd'] >= 2.0 and best_v['pf'] >= 1.5:
        print(f"  KEEP WITH MODIFICATION: apply '{best_nm}' filter, P/D = {best_v['ppd']:+.2f}")
    elif best_v['ppd'] >= 0.5:
        print(f"  MARGINAL: best P/D = {best_v['ppd']:+.2f} — not worth deploying alongside CB (+6.34/d)")
    else:
        print(f"  REMOVE: no variant achieves meaningful positive P/D. NR4 SHORT adds noise, not alpha.")
print()
