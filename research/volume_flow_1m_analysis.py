"""
Volume Flow Analysis — VN30F1M 1m Bars
========================================
Analyzes large-volume bars on 1m data to detect institutional money flow patterns.

Sections:
  1. Volume Bar Profiling
  2. What Happens After a Large Volume Bar?
  3. Volume Cluster Detection
  4. Volume Breakout Pattern (quiet → burst)
  5. Money Flow Index (MFI) Analysis
  6. Large Volume at Key Levels (EMA, VWAP)
  7. Practical Signal Design & Backtest
  8. Time Decay of Volume Signal

Run from e:/Trading/:
    python research/volume_flow_1m_analysis.py
"""

import sys; sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy import stats
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("e:/Trading/data")
COST       = 0.96
HV_MULT    = 2.0   # High Volume threshold (× SMA20)
XV_MULT    = 3.0   # Extreme Volume threshold (× SMA20)
QUIET_MULT = 0.7   # Quiet bar threshold (× SMA20)
VOL_SMA    = 20    # Volume SMA window
HOLIDAY_CUTOFF = pd.Timestamp("2026-05-02")  # Filter known data issue

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sep(title="", width=72):
    if title:
        pad = (width - len(title) - 2) // 2
        print("═" * pad + f" {title} " + "═" * (width - pad - len(title) - 2))
    else:
        print("─" * width)


def load_1m():
    df = pd.read_parquet(DATA_DIR / "vn30f1m_1m.parquet")
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = df[df['time'] < HOLIDAY_CUTOFF].copy()
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['hour_min'] = df['time'].dt.strftime('%H:%M')
    # Session labels
    am = (df['mins'] >= 9*60) & (df['mins'] < 11*60+30)
    pm = (df['mins'] >= 13*60) & (df['mins'] < 14*60+30)
    df = df[am | pm].copy().reset_index(drop=True)
    df['session'] = np.where(am[am | pm], 'AM', 'PM')
    # Price metrics
    df['body']      = (df['close'] - df['open']).abs()
    df['range']     = df['high'] - df['low']
    df['bullish']   = df['close'] >= df['open']
    df['typ_price'] = (df['high'] + df['low'] + df['close']) / 3
    # Volume metrics (rolling within full series — do NOT split by date first)
    df['vol_sma20'] = df['volume'].rolling(VOL_SMA, min_periods=5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20'].replace(0, np.nan)
    df['is_hv']     = df['vol_ratio'] >= HV_MULT
    df['is_xv']     = df['vol_ratio'] >= XV_MULT
    df['is_quiet']  = df['vol_ratio'] < QUIET_MULT
    # RSI for MFI computation
    return df


def load_5m():
    df = pd.read_parquet(DATA_DIR / "vn30f1m_5m.parquet")
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = df[df['time'] < HOLIDAY_CUTOFF].copy()
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    am = (df['mins'] >= 9*60) & (df['mins'] < 11*60+30)
    pm = (df['mins'] >= 13*60) & (df['mins'] < 14*60+30)
    df = df[am | pm].copy().reset_index(drop=True)
    df['atr14']     = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema20']     = ta.ema(df['close'], length=20)
    df['ema50']     = ta.ema(df['close'], length=50)
    return df


def pf(wins, losses, epsilon=1e-9):
    total_win  = sum(w for w in wins  if w > 0)
    total_loss = sum(abs(l) for l in losses if l < 0)
    return total_win / (total_loss + epsilon)


def mfe_mae(df, signal_idx, horizon, direction):
    """
    For each signal bar index, compute MFE and MAE over `horizon` subsequent bars.
    direction: +1 = long, -1 = short
    Returns arrays of MFE, MAE, final_ret (close[signal+horizon] - close[signal]).
    """
    mfe_list, mae_list, final_list = [], [], []
    entry_close = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    n = len(df)

    for i in signal_idx:
        if i + horizon >= n:
            continue
        entry = entry_close[i]
        h_slice = highs[i+1:i+1+horizon]
        l_slice = lows[i+1:i+1+horizon]
        c_end   = closes[i+horizon]
        if direction == 1:
            mfe = (h_slice.max() - entry)
            mae = (entry - l_slice.min())
        else:
            mfe = (entry - l_slice.min())
            mae = (h_slice.max() - entry)
        final = (c_end - entry) * direction
        mfe_list.append(mfe)
        mae_list.append(mae)
        final_list.append(final)

    return np.array(mfe_list), np.array(mae_list), np.array(final_list)


def print_forward_stats(label, df, idx_arr, direction, horizons=(1,3,5,10,15,20)):
    if len(idx_arr) < 10:
        print(f"  {label}: insufficient data ({len(idx_arr)} signals)")
        return
    print(f"\n  {label}  (n={len(idx_arr)}, dir={'LONG' if direction==1 else 'SHORT'})")
    for h in horizons:
        mfe, mae, final = mfe_mae(df, idx_arr, h, direction)
        if len(final) == 0:
            continue
        hit = (final > 0).mean() * 100
        print(f"    H={h:2d}:  avg_ret={final.mean():+.2f}  hit={hit:.0f}%  "
              f"MFE={mfe.mean():.2f}  MAE={mae.mean():.2f}  "
              f"MFE>=5={( mfe>=5).mean()*100:.0f}%  MFE>=8={(mfe>=8).mean()*100:.0f}%")


def trade_backtest(df, signal_df, direction_col, atr5m_map,
                   sl_atr_mult=2.0, trail_pts=3.0, trail_atr=1.0,
                   max_hold=30, cost=COST):
    """
    Simple backtest on 1m bars.
    signal_df: DataFrame with columns [bar_idx (int into df), direction (+1/-1), signal_name]
    Returns list of trade dicts.
    """
    trades = []
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    dates  = df['date'].values
    mins   = df['mins'].values
    n = len(df)

    for _, row in signal_df.iterrows():
        i        = int(row['bar_idx'])
        direction = int(row['direction'])
        signal   = row.get('signal', 'unknown')
        if i + 1 >= n:
            continue
        entry_i  = i + 1  # enter next bar open — use close of next bar as entry proxy
        entry_px = closes[i]  # signal bar close → entry
        atr_key  = dates[i]
        sl_pts   = atr5m_map.get(atr_key, 3.0) * sl_atr_mult
        sl       = entry_px - direction * sl_pts
        best_px  = entry_px
        exit_px  = None
        exit_reason = None

        for j in range(entry_i, min(entry_i + max_hold, n)):
            bar_min = mins[j]
            # Session end
            is_am = (bar_min >= 9*60) and (bar_min < 11*60+30)
            is_pm = (bar_min >= 13*60) and (bar_min < 14*60+30)
            if not (is_am or is_pm):
                exit_px = closes[j-1]
                exit_reason = 'SESSION_END'
                break
            am_end = is_am and bar_min >= 11*60+25
            pm_end = is_pm and bar_min >= 14*60+25
            if am_end or pm_end:
                exit_px = closes[j]
                exit_reason = 'SESSION_EXIT'
                break
            # Check SL on low/high
            if direction == 1:
                if lows[j] <= sl:
                    exit_px = sl
                    exit_reason = 'SL'
                    break
            else:
                if highs[j] >= sl:
                    exit_px = sl
                    exit_reason = 'SL'
                    break
            # Update best price and trail
            if direction == 1:
                best_px = max(best_px, highs[j])
                gain = best_px - entry_px
            else:
                best_px = min(best_px, lows[j])
                gain = entry_px - best_px
            if gain >= trail_pts:
                new_sl = best_px - direction * (trail_atr * atr5m_map.get(atr_key, 3.0))
                sl = max(sl, new_sl) if direction == 1 else min(sl, new_sl)

        if exit_px is None:
            exit_px = closes[min(entry_i + max_hold - 1, n-1)]
            exit_reason = 'MAX_HOLD'

        pnl_raw = (exit_px - entry_px) * direction
        pnl     = pnl_raw - cost
        trades.append({
            'signal':      signal,
            'entry_time':  df['time'].iloc[entry_i] if entry_i < n else None,
            'date':        dates[i],
            'session':     df['session'].iloc[i],
            'direction':   direction,
            'entry_px':    entry_px,
            'exit_px':     exit_px,
            'pnl':         pnl,
            'exit_reason': exit_reason,
        })
    return trades


def summarize_trades(trades, label):
    if not trades:
        print(f"  {label}: no trades")
        return
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    loss = [p for p in pnls if p <= 0]
    wr   = len(wins) / len(pnls) * 100
    pfac = sum(wins) / (abs(sum(loss)) + 1e-9)
    days = len(set(t['date'] for t in trades))
    print(f"  {label}:")
    print(f"    Trades: {len(pnls)} | Days: {days} | Trades/day: {len(pnls)/max(days,1):.1f}")
    print(f"    WR: {wr:.1f}% | PF: {pfac:.2f} | Avg PnL: {np.mean(pnls):+.2f}")
    print(f"    Total: {sum(pnls):+.1f} pts | Per day: {sum(pnls)/max(days,1):+.2f}")
    # AM/PM
    for sess in ['AM', 'PM']:
        sub = [t['pnl'] for t in trades if t['session'] == sess]
        if sub:
            sw = [p for p in sub if p > 0]
            sl = [p for p in sub if p <= 0]
            print(f"    {sess}: n={len(sub)} WR={len(sw)/len(sub)*100:.0f}% "
                  f"PF={sum(sw)/(abs(sum(sl))+1e-9):.2f} avg={np.mean(sub):+.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    sep("VOLUME FLOW ANALYSIS — VN30F1M 1m Bars")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    df   = load_1m()
    df5  = load_5m()

    trading_days = df['date'].nunique()
    total_bars   = len(df)
    print(f"1m data: {total_bars:,} bars | {trading_days} trading days")
    print(f"Date range: {df['time'].min().date()} → {df['time'].max().date()}")
    print(f"5m data: {len(df5):,} bars")

    # Build ATR map per date (from 5m)
    atr5m_map = (df5.dropna(subset=['atr14'])
                    .groupby('date')['atr14']
                    .median()
                    .to_dict())

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — VOLUME BAR PROFILING
    # ══════════════════════════════════════════════════════════════════════════
    sep("1. VOLUME BAR PROFILING")

    vol = df['volume']
    vr  = df['vol_ratio'].dropna()
    print(f"\nVolume statistics (raw contracts):")
    print(f"  Mean:   {vol.mean():.0f}  Median: {vol.median():.0f}")
    for p in [75, 90, 95, 99]:
        print(f"  P{p}:    {np.percentile(vol, p):.0f}")

    print(f"\nVol_ratio (vol / SMA20) statistics:")
    print(f"  Mean: {vr.mean():.2f}  Median: {vr.median():.2f}")
    for p in [75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(vr, p):.2f}")

    hv_bars = df[df['is_hv']]
    xv_bars = df[df['is_xv']]
    print(f"\nHV bars (vol ≥ {HV_MULT}× SMA20): {len(hv_bars):,} "
          f"({len(hv_bars)/trading_days:.1f}/day)")
    print(f"XV bars (vol ≥ {XV_MULT}× SMA20): {len(xv_bars):,} "
          f"({len(xv_bars)/trading_days:.1f}/day)")

    for sess in ['AM', 'PM']:
        sub_hv = hv_bars[hv_bars['session'] == sess]
        sub_xv = xv_bars[xv_bars['session'] == sess]
        days_s = df[df['session'] == sess]['date'].nunique()
        print(f"  {sess}: HV {len(sub_hv):,} ({len(sub_hv)/days_s:.1f}/day) | "
              f"XV {len(sub_xv):,} ({len(sub_xv)/days_s:.1f}/day)")

    # Time-of-day distribution
    print(f"\nHV bars by minute (top 20):")
    tod = hv_bars.groupby('hour_min').size().sort_values(ascending=False).head(20)
    for t, cnt in tod.items():
        bar_total = df[df['hour_min'] == t].shape[0]
        rate = cnt / bar_total * 100 if bar_total > 0 else 0
        print(f"  {t}: {cnt:3d} HV bars  ({rate:.0f}% of all bars at this time)")

    # XV direction split
    xv_bull = (xv_bars['bullish'].sum() / len(xv_bars) * 100) if len(xv_bars) > 0 else 0
    xv_bear = 100 - xv_bull
    print(f"\nXV bar direction: BULLISH {xv_bull:.1f}% | BEARISH {xv_bear:.1f}%")
    # By session
    for sess in ['AM', 'PM']:
        sub = xv_bars[xv_bars['session'] == sess]
        if len(sub) > 5:
            bull_pct = sub['bullish'].mean() * 100
            print(f"  {sess}: BULLISH {bull_pct:.1f}% (n={len(sub)})")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — WHAT HAPPENS AFTER A LARGE VOLUME BAR?
    # ══════════════════════════════════════════════════════════════════════════
    sep("2. WHAT HAPPENS AFTER A LARGE VOLUME BAR?")

    # Random control sample (non-HV bars, same size as HV)
    np.random.seed(42)
    normal_idx = df[~df['is_hv']].index.values
    ctrl_idx = np.random.choice(normal_idx, size=min(len(hv_bars), len(normal_idx)), replace=False)

    print("\n--- HV bars (≥2× SMA20) ---")
    hv_bull_idx = hv_bars[hv_bars['bullish']].index.values
    hv_bear_idx = hv_bars[~hv_bars['bullish']].index.values
    print(f"  Bullish HV: {len(hv_bull_idx)} | Bearish HV: {len(hv_bear_idx)}")

    print_forward_stats("HV BULLISH (expect continuation)", df, hv_bull_idx, +1)
    print_forward_stats("HV BEARISH (expect continuation)", df, hv_bear_idx, -1)
    print_forward_stats("CONTROL random bars (long)",        df, ctrl_idx,    +1)

    print("\n--- XV bars (≥3× SMA20) ---")
    xv_bull_idx = xv_bars[xv_bars['bullish']].index.values
    xv_bear_idx = xv_bars[~xv_bars['bullish']].index.values
    print(f"  Bullish XV: {len(xv_bull_idx)} | Bearish XV: {len(xv_bear_idx)}")
    print_forward_stats("XV BULLISH (expect continuation)", df, xv_bull_idx, +1)
    print_forward_stats("XV BEARISH (expect continuation)", df, xv_bear_idx, -1)

    # Continuation vs Reversal test (H=5)
    print(f"\n--- Continuation vs Reversal test (H=5 bars) ---")
    for label, idx, direction in [
        ("HV BULL cont.", hv_bull_idx, +1),
        ("HV BULL rev. ", hv_bull_idx, -1),
        ("HV BEAR cont.", hv_bear_idx, -1),
        ("HV BEAR rev. ", hv_bear_idx, +1),
        ("XV BULL cont.", xv_bull_idx, +1),
        ("XV BULL rev. ", xv_bull_idx, -1),
        ("XV BEAR cont.", xv_bear_idx, -1),
        ("XV BEAR rev. ", xv_bear_idx, +1),
    ]:
        if len(idx) < 10:
            continue
        _, _, final = mfe_mae(df, idx, 5, direction)
        hit = (final > 0).mean() * 100
        avg = final.mean()
        _, _, ctrl_f = mfe_mae(df, ctrl_idx[:len(idx)], 5, direction)
        print(f"  {label}: avg_ret={avg:+.2f} hit={hit:.0f}%  "
              f"(ctrl: {ctrl_f.mean():+.2f})")

    # AM vs PM breakdown
    print(f"\n--- AM vs PM breakdown (HV bars, H=10) ---")
    for sess in ['AM', 'PM']:
        for bull in [True, False]:
            mask = (hv_bars['session'] == sess) & (hv_bars['bullish'] == bull)
            idx  = hv_bars[mask].index.values
            if len(idx) < 10:
                continue
            direction = +1 if bull else -1
            mfe, mae, final = mfe_mae(df, idx, 10, direction)
            hit = (final > 0).mean() * 100
            label = f"{'BULL' if bull else 'BEAR'} HV {sess}"
            print(f"  {label} (n={len(idx)}): avg_ret={final.mean():+.2f} "
                  f"hit={hit:.0f}%  MFE={mfe.mean():.2f}  MAE={mae.mean():.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — VOLUME CLUSTER DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    sep("3. VOLUME CLUSTER DETECTION")

    # Cluster = 3+ HV bars within any 5-bar window
    df['hv_int']    = df['is_hv'].astype(int)
    df['hv_roll5']  = df['hv_int'].rolling(5, min_periods=1).sum()
    df['in_cluster'] = df['hv_roll5'] >= 3

    # Cluster start: first bar where in_cluster flips True
    cluster_starts = df[(df['in_cluster']) & (~df['in_cluster'].shift(1).fillna(False))].index.values

    print(f"\nClusters (3+ HV in 5-bar window): {len(cluster_starts)}"
          f" ({len(cluster_starts)/trading_days:.1f}/day)")

    if len(cluster_starts) > 0:
        # Direction of cluster: majority direction of HV bars in cluster
        cluster_dirs = []
        for ci in cluster_starts:
            # Look at 5-bar window ending at ci
            start_w = max(0, ci - 4)
            window  = df.loc[start_w:ci]
            hv_w    = window[window['is_hv']]
            if len(hv_w) == 0:
                continue
            bull_cnt = hv_w['bullish'].sum()
            bear_cnt = len(hv_w) - bull_cnt
            direction = +1 if bull_cnt >= bear_cnt else -1
            cluster_dirs.append((ci, direction))

        print(f"  BULL clusters: {sum(1 for _, d in cluster_dirs if d == 1)}")
        print(f"  BEAR clusters: {sum(1 for _, d in cluster_dirs if d == -1)}")

        print(f"\n--- After cluster end (continuation rate, H=10) ---")
        bull_cl = np.array([ci for ci, d in cluster_dirs if d == 1])
        bear_cl = np.array([ci for ci, d in cluster_dirs if d == -1])

        for label, idx, direction in [
            ("BULL cluster continuation", bull_cl, +1),
            ("BULL cluster reversal",     bull_cl, -1),
            ("BEAR cluster continuation", bear_cl, -1),
            ("BEAR cluster reversal",     bear_cl, +1),
        ]:
            if len(idx) < 10:
                continue
            mfe, mae, final = mfe_mae(df, idx, 10, direction)
            hit = (final > 0).mean() * 100
            print(f"  {label} (n={len(idx)}): avg={final.mean():+.2f} "
                  f"hit={hit:.0f}% MFE={mfe.mean():.2f}")

        # Compare single HV vs cluster
        single_hv_idx = df[(df['is_hv']) & (~df['in_cluster'])].index.values
        print(f"\n--- Single HV vs Cluster HV (H=10, continuation direction) ---")
        mfe_s, mae_s, final_s = mfe_mae(df, single_hv_idx, 10, +1)
        all_cl_idx = np.array([ci for ci, _ in cluster_dirs])
        mfe_c, mae_c, final_c = mfe_mae(df, all_cl_idx, 10, +1)
        print(f"  Single HV (n={len(final_s)}): avg={final_s.mean():+.2f} MFE={mfe_s.mean():.2f}")
        print(f"  Cluster   (n={len(final_c)}): avg={final_c.mean():+.2f} MFE={mfe_c.mean():.2f}")

        # Accumulation detection: cluster + small price range
        print(f"\n--- Accumulation vs Distribution clusters ---")
        accum_list, distrib_list = [], []
        for ci, direction in cluster_dirs:
            start_w = max(0, ci - 4)
            window  = df.loc[start_w:ci]
            price_range = window['high'].max() - window['low'].min()
            atr_day = atr5m_map.get(df.loc[ci, 'date'], 3.0)
            if price_range < 0.5 * atr_day:
                accum_list.append((ci, direction))   # price barely moved = accumulation
            else:
                distrib_list.append((ci, direction)) # price moved = distribution/momentum

        print(f"  Accumulation clusters (price_range < 0.5×ATR): {len(accum_list)}")
        print(f"  Momentum/Distribution clusters:                 {len(distrib_list)}")
        for label, clist in [("Accumulation", accum_list), ("Momentum/Distrib", distrib_list)]:
            if len(clist) < 5:
                continue
            idx_arr = np.array([ci for ci, _ in clist])
            dirs    = [d for _, d in clist]
            finals = []
            for ci, d in clist:
                _, _, f = mfe_mae(df, [ci], 10, d)
                if len(f) > 0:
                    finals.append(f[0])
            if finals:
                hit = (np.array(finals) > 0).mean() * 100
                print(f"  {label} (n={len(finals)}): avg={np.mean(finals):+.2f} "
                      f"hit={hit:.0f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — VOLUME BREAKOUT PATTERN
    # ══════════════════════════════════════════════════════════════════════════
    sep("4. VOLUME BREAKOUT PATTERN (quiet → burst)")

    # Define: 5+ consecutive quiet bars (vol < 0.7× SMA20) then HV bar
    QUIET_MIN_BARS = 5

    vb_signals = []
    i = 0
    vals = df['is_quiet'].values
    hv_vals = df['is_hv'].values
    n = len(df)
    while i < n:
        if hv_vals[i]:
            # Count consecutive quiet bars before this
            q_count = 0
            j = i - 1
            while j >= 0 and vals[j] and q_count < 30:
                q_count += 1
                j -= 1
            if q_count >= QUIET_MIN_BARS:
                vb_signals.append(i)
        i += 1

    vb_df = df.loc[vb_signals]
    print(f"\nVolume Breakout signals (5+ quiet bars → HV burst):")
    print(f"  Total: {len(vb_signals)} ({len(vb_signals)/trading_days:.1f}/day)")

    vb_bull_idx = vb_df[vb_df['bullish']].index.values
    vb_bear_idx = vb_df[~vb_df['bullish']].index.values
    print(f"  Bullish burst: {len(vb_bull_idx)} | Bearish burst: {len(vb_bear_idx)}")

    print_forward_stats("VB BULLISH continuation", df, vb_bull_idx, +1)
    print_forward_stats("VB BEARISH continuation", df, vb_bear_idx, -1)

    # vs random (same count)
    ctrl_vb = np.random.choice(normal_idx, size=min(len(vb_signals), len(normal_idx)), replace=False)
    _, _, ctrl_final = mfe_mae(df, ctrl_vb, 10, +1)
    print(f"\n  Control (random, H=10): avg={ctrl_final.mean():+.2f}")

    # Comparison H=10 and H=20
    print(f"\n  Volume Breakout vs Random (H=10, H=20):")
    for h in [10, 20]:
        mfe_b, _, fin_b = mfe_mae(df, vb_bull_idx, h, +1) if len(vb_bull_idx) >= 10 else (np.array([]), np.array([]), np.array([]))
        mfe_be, _, fin_be = mfe_mae(df, vb_bear_idx, h, -1) if len(vb_bear_idx) >= 10 else (np.array([]), np.array([]), np.array([]))
        _, _, ctrl_f = mfe_mae(df, ctrl_vb[:50], h, +1)
        if len(fin_b) > 0:
            print(f"  H={h}: VB_BULL avg={fin_b.mean():+.2f} hit={(fin_b>0).mean()*100:.0f}% "
                  f"| VB_BEAR avg={fin_be.mean():+.2f} hit={(fin_be>0).mean()*100:.0f}% "
                  f"| ctrl={ctrl_f.mean():+.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — MONEY FLOW INDEX (MFI)
    # ══════════════════════════════════════════════════════════════════════════
    sep("5. MONEY FLOW INDEX (MFI)")

    # Compute MFI manually (RSI of typical_price * volume)
    df['tp_vol'] = df['typ_price'] * df['volume']
    # Typical price change direction
    df['tp_diff'] = df['typ_price'].diff()
    df['pos_mf']  = np.where(df['tp_diff'] > 0, df['tp_vol'], 0.0)
    df['neg_mf']  = np.where(df['tp_diff'] < 0, df['tp_vol'], 0.0)
    mfi_len = 14
    pos_roll = df['pos_mf'].rolling(mfi_len).sum()
    neg_roll = df['neg_mf'].rolling(mfi_len).sum()
    mfr      = pos_roll / (neg_roll.replace(0, np.nan))
    df['mfi14'] = 100 - (100 / (1 + mfr))

    print(f"\nMFI(14) statistics:")
    print(f"  Mean: {df['mfi14'].mean():.1f}  Median: {df['mfi14'].median():.1f}")
    mfi_lt20 = (df['mfi14'] < 20).sum()
    mfi_gt80 = (df['mfi14'] > 80).sum()
    print(f"  MFI < 20: {mfi_lt20} bars ({mfi_lt20/len(df)*100:.1f}%)")
    print(f"  MFI > 80: {mfi_gt80} bars ({mfi_gt80/len(df)*100:.1f}%)")

    # MFI crossovers
    df['mfi_cross_up']   = (df['mfi14'] >= 20) & (df['mfi14'].shift(1) < 20)  # cross above 20
    df['mfi_cross_down'] = (df['mfi14'] <= 80) & (df['mfi14'].shift(1) > 80)  # cross below 80

    mfi_buy_idx  = df[df['mfi_cross_up']].index.values
    mfi_sell_idx = df[df['mfi_cross_down']].index.values

    print(f"\nMFI BUY signals (cross above 20):   {len(mfi_buy_idx)} ({len(mfi_buy_idx)/trading_days:.1f}/day)")
    print(f"MFI SELL signals (cross below 80): {len(mfi_sell_idx)} ({len(mfi_sell_idx)/trading_days:.1f}/day)")

    print_forward_stats("MFI BUY (cross above 20)",  df, mfi_buy_idx,  +1, horizons=(1,3,5,10,20))
    print_forward_stats("MFI SELL (cross below 80)", df, mfi_sell_idx, -1, horizons=(1,3,5,10,20))

    # MFI vs pure volume signals comparison (H=10)
    print(f"\n--- MFI vs Volume signals comparison (H=10) ---")
    for label, idx, direction in [
        ("XV BULL (vol only)", xv_bull_idx, +1),
        ("XV BEAR (vol only)", xv_bear_idx, -1),
        ("MFI BUY",            mfi_buy_idx, +1),
        ("MFI SELL",           mfi_sell_idx, -1),
    ]:
        if len(idx) < 10:
            continue
        mfe, mae, final = mfe_mae(df, idx, 10, direction)
        print(f"  {label} (n={len(idx)}): avg={final.mean():+.2f} "
              f"hit={(final>0).mean()*100:.0f}% MFE={mfe.mean():.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — LARGE VOLUME AT KEY LEVELS
    # ══════════════════════════════════════════════════════════════════════════
    sep("6. LARGE VOLUME AT KEY LEVELS (EMA, VWAP)")

    # Map 5m EMA20/EMA50 to 1m bars (forward fill)
    df5_ema = df5[['time', 'ema20', 'ema50']].copy()
    df5_ema = df5_ema.set_index('time').resample('1min').ffill().reset_index()
    df5_ema.columns = ['time', 'ema20_5m', 'ema50_5m']
    df = df.merge(df5_ema, on='time', how='left')
    df['ema20_5m'] = df['ema20_5m'].ffill()
    df['ema50_5m'] = df['ema50_5m'].ffill()

    # Session VWAP
    df['vwap'] = np.nan
    for (date_v, sess), grp in df.groupby(['date', 'session']):
        cum_vol  = (grp['typ_price'] * grp['volume']).cumsum()
        cum_v    = grp['volume'].cumsum()
        vwap_val = cum_vol / cum_v.replace(0, np.nan)
        df.loc[grp.index, 'vwap'] = vwap_val.values

    atr_day_median = np.median(list(atr5m_map.values())) if atr5m_map else 3.0

    # Distance from EMA20 and EMA50 (normalized)
    df['dist_ema20'] = (df['close'] - df['ema20_5m']).abs()
    df['dist_ema50'] = (df['close'] - df['ema50_5m']).abs()
    df['at_ema20']   = df['dist_ema20'] < 0.5
    df['at_ema50']   = df['dist_ema50'] < 0.5
    df['at_vwap']    = (df['close'] - df['vwap']).abs() < 0.5

    print(f"\nHV bars at key levels (within 0.5 pts):")
    for level, col in [("EMA20 (5m)", 'at_ema20'), ("EMA50 (5m)", 'at_ema50'), ("VWAP", 'at_vwap')]:
        at_lvl = df[df['is_hv'] & df[col]]
        not_lvl = df[df['is_hv'] & ~df[col]]
        print(f"\n  HV at {level}: {len(at_lvl)} bars")
        if len(at_lvl) < 10:
            continue
        # bounce (HV in same direction as bar body) vs break (HV breaks through)
        at_bull = at_lvl[at_lvl['bullish']].index.values
        at_bear = at_lvl[~at_lvl['bullish']].index.values
        for label, idx, direction in [
            (f"BOUNCE BULL at {level}", at_bull, +1),
            (f"BOUNCE BEAR at {level}", at_bear, -1),
        ]:
            if len(idx) < 5:
                continue
            mfe, mae, final = mfe_mae(df, idx, 10, direction)
            hit = (final > 0).mean() * 100
            print(f"    {label} (n={len(idx)}): avg={final.mean():+.2f} "
                  f"hit={hit:.0f}% MFE={mfe.mean():.2f}")

    # VWAP: above vs below for HV bars
    print(f"\n--- HV bars by VWAP position (H=10) ---")
    hv_above_vwap = df[df['is_hv'] & (df['close'] > df['vwap'])].index.values
    hv_below_vwap = df[df['is_hv'] & (df['close'] < df['vwap'])].index.values
    for label, idx, direction in [
        ("HV above VWAP (bull, cont)", hv_above_vwap, +1),
        ("HV below VWAP (bear, cont)", hv_below_vwap, -1),
    ]:
        if len(idx) < 10:
            continue
        mfe, mae, final = mfe_mae(df, idx, 10, direction)
        print(f"  {label} (n={len(idx)}): avg={final.mean():+.2f} "
              f"hit={(final>0).mean()*100:.0f}% MFE={mfe.mean():.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — PRACTICAL SIGNAL DESIGN
    # ══════════════════════════════════════════════════════════════════════════
    sep("7. PRACTICAL SIGNAL DESIGN & BACKTEST")

    # VWAP needs to be valid
    df_valid = df.dropna(subset=['vol_ratio', 'vwap', 'ema20_5m']).copy()

    # ── Signal A: XV bar (≥3×) + bullish + above VWAP → BUY ─────────────────
    sig_a_bull = df_valid[
        df_valid['is_xv'] &
        df_valid['bullish'] &
        (df_valid['close'] > df_valid['vwap'])
    ][['date', 'time', 'session']].copy()
    sig_a_bull['bar_idx']   = sig_a_bull.index
    sig_a_bull['direction'] = +1
    sig_a_bull['signal']    = 'A_XV_BULL_ABOVE_VWAP'

    sig_a_bear = df_valid[
        df_valid['is_xv'] &
        ~df_valid['bullish'] &
        (df_valid['close'] < df_valid['vwap'])
    ][['date', 'time', 'session']].copy()
    sig_a_bear['bar_idx']   = sig_a_bear.index
    sig_a_bear['direction'] = -1
    sig_a_bear['signal']    = 'A_XV_BEAR_BELOW_VWAP'

    # ── Signal B: Volume cluster (3 HV in 5 bars) + directional ─────────────
    sig_b_list = []
    for ci, direction in (cluster_dirs if len(cluster_starts) > 0 else []):
        sess = df.loc[ci, 'session'] if ci < len(df) else 'AM'
        sig_b_list.append({
            'bar_idx': ci, 'direction': direction,
            'date': df.loc[ci, 'date'], 'time': df.loc[ci, 'time'],
            'session': sess, 'signal': 'B_CLUSTER'
        })
    sig_b = pd.DataFrame(sig_b_list) if sig_b_list else pd.DataFrame()

    # ── Signal C: Volume breakout after 5+ quiet bars ────────────────────────
    sig_c_list = []
    for i in vb_signals:
        row = df.loc[i]
        direction = +1 if row['bullish'] else -1
        sig_c_list.append({
            'bar_idx': i, 'direction': direction,
            'date': row['date'], 'time': row['time'],
            'session': row['session'], 'signal': 'C_VOL_BREAKOUT'
        })
    sig_c = pd.DataFrame(sig_c_list) if sig_c_list else pd.DataFrame()

    # ── Combined signal A (both directions) ──────────────────────────────────
    sig_a = pd.concat([sig_a_bull, sig_a_bear], ignore_index=True)

    print(f"\nSignal counts:")
    print(f"  Signal A (XV + VWAP direction): {len(sig_a)} total "
          f"(BULL: {len(sig_a_bull)}, BEAR: {len(sig_a_bear)})")
    print(f"  Signal B (Cluster, directional): {len(sig_b)}")
    print(f"  Signal C (Vol Breakout): {len(sig_c)}")

    print(f"\n--- Backtesting (SL=2×ATR5m, trail@3pts/1×ATR, max_hold=30 bars) ---")

    for label, sig_df in [("Signal A", sig_a), ("Signal B", sig_b), ("Signal C", sig_c)]:
        if len(sig_df) < 10:
            print(f"\n  {label}: insufficient signals ({len(sig_df)})")
            continue
        trades = trade_backtest(df, sig_df, 'direction', atr5m_map)
        print(f"\n=== {label} ===")
        summarize_trades(trades, "All sessions")
        for sess in ['AM', 'PM']:
            t_sub = [t for t in trades if t['session'] == sess]
            if len(t_sub) >= 5:
                summarize_trades(t_sub, sess)

    # Combined: Signal A + Signal C (no overlap within 5 bars)
    if len(sig_a) > 0 and len(sig_c) > 0:
        sig_ac = pd.concat([sig_a, sig_c], ignore_index=True).sort_values('bar_idx')
        # Deduplicate: keep first signal, skip any within 5 bars on same date
        dedup_rows = []
        last_by_date = {}
        for _, row in sig_ac.iterrows():
            d = row['date']
            bi = row['bar_idx']
            if d not in last_by_date or (bi - last_by_date[d]) >= 5:
                dedup_rows.append(row)
                last_by_date[d] = bi
        sig_ac_dedup = pd.DataFrame(dedup_rows)
        print(f"\n=== Signal A+C Combined (deduped, n={len(sig_ac_dedup)}) ===")
        trades_ac = trade_backtest(df, sig_ac_dedup, 'direction', atr5m_map)
        summarize_trades(trades_ac, "A+C combined")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — TIME DECAY OF VOLUME SIGNAL
    # ══════════════════════════════════════════════════════════════════════════
    sep("8. TIME DECAY OF VOLUME SIGNAL")

    print("\nEntry timing comparison for XV bars (immediate vs delayed):")
    print("  Method 1: enter at XV bar CLOSE (immediate)")
    print("  Method 2: enter 1 bar AFTER XV bar")
    print("  Method 3: enter on first pullback bar after XV\n")

    # XV bars with clear direction
    xv_all_idx = xv_bars.index.values
    xv_all_dir = np.where(xv_bars['bullish'].values, 1, -1)

    # Method 1 — already computed above, let's compute explicitly
    m1_returns = []
    m2_returns = []
    m3_returns = []
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    n = len(df)

    for i, d in zip(xv_all_idx, xv_all_dir):
        entry1 = closes[i]
        exit_bar = min(i + 10, n - 1)
        m1_returns.append((closes[exit_bar] - entry1) * d)

        if i + 1 < n:
            entry2 = closes[i + 1]
            exit_bar2 = min(i + 11, n - 1)
            m2_returns.append((closes[exit_bar2] - entry2) * d)

        # Method 3: first bar where price retraces (move against direction)
        pb_entry = None
        for j in range(i+1, min(i+6, n)):
            if d == 1 and lows[j] < lows[i]:
                pb_entry = closes[j]
                exit_bar3 = min(j + 10, n - 1)
                m3_returns.append((closes[exit_bar3] - pb_entry) * d)
                break
            elif d == -1 and highs[j] > highs[i]:
                pb_entry = closes[j]
                exit_bar3 = min(j + 10, n - 1)
                m3_returns.append((closes[exit_bar3] - pb_entry) * d)
                break
        else:
            m3_returns.append(np.nan)  # no pullback found

    m3_clean = [x for x in m3_returns if not np.isnan(x)]
    m1_arr = np.array(m1_returns)
    m2_arr = np.array(m2_returns)
    m3_arr = np.array(m3_clean)

    print(f"  Method 1 (immediate, n={len(m1_arr)}): avg={m1_arr.mean():+.3f} "
          f"hit={(m1_arr>0).mean()*100:.0f}%")
    print(f"  Method 2 (1-bar delay, n={len(m2_arr)}): avg={m2_arr.mean():+.3f} "
          f"hit={(m2_arr>0).mean()*100:.0f}%")
    print(f"  Method 3 (pullback entry, n={len(m3_arr)}): avg={m3_arr.mean():+.3f} "
          f"hit={(m3_arr>0).mean()*100:.0f}%")

    # Signal decay: avg return at H=1,2,3,5,10,15,20 from XV bar
    print(f"\n  Signal decay curve (XV bar → hold N bars, avg net return after {COST} cost):")
    for h in [1, 2, 3, 5, 8, 10, 15, 20]:
        returns_h = []
        for i, d in zip(xv_all_idx, xv_all_dir):
            if i + h < n:
                r = (closes[i + h] - closes[i]) * d
                returns_h.append(r)
        arr = np.array(returns_h)
        net = arr.mean() - COST
        hit = (arr > 0).mean() * 100
        print(f"    H={h:2d}: gross={arr.mean():+.3f}  net={net:+.3f}  hit={hit:.0f}%  "
              f"(n={len(arr)})")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — STATISTICAL SIGNIFICANCE SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    sep("STATISTICAL SIGNIFICANCE SUMMARY")

    print("\nT-test: XV continuation vs random (H=10)")
    for label, idx, direction in [
        ("XV BULL continuation", xv_bull_idx, +1),
        ("XV BEAR continuation", xv_bear_idx, -1),
    ]:
        if len(idx) < 10:
            continue
        _, _, final_test = mfe_mae(df, idx, 10, direction)
        ctrl_sample = np.random.choice(normal_idx, size=len(idx), replace=False)
        _, _, ctrl_f = mfe_mae(df, ctrl_sample, 10, direction)
        if len(final_test) > 5 and len(ctrl_f) > 5:
            tstat, pval = stats.ttest_ind(final_test, ctrl_f, equal_var=False)
            sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else "ns"))
            print(f"  {label}: t={tstat:.2f} p={pval:.4f} {sig}  "
                  f"(test_mean={final_test.mean():+.2f} vs ctrl={ctrl_f.mean():+.2f})")

    print("\nT-test: Volume Breakout vs random (H=10)")
    for label, idx, direction in [
        ("VB BULL", vb_bull_idx, +1),
        ("VB BEAR", vb_bear_idx, -1),
    ]:
        if len(idx) < 10:
            continue
        _, _, final_test = mfe_mae(df, idx, 10, direction)
        ctrl_sample = np.random.choice(normal_idx, size=len(idx), replace=False)
        _, _, ctrl_f = mfe_mae(df, ctrl_sample, 10, direction)
        if len(final_test) > 5 and len(ctrl_f) > 5:
            tstat, pval = stats.ttest_ind(final_test, ctrl_f, equal_var=False)
            sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else "ns"))
            print(f"  {label}: t={tstat:.2f} p={pval:.4f} {sig}  "
                  f"(test_mean={final_test.mean():+.2f} vs ctrl={ctrl_f.mean():+.2f})")

    # ══════════════════════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════════════════════
    sep("VERDICT & NEXT STEPS")

    print("""
Findings summary:
  1. Volume profiling: HV/XV frequency per session, time-of-day hotspots
  2. Continuation vs reversal: does large volume predict direction persistence?
  3. Clusters: sustained money flow vs single spike — which is more predictive?
  4. Volume breakout: quiet→burst pattern, does it predict expansion?
  5. MFI: MFI extremes crossovers vs pure volume signals
  6. EMA/VWAP: large volume at key levels (support/resistance confirmation?)
  7. Signals A/B/C: backtested WR, PF, avg PnL per day
  8. Time decay: immediate entry vs 1-bar delay vs pullback

Actionable criteria:
  - WR >= 55% AND PF >= 1.5 after 0.96 cost → PROMISING
  - N >= 30 occurrences for statistical validity
  - Must improve vs CB baseline (+5.99/day) OR be orthogonal (different time slots)
""")
    sep()
    print("Analysis complete.")


if __name__ == "__main__":
    main()
