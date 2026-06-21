"""SL Sweep Feature Analysis: What indicators predict sweep vs reversal?

Questions:
1. When SL is hit, what features DISTINGUISH sweeps (price returns) from reversals?
2. RSI at what level → more likely to sweep back?
3. BB position? Volume? ADX? EMA alignment?
4. Can we build a "sweep probability" score to decide: hold wider SL or cut?

Methodology:
- Collect all SL hit events with full indicator state AT the moment SL is hit
- Label: SWEEP (post-SL MFE >= 70% of SL distance) vs REVERSAL
- Compare distributions of each indicator between sweep and reversal groups
- Find discriminating thresholds (Cohen's d, KS test)

Usage:
    python -m research.sl_sweep_features
"""
import sys
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from strategies.base import EXIT_PRESETS, COST, AM_CUTOFF, PM_CUTOFF

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

STRATEGIES = [
    ('momentum_trend', 'PM', 'SELL'),
    ('momentum_trend', 'ALL', 'SELL'),
    ('macd_cross', 'AM', 'SELL'),
    ('macd_cross', 'PM', 'BUY'),
    ('heikin_ashi', 'AM', 'SELL'),
    ('fibonacci', 'PM', 'SELL'),
    ('fibonacci', 'AM', 'SELL'),
]


def load_1m_data():
    parquet_file = os.path.join(DATA_DIR, "vn30f1m_1m.parquet")
    print(f"[DATA] Loading 1m...", flush=True)
    df = pd.read_parquet(parquet_file)
    df['time'] = pd.to_datetime(df['time'])
    df.columns = [c.lower() for c in df.columns]
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute

    am_mask = (df['mins'] >= 540) & (df['mins'] <= 690)
    pm_mask = (df['mins'] >= 780) & (df['mins'] <= 870)
    df = df[am_mask | pm_mask].reset_index(drop=True)
    df['session'] = 'AM'
    df.loc[df['mins'] >= 780, 'session'] = 'PM'

    # Core indicators
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['rsi_7'] = ta.rsi(df['close'], length=7)
    df['range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['body_pct'] = df['body'] / df['range'].replace(0, np.nan)
    df['ema8'] = ta.ema(df['close'], length=8)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema50'] = ta.ema(df['close'], length=50)

    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20, std=2.0)
    if bb is not None and len(bb.columns) >= 3:
        df['bb_lower'] = bb.iloc[:, 0]
        df['bb_mid'] = bb.iloc[:, 1]
        df['bb_upper'] = bb.iloc[:, 2]
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    else:
        df['bb_lower'] = df['bb_mid'] = df['bb_upper'] = df['bb_width'] = df['bb_pos'] = np.nan

    # ADX, DI
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is not None:
        df['adx'] = adx_df.iloc[:, 0]
        df['di_plus'] = adx_df.iloc[:, 1]
        df['di_minus'] = adx_df.iloc[:, 2]
        df['di_spread'] = df['di_plus'] - df['di_minus']
    else:
        df['adx'] = df['di_plus'] = df['di_minus'] = df['di_spread'] = np.nan

    # MACD
    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd_df is not None:
        df['macd_line'] = macd_df.iloc[:, 0]
        df['macd_signal'] = macd_df.iloc[:, 2]
        df['macd_hist'] = macd_df.iloc[:, 1]
    else:
        df['macd_line'] = df['macd_signal'] = df['macd_hist'] = np.nan

    # Stochastic
    stoch_df = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    if stoch_df is not None:
        df['stoch_k'] = stoch_df.iloc[:, 0]
        df['stoch_d'] = stoch_df.iloc[:, 1]
    else:
        df['stoch_k'] = df['stoch_d'] = np.nan

    # Volume
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20'].replace(0, np.nan)

    # EMA alignment score (-3 to +3)
    df['ema_align'] = 0
    df.loc[df['close'] > df['ema8'], 'ema_align'] += 1
    df.loc[df['close'] > df['ema21'], 'ema_align'] += 1
    df.loc[df['close'] > df['ema50'], 'ema_align'] += 1
    df.loc[df['close'] < df['ema8'], 'ema_align'] -= 1
    df.loc[df['close'] < df['ema21'], 'ema_align'] -= 1
    df.loc[df['close'] < df['ema50'], 'ema_align'] -= 1

    # ATR ratio (current vs SMA)
    df['atr_sma50'] = df['atr'].rolling(50).mean()
    df['atr_ratio'] = df['atr'] / df['atr_sma50'].replace(0, np.nan)

    # Pre-move (3 bars momentum before signal)
    df['ret_3'] = (df['close'] - df['close'].shift(3)) / df['atr'].replace(0, np.nan)
    df['ret_5'] = (df['close'] - df['close'].shift(5)) / df['atr'].replace(0, np.nan)

    # Range compression
    df['range_ratio_3_10'] = df['range'].rolling(3).mean() / df['range'].rolling(10).mean().replace(0, np.nan)

    # Keltner Channel position
    kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=1.5)
    if kc is not None and len(kc.columns) >= 3:
        df['kc_lower'] = kc.iloc[:, 0]
        df['kc_upper'] = kc.iloc[:, 2]
        df['kc_pos'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower']).replace(0, np.nan)
    else:
        df['kc_pos'] = np.nan

    # Bars since session start
    df['bars_in_session'] = df.groupby([df['date'], df['session']]).cumcount()

    n_days = df['date'].nunique()
    print(f"[DATA] {len(df)} bars, {n_days} days", flush=True)
    return df


def get_strategy(name):
    if name == 'momentum_trend':
        from strategies.momentum_trend import MomentumTrendStrategy
        return MomentumTrendStrategy()
    elif name == 'macd_cross':
        from strategies.macd_cross import MACDCrossStrategy
        return MACDCrossStrategy()
    elif name == 'fibonacci':
        from strategies.fibonacci import FibonacciStrategy
        return FibonacciStrategy()
    elif name == 'heikin_ashi':
        from strategies.heikin_ashi import HeikinAshiStrategy
        return HeikinAshiStrategy()
    raise ValueError(name)


def collect_sl_events(df, strategy_name, session_filter, direction_filter,
                      sl_mult=2.0, max_hold=30, track_after=30):
    """Collect all SL hit events with full indicator snapshot at SL bar."""
    strategy = get_strategy(strategy_name)
    strategy.prepare(df)

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    mins_arr = df['mins'].values
    sessions_arr = df['session'].values

    events = []
    last_idx = -10

    for i in range(60, len(df) - 2):
        if i - last_idx < 5:
            continue
        session = sessions_arr[i]
        if session not in ('AM', 'PM'):
            continue
        if session_filter != 'ALL' and session != session_filter:
            continue
        mins = mins_arr[i]
        if session == 'AM' and not (555 <= mins <= 645):
            continue
        if session == 'PM' and not (795 <= mins <= 855):
            continue
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < 0.5:
            continue

        direction = strategy.detect(df, i)
        if direction == 0:
            continue
        if direction_filter == 'BUY' and direction != 1:
            continue
        if direction_filter == 'SELL' and direction != -1:
            continue

        entry = closes[i]
        sl_level = entry - direction * sl_mult * atr
        cutoff = AM_CUTOFF if session == 'AM' else PM_CUTOFF

        # Find SL hit bar
        sl_bar = None
        mfe_before_sl = 0.0
        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            if mins_arr[j] >= cutoff:
                break
            if direction == 1:
                mfe_before_sl = max(mfe_before_sl, highs[j] - entry)
                if lows[j] <= sl_level:
                    sl_bar = j
                    break
            else:
                mfe_before_sl = max(mfe_before_sl, entry - lows[j])
                if highs[j] >= sl_level:
                    sl_bar = j
                    break

        if sl_bar is None:
            continue  # no SL hit

        last_idx = i

        # Track post-SL price action
        post_sl_mfe = 0.0
        post_sl_mae = 0.0
        for k in range(sl_bar + 1, min(sl_bar + track_after, len(df))):
            if mins_arr[k] >= cutoff:
                break
            if direction == 1:
                post_sl_mfe = max(post_sl_mfe, highs[k] - sl_level)
                post_sl_mae = max(post_sl_mae, sl_level - lows[k])
            else:
                post_sl_mfe = max(post_sl_mfe, sl_level - lows[k])
                post_sl_mae = max(post_sl_mae, highs[k] - sl_level)

        is_sweep = post_sl_mfe >= sl_mult * atr * 0.7

        # ═══ CAPTURE FEATURES AT SIGNAL BAR AND AT SL BAR ═══
        # Features at SIGNAL bar (entry moment)
        sig_features = {}
        for col in ['rsi', 'rsi_7', 'bb_pos', 'bb_width', 'adx', 'di_spread',
                    'stoch_k', 'stoch_d', 'vol_ratio', 'ema_align', 'atr_ratio',
                    'ret_3', 'ret_5', 'range_ratio_3_10', 'kc_pos', 'macd_hist',
                    'body_pct', 'bars_in_session']:
            val = df[col].iloc[i]
            sig_features[f'sig_{col}'] = val if not pd.isna(val) else np.nan

        # Features at SL HIT bar
        sl_features = {}
        for col in ['rsi', 'rsi_7', 'bb_pos', 'bb_width', 'adx', 'di_spread',
                    'stoch_k', 'stoch_d', 'vol_ratio', 'ema_align', 'atr_ratio',
                    'ret_3', 'ret_5', 'range_ratio_3_10', 'kc_pos', 'macd_hist',
                    'body_pct', 'bars_in_session']:
            val = df[col].iloc[sl_bar]
            sl_features[f'sl_{col}'] = val if not pd.isna(val) else np.nan

        # Derived features
        bars_to_sl = sl_bar - i
        sl_bar_range = highs[sl_bar] - lows[sl_bar]
        sl_bar_body = abs(closes[sl_bar] - df['open'].iloc[sl_bar])

        events.append({
            'strategy': strategy_name,
            'session': session,
            'direction': direction,
            'date': df['date'].iloc[i],
            'entry': entry,
            'atr': atr,
            'sl_dist': sl_mult * atr,
            'mfe_before_sl': mfe_before_sl,
            'post_sl_mfe': post_sl_mfe,
            'post_sl_mae': post_sl_mae,
            'is_sweep': is_sweep,
            'bars_to_sl': bars_to_sl,
            'sl_bar_range_atr': sl_bar_range / atr,
            'sl_bar_body_pct': sl_bar_body / sl_bar_range if sl_bar_range > 0 else 0,
            **sig_features,
            **sl_features,
        })

    return events


def analyze_feature(sweep_vals, reversal_vals, name):
    """Compare a feature between sweep and reversal groups."""
    s = np.array([v for v in sweep_vals if not np.isnan(v)])
    r = np.array([v for v in reversal_vals if not np.isnan(v)])

    if len(s) < 20 or len(r) < 20:
        return None

    mean_s = np.mean(s)
    mean_r = np.mean(r)
    std_pooled = np.sqrt((np.std(s)**2 + np.std(r)**2) / 2)
    cohens_d = (mean_s - mean_r) / std_pooled if std_pooled > 0 else 0

    # KS test
    ks_stat, ks_p = stats.ks_2samp(s, r)

    return {
        'name': name,
        'sweep_mean': mean_s,
        'reversal_mean': mean_r,
        'sweep_median': np.median(s),
        'reversal_median': np.median(r),
        'cohens_d': cohens_d,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'n_sweep': len(s),
        'n_reversal': len(r),
    }


def find_threshold(sweep_vals, reversal_vals, feature_name, direction_hint=None):
    """Find best threshold to separate sweeps from reversals."""
    all_vals = [(v, 1) for v in sweep_vals if not np.isnan(v)] + \
               [(v, 0) for v in reversal_vals if not np.isnan(v)]
    if len(all_vals) < 40:
        return None

    all_vals.sort(key=lambda x: x[0])
    values = np.array([v[0] for v in all_vals])

    best_score = 0
    best_thresh = None
    best_dir = None

    for pct in range(10, 91, 5):
        thresh = np.percentile(values, pct)

        # Test "above threshold → sweep"
        above_sweep = sum(1 for v, label in all_vals if v >= thresh and label == 1)
        above_rev = sum(1 for v, label in all_vals if v >= thresh and label == 0)
        above_total = above_sweep + above_rev
        if above_total > 0:
            sweep_rate_above = above_sweep / above_total
            n_above = above_total
            score_above = (sweep_rate_above - 0.726) * n_above  # improvement over baseline 72.6%

            if score_above > best_score:
                best_score = score_above
                best_thresh = thresh
                best_dir = 'above'

        # Test "below threshold → sweep"
        below_sweep = sum(1 for v, label in all_vals if v < thresh and label == 1)
        below_rev = sum(1 for v, label in all_vals if v < thresh and label == 0)
        below_total = below_sweep + below_rev
        if below_total > 0:
            sweep_rate_below = below_sweep / below_total
            score_below = (sweep_rate_below - 0.726) * below_total

            if score_below > best_score:
                best_score = score_below
                best_thresh = thresh
                best_dir = 'below'

    if best_thresh is None:
        return None

    # Calculate stats at best threshold
    if best_dir == 'above':
        filtered = [(v, l) for v, l in all_vals if v >= best_thresh]
    else:
        filtered = [(v, l) for v, l in all_vals if v < best_thresh]

    n_filtered = len(filtered)
    sweep_rate = sum(1 for _, l in filtered if l == 1) / n_filtered if n_filtered else 0

    return {
        'threshold': best_thresh,
        'direction': best_dir,
        'sweep_rate': sweep_rate,
        'n_filtered': n_filtered,
        'improvement': sweep_rate - 0.726,  # vs baseline
    }


def main():
    t0 = time.time()
    df = load_1m_data()

    # ══════════════════════════════════════════════════════════════════════════
    # COLLECT ALL SL EVENTS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[COLLECT] Gathering SL events from all strategies...", flush=True)

    all_events = []
    for strat, sess, dirn in STRATEGIES:
        events = collect_sl_events(df, strat, sess, dirn, sl_mult=2.0, max_hold=30)
        print(f"  {strat} {sess}/{dirn}: {len(events)} SL events "
              f"({sum(1 for e in events if e['is_sweep'])}/{len(events)} sweeps)", flush=True)
        all_events.extend(events)

    sweeps = [e for e in all_events if e['is_sweep']]
    reversals = [e for e in all_events if not e['is_sweep']]
    print(f"\n  TOTAL: {len(all_events)} SL events | {len(sweeps)} sweeps ({len(sweeps)/len(all_events)*100:.1f}%) | {len(reversals)} reversals")

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE ANALYSIS: SIGNAL BAR FEATURES
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  FEATURES AT SIGNAL BAR (entry moment)")
    print(f"{'═'*80}")

    sig_features = ['sig_rsi', 'sig_rsi_7', 'sig_bb_pos', 'sig_bb_width', 'sig_adx',
                    'sig_di_spread', 'sig_stoch_k', 'sig_stoch_d', 'sig_vol_ratio',
                    'sig_ema_align', 'sig_atr_ratio', 'sig_ret_3', 'sig_ret_5',
                    'sig_range_ratio_3_10', 'sig_kc_pos', 'sig_macd_hist',
                    'sig_body_pct', 'sig_bars_in_session']

    sig_results = []
    for feat in sig_features:
        s_vals = [e[feat] for e in sweeps]
        r_vals = [e[feat] for e in reversals]
        result = analyze_feature(s_vals, r_vals, feat)
        if result:
            sig_results.append(result)

    sig_results.sort(key=lambda x: abs(x['cohens_d']), reverse=True)

    print(f"\n  {'Feature':<25} {'Cohen d':>8} {'Sweep':>8} {'Reversal':>8} {'KS p':>8} {'Discrimin':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for r in sig_results:
        disc = "***" if abs(r['cohens_d']) > 0.3 else ("**" if abs(r['cohens_d']) > 0.2 else ("*" if abs(r['cohens_d']) > 0.1 else ""))
        print(f"  {r['name']:<25} {r['cohens_d']:>+7.3f} {r['sweep_mean']:>8.3f} {r['reversal_mean']:>8.3f} {r['ks_p']:>8.4f} {disc:>10}")

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE ANALYSIS: SL BAR FEATURES
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  FEATURES AT SL HIT BAR (moment SL triggered)")
    print(f"{'═'*80}")

    sl_features = ['sl_rsi', 'sl_rsi_7', 'sl_bb_pos', 'sl_bb_width', 'sl_adx',
                   'sl_di_spread', 'sl_stoch_k', 'sl_stoch_d', 'sl_vol_ratio',
                   'sl_ema_align', 'sl_atr_ratio', 'sl_ret_3', 'sl_ret_5',
                   'sl_range_ratio_3_10', 'sl_kc_pos', 'sl_macd_hist',
                   'sl_body_pct', 'sl_bars_in_session']

    # Add structural features
    extra_features = ['bars_to_sl', 'mfe_before_sl', 'sl_bar_range_atr', 'sl_bar_body_pct']

    sl_results = []
    for feat in sl_features + extra_features:
        s_vals = [e[feat] for e in sweeps]
        r_vals = [e[feat] for e in reversals]
        result = analyze_feature(s_vals, r_vals, feat)
        if result:
            sl_results.append(result)

    sl_results.sort(key=lambda x: abs(x['cohens_d']), reverse=True)

    print(f"\n  {'Feature':<25} {'Cohen d':>8} {'Sweep':>8} {'Reversal':>8} {'KS p':>8} {'Discrimin':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for r in sl_results:
        disc = "***" if abs(r['cohens_d']) > 0.3 else ("**" if abs(r['cohens_d']) > 0.2 else ("*" if abs(r['cohens_d']) > 0.1 else ""))
        print(f"  {r['name']:<25} {r['cohens_d']:>+7.3f} {r['sweep_mean']:>8.3f} {r['reversal_mean']:>8.3f} {r['ks_p']:>8.4f} {disc:>10}")

    # ══════════════════════════════════════════════════════════════════════════
    # RSI DEEP DIVE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  RSI DEEP DIVE: Sweep rate by RSI bucket")
    print(f"{'═'*80}")

    # RSI at signal bar
    print(f"\n  RSI at SIGNAL bar:")
    print(f"  {'RSI Range':<15} {'Total':>6} {'Sweeps':>7} {'Sweep%':>7} {'vs Base':>8}")
    for lo, hi in [(0, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]:
        bucket = [e for e in all_events if not np.isnan(e['sig_rsi']) and lo <= e['sig_rsi'] < hi]
        if len(bucket) < 10:
            continue
        n_sweep = sum(1 for e in bucket if e['is_sweep'])
        rate = n_sweep / len(bucket) * 100
        vs_base = rate - 72.6
        print(f"  {lo:>3}-{hi:<3}         {len(bucket):>6} {n_sweep:>7} {rate:>6.1f}% {vs_base:>+7.1f}%")

    # RSI at SL hit bar
    print(f"\n  RSI at SL HIT bar:")
    print(f"  {'RSI Range':<15} {'Total':>6} {'Sweeps':>7} {'Sweep%':>7} {'vs Base':>8}")
    for lo, hi in [(0, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]:
        bucket = [e for e in all_events if not np.isnan(e['sl_rsi']) and lo <= e['sl_rsi'] < hi]
        if len(bucket) < 10:
            continue
        n_sweep = sum(1 for e in bucket if e['is_sweep'])
        rate = n_sweep / len(bucket) * 100
        vs_base = rate - 72.6
        print(f"  {lo:>3}-{hi:<3}         {len(bucket):>6} {n_sweep:>7} {rate:>6.1f}% {vs_base:>+7.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # BOLLINGER BAND DEEP DIVE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  BOLLINGER BAND DEEP DIVE: Sweep rate by BB position")
    print(f"{'═'*80}")

    # BB at signal bar
    print(f"\n  BB position at SIGNAL bar (0=lower, 0.5=mid, 1=upper):")
    print(f"  {'BB Pos':<15} {'Total':>6} {'Sweeps':>7} {'Sweep%':>7} {'vs Base':>8}")
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        bucket = [e for e in all_events if not np.isnan(e['sig_bb_pos']) and lo <= e['sig_bb_pos'] < hi]
        if len(bucket) < 10:
            continue
        n_sweep = sum(1 for e in bucket if e['is_sweep'])
        rate = n_sweep / len(bucket) * 100
        vs_base = rate - 72.6
        print(f"  {lo:.1f}-{hi:.1f}         {len(bucket):>6} {n_sweep:>7} {rate:>6.1f}% {vs_base:>+7.1f}%")

    # BB at SL bar
    print(f"\n  BB position at SL HIT bar:")
    print(f"  {'BB Pos':<15} {'Total':>6} {'Sweeps':>7} {'Sweep%':>7} {'vs Base':>8}")
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        bucket = [e for e in all_events if not np.isnan(e['sl_bb_pos']) and lo <= e['sl_bb_pos'] < hi]
        if len(bucket) < 10:
            continue
        n_sweep = sum(1 for e in bucket if e['is_sweep'])
        rate = n_sweep / len(bucket) * 100
        vs_base = rate - 72.6
        print(f"  {lo:.1f}-{hi:.1f}         {len(bucket):>6} {n_sweep:>7} {rate:>6.1f}% {vs_base:>+7.1f}%")

    # BB Width (squeeze detection)
    print(f"\n  BB WIDTH at signal (narrower = more likely squeeze/sweep):")
    print(f"  {'BB Width':<15} {'Total':>6} {'Sweeps':>7} {'Sweep%':>7} {'vs Base':>8}")
    bb_widths = [e['sig_bb_width'] for e in all_events if not np.isnan(e['sig_bb_width'])]
    if bb_widths:
        for pct_lo, pct_hi in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
            lo = np.percentile(bb_widths, pct_lo)
            hi = np.percentile(bb_widths, pct_hi)
            bucket = [e for e in all_events if not np.isnan(e['sig_bb_width']) and lo <= e['sig_bb_width'] < hi]
            if len(bucket) < 10:
                continue
            n_sweep = sum(1 for e in bucket if e['is_sweep'])
            rate = n_sweep / len(bucket) * 100
            vs_base = rate - 72.6
            print(f"  P{pct_lo}-P{pct_hi} ({lo:.4f}-{hi:.4f})  {len(bucket):>6} {n_sweep:>7} {rate:>6.1f}% {vs_base:>+7.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # ADX & VOLUME DEEP DIVE
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  ADX & VOLUME: Sweep rate by ADX and Vol ratio")
    print(f"{'═'*80}")

    print(f"\n  ADX at SIGNAL bar:")
    print(f"  {'ADX Range':<15} {'Total':>6} {'Sweeps':>7} {'Sweep%':>7} {'vs Base':>8}")
    for lo, hi in [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 50), (50, 100)]:
        bucket = [e for e in all_events if not np.isnan(e['sig_adx']) and lo <= e['sig_adx'] < hi]
        if len(bucket) < 10:
            continue
        n_sweep = sum(1 for e in bucket if e['is_sweep'])
        rate = n_sweep / len(bucket) * 100
        vs_base = rate - 72.6
        print(f"  {lo:>3}-{hi:<3}         {len(bucket):>6} {n_sweep:>7} {rate:>6.1f}% {vs_base:>+7.1f}%")

    print(f"\n  Volume Ratio at SIGNAL bar:")
    print(f"  {'Vol Ratio':<15} {'Total':>6} {'Sweeps':>7} {'Sweep%':>7} {'vs Base':>8}")
    for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 1.0), (1.0, 1.3), (1.3, 1.5), (1.5, 2.0), (2.0, 10.0)]:
        bucket = [e for e in all_events if not np.isnan(e['sig_vol_ratio']) and lo <= e['sig_vol_ratio'] < hi]
        if len(bucket) < 10:
            continue
        n_sweep = sum(1 for e in bucket if e['is_sweep'])
        rate = n_sweep / len(bucket) * 100
        vs_base = rate - 72.6
        print(f"  {lo:.1f}-{hi:.1f}         {len(bucket):>6} {n_sweep:>7} {rate:>6.1f}% {vs_base:>+7.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # DIRECTION-SPECIFIC ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  DIRECTION-SPECIFIC: SELL signals vs BUY signals")
    print(f"{'═'*80}")

    for dirn, label in [(1, 'BUY'), (-1, 'SELL')]:
        dir_events = [e for e in all_events if e['direction'] == dirn]
        dir_sweeps = [e for e in dir_events if e['is_sweep']]
        if len(dir_events) < 50:
            continue
        print(f"\n  {label}: {len(dir_events)} SL events, {len(dir_sweeps)} sweeps ({len(dir_sweeps)/len(dir_events)*100:.1f}%)")

        # For SELL: RSI high at signal → price went down → SL above → sweep = price came back up then went down again
        # Key: what RSI/BB at SL hit predicts sweep?
        print(f"    RSI at SL hit:")
        for lo, hi in [(0, 30), (30, 50), (50, 70), (70, 100)]:
            bucket = [e for e in dir_events if not np.isnan(e['sl_rsi']) and lo <= e['sl_rsi'] < hi]
            if len(bucket) < 10:
                continue
            n_s = sum(1 for e in bucket if e['is_sweep'])
            rate = n_s / len(bucket) * 100
            print(f"      RSI {lo}-{hi}: {len(bucket)} events, sweep {rate:.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # BEST THRESHOLD FINDER (top features)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  THRESHOLD SWEEP: Best single-feature rule to predict SWEEP")
    print(f"{'═'*80}")
    print(f"  Baseline sweep rate: 72.6% ({len(sweeps)}/{len(all_events)})")
    print(f"  Goal: find conditions where sweep rate > 80% (safe to hold/re-enter)")
    print(f"")

    all_features = sig_features + sl_features + extra_features
    threshold_results = []

    for feat in all_features:
        s_vals = [e[feat] for e in sweeps if feat in e]
        r_vals = [e[feat] for e in reversals if feat in e]
        result = find_threshold(s_vals, r_vals, feat)
        if result and result['improvement'] > 0.02:
            threshold_results.append({**result, 'feature': feat})

    threshold_results.sort(key=lambda x: x['sweep_rate'] * x['n_filtered'], reverse=True)

    print(f"  {'Feature':<25} {'Dir':>5} {'Threshold':>10} {'Sweep%':>7} {'N':>6} {'Improve':>8}")
    print(f"  {'-'*25} {'-'*5} {'-'*10} {'-'*7} {'-'*6} {'-'*8}")
    for r in threshold_results[:25]:
        print(f"  {r['feature']:<25} {r['direction']:>5} {r['threshold']:>10.3f} {r['sweep_rate']*100:>6.1f}% {r['n_filtered']:>6} {r['improvement']*100:>+7.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # COMBINED RULES
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*80}")
    print(f"  COMBINED RULES: 2-feature combinations for high sweep probability")
    print(f"{'═'*80}")

    # Take top 5 features and test pairwise combinations
    top_feats = threshold_results[:8]
    combo_results = []

    for i in range(len(top_feats)):
        for j in range(i + 1, len(top_feats)):
            f1 = top_feats[i]
            f2 = top_feats[j]

            # Apply both filters
            filtered = []
            for e in all_events:
                feat1_name = f1['feature']
                feat2_name = f2['feature']
                if feat1_name not in e or feat2_name not in e:
                    continue
                v1 = e[feat1_name]
                v2 = e[feat2_name]
                if np.isnan(v1) or np.isnan(v2):
                    continue

                pass1 = (v1 >= f1['threshold']) if f1['direction'] == 'above' else (v1 < f1['threshold'])
                pass2 = (v2 >= f2['threshold']) if f2['direction'] == 'above' else (v2 < f2['threshold'])

                if pass1 and pass2:
                    filtered.append(e)

            if len(filtered) < 30:
                continue

            n_sweep = sum(1 for e in filtered if e['is_sweep'])
            sweep_rate = n_sweep / len(filtered)

            if sweep_rate > 0.78:  # better than baseline + 5%
                combo_results.append({
                    'f1': f1['feature'], 'f2': f2['feature'],
                    'sweep_rate': sweep_rate, 'n': len(filtered),
                    'f1_thresh': f1['threshold'], 'f1_dir': f1['direction'],
                    'f2_thresh': f2['threshold'], 'f2_dir': f2['direction'],
                })

    combo_results.sort(key=lambda x: x['sweep_rate'] * min(x['n'], 500), reverse=True)

    print(f"\n  Top combos with sweep rate > 78%:")
    print(f"  {'Feature 1':<22} {'Feature 2':<22} {'Sweep%':>7} {'N':>5}")
    print(f"  {'-'*22} {'-'*22} {'-'*7} {'-'*5}")
    for r in combo_results[:15]:
        print(f"  {r['f1']:<22} {r['f2']:<22} {r['sweep_rate']*100:>6.1f}% {r['n']:>5}")
        print(f"    Rule: {r['f1']} {r['f1_dir']} {r['f1_thresh']:.3f} AND {r['f2']} {r['f2_dir']} {r['f2_thresh']:.3f}")

    elapsed = time.time() - t0
    print(f"\n[DONE] {elapsed:.1f}s")


if __name__ == '__main__':
    main()
