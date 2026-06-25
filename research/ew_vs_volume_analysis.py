#!/usr/bin/env python3
"""
Elliott Wave vs Volume-Based Structure Analysis -- VN30F1M 5m Futures
======================================================================
Tests whether price action is better explained by:
  (A) Elliott Wave Fibonacci structure (impulse 5-3, retracements)
  (B) Volume-based structure (VPOC, high-volume nodes as S/R)

Run from: e:/Trading/
  python research/ew_vs_volume_analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'data/vn30f1m_5m.parquet'
COST_PER_TRADE = 0.96  # pts round-trip

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values('time').reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])

    # Session filter: only keep trading hours
    tm = df['time'].dt.hour * 60 + df['time'].dt.minute
    am = (tm >= 9*60) & (tm <= 11*60+30)
    pm = (tm >= 13*60) & (tm <= 14*60+30)
    df = df[am | pm].copy().reset_index(drop=True)

    df['date'] = df['time'].dt.date
    df['session'] = df['time'].apply(lambda t: 'AM' if t.hour < 12 else 'PM')
    df['time_min'] = df['time'].dt.hour * 60 + df['time'].dt.minute

    # Derived columns
    df['range'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['bar_dir'] = np.where(df['close'] >= df['open'], 1, -1)
    df['vol_sma20'] = df['volume'].rolling(20, min_periods=5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma20']
    df['atr14'] = df['range'].rolling(14, min_periods=5).mean()
    return df


# -----------------------------------------------------------------------------
# SECTION 1 -- ELLIOTT WAVE ANALYSIS
# -----------------------------------------------------------------------------

def find_swings(prices_high, prices_low, order=5):
    """
    Detect swing highs (local max of high) and swing lows (local min of low).
    Returns sorted list of dicts with keys: idx, price, type (H or L).
    Consecutive same-type points are merged (keep the more extreme one).
    """
    high_arr = prices_high.values
    low_arr  = prices_low.values

    h_idx = argrelextrema(high_arr, np.greater_equal, order=order)[0]
    l_idx = argrelextrema(low_arr,  np.less_equal,    order=order)[0]

    events = []
    for i in h_idx:
        events.append({'idx': int(i), 'price': float(high_arr[i]), 'type': 'H'})
    for i in l_idx:
        events.append({'idx': int(i), 'price': float(low_arr[i]),  'type': 'L'})

    events.sort(key=lambda x: x['idx'])

    # Merge consecutive same-type: keep most extreme
    merged = []
    for ev in events:
        if merged and merged[-1]['type'] == ev['type']:
            prev = merged[-1]
            if ev['type'] == 'H' and ev['price'] > prev['price']:
                merged[-1] = ev
            elif ev['type'] == 'L' and ev['price'] < prev['price']:
                merged[-1] = ev
        else:
            merged.append(ev)

    return merged


def find_5wave_sequences(swings, min_wave1_pts=3.0):
    """
    Scan swing list for candidate 5-wave impulse structures.
    Rules applied:
      - Must alternate H/L/H/L/H/L (bull) or L/H/L/H/L/H (bear)
      - Wave 2 retraces < 100% of Wave 1
      - Wave 3 is not the shortest wave
      - Wave 1 >= min_wave1_pts (noise filter)
    Returns list of dicts with wave sizes and ratios.
    """
    sequences = []
    n = len(swings)

    for i in range(n - 5):
        pts = swings[i:i+6]
        types = [p['type'] for p in pts]

        is_bull = types == ['L', 'H', 'L', 'H', 'L', 'H']
        is_bear = types == ['H', 'L', 'H', 'L', 'H', 'L']
        if not (is_bull or is_bear):
            continue

        p = [q['price'] for q in pts]

        if is_bull:
            w1 = p[1] - p[0]
            w2 = p[1] - p[2]   # retrace of w1
            w3 = p[3] - p[2]
            w4 = p[3] - p[4]   # retrace of w3
            w5 = p[5] - p[4]
            direction = 'bull'
        else:
            w1 = p[0] - p[1]
            w2 = p[2] - p[1]   # retrace of w1
            w3 = p[2] - p[3]
            w4 = p[4] - p[3]   # retrace of w3
            w5 = p[4] - p[5]
            direction = 'bear'

        # All waves must be positive
        if any(w <= 0 for w in [w1, w2, w3, w4, w5]):
            continue
        # Wave 2 cannot retrace more than 100% of Wave 1
        if w2 >= w1:
            continue
        # Wave 3 must not be the shortest
        if w3 <= min(w1, w5):
            continue
        # Minimum wave 1 size
        if w1 < min_wave1_pts:
            continue

        sequences.append({
            'start_idx'     : pts[0]['idx'],
            'end_idx'       : pts[5]['idx'],
            'direction'     : direction,
            'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
            'w2_ret_w1'     : w2 / w1,
            'w3_ext_w1'     : w3 / w1,
            'w4_ret_w3'     : w4 / w3,
            'w5_vs_w1'      : w5 / w1,
        })

    return sequences


def fib_ratio_statistical_test(sequences):
    """
    Test whether key wave ratios cluster at Fibonacci levels.
    Uses binomial test: observed hit rate vs expected if uniform distribution.
    Tolerance: +/-10% relative to each Fibonacci level.
    """
    if len(sequences) < 20:
        return {}

    df_s = pd.DataFrame(sequences)

    # Fibonacci targets per ratio and their typical observed range
    tests = {
        'w3_ext_w1' : {
            'fibs': [1.0, 1.272, 1.618, 2.0, 2.618],
            'obs_range': (0.3, 4.0),
            'label': 'W3 extension of W1'
        },
        'w2_ret_w1' : {
            'fibs': [0.236, 0.382, 0.500, 0.618, 0.764],
            'obs_range': (0.05, 1.0),
            'label': 'W2 retrace of W1'
        },
        'w4_ret_w3' : {
            'fibs': [0.236, 0.382, 0.500, 0.618],
            'obs_range': (0.05, 1.0),
            'label': 'W4 retrace of W3'
        },
        'w5_vs_w1'  : {
            'fibs': [0.618, 1.0, 1.618],
            'obs_range': (0.1, 3.0),
            'label': 'W5 relative to W1'
        },
    }

    results = {}
    TOL = 0.10  # 10% relative tolerance

    for col, cfg in tests.items():
        if col not in df_s:
            continue
        vals = df_s[col].dropna().values
        if len(vals) < 15:
            continue

        # Count hits
        hits = sum(
            any(abs(v - f) / f < TOL for f in cfg['fibs'])
            for v in vals
        )
        hit_rate = hits / len(vals)

        # Estimate random baseline: fraction of obs_range covered by fib zones
        lo, hi = cfg['obs_range']
        total_range = hi - lo
        covered = 0.0
        for f in cfg['fibs']:
            zone_lo = f * (1 - TOL)
            zone_hi = f * (1 + TOL)
            overlap = max(0, min(zone_hi, hi) - max(zone_lo, lo))
            covered += overlap
        # Correct for overlapping zones
        random_p = min(covered / total_range, 0.95)

        btest = stats.binomtest(hits, len(vals), random_p, alternative='greater')

        results[col] = {
            'label'        : cfg['label'],
            'n'            : len(vals),
            'mean'         : float(np.mean(vals)),
            'median'       : float(np.median(vals)),
            'std'          : float(np.std(vals)),
            'hit_rate'     : hit_rate,
            'random_p'     : random_p,
            'lift'         : hit_rate / random_p if random_p > 0 else None,
            'p_value'      : float(btest.pvalue),
            'significant'  : btest.pvalue < 0.05,
        }

    return results


def corrective_retrace_analysis(swings):
    """
    For every impulse A->B, measure retrace B->C.
    Test: does C land at Fibonacci retrace level of AB?
    Fibonacci targets: 38.2%, 50%, 61.8%  (+/-8% absolute tolerance)
    """
    FIB_LEVELS = [0.382, 0.500, 0.618]
    TOL        = 0.08   # absolute tolerance on retrace ratio

    records = []
    min_impulse = 3.0

    for i in range(len(swings) - 2):
        a, b, c = swings[i], swings[i+1], swings[i+2]
        if a['type'] == b['type']:
            continue
        impulse = abs(b['price'] - a['price'])
        if impulse < min_impulse:
            continue
        retrace = abs(c['price'] - b['price'])
        ratio   = retrace / impulse

        is_fib  = any(abs(ratio - f) < TOL for f in FIB_LEVELS)
        records.append({'ratio': ratio, 'is_fib': is_fib, 'impulse': impulse})

    if len(records) < 30:
        return {}

    df_r = pd.DataFrame(records)
    hit_rate = df_r['is_fib'].mean()
    n        = len(df_r)
    hits     = int(df_r['is_fib'].sum())

    # Random baseline: 3 zones of width 0.16 over [0, 2.0] range
    random_p = (3 * TOL * 2) / 2.0   # = 3 * 0.16 / 2.0 = 0.24
    btest    = stats.binomtest(hits, n, random_p, alternative='greater')

    # Distribution
    bins_  = [0, 0.236, 0.382, 0.500, 0.618, 0.764, 1.0, 1.5, 2.5]
    counts_= np.histogram(df_r['ratio'].clip(0, 2.4), bins=bins_)[0]
    dist   = {f'[{bins_[k]:.3f}-{bins_[k+1]:.3f})': int(counts_[k]) for k in range(len(counts_))}

    return {
        'n'            : n,
        'hit_rate'     : hit_rate,
        'random_p'     : random_p,
        'lift'         : hit_rate / random_p,
        'p_value'      : float(btest.pvalue),
        'significant'  : btest.pvalue < 0.05,
        'median_ratio' : float(df_r['ratio'].median()),
        'distribution' : dist,
    }


def ew_post_impulse_prediction(sequences, df):
    """
    After identifying a completed 5-wave impulse, predict that price will
    retrace to 38.2-61.8% of the total impulse magnitude.
    Measure hit rate over next 20 bars. Compare to random baseline.
    """
    if len(sequences) < 15:
        return {}

    FIB_ZONE_LO = 0.30
    FIB_ZONE_HI = 0.70
    LOOKAHEAD   = 20

    records = []

    for seq in sequences:
        end_i = seq['end_idx']
        if end_i + LOOKAHEAD >= len(df):
            continue

        # Total impulse: net move of 5 wave up-legs
        total_impulse = seq['w1'] + seq['w3'] + seq['w5']
        end_price     = df.loc[end_i, 'close']

        future = df.iloc[end_i+1 : end_i+LOOKAHEAD+1]

        if seq['direction'] == 'bull':
            extreme      = future['low'].min()
            actual_retr  = end_price - extreme
        else:
            extreme      = future['high'].max()
            actual_retr  = extreme - end_price

        ratio = actual_retr / total_impulse if total_impulse > 0 else 0
        predicted_hit = FIB_ZONE_LO <= ratio <= FIB_ZONE_HI

        records.append({
            'ratio'         : ratio,
            'predicted_hit' : predicted_hit,
            'total_impulse' : total_impulse,
        })

    if len(records) < 10:
        return {}

    df_r     = pd.DataFrame(records)
    hit_rate = df_r['predicted_hit'].mean()
    n        = len(df_r)
    # Random baseline: FIB_ZONE occupies 40% of [0, 1.0] range (plus beyond 1.0 cases)
    # Empirically, about 40% of random retraces land in [0.3, 0.7]
    # Use actual Uniform[0,1.5] -> P[0.3,0.7] = 0.4/1.5 = 0.267
    random_p = 0.267
    btest    = stats.binomtest(int(df_r['predicted_hit'].sum()), n, random_p, alternative='greater')

    return {
        'n'           : n,
        'hit_rate'    : hit_rate,
        'random_p'    : random_p,
        'lift'        : hit_rate / random_p,
        'p_value'     : float(btest.pvalue),
        'significant' : btest.pvalue < 0.05,
        'avg_ratio'   : float(df_r['ratio'].mean()),
        'median_ratio': float(df_r['ratio'].median()),
    }


def ew_ambiguity_score(df, order_lo=3, order_hi=10):
    """
    Measure how many different valid 5-wave counts exist for the same
    data when swing detection order parameter varies.
    Higher count = more ambiguous = harder to automate.
    """
    counts_by_order = {}
    for order in range(order_lo, order_hi+1):
        sw = find_swings(df['high'], df['low'], order=order)
        seqs = find_5wave_sequences(sw, min_wave1_pts=3.0)
        counts_by_order[order] = len(seqs)

    total_across_orders = sum(counts_by_order.values())
    return {
        'counts_by_order' : counts_by_order,
        'total'           : total_across_orders,
        'avg_per_order'   : total_across_orders / len(counts_by_order),
        'min'             : min(counts_by_order.values()),
        'max'             : max(counts_by_order.values()),
        'ratio_max_min'   : (max(counts_by_order.values()) /
                             max(min(counts_by_order.values()), 1)),
    }


# -----------------------------------------------------------------------------
# SECTION 2 -- VOLUME-BASED STRUCTURE
# -----------------------------------------------------------------------------

def compute_session_vpoc(df, price_bin=1.0):
    """
    Compute VPOC (highest-volume price level) for each session (AM/PM per day).
    Uses close price binned to price_bin granularity.
    """
    df = df.copy()
    df['price_bin'] = (df['close'] / price_bin).round() * price_bin
    df['session_key'] = df['date'].astype(str) + '_' + df['session']

    vpocs = []
    for sk, grp in df.groupby('session_key'):
        if len(grp) < 6:
            continue
        vol_profile = grp.groupby('price_bin')['volume'].sum()
        vpoc_price  = vol_profile.idxmax()
        vpoc_vol    = vol_profile.max()
        session_high = grp['high'].max()
        session_low  = grp['low'].min()
        session_mid  = (session_high + session_low) / 2

        vpocs.append({
            'session_key'  : sk,
            'date'         : grp['date'].iloc[0],
            'session'      : grp['session'].iloc[0],
            'vpoc'         : vpoc_price,
            'vpoc_vol'     : vpoc_vol,
            'session_range': session_high - session_low,
            'session_mid'  : session_mid,
            'vpoc_vs_mid'  : vpoc_price - session_mid,
        })

    return pd.DataFrame(vpocs)


def test_vpoc_intra_session_magnet(df, price_bin=1.0, lookahead=10, tol_pts=1.0):
    """
    Within a session, check if price returns to the developing VPOC.
    For each bar i, compute VPOC of bars [0..i-1] in the session.
    Check if any of the next `lookahead` bars touch within tol_pts of that VPOC.
    This tests the MAGNET hypothesis: price is attracted to volume center.
    """
    df = df.copy()
    df['price_bin'] = (df['close'] / price_bin).round() * price_bin
    df['session_key'] = df['date'].astype(str) + '_' + df['session']

    records = []

    for sk, grp in df.groupby('session_key'):
        grp = grp.reset_index(drop=True)
        n   = len(grp)
        if n < lookahead + 6:
            continue

        # Build rolling volume profile
        vol_accum = {}

        for i in range(n):
            pb = grp.loc[i, 'price_bin']
            vl = grp.loc[i, 'volume']
            vol_accum[pb] = vol_accum.get(pb, 0) + vl

            if i < 5:
                continue
            if i + lookahead >= n:
                break

            current_vpoc  = max(vol_accum, key=vol_accum.get)
            current_close = grp.loc[i, 'close']
            dist_from_vpoc= abs(current_close - current_vpoc)

            # Does price touch VPOC zone in next bars?
            future = grp.iloc[i+1 : i+lookahead+1]
            touches = (
                (future['low']  <= current_vpoc + tol_pts) &
                (future['high'] >= current_vpoc - tol_pts)
            ).any()

            above_vpoc = current_close > current_vpoc

            records.append({
                'dist_from_vpoc' : dist_from_vpoc,
                'above_vpoc'     : above_vpoc,
                'touches_vpoc'   : touches,
                'dist_atr_norm'  : dist_from_vpoc / (grp['range'].mean() * 2 + 0.01),
            })

    if len(records) < 100:
        return {}

    df_r = pd.DataFrame(records)
    overall = df_r['touches_vpoc'].mean()

    # Split by distance quartile
    df_r['dist_q'] = pd.qcut(df_r['dist_from_vpoc'], q=4,
                              labels=['Q1 (close)', 'Q2', 'Q3', 'Q4 (far)'])
    by_dist = df_r.groupby('dist_q')['touches_vpoc'].agg(['mean', 'count'])

    # Test if "close to VPOC" touch rate > "far from VPOC" touch rate
    q1 = df_r[df_r['dist_q'] == 'Q1 (close)']['touches_vpoc']
    q4 = df_r[df_r['dist_q'] == 'Q4 (far)']['touches_vpoc']
    _, pval = stats.ttest_ind(q1, q4, alternative='two-sided')

    return {
        'n_obs'          : len(df_r),
        'overall_touch_rate' : float(overall),
        'by_distance_quartile' : by_dist.to_dict(),
        'q1_touch_rate'  : float(q1.mean()),
        'q4_touch_rate'  : float(q4.mean()),
        'pval_q1_vs_q4'  : float(pval),
        'magnet_signal'  : overall > 0.5,
    }


def test_hv_bar_sr_role(df, vol_mult=2.0, lookahead=15, tol_pts=0.8):
    """
    High-volume bars (>vol_mult x avg20): do they act as S/R?
    Logic:
      1. Identify each HV bar.
      2. In next `lookahead` bars, check if price returns to HV zone.
      3. When price returns, does it reverse (barrier) or continue (breaks through)?
    Random baseline: same test on non-HV bars.
    """
    hv_mask = df['vol_ratio'] >= vol_mult

    def _test_bars(mask):
        recs = []
        indices = df.index[mask].tolist()
        for idx in indices:
            pos = df.index.get_loc(idx)
            if pos + lookahead >= len(df):
                continue

            bar     = df.iloc[pos]
            zone_hi = bar['high'] + tol_pts
            zone_lo = bar['low']  - tol_pts
            hv_mid  = (bar['high'] + bar['low']) / 2

            future = df.iloc[pos+1 : pos+lookahead+1]

            # Does price re-enter the zone?
            in_zone = (future['low'] <= zone_hi) & (future['high'] >= zone_lo)
            if not in_zone.any():
                continue

            first_return_pos = in_zone.idxmax()
            fpos_loc = df.index.get_loc(first_return_pos)

            # Approach direction (what was price doing before returning?)
            pre = df.iloc[max(0, pos+1) : fpos_loc]
            if len(pre) == 0:
                continue
            approach_dir = 1 if pre.iloc[-1]['close'] > hv_mid else -1

            # What happens after the touch?
            post = df.iloc[fpos_loc : min(len(df), fpos_loc+5)]
            if len(post) < 2:
                continue
            after_move = post.iloc[-1]['close'] - post.iloc[0]['open']

            # Reversal: if approaching from above, does price bounce up?
            reversal = (
                (approach_dir == 1  and after_move > 0) or
                (approach_dir == -1 and after_move < 0)
            )

            recs.append({
                'vol_ratio' : float(bar['vol_ratio']),
                'hv_range'  : float(bar['high'] - bar['low']),
                'reversal'  : reversal,
                'approach_dir': approach_dir,
            })
        return recs

    hv_recs  = _test_bars(hv_mask)
    # Randomly sample non-HV bars as baseline
    rng = np.random.default_rng(42)
    non_hv_idx = df.index[~hv_mask & df['vol_ratio'].notna()].tolist()
    sample_idx = rng.choice(non_hv_idx, size=min(500, len(non_hv_idx)), replace=False)
    sample_mask = df.index.isin(sample_idx)
    lv_recs = _test_bars(sample_mask)

    if len(hv_recs) < 30 or len(lv_recs) < 30:
        return {}

    df_hv = pd.DataFrame(hv_recs)
    df_lv = pd.DataFrame(lv_recs)

    hv_rev  = float(df_hv['reversal'].mean())
    lv_rev  = float(df_lv['reversal'].mean())

    # Chi-square test
    ct = np.array([
        [df_hv['reversal'].sum(),   len(df_hv) - df_hv['reversal'].sum()],
        [df_lv['reversal'].sum(),   len(df_lv) - df_lv['reversal'].sum()],
    ])
    chi2, pval, _, _ = stats.chi2_contingency(ct)

    # Breakdown by vol ratio band
    bands = {}
    for lo, hi, label in [(2, 3, '2-3x'), (3, 4, '3-4x'), (4, 99, '4x+')]:
        sub = df_hv[(df_hv['vol_ratio'] >= lo) & (df_hv['vol_ratio'] < hi)]
        if len(sub) >= 10:
            bands[label] = {'n': len(sub), 'reversal_rate': float(sub['reversal'].mean())}

    return {
        'total_hv_bars'       : int(hv_mask.sum()),
        'hv_returns_to_zone'  : len(hv_recs),
        'lv_returns_to_zone'  : len(lv_recs),
        'hv_reversal_rate'    : hv_rev,
        'lv_reversal_rate'    : lv_rev,
        'lift_vs_random'      : hv_rev / lv_rev if lv_rev > 0 else None,
        'chi2'                : float(chi2),
        'p_value'             : float(pval),
        'significant'         : pval < 0.05,
        'interpretation'      : 'BARRIER (reversal > baseline)' if hv_rev > lv_rev + 0.05
                                else 'NEUTRAL',
        'by_vol_band'         : bands,
    }


def test_volume_surge_direction(df, vol_mult=2.5, lookahead=5):
    """
    Volume surge bars (>vol_mult x avg): does price continue in bar direction?
    Tests both sessions separately.
    Baseline: all bars regardless of volume.
    """
    def _test(mask):
        recs = []
        for pos in df.index[mask].tolist():
            p = df.index.get_loc(pos)
            if p + lookahead >= len(df):
                continue
            bar = df.iloc[p]
            future = df.iloc[p+1 : p+lookahead+1]
            move  = future.iloc[-1]['close'] - bar['close']
            bdir  = bar['bar_dir']
            cont  = (bdir == 1 and move > 0) or (bdir == -1 and move < 0)
            mfe   = (future['high'].max() - bar['close']) if bdir == 1 \
                     else (bar['close'] - future['low'].min())
            recs.append({'cont': cont, 'mfe': mfe, 'session': bar['session']})
        return pd.DataFrame(recs) if recs else pd.DataFrame()

    surge_mask = df['vol_ratio'] >= vol_mult
    df_surge   = _test(surge_mask)

    # Baseline: all bars
    rng = np.random.default_rng(42)
    all_valid = df.index[df['vol_ratio'].notna()].tolist()
    sample    = rng.choice(all_valid, size=min(2000, len(all_valid)), replace=False)
    df_base   = _test(df.index.isin(sample))

    if len(df_surge) < 30:
        return {}

    surge_cont  = float(df_surge['cont'].mean())
    base_cont   = float(df_base['cont'].mean()) if len(df_base) > 0 else 0.5

    btest = stats.binomtest(int(df_surge['cont'].sum()), len(df_surge), 0.5)

    by_session = {}
    for sess in ['AM', 'PM']:
        sub = df_surge[df_surge['session'] == sess]
        if len(sub) >= 15:
            by_session[sess] = {
                'n'           : len(sub),
                'cont_rate'   : float(sub['cont'].mean()),
                'avg_mfe'     : float(sub['mfe'].mean()),
            }

    return {
        'n_surges'     : len(df_surge),
        'surge_cont'   : surge_cont,
        'base_cont'    : base_cont,
        'avg_mfe_surge': float(df_surge['mfe'].mean()),
        'avg_mfe_base' : float(df_base['mfe'].mean()) if len(df_base) > 0 else None,
        'p_value'      : float(btest.pvalue),
        'significant'  : btest.pvalue < 0.05,
        'by_session'   : by_session,
        'interpretation': 'CONTINUATION' if surge_cont > 0.55
                          else ('REVERSAL' if surge_cont < 0.45 else 'NEUTRAL'),
    }


def test_hv_level_revisit_vs_random(df, vol_mult=2.0, window=20, tol=1.0):
    """
    Predictive test:
      HV hypothesis: price will revisit a HV bar's level within `window` bars.
      Random baseline: same test on random price levels.
    Returns lift ratio and chi-square p-value.
    """
    hv_bars = df[df['vol_ratio'] >= vol_mult].index.tolist()

    def _revisit_rate(level_bars, level_fn):
        hits = 0
        total = 0
        for pos_idx in level_bars:
            pos = df.index.get_loc(pos_idx)
            if pos + window >= len(df):
                continue
            level = level_fn(pos_idx)
            future = df.iloc[pos+1 : pos+window+1]
            revisits = (
                (future['low']  <= level + tol) &
                (future['high'] >= level - tol)
            ).any()
            hits  += int(revisits)
            total += 1
        return hits, total

    # HV level = midpoint of HV bar
    hv_hits, hv_total = _revisit_rate(
        hv_bars,
        lambda i: (df.loc[i, 'high'] + df.loc[i, 'low']) / 2
    )

    # Random levels: randomly chosen bars, level = close +/- random offset
    rng = np.random.default_rng(42)
    rand_idx = rng.choice(
        df.index[df['vol_ratio'].notna()].tolist(),
        size=min(600, len(df) - window - 1), replace=False
    )
    rand_offsets = rng.uniform(-5.0, 5.0, size=len(rand_idx))

    rand_hits, rand_total = 0, 0
    for pos_idx, offset in zip(rand_idx, rand_offsets):
        pos = df.index.get_loc(pos_idx)
        if pos + window >= len(df):
            continue
        level  = df.loc[pos_idx, 'close'] + offset
        future = df.iloc[pos+1 : pos+window+1]
        revisits = (
            (future['low']  <= level + tol) &
            (future['high'] >= level - tol)
        ).any()
        rand_hits  += int(revisits)
        rand_total += 1

    hv_rate   = hv_hits / hv_total   if hv_total   > 0 else 0
    rand_rate = rand_hits / rand_total if rand_total > 0 else 0

    ct = np.array([
        [hv_hits,   hv_total   - hv_hits],
        [rand_hits, rand_total - rand_hits],
    ])
    chi2, pval, _, _ = stats.chi2_contingency(ct)

    return {
        'hv_n'          : hv_total,
        'hv_revisit'    : hv_rate,
        'rand_n'        : rand_total,
        'rand_revisit'  : rand_rate,
        'lift'          : hv_rate / rand_rate if rand_rate > 0 else None,
        'p_value'       : float(pval),
        'significant'   : pval < 0.05,
    }


# -----------------------------------------------------------------------------
# SECTION 3 -- SIMPLE VOLUME NODE RULE BACKTEST
# -----------------------------------------------------------------------------

def backtest_hv_fade_rule(df, vol_mult=2.0, lookback=78,
                           entry_tol=0.6, sl_atr_mult=1.5,
                           tp_pts=5.0, max_hold=12):
    """
    Simple automatable rule:
      - Identify the most recent HV bar within last `lookback` bars.
      - When current price enters the HV bar's range (+/-entry_tol),
        fade the move (mean-reversion bet).
      - Exit: TP at tp_pts, SL at sl_atr_mult x ATR14, or max_hold bars.
    Also tests momentum variant (breakout continuation).
    """
    df = df.copy()
    signals_fade = []
    signals_mo   = []

    for i in range(lookback + 15, len(df) - max_hold - 2):
        cur = df.iloc[i]
        if pd.isna(cur['vol_ratio']) or pd.isna(cur['atr14']):
            continue

        hist = df.iloc[i - lookback : i]
        hv   = hist[hist['vol_ratio'] >= vol_mult]
        if len(hv) == 0:
            continue

        # Use most recent HV bar
        last_hv  = hv.iloc[-1]
        hv_mid   = (last_hv['high'] + last_hv['low']) / 2
        hv_range = last_hv['high'] - last_hv['low']

        dist = abs(cur['close'] - hv_mid)
        if dist > entry_tol + hv_range / 2:
            continue

        atr   = float(cur['atr14'])
        entry = float(cur['close'])
        apdir = 1 if entry > hv_mid else -1   # approach direction

        future = df.iloc[i+1 : i+max_hold+1]
        if len(future) < 3:
            continue

        def _exit(entry, sl, tp, direction, future_bars):
            """Simulate bar-by-bar exit. direction: +1=long, -1=short."""
            for _, fb in future_bars.iterrows():
                if direction == 1:
                    if fb['high'] >= tp:
                        return tp - entry
                    if fb['low']  <= sl:
                        return sl - entry   # negative
                else:
                    if fb['low']  <= tp:
                        return entry - tp
                    if fb['high'] >= sl:
                        return entry - sl   # negative
            # Timeout: use last close
            last = future_bars.iloc[-1]['close']
            return (last - entry) * direction

        # FADE: bet against approach direction
        fade_dir = -apdir
        fade_sl  = entry + apdir * atr * sl_atr_mult  # SL in approach direction
        fade_tp  = entry - apdir * tp_pts              # TP against approach

        pnl_fade = _exit(entry, fade_sl, fade_tp, fade_dir, future)
        signals_fade.append({
            'pnl_gross': float(pnl_fade),
            'pnl_net'  : float(pnl_fade) - COST_PER_TRADE,
            'win'      : pnl_fade > COST_PER_TRADE,
        })

        # MOMENTUM: bet with approach direction
        mo_sl  = entry - apdir * atr * sl_atr_mult
        mo_tp  = entry + apdir * tp_pts

        pnl_mo = _exit(entry, mo_sl, mo_tp, apdir, future)
        signals_mo.append({
            'pnl_gross': float(pnl_mo),
            'pnl_net'  : float(pnl_mo) - COST_PER_TRADE,
            'win'      : pnl_mo > COST_PER_TRADE,
        })

    if len(signals_fade) < 30:
        return {}

    def _summarise(recs, label):
        df_r = pd.DataFrame(recs)
        wins   = df_r['pnl_net'][df_r['win']]
        losses = df_r['pnl_net'][~df_r['win']]
        gross_w = df_r['pnl_gross'][df_r['win']].sum()
        gross_l = abs(df_r['pnl_gross'][~df_r['win']].sum())
        pf = gross_w / gross_l if gross_l > 0 else float('inf')
        n_days  = df['date'].nunique()
        return {
            'label'     : label,
            'n_signals' : len(df_r),
            'per_day'   : len(df_r) / n_days,
            'wr'        : float(df_r['win'].mean()),
            'pf'        : pf,
            'avg_pnl'   : float(df_r['pnl_net'].mean()),
            'total_net' : float(df_r['pnl_net'].sum()),
            'per_day_net': float(df_r['pnl_net'].sum()) / n_days,
        }

    return {
        'fade'    : _summarise(signals_fade, 'FADE (mean-reversion)'),
        'momentum': _summarise(signals_mo,   'MOMENTUM (breakout)'),
    }


# -----------------------------------------------------------------------------
# PRINTING HELPERS
# -----------------------------------------------------------------------------

def hdr(title, width=72):
    pad = max(0, (width - len(title) - 2) // 2)
    print(f"\n{'='*pad} {title} {'='*(width - pad - len(title) - 2)}")

def sig_tag(p):
    if p < 0.001: return '*** p<0.001'
    if p < 0.01:  return '** p<0.01'
    if p < 0.05:  return '* p<0.05'
    return 'ns (p={:.3f})'.format(p)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print('='*72)
    print('  VN30F1M FUTURES: ELLIOTT WAVE vs VOLUME STRUCTURE ANALYSIS')
    print('='*72)

    print('\nLoading data...')
    df = load_data()
    n_days = df['date'].nunique()
    print(f'  Bars: {len(df):,}  |  Trading days: {n_days}')
    print(f'  Period: {df["time"].iloc[0].date()} to {df["time"].iloc[-1].date()}')
    print(f'  Price range: {df["close"].min():.0f} - {df["close"].max():.0f} pts')
    print(f'  ATR14 (avg): {df["atr14"].mean():.2f} pts  |  '
          f'Bar range (avg): {df["range"].mean():.2f} pts')
    print(f'  Volume (mean): {int(df["volume"].mean())}  |  '
          f'High-vol bars (>2x): {int((df["vol_ratio"] >= 2.0).sum())}')

    # --- SECTION 1: ELLIOTT WAVE ----------------------------------------------
    hdr('1. ELLIOTT WAVE VIABILITY TEST')

    # 1a. Swing detection with primary order=5
    print('\n[1a] Swing detection')
    swings_primary = find_swings(df['high'], df['low'], order=5)
    n_h = sum(1 for s in swings_primary if s['type'] == 'H')
    n_l = sum(1 for s in swings_primary if s['type'] == 'L')
    print(f'  order=5 swings: {len(swings_primary)} ({n_h} highs + {n_l} lows)')
    print(f'  Avg swing size: '
          f'{np.mean([abs(swings_primary[i]["price"] - swings_primary[i-1]["price"]) for i in range(1, len(swings_primary))]):.2f} pts')

    # 1b. 5-wave sequence detection
    print('\n[1b] 5-wave impulse sequence search')
    seqs = find_5wave_sequences(swings_primary, min_wave1_pts=3.0)
    print(f'  Candidate sequences (order=5, W1>=3pts): {len(seqs)}')

    if len(seqs) >= 20:
        df_s = pd.DataFrame(seqs)
        print(f'  Bull / Bear split: {(df_s["direction"]=="bull").sum()} / '
              f'{(df_s["direction"]=="bear").sum()}')
        print(f'  Wave size summary (pts):')
        for col in ['w1', 'w3', 'w5']:
            vals = df_s[col]
            print(f'    {col}: mean={vals.mean():.2f} | median={vals.median():.2f} | '
                  f'min={vals.min():.2f} | max={vals.max():.2f}')
        print(f'  Key ratio summary:')
        for col, label in [('w3_ext_w1','W3/W1'), ('w2_ret_w1','W2 retrace'),
                           ('w4_ret_w3','W4 retrace'), ('w5_vs_w1','W5/W1')]:
            vals = df_s[col]
            print(f'    {label}: mean={vals.mean():.3f} | median={vals.median():.3f} | '
                  f'std={vals.std():.3f}')

    # 1c. Fibonacci ratio statistical test
    print('\n[1c] Fibonacci ratio test (does W3/W1 cluster at 1.618 etc?)')
    fib_res = fib_ratio_statistical_test(seqs)

    for col, res in fib_res.items():
        print(f'\n  {res["label"]} (n={res["n"]}):')
        print(f'    Observed Fib hit rate : {res["hit_rate"]:.1%}')
        print(f'    Random baseline       : {res["random_p"]:.1%}')
        print(f'    Lift                  : {res["lift"]:.2f}x')
        print(f'    Statistical test      : {sig_tag(res["p_value"])}')
        print(f'    Mean ratio            : {res["mean"]:.3f}  (median: {res["median"]:.3f})')

    if not fib_res:
        print('  Insufficient data for ratio test (<20 sequences)')

    # 1d. Corrective retrace test
    print('\n[1d] Corrective retrace test (does ABC land at 38.2/50/61.8%?)')
    retr_res = corrective_retrace_analysis(swings_primary)

    if retr_res:
        print(f'  n impulse-correction pairs: {retr_res["n"]}')
        print(f'  Fib zone hit rate : {retr_res["hit_rate"]:.1%}  '
              f'(random baseline: {retr_res["random_p"]:.1%})')
        print(f'  Lift              : {retr_res["lift"]:.2f}x')
        print(f'  Statistical test  : {sig_tag(retr_res["p_value"])}')
        print(f'  Median retrace    : {retr_res["median_ratio"]:.3f}')
        print(f'  Retrace distribution:')
        for zone, cnt in retr_res['distribution'].items():
            bar = '#' * int(cnt / max(retr_res['distribution'].values()) * 30)
            print(f'    {zone}: {cnt:4d}  {bar}')

    # 1e. Post-impulse correction accuracy
    print('\n[1e] EW prediction test (after 5-wave, does price retrace 38-62%?)')
    ew_pred = ew_post_impulse_prediction(seqs, df)
    if ew_pred:
        print(f'  n completed sequences  : {ew_pred["n"]}')
        print(f'  EW prediction hit rate : {ew_pred["hit_rate"]:.1%}  '
              f'(random: {ew_pred["random_p"]:.1%})')
        print(f'  Lift                   : {ew_pred["lift"]:.2f}x')
        print(f'  Statistical test       : {sig_tag(ew_pred["p_value"])}')
        print(f'  Avg actual retrace     : {ew_pred["avg_ratio"]:.3f}  '
              f'(median: {ew_pred["median_ratio"]:.3f})')

    # 1f. Ambiguity across swing detection parameters
    print('\n[1f] EW ambiguity score (different order params = different counts)')
    amb = ew_ambiguity_score(df, order_lo=3, order_hi=10)
    print(f'  5-wave sequences found per detection order:')
    for order, cnt in amb['counts_by_order'].items():
        bar = '#' * min(cnt // 2, 40)
        print(f'    order={order}: {cnt:4d}  {bar}')
    print(f'  Max/Min ratio: {amb["ratio_max_min"]:.1f}x  '
          f'(higher = more ambiguous)')
    print(f'  Conclusion: Same data, {amb["ratio_max_min"]:.0f}x more sequences'
          f' from different parameter choices')

    # --- SECTION 2: VOLUME STRUCTURE -----------------------------------------
    hdr('2. VOLUME-BASED STRUCTURE ANALYSIS')

    # 2a. Session VPOC computation
    print('\n[2a] Session VPOC computation')
    vpoc_df = compute_session_vpoc(df, price_bin=1.0)
    print(f'  Sessions with VPOC: {len(vpoc_df)}')
    print(f'  VPOC vs session midpoint (avg deviation): '
          f'{vpoc_df["vpoc_vs_mid"].abs().mean():.2f} pts')
    above_mid = (vpoc_df['vpoc_vs_mid'] > 0).mean()
    print(f'  VPOC above session midpoint: {above_mid:.1%}  '
          f'(50% = random, >55% = skew)')

    # 2b. Intra-session VPOC magnet test
    print('\n[2b] Intra-session VPOC magnet test (does price orbit VPOC?)')
    magnet_res = test_vpoc_intra_session_magnet(df, price_bin=1.0, lookahead=10)
    if magnet_res:
        print(f'  n bar observations: {magnet_res["n_obs"]:,}')
        print(f'  Overall VPOC touch rate (next 10 bars): '
              f'{magnet_res["overall_touch_rate"]:.1%}')
        print(f'  By distance quartile (Q1=closest, Q4=farthest):')
        for q, vals in magnet_res['by_distance_quartile']['mean'].items():
            cnt = magnet_res['by_distance_quartile']['count'][q]
            print(f'    {q}: {vals:.1%} touch rate (n={cnt:.0f})')
        print(f'  Q1 vs Q4 t-test: {sig_tag(magnet_res["pval_q1_vs_q4"])}')
        print(f'  Magnet signal: {"YES" if magnet_res["magnet_signal"] else "WEAK"}')

    # 2c. High-volume bar as S/R
    print('\n[2c] High-volume bars (>2x avg) as support/resistance zones')
    hv_sr = test_hv_bar_sr_role(df, vol_mult=2.0, lookahead=15)
    if hv_sr:
        print(f'  Total HV bars (>2x avg): {hv_sr["total_hv_bars"]}')
        print(f'  HV bars with price return (15-bar window): {hv_sr["hv_returns_to_zone"]}')
        print(f'  Low-vol baseline returns: {hv_sr["lv_returns_to_zone"]}')
        print(f'  HV reversal rate : {hv_sr["hv_reversal_rate"]:.1%}')
        print(f'  LV reversal rate : {hv_sr["lv_reversal_rate"]:.1%}  (baseline)')
        print(f'  Lift vs baseline : {hv_sr["lift_vs_random"]:.2f}x')
        print(f'  Statistical test : {sig_tag(hv_sr["p_value"])}')
        print(f'  Role: {hv_sr["interpretation"]}')
        if hv_sr['by_vol_band']:
            print(f'  Breakdown by volume intensity:')
            for band, d in hv_sr['by_vol_band'].items():
                print(f'    {band}: reversal rate {d["reversal_rate"]:.1%} (n={d["n"]})')

    # 2d. Volume surge direction
    print('\n[2d] Volume surge (>2.5x avg) -> continuation or reversal? (5-bar)')
    surge_res = test_volume_surge_direction(df, vol_mult=2.5, lookahead=5)
    if surge_res:
        print(f'  n surges: {surge_res["n_surges"]}')
        print(f'  Surge continuation rate : {surge_res["surge_cont"]:.1%}')
        print(f'  All-bar baseline        : {surge_res["base_cont"]:.1%}')
        print(f'  Avg MFE after surge     : {surge_res["avg_mfe_surge"]:.2f} pts')
        if surge_res['avg_mfe_base']:
            print(f'  Avg MFE baseline        : {surge_res["avg_mfe_base"]:.2f} pts')
        print(f'  Statistical test        : {sig_tag(surge_res["p_value"])}')
        print(f'  Interpretation: {surge_res["interpretation"]}')
        for sess, d in surge_res['by_session'].items():
            print(f'    {sess}: {d["cont_rate"]:.1%} cont. | avg MFE {d["avg_mfe"]:.2f} | '
                  f'n={d["n"]}')

    # 2e. HV level revisit vs random
    print('\n[2e] HV level revisit test vs random price levels (20-bar window)')
    revisit = test_hv_level_revisit_vs_random(df, vol_mult=2.0, window=20)
    if revisit:
        print(f'  HV level revisit rate   : {revisit["hv_revisit"]:.1%} (n={revisit["hv_n"]})')
        print(f'  Random level revisit    : {revisit["rand_revisit"]:.1%} (n={revisit["rand_n"]})')
        print(f'  Lift                    : {revisit["lift"]:.2f}x')
        print(f'  Statistical test        : {sig_tag(revisit["p_value"])}')

    # --- SECTION 3: COMPARISON -----------------------------------------------
    hdr('3. COMPARISON -- EW vs VOLUME PREDICTIVE ACCURACY')

    print('\n  Metric                         EW-based           Volume-based')
    print('  ' + '-'*60)

    ew_hit  = ew_pred.get('hit_rate', 0)  if ew_pred  else 0
    ew_rnd  = ew_pred.get('random_p', 0) if ew_pred   else 0
    ew_pval = ew_pred.get('p_value', 1)  if ew_pred   else 1
    ew_lift = ew_pred.get('lift', 1)     if ew_pred   else 1

    rv_hit  = revisit.get('hv_revisit', 0)   if revisit else 0
    rv_rnd  = revisit.get('rand_revisit', 0) if revisit else 0
    rv_pval = revisit.get('p_value', 1)      if revisit else 1
    rv_lift = revisit.get('lift', 1)         if revisit else 1

    print(f'  Hit rate (observed)            {ew_hit:.1%}               {rv_hit:.1%}')
    print(f'  Random baseline                {ew_rnd:.1%}               {rv_rnd:.1%}')
    print(f'  Lift vs random                 {ew_lift:.2f}x              {rv_lift:.2f}x')
    print(f'  Statistical significance       {sig_tag(ew_pval)[:12]:<14} {sig_tag(rv_pval)}')

    print('\n  Signal detection:')
    n_ew_signals = len(seqs) if seqs else 0
    n_vol_signals = int((df['vol_ratio'] >= 2.0).sum())
    print(f'    EW: {n_ew_signals} sequences in {n_days}d = '
          f'{n_ew_signals/n_days:.2f}/day')
    print(f'    Volume: {n_vol_signals} HV bars in {n_days}d = '
          f'{n_vol_signals/n_days:.1f}/day')

    # --- SECTION 4: AUTOMATION & SIMPLE RULE ---------------------------------
    hdr('4. AUTOMATION FEASIBILITY')

    print("""
VOLUME NODE DETECTION -- Real-time viability: YES
  - HV bar flag: volume > 2.0 * rolling_mean(volume, 20)   [O(1) per bar]
  - Session VPOC: running argmax of rolling volume profile  [O(levels) per bar]
  - No look-ahead required -- uses only completed past bars
  - Latency: microseconds per bar, trivial on 5m data
  - Key metrics are unambiguous: one number per bar, fully deterministic

ELLIOTT WAVE DETECTION -- Real-time viability: PROBLEMATIC
  - Wave count requires knowing where swings END (unknown in real-time)
  - Wave 3 only confirmed once wave 4 begins (structural lag)
  - Every new swing high/low can invalidate the prior count
  - Detection order parameter changes count by {:.0f}x (see Section 1f)
  - Professional EW practitioners routinely disagree on live counts
  - Algorithmic implementations require subjective parameter choices
  - Any backtest on EW uses look-ahead bias (you know the full wave)
  - Conclusion: NOT reliably automatable without severe ambiguity
""".format(amb.get('ratio_max_min', 0)))

    print('[4a] Simple volume node rule backtest (price at HV zone)')
    vol_rule = backtest_hv_fade_rule(
        df, vol_mult=2.0, lookback=78, entry_tol=0.6,
        sl_atr_mult=1.5, tp_pts=5.0, max_hold=12
    )
    if vol_rule:
        for variant_key in ['fade', 'momentum']:
            r = vol_rule[variant_key]
            print(f'\n  Variant: {r["label"]}')
            print(f'    Signals  : {r["n_signals"]} total ({r["per_day"]:.2f}/day)')
            print(f'    Win rate : {r["wr"]:.1%}')
            print(f'    Profit F : {r["pf"]:.2f}')
            print(f'    Avg P&L  : {r["avg_pnl"]:+.3f} pts/trade (after {COST_PER_TRADE} cost)')
            print(f'    Net/day  : {r["per_day_net"]:+.2f} pts/day')
            verdict = 'POSITIVE EDGE' if r['per_day_net'] > 0 and r['wr'] > 0.45 else 'NO EDGE'
            print(f'    Verdict  : {verdict}')
    else:
        print('  Insufficient signals for backtest')

    # --- FINAL VERDICT -------------------------------------------------------
    hdr('FINAL VERDICT')

    ew_sig   = retr_res.get('significant', False) if retr_res else False
    vol_sig  = revisit.get('significant', False)  if revisit  else False
    vol_lift_val = revisit.get('lift', 1.0)       if revisit  else 1.0
    ew_lift_val  = retr_res.get('lift', 1.0)      if retr_res else 1.0

    print(f"""
HYPOTHESIS TEST: Does VN30F1M follow Elliott Wave or Volume Structure?

ELLIOTT WAVE FINDINGS:
  - Fibonacci ratios in wave relationships:
      Significant? {ew_sig}
      Key issue: {amb.get("ratio_max_min", 0):.0f}x ambiguity across detection parameters
  - Corrective retraces at Fib levels:
      Hit rate {retr_res.get("hit_rate", 0):.1%} vs random {retr_res.get("random_p", 0):.1%}
      Lift {ew_lift_val:.2f}x -- Significant? {ew_sig}
  - Post-impulse prediction:
      Hit rate {ew_pred.get("hit_rate", 0):.1%} vs random {ew_pred.get("random_p", 0):.1%}
  - Automation: NOT VIABLE -- ambiguous counts, look-ahead dependency
  - Signal density: {n_ew_signals/n_days:.2f}/day (very low, highly subjective)

VOLUME STRUCTURE FINDINGS:
  - VPOC intra-session: {'MAGNET confirmed' if magnet_res and magnet_res.get("magnet_signal") else 'Weak magnet'}
      Touch rate {magnet_res.get("overall_touch_rate", 0):.1%} within 10 bars
  - HV bar S/R: reversal {hv_sr.get("hv_reversal_rate", 0):.1%} vs baseline {hv_sr.get("lv_reversal_rate", 0):.1%}
      Lift {hv_sr.get("lift_vs_random", 1):.2f}x -- Significant? {hv_sr.get("significant", False)}
  - HV level revisit: {rv_hit:.1%} vs random {rv_rnd:.1%} -- lift {vol_lift_val:.2f}x
  - Volume surges: {surge_res.get("interpretation", "N/A")} bias
      ({surge_res.get("surge_cont", 0):.1%} continuation rate, {surge_res.get("n_surges", 0)} events)
  - Automation: FULLY VIABLE -- real-time, deterministic, no ambiguity

WINNER: {'VOLUME STRUCTURE' if vol_lift_val > ew_lift_val else 'TIED / INCONCLUSIVE'}
  Volume provides {vol_lift_val:.2f}x lift vs {ew_lift_val:.2f}x for EW retraces

RECOMMENDED INTEGRATION INTO TRADING BOT:
  1. Use VPOC as intraday mean-reversion context filter
     - CB signals near VPOC: higher confirmation (price is at volume center)
     - CB signals far from VPOC: weaker (>2 ATR from session VPOC -> skip)
  2. Use HV bars as dynamic S/R levels
     - Block CB BUY entries when price is within 1pt of major resistance HV
     - Block CB SELL entries when price is within 1pt of major support HV
  3. Volume surge filter: use as momentum confirmation
     - After volume surge in CB direction -> increase confidence
  4. Do NOT implement EW counting as systematic signal
     - Subjectivity and look-ahead bias make it untestable and unautomatable
""")

    print('='*72)
    print('Analysis complete.')
    print('='*72)


if __name__ == '__main__':
    import os
    os.chdir('e:/Trading')
    main()
