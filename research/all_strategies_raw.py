"""Test ALL strategies with the winning approach:
  - Wide SL (3.0x ATR)
  - No trailing, no TP
  - Exit at session end only
  - Optional regime filter (price_below EMA50)

Also analyze opportunity coverage:
  - How many "real moves" (>5pts in 1 direction within session) exist?
  - How many does each strategy capture?
  - What % of total opportunities are covered by all strategies combined?
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import pandas_ta as ta
import importlib
from collections import defaultdict

COST = 0.96

STRATEGY_MAP = {
    'momentum_trend': ('strategies.momentum_trend', 'MomentumTrendStrategy'),
    'macd_cross': ('strategies.macd_cross', 'MACDCrossStrategy'),
    'heikin_ashi': ('strategies.heikin_ashi', 'HeikinAshiStrategy'),
    'fibonacci': ('strategies.fibonacci', 'FibonacciStrategy'),
    'market_structure': ('strategies.market_structure', 'MarketStructureStrategy'),
    'bos': ('strategies.bos', 'BOSStrategy'),
    'choch': ('strategies.choch', 'CHoCHStrategy'),
    'bb_squeeze': ('strategies.bb_squeeze', 'BBSqueezeStrategy'),
    'ma_crossover': ('strategies.ma_crossover', 'MACrossoverStrategy'),
    'narrow_range': ('strategies.narrow_range', 'NarrowRangeStrategy'),
}


def _get_strategy(name):
    module_path, class_name = STRATEGY_MAP[name]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


def load_data():
    df = pd.read_parquet('data/vn30f1m_1m.parquet')
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['session'] = df['mins'].apply(lambda m: 'AM' if m < 720 else 'PM')
    return df


def compute_indicators(df):
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi_7'] = ta.rsi(df['close'], length=7)
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['rsi'] = df['rsi_14']

    bb = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    df['bb_mid'] = bb['BBM_20_2.0']
    df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_pos'] = df['bb_pos'].clip(0, 1)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=1.5)
    df['kc_upper'] = kc['KCUe_20_1.5']
    df['kc_lower'] = kc['KCLe_20_1.5']
    df['kc_pos'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
    df['kc_pos'] = df['kc_pos'].clip(0, 1)

    df['ema8'] = ta.ema(df['close'], length=8)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema_8'] = df['ema8']
    df['ema_21'] = df['ema21']
    df['ema_50'] = df['ema50']

    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd_hist'] = macd['MACDh_12_26_9']
    df['macd_line'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']

    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']

    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14']
    df['di_plus'] = adx_df['DMP_14']
    df['di_minus'] = adx_df['DMN_14']
    df['di_spread'] = df['di_plus'] - df['di_minus']

    df['body_pct'] = (abs(df['close'] - df['open']) /
                      (df['high'] - df['low']).replace(0, np.nan)).fillna(0)

    df['ema50_slope'] = (df['ema50'] - df['ema50'].shift(10)) / df['atr_14']
    df['price_vs_ema50'] = (df['close'] - df['ema50']) / df['atr_14']

    # For narrow_range / bb_squeeze
    df['range'] = df['high'] - df['low']
    df['atr'] = df['atr_14']

    return df


def check_regime_sell(df, idx):
    """price_below: price < EMA50 - 0.5*ATR"""
    row = df.iloc[idx]
    return row['price_vs_ema50'] < -0.5


def check_regime_buy(df, idx):
    """price_above: price > EMA50 + 0.5*ATR"""
    row = df.iloc[idx]
    return row['price_vs_ema50'] > 0.5


def backtest_raw(df, strategy_name, session_filter, dir_filter,
                 sl_mult=3.0, use_regime=False, max_hold_bars=60):
    """Backtest with wide SL, no trail, no TP, session exit."""
    try:
        strategy = _get_strategy(strategy_name)
    except Exception as e:
        return pd.DataFrame()

    trades = []
    i = 60  # warmup
    n = len(df)
    last_signal_idx = -10

    while i < n:
        row = df.iloc[i]

        # Session filter
        if session_filter != 'ALL' and row['session'] != session_filter:
            i += 1
            continue

        # Time filter
        if row['session'] == 'AM' and not (555 <= row['mins'] <= 645):
            i += 1
            continue
        if row['session'] == 'PM' and not (795 <= row['mins'] <= 855):
            i += 1
            continue

        # Dedup
        if i - last_signal_idx < 5:
            i += 1
            continue

        # Detect signal
        try:
            signal = strategy.detect(df, i)
        except Exception:
            i += 1
            continue

        if dir_filter == 'BUY' and signal != 1:
            i += 1
            continue
        if dir_filter == 'SELL' and signal != -1:
            i += 1
            continue
        if signal == 0:
            i += 1
            continue

        direction = signal

        # Regime filter
        if use_regime:
            if direction == -1 and not check_regime_sell(df, i):
                i += 1
                continue
            if direction == 1 and not check_regime_buy(df, i):
                i += 1
                continue

        entry_price = df['close'].iloc[i]
        atr = df['atr_14'].iloc[i]
        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        sl_dist = sl_mult * atr
        sl_price = entry_price - direction * sl_dist

        # Simulate
        exit_reason = 'MAX_HOLD'
        exit_price = entry_price
        exit_bar = i
        mfe = 0

        for j in range(i + 1, min(i + max_hold_bars + 1, n)):
            bar = df.iloc[j]

            # Session boundary
            if bar['date'] != df.iloc[i]['date'] or bar['session'] != row['session']:
                exit_reason = 'SESSION'
                exit_price = df['close'].iloc[j - 1]
                exit_bar = j - 1
                break
            if bar['session'] == 'AM' and bar['mins'] >= 685:
                exit_reason = 'SESSION'
                exit_price = bar['close']
                exit_bar = j
                break
            if bar['session'] == 'PM' and bar['mins'] >= 865:
                exit_reason = 'SESSION'
                exit_price = bar['close']
                exit_bar = j
                break

            # Update MFE
            if direction == 1:
                mfe = max(mfe, bar['high'] - entry_price)
            else:
                mfe = max(mfe, entry_price - bar['low'])

            # Check SL
            if direction == 1 and bar['low'] <= sl_price:
                exit_reason = 'SL'
                exit_price = sl_price
                exit_bar = j
                break
            if direction == -1 and bar['high'] >= sl_price:
                exit_reason = 'SL'
                exit_price = sl_price
                exit_bar = j
                break
        else:
            exit_bar = min(i + max_hold_bars, n - 1)
            exit_price = df['close'].iloc[exit_bar]

        pnl = (exit_price - entry_price) * direction - COST
        trades.append({
            'entry_time': df['time'].iloc[i],
            'entry_idx': i,
            'exit_time': df['time'].iloc[exit_bar],
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'atr': atr,
            'mfe': mfe,
            'exit_reason': exit_reason,
            'bars_held': exit_bar - i,
            'session': row['session'],
            'date': str(df['date'].iloc[i]),
        })

        last_signal_idx = i
        i = exit_bar + 1
        continue
        i += 1

    return pd.DataFrame(trades)


def find_real_opportunities(df, min_move_pts=5.0):
    """Find all 'real moves' in data: price moves >= min_move_pts within a session.

    A real opportunity is a move from local high to low (SELL) or low to high (BUY)
    that covers at least min_move_pts within a single session.

    Returns list of opportunities with time, direction, start_price, end_price, magnitude.
    """
    opportunities = []
    dates = df['date'].unique()

    for date in dates:
        for session in ['AM', 'PM']:
            if session == 'AM':
                mask = (df['date'] == date) & (df['mins'] >= 540) & (df['mins'] < 690)
            else:
                mask = (df['date'] == date) & (df['mins'] >= 780) & (df['mins'] < 870)

            session_df = df[mask]
            if len(session_df) < 10:
                continue

            prices = session_df['close'].values
            times = session_df['time'].values
            indices = session_df.index.values
            highs = session_df['high'].values
            lows = session_df['low'].values

            # Find swings using rolling window approach
            # Look for moves of min_move_pts from any point to subsequent point
            n = len(prices)

            # Track running max and min from each start point
            for start in range(0, n - 5, 3):  # check every 3 bars for speed
                max_after = prices[start]
                min_after = prices[start]

                for end in range(start + 1, n):
                    max_after = max(max_after, highs[end])
                    min_after = min(min_after, lows[end])

                    # BUY opportunity: price went up >= min_move from start price
                    up_move = max_after - prices[start]
                    if up_move >= min_move_pts:
                        opportunities.append({
                            'date': str(date),
                            'session': session,
                            'time': times[start],
                            'idx': indices[start],
                            'direction': 'BUY',
                            'magnitude': up_move,
                            'start_bar': start,
                        })
                        break

                    # SELL opportunity: price went down >= min_move from start price
                    down_move = prices[start] - min_after
                    if down_move >= min_move_pts:
                        opportunities.append({
                            'date': str(date),
                            'session': session,
                            'time': times[start],
                            'idx': indices[start],
                            'direction': 'SELL',
                            'magnitude': down_move,
                            'start_bar': start,
                        })
                        break

    # Dedup: keep the largest move per (date, session, direction)
    opp_df = pd.DataFrame(opportunities)
    if opp_df.empty:
        return opp_df

    opp_df = opp_df.sort_values('magnitude', ascending=False)
    opp_df = opp_df.drop_duplicates(subset=['date', 'session', 'direction'], keep='first')
    return opp_df.reset_index(drop=True)


def check_coverage(trades_df, opportunities_df, window_bars=10):
    """Check what % of opportunities were captured by a strategy.

    A trade "captures" an opportunity if:
    - Same date + session + direction
    - Trade entry within window_bars of opportunity start
    - Trade PnL > 0 (actually captured the move)
    """
    if trades_df.empty or opportunities_df.empty:
        return 0, 0, 0

    captured = 0
    captured_profit = 0
    total_opps = len(opportunities_df)

    for _, opp in opportunities_df.iterrows():
        matching = trades_df[
            (trades_df['date'] == opp['date']) &
            (trades_df['session'] == opp['session']) &
            (trades_df['direction'] == opp['direction'])
        ]
        if not matching.empty:
            captured += 1
            if (matching['pnl'] > 0).any():
                captured_profit += 1

    return captured, captured_profit, total_opps


def main():
    print("=" * 80)
    print("  ALL STRATEGIES: RAW APPROACH (SL3.0, no trail, session exit)")
    print("=" * 80)

    df = load_data()
    df = compute_indicators(df)
    print(f"  Data: {len(df)} bars, {df['date'].nunique()} days")
    print(f"  Period: {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}")

    # ═══════════════════════════════════════════════════════════
    # TEST ALL STRATEGIES
    # ═══════════════════════════════════════════════════════════
    strategies = list(STRATEGY_MAP.keys())
    sessions = ['AM', 'PM', 'ALL']
    directions = ['BUY', 'SELL']

    results = []

    print(f"\n{'='*80}")
    print(f"  FULL GRID: {len(strategies)} strategies x sessions x directions")
    print(f"{'='*80}\n")

    for strat_name in strategies:
        for sess in sessions:
            for dirn in directions:
                # Without regime
                trades = backtest_raw(df, strat_name, sess, dirn, sl_mult=3.0, use_regime=False)
                if not trades.empty and len(trades) >= 20:
                    n = len(trades)
                    wr = (trades['pnl'] > 0).mean() * 100
                    total = trades['pnl'].sum()
                    gross_w = trades[trades['pnl'] > 0]['pnl'].sum()
                    gross_l = abs(trades[trades['pnl'] <= 0]['pnl'].sum())
                    pf = gross_w / gross_l if gross_l > 0 else 99
                    days = max((trades['entry_time'].iloc[-1] - trades['entry_time'].iloc[0]).days, 1)
                    per_day = total / days

                    results.append({
                        'strategy': strat_name,
                        'session': sess,
                        'direction': dirn,
                        'regime': False,
                        'n': n,
                        'wr': wr,
                        'pf': pf,
                        'pnl': total,
                        'per_day': per_day,
                    })

                # With regime filter
                trades_r = backtest_raw(df, strat_name, sess, dirn, sl_mult=3.0, use_regime=True)
                if not trades_r.empty and len(trades_r) >= 15:
                    n = len(trades_r)
                    wr = (trades_r['pnl'] > 0).mean() * 100
                    total = trades_r['pnl'].sum()
                    gross_w = trades_r[trades_r['pnl'] > 0]['pnl'].sum()
                    gross_l = abs(trades_r[trades_r['pnl'] <= 0]['pnl'].sum())
                    pf = gross_w / gross_l if gross_l > 0 else 99
                    days = max((trades_r['entry_time'].iloc[-1] - trades_r['entry_time'].iloc[0]).days, 1)
                    per_day = total / days

                    results.append({
                        'strategy': strat_name,
                        'session': sess,
                        'direction': dirn,
                        'regime': True,
                        'n': n,
                        'wr': wr,
                        'pf': pf,
                        'pnl': total,
                        'per_day': per_day,
                    })

    # Sort by PF and show profitable ones
    results_df = pd.DataFrame(results)
    profitable = results_df[results_df['pf'] > 1.0].sort_values('pf', ascending=False)

    print("  PROFITABLE CONFIGS (PF > 1.0, sorted by PF):")
    print(f"  {'Strategy':<18} {'Sess':<4} {'Dir':<5} {'Rgm':<4} {'N':>4} {'WR':>5} {'PF':>5} {'PnL':>7} {'/d':>6}")
    print(f"  {'-'*18} {'-'*4} {'-'*5} {'-'*4} {'-'*4} {'-'*5} {'-'*5} {'-'*7} {'-'*6}")
    for _, r in profitable.head(40).iterrows():
        rgm = 'Y' if r['regime'] else 'N'
        print(f"  {r['strategy']:<18} {r['session']:<4} {r['direction']:<5} {rgm:<4} {int(r['n']):>4} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>7.1f} {r['per_day']:>6.2f}")

    # ═══════════════════════════════════════════════════════════
    # OPPORTUNITY COVERAGE ANALYSIS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  OPPORTUNITY COVERAGE ANALYSIS")
    print(f"{'='*80}\n")

    # Find real moves >= 5pts
    print("  Finding real opportunities (moves >= 5pts within session)...")
    opps = find_real_opportunities(df, min_move_pts=5.0)
    print(f"  Total opportunities found: {len(opps)}")
    print(f"    BUY: {len(opps[opps['direction']=='BUY'])}")
    print(f"    SELL: {len(opps[opps['direction']=='SELL'])}")
    print(f"    AM: {len(opps[opps['session']=='AM'])}")
    print(f"    PM: {len(opps[opps['session']=='PM'])}")
    print(f"    Avg magnitude: {opps['magnitude'].mean():.1f} pts")
    print(f"    Per day: {len(opps) / df['date'].nunique():.1f} opportunities")
    print()

    # Check coverage for top strategies
    print("  Coverage by top profitable strategies:")
    print(f"  {'Strategy':<18} {'Sess':<4} {'Dir':<5} {'Rgm':<4} | {'Caught':>6} {'Profit':>6} {'Total':>6} {'Cover%':>6} {'ProfCov%':>7}")
    print(f"  {'-'*70}")

    top_configs = profitable.head(15)
    all_caught_dates = set()

    for _, r in top_configs.iterrows():
        trades = backtest_raw(df, r['strategy'], r['session'], r['direction'],
                              sl_mult=3.0, use_regime=r['regime'])
        # Filter opportunities by session and direction
        opp_filtered = opps.copy()
        if r['session'] != 'ALL':
            opp_filtered = opp_filtered[opp_filtered['session'] == r['session']]
        opp_filtered = opp_filtered[opp_filtered['direction'] == r['direction']]

        caught, profit, total = check_coverage(trades, opp_filtered)
        cover_pct = caught / total * 100 if total > 0 else 0
        profit_pct = profit / total * 100 if total > 0 else 0

        rgm = 'Y' if r['regime'] else 'N'
        print(f"  {r['strategy']:<18} {r['session']:<4} {r['direction']:<5} {rgm:<4} | {caught:>6} {profit:>6} {total:>6} {cover_pct:>5.1f}% {profit_pct:>6.1f}%")

        # Track which date+session+direction combinations are covered
        if not trades.empty:
            for _, t in trades[trades['pnl'] > 0].iterrows():
                all_caught_dates.add((t['date'], t['session'], t['direction']))

    # Combined coverage
    print()
    total_opps_unique = len(opps.drop_duplicates(subset=['date', 'session', 'direction']))
    combined_covered = len(all_caught_dates)
    print(f"  COMBINED coverage (all profitable strategies together):")
    print(f"    Unique opportunities: {total_opps_unique}")
    print(f"    Covered by at least 1 strategy: {combined_covered}")
    print(f"    Coverage: {combined_covered/total_opps_unique*100:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # UNCOVERED OPPORTUNITIES ANALYSIS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  UNCOVERED OPPORTUNITIES: What we're missing")
    print(f"{'='*80}\n")

    # Find opportunities not captured
    opps_with_key = opps.copy()
    opps_with_key['key'] = opps_with_key['date'] + '_' + opps_with_key['session'] + '_' + opps_with_key['direction']
    caught_keys = set(f"{d}_{s}_{dr}" for d, s, dr in all_caught_dates)
    uncovered = opps_with_key[~opps_with_key['key'].isin(caught_keys)]

    print(f"  Uncovered: {len(uncovered)} / {len(opps)} total ({len(uncovered)/len(opps)*100:.1f}%)")
    print(f"    By session: AM={len(uncovered[uncovered['session']=='AM'])}, PM={len(uncovered[uncovered['session']=='PM'])}")
    print(f"    By direction: BUY={len(uncovered[uncovered['direction']=='BUY'])}, SELL={len(uncovered[uncovered['direction']=='SELL'])}")
    print(f"    Avg magnitude: {uncovered['magnitude'].mean():.1f} pts (vs {opps['magnitude'].mean():.1f} all)")

    # Time distribution of uncovered
    if not uncovered.empty and 'idx' in uncovered.columns:
        uncov_mins = df.loc[uncovered['idx'], 'mins'].values
        print(f"\n  Time distribution of uncovered opportunities:")
        for label, lo, hi in [('09:00-09:30', 540, 570), ('09:30-10:00', 570, 600),
                               ('10:00-10:30', 600, 630), ('10:30-11:00', 630, 660),
                               ('11:00-11:30', 660, 690), ('13:00-13:30', 780, 810),
                               ('13:30-14:00', 810, 840), ('14:00-14:30', 840, 870)]:
            count = ((uncov_mins >= lo) & (uncov_mins < hi)).sum()
            print(f"    {label}: {count}")


if __name__ == '__main__':
    main()
