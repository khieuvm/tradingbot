"""Wide SL + Two approaches comparison:
  1. Wide SL + Trailing after MFE (lock profits when right)
  2. Wide SL + Regime filter (only trade in SELL-dominant regime)
  3. Combined: both trailing + regime filter

Test on momentum_trend PM/SELL and ALL/SELL with full 682d data.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
import importlib


COST = 0.96

STRATEGY_MAP = {
    'momentum_trend': 'strategies.momentum_trend.MomentumTrendStrategy',
    'macd_cross': 'strategies.macd_cross.MACDCrossStrategy',
    'heikin_ashi': 'strategies.heikin_ashi.HeikinAshiStrategy',
}


def _get_strategy(name):
    path = STRATEGY_MAP[name]
    module_path, class_name = path.rsplit('.', 1)
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

    # Higher TF regime indicators (rolling windows on 1m)
    # 50-bar EMA slope (normalized) - proxy for short-term trend
    df['ema50_slope'] = (df['ema50'] - df['ema50'].shift(10)) / df['atr_14']
    # 100-bar momentum
    df['mom_100'] = df['close'] - df['close'].shift(100)
    # DI spread smoothed (20-bar SMA of DI spread)
    df['di_spread_smooth'] = df['di_spread'].rolling(20).mean()
    # Price vs EMA50 normalized
    df['price_vs_ema50'] = (df['close'] - df['ema50']) / df['atr_14']

    return df


def check_regime_sell(df, idx, method='di_spread'):
    """Check if current regime favors SELL.

    Methods:
      'di_spread': DI- > DI+ (smoothed), bearish pressure
      'ema_slope': EMA50 slope negative
      'price_below': price below EMA50
      'combined': at least 2 of 3 conditions met
    """
    row = df.iloc[idx]

    if method == 'di_spread':
        # DI- dominant (smoothed to reduce whipsaw)
        return row['di_spread_smooth'] < -3

    elif method == 'ema_slope':
        # EMA50 trending down
        return row['ema50_slope'] < -0.3

    elif method == 'price_below':
        # Price below EMA50 by > 0.5 ATR
        return row['price_vs_ema50'] < -0.5

    elif method == 'combined_2of3':
        score = 0
        if row['di_spread_smooth'] < -3:
            score += 1
        if row['ema50_slope'] < -0.3:
            score += 1
        if row['price_vs_ema50'] < -0.5:
            score += 1
        return score >= 2

    elif method == 'combined_1of3':
        # Looser: any 1 of 3
        if row['di_spread_smooth'] < -3:
            return True
        if row['ema50_slope'] < -0.3:
            return True
        if row['price_vs_ema50'] < -0.5:
            return True
        return False

    elif method == 'adx_di':
        # ADX > 20 and DI- > DI+ (trending + bearish)
        return row['adx'] > 20 and row['di_spread'] < -5

    return True  # no filter


def backtest_wide_sl_trail(df, strategy_name, session_filter, dir_filter,
                           sl_mult=3.0, tp_mult=None,
                           trail_trigger_pts=None, trail_mult=None,
                           be_trigger_pts=None,
                           regime_method=None,
                           max_hold_bars=60):
    """Backtest with wide SL + optional trailing + optional regime filter.

    Args:
        sl_mult: wide SL multiplier
        tp_mult: fixed TP multiplier (None = no fixed TP)
        trail_trigger_pts: MFE in pts to activate trail (None = no trail)
        trail_mult: trail distance as ATR multiplier (once activated)
        be_trigger_pts: MFE to move SL to breakeven (None = no BE)
        regime_method: regime filter method (None = no filter)
        max_hold_bars: max hold
    """
    strategy = _get_strategy(strategy_name)
    trades = []
    i = 0
    n = len(df)

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

        # Regime filter
        if regime_method and dir_filter == 'SELL':
            if not check_regime_sell(df, i, regime_method):
                i += 1
                continue

        # Detect signal
        signal = strategy.detect(df, i)
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
        entry_price = df['close'].iloc[i]
        atr = df['atr_14'].iloc[i]
        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        sl_dist = sl_mult * atr
        sl_price = entry_price - direction * sl_dist
        tp_price = entry_price + direction * tp_mult * atr if tp_mult else None

        # Simulate
        exit_reason = 'MAX_HOLD'
        exit_price = entry_price
        exit_bar = i
        mfe = 0
        best_price = entry_price
        trail_active = False
        be_active = False

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
                cur_mfe = bar['high'] - entry_price
                if bar['high'] > best_price:
                    best_price = bar['high']
            else:
                cur_mfe = entry_price - bar['low']
                if bar['low'] < best_price:
                    best_price = bar['low']
            mfe = max(mfe, cur_mfe)

            # Breakeven activation
            if be_trigger_pts and not be_active and mfe >= be_trigger_pts:
                be_active = True
                sl_price = entry_price  # move SL to entry

            # Trail activation
            if trail_trigger_pts and mfe >= trail_trigger_pts:
                trail_active = True
                if direction == 1:
                    new_sl = best_price - trail_mult * atr
                    sl_price = max(sl_price, new_sl)
                else:
                    new_sl = best_price + trail_mult * atr
                    sl_price = min(sl_price, new_sl)

            # Check SL
            if direction == 1 and bar['low'] <= sl_price:
                exit_reason = 'TRAIL' if trail_active else ('BE' if be_active else 'SL')
                exit_price = sl_price
                exit_bar = j
                break
            if direction == -1 and bar['high'] >= sl_price:
                exit_reason = 'TRAIL' if trail_active else ('BE' if be_active else 'SL')
                exit_price = sl_price
                exit_bar = j
                break

            # Check TP
            if tp_price:
                if direction == 1 and bar['high'] >= tp_price:
                    exit_reason = 'TP'
                    exit_price = tp_price
                    exit_bar = j
                    break
                if direction == -1 and bar['low'] <= tp_price:
                    exit_reason = 'TP'
                    exit_price = tp_price
                    exit_bar = j
                    break
        else:
            exit_bar = min(i + max_hold_bars, n - 1)
            exit_price = df['close'].iloc[exit_bar]

        pnl = (exit_price - entry_price) * direction - COST
        trades.append({
            'entry_time': df['time'].iloc[i],
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
        })

        i = exit_bar + 1
        continue
        i += 1

    return pd.DataFrame(trades)


def print_results(trades_df, label):
    if trades_df.empty:
        print(f"  {label}: NO TRADES")
        return

    n = len(trades_df)
    winners = trades_df[trades_df['pnl'] > 0]
    wr = len(winners) / n * 100
    total_pnl = trades_df['pnl'].sum()
    gross_win = winners['pnl'].sum() if len(winners) > 0 else 0
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else 999

    days = (trades_df['entry_time'].iloc[-1] - trades_df['entry_time'].iloc[0]).days
    days = max(days, 1)
    per_day = total_pnl / days

    print(f"  {label}:")
    print(f"    N={n} | WR={wr:.1f}% | PF={pf:.2f} | PnL={total_pnl:.1f} | /d={per_day:.2f}")
    print(f"    Avg win={winners['pnl'].mean():.2f} | Avg loss={trades_df[trades_df['pnl']<=0]['pnl'].mean():.2f} | Avg bars={trades_df['bars_held'].mean():.0f}")

    reasons = trades_df.groupby('exit_reason').agg(
        cnt=('pnl', 'count'),
        avg_pnl=('pnl', 'mean'),
        wr=('pnl', lambda x: (x > 0).mean() * 100)
    ).reset_index()
    parts = []
    for _, r in reasons.iterrows():
        parts.append(f"{r['exit_reason']}:{int(r['cnt'])}({r['wr']:.0f}%,{r['avg_pnl']:.1f})")
    print(f"    Exits: {' | '.join(parts)}")
    print()


def main():
    print("=" * 80)
    print("  COMPARISON: TRAILING vs REGIME FILTER vs COMBINED")
    print("=" * 80)

    df = load_data()
    df = compute_indicators(df)
    print(f"  Data: {len(df)} bars, {df['date'].nunique()} days\n")

    # ═══════════════════════════════════════════════════════════════════════
    # APPROACH 1: Wide SL + Trailing (lock profit when right)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("  APPROACH 1: WIDE SL + TRAILING (lock profits after MFE)")
    print("=" * 80)
    print()

    trail_configs = [
        # (sl_mult, tp_mult, trail_trigger, trail_mult, be_trigger, label)
        (3.0, None, None, None, None, 'SL3.0 no trail no TP (raw)'),
        (3.0, 5.0, None, None, None, 'SL3.0 TP5.0 (fixed TP)'),
        (3.0, None, 3.0, 2.0, None, 'SL3.0 trail@3pts/2.0x'),
        (3.0, None, 4.0, 2.0, None, 'SL3.0 trail@4pts/2.0x'),
        (3.0, None, 5.0, 2.0, None, 'SL3.0 trail@5pts/2.0x'),
        (3.0, None, 3.0, 1.5, None, 'SL3.0 trail@3pts/1.5x'),
        (3.0, None, 4.0, 1.5, None, 'SL3.0 trail@4pts/1.5x'),
        (3.0, None, 5.0, 1.5, None, 'SL3.0 trail@5pts/1.5x'),
        (3.0, None, 3.0, 1.0, None, 'SL3.0 trail@3pts/1.0x (tight)'),
        (3.0, None, 4.0, 1.0, None, 'SL3.0 trail@4pts/1.0x (tight)'),
        (3.0, None, 4.0, 2.0, 3.0, 'SL3.0 BE@3 trail@4pts/2.0x'),
        (3.0, None, 5.0, 2.0, 4.0, 'SL3.0 BE@4 trail@5pts/2.0x'),
        (3.0, 8.0, 4.0, 1.5, 3.0, 'SL3.0 TP8.0 BE@3 trail@4/1.5x'),
        (2.5, None, 3.0, 1.5, None, 'SL2.5 trail@3pts/1.5x'),
        (2.5, None, 4.0, 1.5, None, 'SL2.5 trail@4pts/1.5x'),
        (2.0, None, 3.0, 1.5, None, 'SL2.0 trail@3pts/1.5x'),
        (2.0, None, 4.0, 1.5, 3.0, 'SL2.0 BE@3 trail@4/1.5x'),
    ]

    print("  --- momentum_trend PM/SELL ---")
    for sl, tp, trail_t, trail_m, be, label in trail_configs:
        trades = backtest_wide_sl_trail(df, 'momentum_trend', 'PM', 'SELL',
                                        sl_mult=sl, tp_mult=tp,
                                        trail_trigger_pts=trail_t,
                                        trail_mult=trail_m,
                                        be_trigger_pts=be,
                                        regime_method=None)
        print_results(trades, label)

    print("\n  --- momentum_trend ALL/SELL ---")
    best_trail_configs = [
        (3.0, None, None, None, None, 'SL3.0 no trail (raw)'),
        (3.0, 5.0, None, None, None, 'SL3.0 TP5.0'),
        (3.0, None, 4.0, 1.5, None, 'SL3.0 trail@4pts/1.5x'),
        (3.0, None, 5.0, 2.0, 4.0, 'SL3.0 BE@4 trail@5pts/2.0x'),
        (2.5, None, 4.0, 1.5, None, 'SL2.5 trail@4pts/1.5x'),
    ]
    for sl, tp, trail_t, trail_m, be, label in best_trail_configs:
        trades = backtest_wide_sl_trail(df, 'momentum_trend', 'ALL', 'SELL',
                                        sl_mult=sl, tp_mult=tp,
                                        trail_trigger_pts=trail_t,
                                        trail_mult=trail_m,
                                        be_trigger_pts=be,
                                        regime_method=None)
        print_results(trades, label)

    # ═══════════════════════════════════════════════════════════════════════
    # APPROACH 2: Wide SL + Regime Filter
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  APPROACH 2: WIDE SL + REGIME FILTER (only trade when SELL-favorable)")
    print("=" * 80)
    print()

    regime_methods = ['di_spread', 'ema_slope', 'price_below',
                      'combined_2of3', 'combined_1of3', 'adx_di']

    print("  --- momentum_trend PM/SELL + regime filter ---")
    # Baseline first
    trades_base = backtest_wide_sl_trail(df, 'momentum_trend', 'PM', 'SELL',
                                         sl_mult=3.0, tp_mult=5.0)
    print_results(trades_base, 'BASELINE (no regime filter)')

    for method in regime_methods:
        trades = backtest_wide_sl_trail(df, 'momentum_trend', 'PM', 'SELL',
                                        sl_mult=3.0, tp_mult=5.0,
                                        regime_method=method)
        print_results(trades, f'regime={method}')

    print("\n  --- momentum_trend ALL/SELL + regime filter ---")
    trades_base = backtest_wide_sl_trail(df, 'momentum_trend', 'ALL', 'SELL',
                                         sl_mult=3.0, tp_mult=5.0)
    print_results(trades_base, 'BASELINE (no regime filter)')

    for method in regime_methods:
        trades = backtest_wide_sl_trail(df, 'momentum_trend', 'ALL', 'SELL',
                                        sl_mult=3.0, tp_mult=5.0,
                                        regime_method=method)
        print_results(trades, f'regime={method}')

    # ═══════════════════════════════════════════════════════════════════════
    # APPROACH 3: COMBINED (trailing + regime filter)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  APPROACH 3: COMBINED (best trailing + best regime filter)")
    print("=" * 80)
    print()

    combined_configs = [
        # (sl, tp, trail_t, trail_m, be, regime, label)
        (3.0, 5.0, None, None, None, None, 'BASELINE: SL3.0 TP5.0'),
        (3.0, None, 4.0, 1.5, None, 'di_spread', 'SL3.0 trail@4/1.5 + DI regime'),
        (3.0, None, 4.0, 1.5, None, 'ema_slope', 'SL3.0 trail@4/1.5 + EMA slope'),
        (3.0, None, 4.0, 1.5, None, 'combined_1of3', 'SL3.0 trail@4/1.5 + any1of3'),
        (3.0, None, 4.0, 1.5, None, 'combined_2of3', 'SL3.0 trail@4/1.5 + 2of3'),
        (3.0, None, 4.0, 1.5, None, 'adx_di', 'SL3.0 trail@4/1.5 + ADX+DI'),
        (3.0, None, 5.0, 2.0, 4.0, 'di_spread', 'SL3.0 BE@4 trail@5/2.0 + DI'),
        (3.0, None, 5.0, 2.0, 4.0, 'ema_slope', 'SL3.0 BE@4 trail@5/2.0 + EMA'),
        (3.0, None, 5.0, 2.0, 4.0, 'combined_1of3', 'SL3.0 BE@4 trail@5/2.0 + any1of3'),
        (2.5, None, 4.0, 1.5, None, 'di_spread', 'SL2.5 trail@4/1.5 + DI'),
        (2.5, None, 4.0, 1.5, None, 'ema_slope', 'SL2.5 trail@4/1.5 + EMA slope'),
        (2.5, None, 4.0, 1.5, None, 'combined_1of3', 'SL2.5 trail@4/1.5 + any1of3'),
    ]

    print("  --- momentum_trend PM/SELL ---")
    for sl, tp, trail_t, trail_m, be, regime, label in combined_configs:
        trades = backtest_wide_sl_trail(df, 'momentum_trend', 'PM', 'SELL',
                                        sl_mult=sl, tp_mult=tp,
                                        trail_trigger_pts=trail_t,
                                        trail_mult=trail_m,
                                        be_trigger_pts=be,
                                        regime_method=regime)
        print_results(trades, label)

    print("\n  --- momentum_trend ALL/SELL ---")
    for sl, tp, trail_t, trail_m, be, regime, label in combined_configs:
        trades = backtest_wide_sl_trail(df, 'momentum_trend', 'ALL', 'SELL',
                                        sl_mult=sl, tp_mult=tp,
                                        trail_trigger_pts=trail_t,
                                        trail_mult=trail_m,
                                        be_trigger_pts=be,
                                        regime_method=regime)
        print_results(trades, label)

    # ═══════════════════════════════════════════════════════════════════════
    # PERIOD BREAKDOWN: Best configs on Full vs 6M vs 1M
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  PERIOD BREAKDOWN: Best configs across timeframes")
    print("=" * 80)
    print()

    cutoff_6m = df['time'].iloc[-1] - pd.Timedelta(days=180)
    cutoff_1m = df['time'].iloc[-1] - pd.Timedelta(days=30)

    df_6m = df[df['time'] >= cutoff_6m].reset_index(drop=True)
    df_6m = compute_indicators(df_6m)
    df_1m = df[df['time'] >= cutoff_1m].reset_index(drop=True)
    df_1m = compute_indicators(df_1m)

    period_configs = [
        # (strat, sess, dir, sl, tp, trail_t, trail_m, be, regime, label)
        ('momentum_trend', 'PM', 'SELL', 3.0, 5.0, None, None, None, None, 'BASE: SL3 TP5'),
        ('momentum_trend', 'PM', 'SELL', 3.0, None, 4.0, 1.5, None, None, 'TRAIL: SL3 trail@4/1.5'),
        ('momentum_trend', 'PM', 'SELL', 3.0, None, 4.0, 1.5, None, 'di_spread', 'TRAIL+DI: SL3 trail@4/1.5 + DI'),
        ('momentum_trend', 'PM', 'SELL', 3.0, None, 4.0, 1.5, None, 'ema_slope', 'TRAIL+EMA: SL3 trail@4/1.5 + EMA'),
        ('momentum_trend', 'PM', 'SELL', 3.0, None, 5.0, 2.0, 4.0, 'di_spread', 'TRAIL+BE+DI: SL3 BE@4 trail@5/2 + DI'),
        ('momentum_trend', 'ALL', 'SELL', 3.0, None, 4.0, 1.5, None, 'di_spread', 'ALL/SELL TRAIL+DI: SL3 trail@4/1.5'),
    ]

    for strat, sess, dirn, sl, tp, trail_t, trail_m, be, regime, label in period_configs:
        print(f"  {label}:")
        for period_label, period_df in [('FULL', df), ('6M', df_6m), ('1M', df_1m)]:
            trades = backtest_wide_sl_trail(period_df, strat, sess, dirn,
                                            sl_mult=sl, tp_mult=tp,
                                            trail_trigger_pts=trail_t,
                                            trail_mult=trail_m,
                                            be_trigger_pts=be,
                                            regime_method=regime)
            if trades.empty:
                print(f"    {period_label}: NO TRADES")
            else:
                n = len(trades)
                wr = (trades['pnl'] > 0).mean() * 100
                total = trades['pnl'].sum()
                gross_w = trades[trades['pnl'] > 0]['pnl'].sum()
                gross_l = abs(trades[trades['pnl'] <= 0]['pnl'].sum())
                pf = gross_w / gross_l if gross_l > 0 else 99
                days = max((trades['entry_time'].iloc[-1] - trades['entry_time'].iloc[0]).days, 1)
                print(f"    {period_label}: N={n} WR={wr:.1f}% PF={pf:.2f} PnL={total:.1f} /d={total/days:.2f}")
        print()


if __name__ == '__main__':
    main()
