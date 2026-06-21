"""Wide SL + Smart Early Exit based on reversal indicators.

Approach:
  - Enter with wide SL (3.0×ATR) to avoid 70%+ sweeps
  - Every bar after entry, compute a "reversal score" from indicators
  - If reversal score exceeds threshold → exit early (loss smaller than full SL)
  - If indicators stay normal → hold until TP or session end

Reversal indicators (from sl_sweep_features.py findings):
  - RSI(7) > 67 at SL zone → 80% are real reversals
  - BB_pos > 0.85 → price at band extreme, less likely to return
  - KC_pos > 0.89 → same as BB
  - EMA_align flipped (>= 2 for SELL) → trend changing
  - DI_spread against position (> +6 for SELL) → directional pressure
  - Body % > 0.65 → strong momentum bar (not wick rejection)
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path


def load_data():
    df = pd.read_parquet('data/vn30f1m_1m.parquet')
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['session'] = df['mins'].apply(lambda m: 'AM' if m < 720 else 'PM')
    return df


def compute_indicators(df):
    """Add all needed indicators to 1m dataframe."""
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi_7'] = ta.rsi(df['close'], length=7)
    df['rsi_14'] = ta.rsi(df['close'], length=14)

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

    # EMA columns needed by strategies (ema8, ema21, ema50)
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

    # RSI for strategies (rsi column)
    df['rsi'] = df['rsi_14']

    df['body_pct'] = (abs(df['close'] - df['open']) /
                      (df['high'] - df['low']).replace(0, np.nan)).fillna(0)

    return df


def compute_ema_align(row, direction):
    """EMA alignment score relative to position direction.
    For SELL: negative = bearish aligned (good), positive = bullish (bad)
    For BUY: positive = bullish aligned (good), negative = bearish (bad)
    """
    score = 0
    if row['ema_8'] > row['ema_21']:
        score += 1
    if row['ema_21'] > row['ema_50']:
        score += 1
    if row['close'] > row['ema_8']:
        score += 1
    # For SELL, positive EMA align = bullish = bad (reversal signal)
    # For BUY, negative EMA align = bearish = bad (reversal signal)
    if direction == -1:
        return score  # high = bullish = bad for short
    else:
        return -score  # low (negative) = bearish = bad for long


def detect_reversal(row, direction, entry_price, atr_at_entry):
    """Score how likely current bar indicates real reversal (not sweep).

    Returns reversal_score 0-6. Higher = more likely real reversal.
    Each indicator contributes 0 or 1 based on thresholds from analysis.
    """
    score = 0

    if direction == -1:  # SHORT position
        # RSI(7) high = bullish momentum against us
        if row['rsi_7'] > 67:
            score += 1
        if row['rsi_7'] > 75:
            score += 1  # extra point for extreme

        # BB position high = price at upper band
        if row['bb_pos'] > 0.85:
            score += 1

        # KC position high
        if row['kc_pos'] > 0.89:
            score += 1

        # EMA align bullish (against short)
        ema_score = compute_ema_align(row, direction)
        if ema_score >= 2:
            score += 1

        # DI spread positive (bullish) and large
        if row['di_spread'] > 6:
            score += 1

        # Strong bullish body (momentum against us)
        if row['body_pct'] > 0.65 and row['close'] > row['open']:
            score += 1

    else:  # LONG position
        # RSI(7) low = bearish momentum against us
        if row['rsi_7'] < 33:
            score += 1
        if row['rsi_7'] < 25:
            score += 1

        # BB position low = price at lower band
        if row['bb_pos'] < 0.15:
            score += 1

        # KC position low
        if row['kc_pos'] < 0.11:
            score += 1

        # EMA align bearish (against long)
        ema_score = compute_ema_align(row, direction)
        if ema_score >= 2:
            score += 1

        # DI spread negative (bearish) and large
        if row['di_spread'] < -6:
            score += 1

        # Strong bearish body
        if row['body_pct'] > 0.65 and row['close'] < row['open']:
            score += 1

    return score


STRATEGY_MAP = {
    'momentum_trend': 'strategies.momentum_trend.MomentumTrendStrategy',
    'macd_cross': 'strategies.macd_cross.MACDCrossStrategy',
    'heikin_ashi': 'strategies.heikin_ashi.HeikinAshiStrategy',
    'fibonacci': 'strategies.fibonacci.FibonacciStrategy',
    'market_structure': 'strategies.market_structure.MarketStructureStrategy',
}


def _get_strategy(name):
    path = STRATEGY_MAP[name]
    module_path, class_name = path.rsplit('.', 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


def run_strategy_backtest(df, strategy_name, session_filter, dir_filter,
                          sl_mult=3.0, tp_mult=5.0, reversal_threshold=3,
                          max_hold_bars=60, min_bars_before_check=3):
    """Backtest a strategy with wide SL + smart reversal exit.

    Args:
        sl_mult: SL as multiple of ATR (wide)
        tp_mult: TP as multiple of ATR
        reversal_threshold: score needed to trigger early exit (3-5)
        max_hold_bars: max bars to hold before session exit
        min_bars_before_check: don't check reversal in first N bars (let trade develop)
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

        # Time filter: AM 09:15-10:45, PM 13:15-14:15
        if row['session'] == 'AM' and not (555 <= row['mins'] <= 645):
            i += 1
            continue
        if row['session'] == 'PM' and not (795 <= row['mins'] <= 855):
            i += 1
            continue

        # Detect signal
        signal = strategy.detect(df, i)

        # Direction filter
        if dir_filter == 'BUY' and signal != 1:
            i += 1
            continue
        if dir_filter == 'SELL' and signal != -1:
            i += 1
            continue
        if signal == 0:
            i += 1
            continue

        direction = signal  # 1 = long, -1 = short
        entry_price = df['close'].iloc[i]
        atr = df['atr_14'].iloc[i]

        if pd.isna(atr) or atr <= 0:
            i += 1
            continue

        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr

        if direction == 1:
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        # Simulate trade
        exit_reason = 'MAX_HOLD'
        exit_price = entry_price
        exit_bar = i
        mfe = 0
        reversal_scores = []

        for j in range(i + 1, min(i + max_hold_bars + 1, n)):
            bar = df.iloc[j]

            # Session boundary — exit
            if bar['date'] != df.iloc[i]['date']:
                exit_reason = 'SESSION'
                exit_price = df['close'].iloc[j - 1]
                exit_bar = j - 1
                break
            if bar['session'] != row['session']:
                exit_reason = 'SESSION'
                exit_price = df['close'].iloc[j - 1]
                exit_bar = j - 1
                break
            # Hard session exit
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

            # Check SL
            if direction == 1:
                if bar['low'] <= sl_price:
                    exit_reason = 'SL'
                    exit_price = sl_price
                    exit_bar = j
                    break
                mfe = max(mfe, bar['high'] - entry_price)
            else:
                if bar['high'] >= sl_price:
                    exit_reason = 'SL'
                    exit_price = sl_price
                    exit_bar = j
                    break
                mfe = max(mfe, entry_price - bar['low'])

            # Check TP
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

            # Smart reversal check (after min_bars AND only when in drawdown)
            bars_held = j - i
            if bars_held >= min_bars_before_check:
                # Only check reversal when position is losing > 1.0×ATR
                cur_pnl = (bar['close'] - entry_price) * direction
                if cur_pnl < -1.0 * atr:
                    rev_score = detect_reversal(bar, direction, entry_price, atr)
                    reversal_scores.append(rev_score)

                    if rev_score >= reversal_threshold:
                        exit_reason = 'SMART_EXIT'
                        exit_price = bar['close']
                        exit_bar = j
                        break
        else:
            # Loop ended without break = max hold
            exit_bar = min(i + max_hold_bars, n - 1)
            exit_price = df['close'].iloc[exit_bar]

        # Calculate PnL
        pnl = (exit_price - entry_price) * direction - 0.96  # cost

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
            'max_rev_score': max(reversal_scores) if reversal_scores else 0,
        })

        # Skip forward past this trade
        i = exit_bar + 1
        continue

        i += 1

    return pd.DataFrame(trades)


def analyze_results(trades_df, label):
    """Print summary statistics."""
    if trades_df.empty:
        print(f"\n  {label}: NO TRADES")
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

    print(f"\n  {label}:")
    print(f"    Trades: {n} | WR: {wr:.1f}% | PF: {pf:.2f}")
    print(f"    Total PnL: {total_pnl:.1f} pts | Per day: {per_day:.2f} pts/d")
    print(f"    Avg win: {winners['pnl'].mean():.2f} | Avg loss: {trades_df[trades_df['pnl'] <= 0]['pnl'].mean():.2f}")
    print(f"    Avg bars held: {trades_df['bars_held'].mean():.1f}")

    # Exit reason breakdown
    print(f"    Exit reasons:")
    for reason, group in trades_df.groupby('exit_reason'):
        r_wr = (group['pnl'] > 0).mean() * 100
        print(f"      {reason}: {len(group)} ({len(group)/n*100:.1f}%) | WR {r_wr:.0f}% | avg PnL {group['pnl'].mean():.2f}")


def main():
    print("=" * 80)
    print("  WIDE SL + SMART REVERSAL EXIT BACKTEST")
    print("=" * 80)

    df = load_data()
    df = compute_indicators(df)
    print(f"  Data: {len(df)} bars, {df['date'].nunique()} days")
    print(f"  Period: {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}")

    # Test configurations
    configs = [
        # (strategy, session, direction, sl_mult, tp_mult, rev_threshold, label)
        ('momentum_trend', 'PM', 'SELL', 3.0, 5.0, 3, 'MT PM/SELL SL3.0 TP5.0 rev>=3'),
        ('momentum_trend', 'PM', 'SELL', 3.0, 5.0, 4, 'MT PM/SELL SL3.0 TP5.0 rev>=4'),
        ('momentum_trend', 'PM', 'SELL', 3.0, 5.0, 5, 'MT PM/SELL SL3.0 TP5.0 rev>=5'),
        ('momentum_trend', 'PM', 'SELL', 4.0, 5.0, 3, 'MT PM/SELL SL4.0 TP5.0 rev>=3'),
        ('momentum_trend', 'PM', 'SELL', 4.0, 5.0, 4, 'MT PM/SELL SL4.0 TP5.0 rev>=4'),
        ('momentum_trend', 'ALL', 'SELL', 3.0, 5.0, 3, 'MT ALL/SELL SL3.0 TP5.0 rev>=3'),
        ('momentum_trend', 'ALL', 'SELL', 3.0, 5.0, 4, 'MT ALL/SELL SL3.0 TP5.0 rev>=4'),
        ('macd_cross', 'AM', 'SELL', 3.0, 5.0, 3, 'MACD AM/SELL SL3.0 TP5.0 rev>=3'),
        ('macd_cross', 'AM', 'SELL', 3.0, 5.0, 4, 'MACD AM/SELL SL3.0 TP5.0 rev>=4'),
        ('heikin_ashi', 'AM', 'SELL', 3.0, 5.0, 3, 'HA AM/SELL SL3.0 TP5.0 rev>=3'),
        ('heikin_ashi', 'AM', 'SELL', 3.0, 5.0, 4, 'HA AM/SELL SL3.0 TP5.0 rev>=4'),
    ]

    # Also test baseline (wide SL, no smart exit) for comparison
    baseline_configs = [
        ('momentum_trend', 'PM', 'SELL', 3.0, 5.0, 99, 'BASELINE: MT PM/SELL SL3.0 TP5.0 (no smart exit)'),
        ('momentum_trend', 'PM', 'SELL', 4.0, 5.0, 99, 'BASELINE: MT PM/SELL SL4.0 TP5.0 (no smart exit)'),
        ('momentum_trend', 'ALL', 'SELL', 3.0, 5.0, 99, 'BASELINE: MT ALL/SELL SL3.0 TP5.0 (no smart exit)'),
    ]

    print("\n" + "=" * 80)
    print("  BASELINES (wide SL, NO smart exit — threshold=99 so never triggers)")
    print("=" * 80)

    for strat, sess, dirn, sl, tp, rev_th, label in baseline_configs:
        trades = run_strategy_backtest(df, strat, sess, dirn,
                                       sl_mult=sl, tp_mult=tp,
                                       reversal_threshold=rev_th,
                                       max_hold_bars=60,
                                       min_bars_before_check=3)
        analyze_results(trades, label)

    print("\n" + "=" * 80)
    print("  SMART EXIT CONFIGS (wide SL + reversal detection)")
    print("=" * 80)

    for strat, sess, dirn, sl, tp, rev_th, label in configs:
        trades = run_strategy_backtest(df, strat, sess, dirn,
                                       sl_mult=sl, tp_mult=tp,
                                       reversal_threshold=rev_th,
                                       max_hold_bars=60,
                                       min_bars_before_check=3)
        analyze_results(trades, label)

    # Deep dive: best config on recent 6 months vs full
    print("\n" + "=" * 80)
    print("  PERIOD COMPARISON: Full vs Last 6 Months vs Last 1 Month")
    print("=" * 80)

    best_configs = [
        ('momentum_trend', 'PM', 'SELL', 3.0, 5.0, 3),
        ('momentum_trend', 'PM', 'SELL', 3.0, 5.0, 4),
        ('momentum_trend', 'ALL', 'SELL', 3.0, 5.0, 3),
    ]

    cutoff_6m = df['time'].iloc[-1] - pd.Timedelta(days=180)
    cutoff_1m = df['time'].iloc[-1] - pd.Timedelta(days=30)

    for strat, sess, dirn, sl, tp, rev_th in best_configs:
        label = f"{strat} {sess}/{dirn} SL{sl} rev≥{rev_th}"
        print(f"\n  {'-' * 60}")
        print(f"  {label}")

        # Full
        trades_full = run_strategy_backtest(df, strat, sess, dirn,
                                            sl_mult=sl, tp_mult=tp,
                                            reversal_threshold=rev_th)
        analyze_results(trades_full, "FULL (682d)")

        # Last 6m
        df_6m = df[df['time'] >= cutoff_6m].reset_index(drop=True)
        df_6m = compute_indicators(df_6m)
        trades_6m = run_strategy_backtest(df_6m, strat, sess, dirn,
                                          sl_mult=sl, tp_mult=tp,
                                          reversal_threshold=rev_th)
        analyze_results(trades_6m, "LAST 6M")

        # Last 1m
        df_1m = df[df['time'] >= cutoff_1m].reset_index(drop=True)
        df_1m = compute_indicators(df_1m)
        trades_1m = run_strategy_backtest(df_1m, strat, sess, dirn,
                                          sl_mult=sl, tp_mult=tp,
                                          reversal_threshold=rev_th)
        analyze_results(trades_1m, "LAST 1M")

    print("\n" + "=" * 80)
    print("  SMART EXIT ANALYSIS: When does smart exit save us?")
    print("=" * 80)

    # Run best config and analyze smart exit trades specifically
    trades = run_strategy_backtest(df, 'momentum_trend', 'PM', 'SELL',
                                   sl_mult=3.0, tp_mult=5.0,
                                   reversal_threshold=3)
    if not trades.empty:
        smart = trades[trades['exit_reason'] == 'SMART_EXIT']
        sl_trades = trades[trades['exit_reason'] == 'SL']
        tp_trades = trades[trades['exit_reason'] == 'TP']

        print(f"\n  MT PM/SELL SL3.0 rev≥3:")
        print(f"    SMART_EXIT trades: {len(smart)}")
        if len(smart) > 0:
            print(f"      Avg PnL: {smart['pnl'].mean():.2f} | WR: {(smart['pnl']>0).mean()*100:.1f}%")
            print(f"      Avg bars held: {smart['bars_held'].mean():.1f}")
            print(f"      → Saved from SL: {(smart['pnl'] > -sl_mult * trades['atr'].mean()).sum()} trades")
            # Compare: if these had hit SL instead
            theoretical_sl_loss = -3.0 * trades['atr'].mean() - 0.96
            actual_smart_loss = smart[smart['pnl'] < 0]['pnl'].mean() if (smart['pnl'] < 0).any() else 0
            print(f"      Avg loss if hit SL: {theoretical_sl_loss:.2f} vs actual smart exit loss: {actual_smart_loss:.2f}")

        print(f"\n    Full SL hits: {len(sl_trades)}")
        if len(sl_trades) > 0:
            print(f"      Avg PnL: {sl_trades['pnl'].mean():.2f}")
            # What was their max rev score? (didn't trigger)
            print(f"      Max rev score seen: {sl_trades['max_rev_score'].mean():.1f} avg")


if __name__ == '__main__':
    main()
