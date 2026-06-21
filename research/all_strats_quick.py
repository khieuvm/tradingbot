"""Quick scan: all strategies with raw approach (SL3.0, no trail, session exit)."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import pandas_ta as ta
import importlib

COST = 0.96

ALL_STRATEGIES = {
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
    'sr_horizontal': ('strategies.sr_horizontal', 'SRHorizontalStrategy'),
    'reversal_patterns': ('strategies.reversal_patterns', 'ReversalPatternsStrategy'),
    'divergence': ('strategies.divergence', 'DivergenceStrategy'),
    'fvg': ('strategies.fvg', 'FVGStrategy'),
    'rsi2': ('strategies.rsi2', 'RSI2Strategy'),
    'volume': ('strategies.volume', 'VolumeStrategy'),
    'ha_stoch': ('strategies.ha_stoch', 'HAStochStrategy'),
    'nbar_breakout': ('strategies.nbar_breakout', 'NBarBreakoutStrategy'),
    'candlestick': ('strategies.candlestick', 'CandlestickStrategy'),
    'dynamic_sr': ('strategies.dynamic_sr', 'DynamicSRStrategy'),
    'renko': ('strategies.renko', 'RenkoStrategy'),
    'harmonic': ('strategies.harmonic', 'HarmonicStrategy'),
    'role_reversal': ('strategies.role_reversal', 'RoleReversalStrategy'),
    'oscillators': ('strategies.oscillators', 'OscillatorsStrategy'),
}


def _get_strategy(name):
    module_path, class_name = ALL_STRATEGIES[name]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


def load_and_prepare():
    print('[LOAD] Loading data...')
    df = pd.read_parquet('data/vn30f1m_1m.parquet')
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['session'] = df['mins'].apply(lambda m: 'AM' if m < 720 else 'PM')

    print('[INDICATORS] Computing...')
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr'] = df['atr_14']
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['rsi_14'] = df['rsi']
    df['rsi_7'] = ta.rsi(df['close'], length=7)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    df['bb_mid'] = bb['BBM_20_2.0']
    df['bb_pos'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])).clip(0, 1)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=1.5)
    df['kc_upper'] = kc['KCUe_20_1.5']
    df['kc_lower'] = kc['KCLe_20_1.5']
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
    df['body_pct'] = (abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)).fillna(0)
    df['range'] = df['high'] - df['low']
    df['price_vs_ema50'] = (df['close'] - df['ema50']) / df['atr_14']

    print(f'[DATA] {len(df)} bars, {df["date"].nunique()} days')
    return df


def backtest_raw(df, strategy_name, sess_filter, dir_filter, use_regime=False):
    try:
        strategy = _get_strategy(strategy_name)
    except Exception:
        return []

    trades = []
    last_sig = -10
    n = len(df)

    for i in range(60, n):
        row = df.iloc[i]
        if sess_filter != 'ALL' and row['session'] != sess_filter:
            continue
        if row['session'] == 'AM' and not (555 <= row['mins'] <= 645):
            continue
        if row['session'] == 'PM' and not (795 <= row['mins'] <= 855):
            continue
        if i - last_sig < 5:
            continue

        try:
            signal = strategy.detect(df, i)
        except Exception:
            continue

        if dir_filter == 'BUY' and signal != 1:
            continue
        if dir_filter == 'SELL' and signal != -1:
            continue
        if signal == 0:
            continue

        direction = signal

        # Regime filter
        if use_regime:
            pve = row['price_vs_ema50']
            if pd.isna(pve):
                continue
            if direction == -1 and pve >= -0.5:
                continue
            if direction == 1 and pve <= 0.5:
                continue

        entry = df['close'].iloc[i]
        atr = df['atr_14'].iloc[i]
        if pd.isna(atr) or atr <= 0:
            continue

        sl_price = entry - direction * 3.0 * atr
        exit_price = entry
        exit_reason = 'MAX_HOLD'
        exit_bar = i

        for j in range(i + 1, min(i + 61, n)):
            bar = df.iloc[j]
            if bar['date'] != row['date'] or bar['session'] != row['session']:
                exit_price = df['close'].iloc[j - 1]
                exit_reason = 'SESSION'
                exit_bar = j - 1
                break
            if bar['session'] == 'AM' and bar['mins'] >= 685:
                exit_price = bar['close']
                exit_reason = 'SESSION'
                exit_bar = j
                break
            if bar['session'] == 'PM' and bar['mins'] >= 865:
                exit_price = bar['close']
                exit_reason = 'SESSION'
                exit_bar = j
                break
            if direction == 1 and bar['low'] <= sl_price:
                exit_price = sl_price
                exit_reason = 'SL'
                exit_bar = j
                break
            if direction == -1 and bar['high'] >= sl_price:
                exit_price = sl_price
                exit_reason = 'SL'
                exit_bar = j
                break
        else:
            exit_bar = min(i + 60, n - 1)
            exit_price = df['close'].iloc[exit_bar]

        pnl = (exit_price - entry) * direction - COST
        trades.append({
            'pnl': pnl,
            'date': str(row['date']),
            'session': row['session'],
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry_idx': i,
        })
        last_sig = i

    return trades


def main():
    df = load_and_prepare()

    print()
    print('=' * 75)
    print('  ALL 24 STRATEGIES x [AM/PM/ALL] x [BUY/SELL] x [regime/no-regime]')
    print('  Raw approach: SL=3.0xATR, no trail, no TP, session exit')
    print('=' * 75)
    print()

    results = []
    total_tested = 0

    for sname in ALL_STRATEGIES:
        print(f'  Testing {sname}...', end=' ')
        count = 0
        for sess in ['AM', 'PM', 'ALL']:
            for dirn in ['BUY', 'SELL']:
                for regime in [False, True]:
                    trades = backtest_raw(df, sname, sess, dirn, regime)
                    total_tested += 1
                    if len(trades) < 15:
                        continue
                    pnls = [t['pnl'] for t in trades]
                    wins = [p for p in pnls if p > 0]
                    losses = [p for p in pnls if p <= 0]
                    n_trades = len(pnls)
                    wr = len(wins) / n_trades * 100
                    gw = sum(wins) if wins else 0
                    gl = abs(sum(losses)) if losses else 0.001
                    pf = gw / gl
                    total_pnl = sum(pnls)
                    per_day = total_pnl / 682

                    results.append({
                        'strategy': sname, 'session': sess, 'direction': dirn,
                        'regime': regime, 'n': n_trades, 'wr': wr, 'pf': pf,
                        'pnl': total_pnl, 'per_day': per_day, 'trades': trades,
                    })
                    if pf > 1.0:
                        count += 1
        print(f'{count} profitable')

    # Sort and print
    profitable = sorted([r for r in results if r['pf'] > 1.0], key=lambda x: -x['pf'])

    print()
    print(f'Total configs tested: {total_tested}')
    print(f'Total with >= 15 trades: {len(results)}')
    print(f'Total profitable (PF>1.0): {len(profitable)}')
    print()
    print('TOP PROFITABLE CONFIGS (sorted by PF):')
    print(f"  {'Strategy':<18} {'Sess':<4} {'Dir':<5} {'Rgm':<4} {'N':>4} {'WR':>5} {'PF':>5} {'PnL':>8} {'/d':>6}")
    print(f"  {'-'*70}")
    for r in profitable[:60]:
        rgm = 'Y' if r['regime'] else 'N'
        print(f"  {r['strategy']:<18} {r['session']:<4} {r['direction']:<5} {rgm:<4} {r['n']:>4} {r['wr']:>5.1f} {r['pf']:>5.2f} {r['pnl']:>8.1f} {r['per_day']:>6.2f}")

    # ═══════════════════════════════════════════════════════════
    # OPPORTUNITY COVERAGE
    # ═══════════════════════════════════════════════════════════
    print()
    print('=' * 75)
    print('  OPPORTUNITY COVERAGE ANALYSIS')
    print('=' * 75)
    print()

    # Find real moves >= 5pts per session
    # Simplified: for each (date, session), check if max(high) - min(low) >= 5pts
    print('  Counting sessions with >= 5pt moves...')
    session_moves = []
    for date in df['date'].unique():
        for sess in ['AM', 'PM']:
            if sess == 'AM':
                mask = (df['date'] == date) & (df['mins'] >= 540) & (df['mins'] < 690)
            else:
                mask = (df['date'] == date) & (df['mins'] >= 780) & (df['mins'] < 870)

            sdf = df[mask]
            if len(sdf) < 5:
                continue

            prices = sdf['close'].values
            high_max = sdf['high'].max()
            low_min = sdf['low'].min()

            # Check for BUY opportunity (swing low to high >= 5pts)
            # Simple: from first half low to second half high
            mid = len(sdf) // 2
            first_half_low = sdf['low'].iloc[:mid].min()
            second_half_high = sdf['high'].iloc[mid:].max()
            buy_move = second_half_high - first_half_low

            # Check for SELL opportunity
            first_half_high = sdf['high'].iloc[:mid].max()
            second_half_low = sdf['low'].iloc[mid:].min()
            sell_move = first_half_high - second_half_low

            # More generous: any window
            total_range = high_max - low_min

            session_moves.append({
                'date': str(date),
                'session': sess,
                'total_range': total_range,
                'buy_move': buy_move,
                'sell_move': sell_move,
            })

    moves_df = pd.DataFrame(session_moves)
    buy_opps = moves_df[moves_df['buy_move'] >= 5.0]
    sell_opps = moves_df[moves_df['sell_move'] >= 5.0]
    any_opps = moves_df[moves_df['total_range'] >= 5.0]

    print(f'  Total sessions: {len(moves_df)}')
    print(f'  Sessions with >= 5pt range: {len(any_opps)} ({len(any_opps)/len(moves_df)*100:.1f}%)')
    print(f'  Sessions with >= 5pt BUY move: {len(buy_opps)} ({len(buy_opps)/len(moves_df)*100:.1f}%)')
    print(f'  Sessions with >= 5pt SELL move: {len(sell_opps)} ({len(sell_opps)/len(moves_df)*100:.1f}%)')
    print(f'  Avg total range: {moves_df["total_range"].mean():.1f} pts')
    print()

    # Check how many of these sessions our top strategies capture
    print('  COVERAGE by top 10 profitable strategies:')
    top10 = profitable[:10]

    all_profitable_dates = set()
    for r in top10:
        winning_trades = [t for t in r['trades'] if t['pnl'] > 0]
        strategy_dates = set()
        for t in winning_trades:
            key = (t['date'], t['session'], t['direction'])
            strategy_dates.add(key)
            all_profitable_dates.add(key)

        # How many 5pt sessions did this strategy win in?
        covered_sell = len([k for k in strategy_dates if k[2] == 'SELL'])
        covered_buy = len([k for k in strategy_dates if k[2] == 'BUY'])
        rgm = '+rgm' if r['regime'] else ''
        print(f"    {r['strategy']:<16} {r['session']}/{r['direction']}{rgm:<8}: "
              f"{len(winning_trades)} wins ({covered_sell} SELL, {covered_buy} BUY sessions)")

    # Combined coverage
    total_sell_opps = len(sell_opps)
    total_buy_opps = len(buy_opps)
    covered_sell_sessions = len([k for k in all_profitable_dates if k[2] == 'SELL'])
    covered_buy_sessions = len([k for k in all_profitable_dates if k[2] == 'BUY'])

    print()
    print(f'  COMBINED COVERAGE (top 10 strategies):')
    print(f'    SELL: captured {covered_sell_sessions} winning sessions / {total_sell_opps} opportunities = {covered_sell_sessions/total_sell_opps*100:.1f}%')
    print(f'    BUY:  captured {covered_buy_sessions} winning sessions / {total_buy_opps} opportunities = {covered_buy_sessions/total_buy_opps*100:.1f}%')
    print(f'    TOTAL: {len(all_profitable_dates)} unique wins / {total_sell_opps + total_buy_opps} total opps = {len(all_profitable_dates)/(total_sell_opps+total_buy_opps)*100:.1f}%')

    # Overlap analysis
    print()
    print('  OVERLAP: Do top strategies fire on SAME or DIFFERENT days?')
    top5_sell = [r for r in profitable if r['direction'] == 'SELL'][:5]
    if len(top5_sell) >= 2:
        sets = []
        for r in top5_sell:
            dates = set(t['date'] + '_' + t['session'] for t in r['trades'] if t['pnl'] > 0)
            sets.append((r['strategy'] + ' ' + r['session'], dates))

        for i in range(min(3, len(sets))):
            for j in range(i + 1, min(4, len(sets))):
                overlap = len(sets[i][1] & sets[j][1])
                union = len(sets[i][1] | sets[j][1])
                print(f"    {sets[i][0]} vs {sets[j][0]}: overlap {overlap}/{union} = {overlap/union*100:.0f}%")


if __name__ == '__main__':
    main()
