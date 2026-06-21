"""Day-of-Week deep analysis: session patterns, best strategies per day.

Tests all opening + momentum strategies split by Mon/Tue/Wed/Thu/Fri.
Both AM and PM sessions. Uses 33 days of 1m data.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import pandas_ta as ta
import importlib
from src.data_fetcher import DataFetcher

COST = 0.96
DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

ALL_STRATEGIES = {
    'momentum_trend': ('strategies.momentum_trend', 'MomentumTrendStrategy'),
    'macd_cross': ('strategies.macd_cross', 'MACDCrossStrategy'),
    'divergence': ('strategies.divergence', 'DivergenceStrategy'),
    'choch': ('strategies.choch', 'CHoCHStrategy'),
    'market_structure': ('strategies.market_structure', 'MarketStructureStrategy'),
    'bos': ('strategies.bos', 'BOSStrategy'),
    'volume': ('strategies.volume', 'VolumeStrategy'),
    'heikin_ashi': ('strategies.heikin_ashi', 'HeikinAshiStrategy'),
}


def load_data():
    fetcher = DataFetcher()
    df = fetcher.get_futures_ohlcv('VN30F1M', '2026-05-02', '2026-06-17', interval='1m')
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['session'] = df['mins'].apply(lambda m: 'AM' if m < 720 else 'PM')
    df['weekday'] = df['time'].dt.weekday
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_14'] = df['atr']
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
    df['price_vs_ema50'] = (df['close'] - df['ema50']) / df['atr']

    print(f'[DATA] {len(df)} bars, {df["date"].nunique()} days')
    return df


def simulate_exit(df, entry_idx, direction, sl_price):
    row = df.iloc[entry_idx]
    session = row['session']
    date = row['date']
    for j in range(entry_idx + 1, min(entry_idx + 200, len(df))):
        bar = df.iloc[j]
        if bar['date'] != date or bar['session'] != session:
            return df.iloc[j - 1]['close'], 'SESSION'
        if session == 'AM' and bar['mins'] >= 685:
            return bar['close'], 'SESSION'
        if session == 'PM' and bar['mins'] >= 865:
            return bar['close'], 'SESSION'
        if direction == 1 and bar['low'] <= sl_price:
            return sl_price, 'SL'
        if direction == -1 and bar['high'] >= sl_price:
            return sl_price, 'SL'
    return df.iloc[min(entry_idx + 199, len(df) - 1)]['close'], 'MAX'


def main():
    df = load_data()
    dates = sorted(df['date'].unique())
    n_days = len(dates)

    # ═══════════════════════════════════════════════════════════
    # PART 1: Session characteristics per weekday
    # ═══════════════════════════════════════════════════════════
    print()
    print('=' * 80)
    print('  PART 1: SESSION CHARACTERISTICS BY WEEKDAY')
    print('=' * 80)
    print()

    for sess_name in ['AM', 'PM']:
        print(f'  --- {sess_name} SESSION ---')
        print(f'  {"Day":<5} {"N":>3} {"Avg Net":>8} {"Avg Rng":>8} '
              f'{"Drop>3":>7} {"Rise>3":>7} {"SELL%":>6} {"Avg ATR":>8}')
        print(f'  {"-"*55}')

        for dow in range(5):
            day_dates = [d for d in dates if pd.Timestamp(d).weekday() == dow]
            nets, rngs, atrs = [], [], []
            drops, rises = 0, 0

            for date in day_dates:
                if sess_name == 'AM':
                    mask = (df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 690)
                else:
                    mask = (df['date'] == date) & (df['mins'] >= 780) & (df['mins'] <= 870)

                sdf = df[mask]
                if len(sdf) < 10:
                    continue

                net = sdf['close'].iloc[-1] - sdf['open'].iloc[0]
                rng = sdf['high'].max() - sdf['low'].min()
                avg_atr = sdf['atr'].mean()
                nets.append(net)
                rngs.append(rng)
                atrs.append(avg_atr)
                if net < -3:
                    drops += 1
                if net > 3:
                    rises += 1

            if not nets:
                continue
            n = len(nets)
            sell_pct = (drops / n * 100) if n > 0 else 0
            print(f'  {DOW[dow]:<5} {n:>3} {np.mean(nets):>+8.2f} {np.mean(rngs):>8.1f} '
                  f'{drops:>7} {rises:>7} {sell_pct:>5.0f}% {np.nanmean(atrs):>8.2f}')

        print()

    # ═══════════════════════════════════════════════════════════
    # PART 2: All strategies x weekday x session x direction
    # ═══════════════════════════════════════════════════════════
    print('=' * 80)
    print('  PART 2: ALL STRATEGIES BY WEEKDAY (raw exit, SL=3.0xATR)')
    print('=' * 80)
    print()

    # Load all strategies
    strats = {}
    for sname, (mod_path, class_name) in ALL_STRATEGIES.items():
        try:
            mod = importlib.import_module(mod_path)
            strats[sname] = getattr(mod, class_name)()
        except:
            pass

    # Run all strategies, record trades with weekday
    all_trades = []
    for sname, strategy in strats.items():
        last_sig = -10
        for i in range(60, len(df)):
            row = df.iloc[i]
            if row['session'] == 'AM' and not (555 <= row['mins'] <= 645):
                continue
            if row['session'] == 'PM' and not (795 <= row['mins'] <= 855):
                continue
            if i - last_sig < 5:
                continue

            atr = row['atr']
            if pd.isna(atr) or atr <= 0:
                continue

            try:
                signal = strategy.detect(df, i)
            except:
                continue
            if signal == 0:
                continue

            last_sig = i
            direction = signal
            entry = row['close']
            sl = entry - direction * 3.0 * atr
            exit_p, exit_r = simulate_exit(df, i, direction, sl)
            pnl = (exit_p - entry) * direction - COST

            all_trades.append({
                'strategy': sname,
                'date': str(row['date']),
                'session': row['session'],
                'direction': 'BUY' if direction == 1 else 'SELL',
                'weekday': row['weekday'],
                'pnl': pnl,
                'exit_reason': exit_r,
            })

    # Print results: best strategy for each weekday
    print(f'  Total trades: {len(all_trades)}')
    print()

    # Summary table: each weekday, best AM and PM strategy
    for sess in ['AM', 'PM']:
        print(f'  === {sess} SESSION ===')
        print(f'  {"Day":<5} {"Strategy":<16} {"Dir":<5} {"N":>3} {"WR":>5} {"PF":>5} {"PnL":>7}')
        print(f'  {"-"*50}')

        for dow in range(5):
            best_pf = 0
            best_row = None

            for sname in strats:
                for dirn in ['BUY', 'SELL']:
                    trades = [t for t in all_trades
                              if t['strategy'] == sname
                              and t['session'] == sess
                              and t['direction'] == dirn
                              and t['weekday'] == dow]
                    if len(trades) < 2:
                        continue
                    pnls = [t['pnl'] for t in trades]
                    wins = [p for p in pnls if p > 0]
                    gl = abs(sum([p for p in pnls if p <= 0])) or 0.001
                    pf = sum(wins) / gl
                    wr = len(wins) / len(pnls) * 100
                    total = sum(pnls)

                    if pf > best_pf:
                        best_pf = pf
                        best_row = (DOW[dow], sname, dirn, len(pnls), wr, pf, total)

            if best_row and best_row[5] > 0.5:
                d, s, di, n, wr, pf, total = best_row
                marker = ' ***' if pf > 1.5 else (' *' if pf > 1.0 else '')
                print(f'  {d:<5} {s:<16} {di:<5} {n:>3} {wr:>4.0f}% {pf:>5.2f} {total:>+7.1f}{marker}')

        print()

    # ═══════════════════════════════════════════════════════════
    # PART 3: Opening strategies per weekday
    # ═══════════════════════════════════════════════════════════
    print('=' * 80)
    print('  PART 3: OPENING STRATEGIES BY WEEKDAY')
    print('=' * 80)
    print()

    # ORB Breakdown SELL
    print('  --- ORB Breakdown SELL (15min range, SL=3xATR) ---')
    print(f'  {"Day":<5} {"N":>3} {"WR":>5} {"PF":>5} {"PnL":>7} {"Trades":<50}')
    print(f'  {"-"*75}')

    for dow in range(5):
        trades = []
        for date in dates:
            if pd.Timestamp(date).weekday() != dow:
                continue
            am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
            if len(am) < 20:
                continue
            opening = am.iloc[:15]
            or_low = opening['low'].min()
            atr = opening['atr'].iloc[-1]
            if pd.isna(atr) or atr <= 0:
                continue

            post = am.iloc[15:]
            if len(post) < 5:
                continue
            traded = False
            for j in range(len(post)):
                if traded:
                    break
                bar = post.iloc[j]
                if bar['mins'] > 645:
                    break
                if bar['close'] < or_low:
                    traded = True
                    sl = or_low + 3.0 * atr
                    exit_p, exit_r = simulate_exit(df, post.index[j], -1, sl)
                    pnl = (or_low - exit_p) - COST
                    trades.append({'date': str(date), 'pnl': pnl})

        if not trades:
            print(f'  {DOW[dow]:<5}   0   n/a   n/a     n/a')
            continue
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        gl = abs(sum([p for p in pnls if p <= 0])) or 0.001
        pf = sum(wins) / gl
        wr = len(wins) / len(pnls) * 100
        detail = ', '.join([f'{t["date"][-5:]}({t["pnl"]:+.0f})' for t in trades])
        print(f'  {DOW[dow]:<5} {len(pnls):>3} {wr:>4.0f}% {pf:>5.2f} {sum(pnls):>+7.1f} {detail}')

    print()

    # Crabel Stretch SELL
    d_ranges = {}
    for date in dates:
        day_df = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 690)]
        if len(day_df) < 10:
            continue
        d_ranges[date] = day_df['high'].max() - day_df['low'].min()

    dates_list = sorted(d_ranges.keys())

    print('  --- Crabel Stretch SELL (SMA5, 50%, stop=open) ---')
    print(f'  {"Day":<5} {"N":>3} {"WR":>5} {"PF":>5} {"PnL":>7} {"Trades":<50}')
    print(f'  {"-"*75}')

    for dow in range(5):
        trades = []
        for i, date in enumerate(dates_list):
            if i < 5:
                continue
            if pd.Timestamp(date).weekday() != dow:
                continue
            recent = [d_ranges[dates_list[k]] for k in range(i - 5, i)]
            stretch = np.mean(recent)
            am = df[(df['date'] == date) & (df['mins'] >= 540) & (df['mins'] <= 685)]
            if len(am) < 20:
                continue
            am_open = am['open'].iloc[0]
            short_trigger = am_open - stretch * 0.5
            traded = False
            for j in range(5, len(am)):
                if traded:
                    break
                bar = am.iloc[j]
                if bar['low'] < short_trigger:
                    traded = True
                    sl = am_open
                    exit_p, exit_r = simulate_exit(df, am.index[j], -1, sl)
                    pnl = (short_trigger - exit_p) - COST
                    trades.append({'date': str(date), 'pnl': pnl})

        if not trades:
            print(f'  {DOW[dow]:<5}   0   n/a   n/a     n/a')
            continue
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        gl = abs(sum([p for p in pnls if p <= 0])) or 0.001
        pf = sum(wins) / gl
        wr = len(wins) / len(pnls) * 100
        detail = ', '.join([f'{t["date"][-5:]}({t["pnl"]:+.0f})' for t in trades])
        print(f'  {DOW[dow]:<5} {len(pnls):>3} {wr:>4.0f}% {pf:>5.2f} {sum(pnls):>+7.1f} {detail}')

    print()

    # PM Momentum BUY per weekday
    print('  --- PM Momentum BUY (raw exit, regime filter) ---')
    print(f'  {"Day":<5} {"N":>3} {"WR":>5} {"PF":>5} {"PnL":>7}')
    print(f'  {"-"*35}')

    for dow in range(5):
        trades = [t for t in all_trades
                  if t['strategy'] == 'momentum_trend'
                  and t['session'] == 'PM' and t['direction'] == 'BUY'
                  and t['weekday'] == dow]
        if not trades:
            print(f'  {DOW[dow]:<5}   0')
            continue
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        gl = abs(sum([p for p in pnls if p <= 0])) or 0.001
        print(f'  {DOW[dow]:<5} {len(pnls):>3} {len(wins)/len(pnls)*100:>4.0f}% '
              f'{sum(wins)/gl:>5.2f} {sum(pnls):>+7.1f}')

    print()

    # PM Momentum SELL per weekday
    print('  --- PM Momentum SELL (raw exit, no regime) ---')
    print(f'  {"Day":<5} {"N":>3} {"WR":>5} {"PF":>5} {"PnL":>7}')
    print(f'  {"-"*35}')

    for dow in range(5):
        trades = [t for t in all_trades
                  if t['strategy'] == 'momentum_trend'
                  and t['session'] == 'PM' and t['direction'] == 'SELL'
                  and t['weekday'] == dow]
        if not trades:
            print(f'  {DOW[dow]:<5}   0')
            continue
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        gl = abs(sum([p for p in pnls if p <= 0])) or 0.001
        print(f'  {DOW[dow]:<5} {len(pnls):>3} {len(wins)/len(pnls)*100:>4.0f}% '
              f'{sum(wins)/gl:>5.2f} {sum(pnls):>+7.1f}')

    # ═══════════════════════════════════════════════════════════
    # PART 4: RECOMMENDATION PER WEEKDAY
    # ═══════════════════════════════════════════════════════════
    print()
    print('=' * 80)
    print('  PART 4: RECOMMENDED STRATEGY BY WEEKDAY')
    print('=' * 80)
    print()
    print('  Based on 33 days backtest (2026-05-04 to 2026-06-17)')
    print()


if __name__ == '__main__':
    main()
