"""Simulate today's trading with all strategies (RAW exit approach)."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import pandas_ta as ta
import importlib
from datetime import datetime, timedelta, timezone

COST = 0.96
VN_TZ = timezone(timedelta(hours=7))
TODAY = '2026-06-17'
YESTERDAY = '2026-06-14'

# Load strategies
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

# RAW exit configs (the ones in shadow tracker)
RAW_CONFIGS = [
    ('momentum_trend', 'PM', 'BUY', 'buy', 'RAW Momentum PM BUY'),
    ('momentum_trend', 'PM', 'SELL', 'sell', 'RAW Momentum PM SELL'),
    ('macd_cross', 'PM', 'BUY', 'buy', 'RAW MACD PM BUY'),
    ('divergence', 'AM', 'SELL', 'sell', 'RAW Divergence AM SELL'),
]


def load_and_prepare():
    from src.data_fetcher import DataFetcher
    fetcher = DataFetcher()

    print(f'[LOAD] Fetching 1m data...')
    df = fetcher.get_futures_ohlcv('VN30F1M', YESTERDAY, TODAY, interval='1m')
    if df is None or len(df) == 0:
        print('No data available')
        sys.exit(1)

    print(f'[DATA] {len(df)} bars, range: {df["time"].iloc[0]} to {df["time"].iloc[-1]}')

    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['session'] = df['mins'].apply(lambda m: 'AM' if m < 720 else 'PM')
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

    return df


def simulate_today(df):
    today_date = pd.Timestamp(TODAY).date()
    today_mask = df['date'] == today_date
    today_indices = df[today_mask].index.tolist()

    if not today_indices:
        print('No data for today')
        return

    # Session overview
    am_bars = df[(df['date'] == today_date) & (df['session'] == 'AM')]
    pm_bars = df[(df['date'] == today_date) & (df['session'] == 'PM')]

    print()
    print('=' * 70)
    print(f'  SESSION OVERVIEW: {TODAY}')
    print('=' * 70)
    if len(am_bars) > 0:
        am_range = am_bars['high'].max() - am_bars['low'].min()
        am_move = am_bars['close'].iloc[-1] - am_bars['open'].iloc[0]
        print(f'  AM: {am_bars["time"].iloc[0].strftime("%H:%M")}-{am_bars["time"].iloc[-1].strftime("%H:%M")} '
              f'| Range: {am_range:.1f}pts | Net: {am_move:+.1f}pts')
        print(f'      Low: {am_bars["low"].min():.1f} | High: {am_bars["high"].max():.1f}')
    if len(pm_bars) > 0:
        pm_range = pm_bars['high'].max() - pm_bars['low'].min()
        pm_move = pm_bars['close'].iloc[-1] - pm_bars['open'].iloc[0]
        print(f'  PM: {pm_bars["time"].iloc[0].strftime("%H:%M")}-{pm_bars["time"].iloc[-1].strftime("%H:%M")} '
              f'| Range: {pm_range:.1f}pts | Net: {pm_move:+.1f}pts')
        print(f'      Low: {pm_bars["low"].min():.1f} | High: {pm_bars["high"].max():.1f}')
    else:
        print('  PM: No data yet (market may still be open or data not available)')

    # Load strategies
    strats = {}
    for sname, (mod_path, class_name) in ALL_STRATEGIES.items():
        try:
            mod = importlib.import_module(mod_path)
            strats[sname] = getattr(mod, class_name)()
        except Exception as e:
            pass

    # Find all signals
    signals = []
    last_sig_idx = {}

    for i in today_indices:
        row = df.iloc[i]
        mins = row['mins']
        session = row['session']
        atr = row['atr']

        if session == 'AM' and not (555 <= mins <= 645):
            continue
        if session == 'PM' and not (795 <= mins <= 855):
            continue
        if pd.isna(atr) or atr < 1.5:
            continue

        for sname, strategy in strats.items():
            if sname in last_sig_idx and (i - last_sig_idx[sname]) < 5:
                continue

            try:
                signal = strategy.detect(df, i)
            except:
                continue

            if signal == 0:
                continue

            last_sig_idx[sname] = i
            direction = signal
            dir_str = 'BUY' if direction == 1 else 'SELL'

            # Regime check
            pve = row['price_vs_ema50']
            if pd.isna(pve):
                regime_pass = False
            elif direction == -1 and pve >= -0.5:
                regime_pass = False
            elif direction == 1 and pve <= 0.5:
                regime_pass = False
            else:
                regime_pass = True

            # Compute exit (raw: SL=3.0xATR, hold to session end)
            entry = row['close']
            sl_price = entry - direction * 3.0 * atr
            exit_price = None
            exit_reason = 'OPEN'
            exit_time = ''

            for j in range(i + 1, today_indices[-1] + 1):
                if j >= len(df):
                    break
                bar = df.iloc[j]
                if bar['date'] != today_date:
                    exit_price = df.iloc[j - 1]['close']
                    exit_reason = 'SESSION'
                    exit_time = df.iloc[j - 1]['time'].strftime('%H:%M')
                    break
                if bar['session'] != session:
                    exit_price = df.iloc[j - 1]['close']
                    exit_reason = 'SESSION'
                    exit_time = df.iloc[j - 1]['time'].strftime('%H:%M')
                    break
                if session == 'AM' and bar['mins'] >= 685:
                    exit_price = bar['close']
                    exit_reason = 'SESSION'
                    exit_time = bar['time'].strftime('%H:%M')
                    break
                if session == 'PM' and bar['mins'] >= 865:
                    exit_price = bar['close']
                    exit_reason = 'SESSION'
                    exit_time = bar['time'].strftime('%H:%M')
                    break
                if direction == 1 and bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    exit_time = bar['time'].strftime('%H:%M')
                    break
                if direction == -1 and bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    exit_time = bar['time'].strftime('%H:%M')
                    break

            if exit_price is None:
                # Still open or at end of data
                exit_price = df.iloc[today_indices[-1]]['close']
                exit_reason = 'STILL_OPEN'
                exit_time = df.iloc[today_indices[-1]]['time'].strftime('%H:%M')

            pnl = (exit_price - entry) * direction - COST

            signals.append({
                'time': row['time'].strftime('%H:%M'),
                'strategy': sname,
                'direction': dir_str,
                'session': session,
                'entry': entry,
                'sl': sl_price,
                'exit_price': exit_price,
                'exit_time': exit_time,
                'exit_reason': exit_reason,
                'pnl': pnl,
                'atr': atr,
                'regime_pass': regime_pass,
                'pve': pve if not pd.isna(pve) else 0,
            })

    signals.sort(key=lambda x: x['time'])

    # Print all signals
    print()
    print('=' * 70)
    print(f'  ALL SIGNALS TODAY: {len(signals)} total')
    print('=' * 70)
    print()

    if not signals:
        print('  No signals fired today.')
        return

    print(f'{"Time":<6} {"Strategy":<16} {"Dir":<5} {"S":<3} {"Entry":>7} {"SL":>7} '
          f'{"Exit":>7} {"ExT":<6} {"Why":<8} {"PnL":>6} {"R":<2}')
    print('-' * 85)

    total_all = 0
    for s in signals:
        rgm = '*' if s['regime_pass'] else ' '
        print(f'{s["time"]:<6} {s["strategy"]:<16} {s["direction"]:<5} {s["session"]:<3} '
              f'{s["entry"]:>7.1f} {s["sl"]:>7.1f} {s["exit_price"]:>7.1f} {s["exit_time"]:<6} '
              f'{s["exit_reason"]:<8} {s["pnl"]:>+6.1f} {rgm}')
        total_all += s['pnl']

    print('-' * 85)
    print(f'Total all signals: {total_all:+.1f} pts | (* = regime filter pass)')

    # RAW EXIT strategies (the shadow tracked ones)
    print()
    print('=' * 70)
    print('  RAW EXIT SHADOW STRATEGIES (regime-filtered)')
    print('=' * 70)
    print()

    raw_signals = []
    for s in signals:
        for (strat, sess, dirn, regime_type, label) in RAW_CONFIGS:
            if (s['strategy'] == strat and s['session'] == sess
                    and s['direction'] == dirn and s['regime_pass']):
                raw_signals.append({**s, 'label': label})
                break

    if raw_signals:
        print(f'{"Time":<6} {"Label":<26} {"Entry":>7} {"SL":>7} {"Exit":>7} '
              f'{"ExT":<6} {"Why":<8} {"PnL":>6}')
        print('-' * 75)
        raw_total = 0
        for s in raw_signals:
            print(f'{s["time"]:<6} {s["label"]:<26} {s["entry"]:>7.1f} {s["sl"]:>7.1f} '
                  f'{s["exit_price"]:>7.1f} {s["exit_time"]:<6} {s["exit_reason"]:<8} {s["pnl"]:>+6.1f}')
            raw_total += s['pnl']
        print('-' * 75)
        print(f'RAW EXIT total: {raw_total:+.1f} pts ({len(raw_signals)} trades)')
    else:
        print('  No RAW exit signals passed regime filter today.')
        print()
        # Show why
        potential = []
        for s in signals:
            for (strat, sess, dirn, regime_type, label) in RAW_CONFIGS:
                if s['strategy'] == strat and s['session'] == sess and s['direction'] == dirn:
                    potential.append({**s, 'label': label})
                    break
        if potential:
            print('  Signals that WOULD have fired without regime filter:')
            for s in potential:
                print(f'    {s["time"]} {s["label"]}: entry={s["entry"]:.1f}, '
                      f'PnL={s["pnl"]:+.1f}, pve={s["pve"]:.2f} (need {"<-0.5" if "SELL" in s["label"] else ">0.5"})')

    # Also show all shadow strategies (non-RAW)
    print()
    print('=' * 70)
    print('  OTHER SHADOW STRATEGIES (standard exit)')
    print('=' * 70)
    print()

    shadow_configs = [
        ('choch', 'AM', None, 'CHoCH 1m'),
        ('choch', 'PM', None, 'CHoCH 1m'),
        ('momentum_trend', 'PM', 'SELL', 'Momentum PM SELL'),
        ('momentum_trend', 'AM', 'SELL', 'Momentum AM SELL'),
        ('heikin_ashi', 'PM', 'SELL', 'Heikin Ashi PM SELL'),
        ('volume', 'PM', 'SELL', 'Volume PM SELL'),
    ]

    shadow_signals = []
    for s in signals:
        for (strat, sess, dirn, label) in shadow_configs:
            if s['strategy'] == strat and s['session'] == sess:
                if dirn is None or s['direction'] == dirn:
                    shadow_signals.append({**s, 'label': label})
                    break

    if shadow_signals:
        for s in shadow_signals:
            print(f'  {s["time"]} {s["label"]} {s["direction"]}: '
                  f'entry={s["entry"]:.1f} exit={s["exit_price"]:.1f} ({s["exit_reason"]}) '
                  f'PnL={s["pnl"]:+.1f}')
    else:
        print('  No shadow strategy signals today.')


if __name__ == '__main__':
    df = load_and_prepare()
    simulate_today(df)
