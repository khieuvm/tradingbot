"""Simulate this week's shadow strategies — all active configs.

Week: Mon 2026-06-16 through Thu 2026-06-19 (Fri not yet).
Tests:
  - DOW-filtered 1m strategies
  - AM opening: ORB Breakdown, Crabel Stretch
  - PM opening: PM ORB (Mon SELL, Tue BUY, Wed BUY, Fri SELL)
  - Raw exit strategies (regime-filtered)

Exit rule: SL=3.0×ATR (unless custom_sl), session end.
Cost: 0.96 pts/trade.
"""
import sys
sys.path.insert(0, '.')
import importlib
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import date

COST = 0.96
DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
WEEK_START = '2026-06-15'
WEEK_END   = '2026-06-19'

# ── Strategy configs (mirrors shadow_scanner.py) ─────────────────────────────
# (name, module, class, session, direction, weekdays, regime_filter, exit)
STRATEGY_CONFIGS = [
    # ── DOW-filtered indicator strategies ──
    dict(key='dow_momentum_am_sell_tue',   mod='strategies.momentum_trend', cls='MomentumTrendStrategy',
         session='AM', direction=-1, weekdays=[1],    regime='sell', label='DOW Momentum AM SELL Tue'),
    dict(key='dow_heikin_ashi_am_buy_fri', mod='strategies.heikin_ashi', cls='HeikinAshiStrategy',
         session='AM', direction=1,  weekdays=[4],    regime='buy',  label='DOW Heikin Ashi AM BUY Fri'),
    dict(key='dow_momentum_pm_sell_monfri',mod='strategies.momentum_trend', cls='MomentumTrendStrategy',
         session='PM', direction=-1, weekdays=[0, 4], regime='sell', label='DOW Momentum PM SELL Mon+Fri'),
    dict(key='dow_divergence_pm_sell_mon', mod='strategies.divergence', cls='DivergenceStrategy',
         session='PM', direction=-1, weekdays=[0],    regime=None,   label='DOW Divergence PM SELL Mon'),
    dict(key='dow_macd_pm_buy_tuewed',     mod='strategies.macd_cross', cls='MACDCrossStrategy',
         session='PM', direction=1,  weekdays=[1, 2], regime='buy',  label='DOW MACD PM BUY Tue+Wed'),
    # ── Raw exit (all-week, regime-filtered) ──
    dict(key='raw_momentum_pm_buy',        mod='strategies.momentum_trend', cls='MomentumTrendStrategy',
         session='PM', direction=1,  weekdays=None,   regime='buy',  label='RAW Momentum PM BUY'),
    dict(key='raw_momentum_pm_sell',       mod='strategies.momentum_trend', cls='MomentumTrendStrategy',
         session='PM', direction=-1, weekdays=None,   regime='sell', label='RAW Momentum PM SELL'),
    dict(key='raw_macd_pm_buy',            mod='strategies.macd_cross', cls='MACDCrossStrategy',
         session='PM', direction=1,  weekdays=None,   regime='buy',  label='RAW MACD PM BUY'),
    dict(key='raw_divergence_am_sell',     mod='strategies.divergence', cls='DivergenceStrategy',
         session='AM', direction=-1, weekdays=None,   regime='sell', label='RAW Divergence AM SELL'),
]

# ── Opening strategies (separate logic) ──────────────────────────────────────
OPENING_CONFIGS = [
    dict(key='orb_am_sell', mod='strategies.orb_breakdown', cls='ORBBreakdownStrategy',
         kwargs=dict(or_bars=15, weekday_filter=[0, 2]),
         session='AM', direction=-1, custom_sl=None, label='AM ORB SELL Mon+Wed'),
    dict(key='crabel_sell', mod='strategies.crabel_stretch', cls='CrabelStretchStrategy',
         kwargs=dict(stretch_sma=5, stretch_pct=0.5, weekday_filter=[1, 2]),
         session='AM', direction=-1, custom_sl='open', label='Crabel Stretch SELL Tue+Wed'),
    dict(key='pm_orb_sell_mon', mod='strategies.pm_orb', cls='PMORBStrategy',
         kwargs=dict(or_bars=15, direction=-1, weekday_filter=[0]),
         session='PM', direction=-1, custom_sl=None, label='PM ORB SELL Mon'),
    dict(key='pm_orb_buy_tue', mod='strategies.pm_orb', cls='PMORBStrategy',
         kwargs=dict(or_bars=15, direction=1, weekday_filter=[1]),
         session='PM', direction=1,  custom_sl=None, label='PM ORB BUY Tue'),
    dict(key='pm_orb_buy_wed', mod='strategies.pm_orb', cls='PMORBStrategy',
         kwargs=dict(or_bars=15, direction=1, weekday_filter=[2]),
         session='PM', direction=1,  custom_sl=None, label='PM ORB BUY Wed'),
    dict(key='pm_orb_sell_fri', mod='strategies.pm_orb', cls='PMORBStrategy',
         kwargs=dict(or_bars=15, direction=-1, weekday_filter=[4]),
         session='PM', direction=-1, custom_sl=None, label='PM ORB SELL Fri'),
]


def fetch_and_prepare():
    from src.data_fetcher import DataFetcher
    fetcher = DataFetcher()
    print(f'[LOAD] Fetching 1m data {WEEK_START} to {WEEK_END}...')
    df = fetcher.get_futures_ohlcv('VN30F1M', WEEK_START, WEEK_END, interval='1m')
    if df is None or len(df) == 0:
        print('No data'); sys.exit(1)

    df = df.reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df['weekday'] = df['time'].dt.weekday
    df['session'] = df['mins'].apply(lambda m: 'AM' if m < 720 else 'PM')

    # Filter to trading hours
    am = (df['mins'] >= 540) & (df['mins'] <= 690)
    pm = (df['mins'] >= 780) & (df['mins'] <= 870)
    df = df[am | pm].reset_index(drop=True)

    # Indicators
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['rsi_7'] = ta.rsi(df['close'], length=7)
    df['range'] = df['high'] - df['low']

    bb = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    df['bb_mid']   = bb['BBM_20_2.0']
    df['bb_pos']   = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])).clip(0, 1)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=1.5)
    df['kc_upper'] = kc['KCUe_20_1.5']
    df['kc_lower'] = kc['KCLe_20_1.5']

    df['ema8']  = ta.ema(df['close'], length=8)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema_8']  = df['ema8']
    df['ema_21'] = df['ema21']
    df['ema_50'] = df['ema50']

    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd_hist']   = macd['MACDh_12_26_9']
    df['macd_line']   = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']

    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']

    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx']      = adx_df['ADX_14']
    df['di_plus']  = adx_df['DMP_14']
    df['di_minus'] = adx_df['DMN_14']
    df['di_spread'] = df['di_plus'] - df['di_minus']

    df['body_pct'] = (abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)).fillna(0)
    df['price_vs_ema50'] = (df['close'] - df['ema50']) / df['atr'].replace(0, np.nan)

    print(f'[DATA] {len(df)} bars across {df["date"].nunique()} days')
    return df


def simulate_exit(df, entry_idx, direction, sl_price, session, today_date):
    """Simulate raw exit: SL or session end."""
    session_end = 685 if session == 'AM' else 865
    for j in range(entry_idx + 1, min(entry_idx + 200, len(df))):
        bar = df.iloc[j]
        if bar['date'] != today_date or bar['session'] != session:
            return df.iloc[j - 1]['close'], 'SESSION', df.iloc[j - 1]['time'].strftime('%H:%M')
        if bar['mins'] >= session_end:
            return bar['close'], 'SESSION', bar['time'].strftime('%H:%M')
        if direction == 1 and bar['low'] <= sl_price:
            return sl_price, 'SL', bar['time'].strftime('%H:%M')
        if direction == -1 and bar['high'] >= sl_price:
            return sl_price, 'SL', bar['time'].strftime('%H:%M')
    last = df.iloc[min(entry_idx + 199, len(df) - 1)]
    return last['close'], 'EOD', last['time'].strftime('%H:%M')


def run_week(df):
    # Load strategy instances
    indicator_strats = {}
    for cfg in STRATEGY_CONFIGS:
        try:
            mod = importlib.import_module(cfg['mod'])
            indicator_strats[cfg['key']] = getattr(mod, cfg['cls'])()
        except Exception as e:
            print(f"  Skip {cfg['key']}: {e}")

    opening_strats = {}
    for cfg in OPENING_CONFIGS:
        try:
            mod = importlib.import_module(cfg['mod'])
            opening_strats[cfg['key']] = getattr(mod, cfg['cls'])(**cfg['kwargs'])
        except Exception as e:
            print(f"  Skip {cfg['key']}: {e}")

    all_trades = []
    daily_open_prices = {}

    dates = sorted(df['date'].unique())

    for day_date in dates:
        dow = pd.Timestamp(day_date).weekday()
        day_df_mask = df['date'] == day_date
        day_indices = df[day_df_mask].index.tolist()
        if not day_indices:
            continue

        # Record open price (for Crabel SL)
        first_am = df[(df['date'] == day_date) & (df['session'] == 'AM')]
        day_open = float(first_am['open'].iloc[0]) if len(first_am) > 0 else None
        daily_open_prices[day_date] = day_open

        last_sig_idx = {}  # per-strategy dedup within day

        for i in day_indices:
            row = df.iloc[i]
            mins = row['mins']
            session = row['session']
            atr = row['atr']
            if pd.isna(atr) or atr < 1.5:
                continue

            in_indicator = (
                (session == 'AM' and 555 <= mins <= 645) or
                (session == 'PM' and 795 <= mins <= 855)
            )
            in_opening = (
                (session == 'AM' and 540 <= mins <= 685) or
                (session == 'PM' and 795 <= mins <= 855)
            )

            # ── Indicator strategies ──
            if in_indicator:
                for cfg in STRATEGY_CONFIGS:
                    key = cfg['key']
                    if cfg['session'] != session:
                        continue
                    if cfg['weekdays'] is not None and dow not in cfg['weekdays']:
                        continue
                    # Dedup: 5 bars
                    if key in last_sig_idx and (i - last_sig_idx[key]) < 5:
                        continue

                    strat = indicator_strats.get(key)
                    if strat is None:
                        continue

                    try:
                        sig = strat.detect(df, i)
                    except:
                        continue

                    if sig == 0 or sig != cfg['direction']:
                        continue

                    # Regime filter
                    pve = row['price_vs_ema50']
                    if not pd.isna(pve) and cfg['regime'] == 'sell' and pve >= -0.5:
                        continue
                    if not pd.isna(pve) and cfg['regime'] == 'buy' and pve <= 0.5:
                        continue

                    last_sig_idx[key] = i
                    entry = float(row['close'])
                    sl = entry - sig * 3.0 * float(atr)
                    exit_p, exit_r, exit_t = simulate_exit(df, i, sig, sl, session, day_date)
                    pnl = (exit_p - entry) * sig - COST

                    all_trades.append({
                        'date': str(day_date), 'dow': DOW_NAMES[dow],
                        'time': row['time'].strftime('%H:%M'),
                        'strategy': cfg['label'], 'type': 'indicator',
                        'session': session, 'direction': 'BUY' if sig == 1 else 'SELL',
                        'entry': entry, 'exit': exit_p, 'exit_r': exit_r, 'exit_t': exit_t,
                        'pnl': round(pnl, 2), 'atr': round(float(atr), 2),
                    })

            # ── Opening strategies ──
            if in_opening:
                for cfg in OPENING_CONFIGS:
                    key = cfg['key']
                    if cfg['session'] != session:
                        continue
                    # Dedup: once per day (handled internally by strategy)
                    strat = opening_strats.get(key)
                    if strat is None:
                        continue

                    try:
                        sig = strat.detect(df, i)
                    except:
                        continue

                    if sig == 0:
                        continue

                    # Get trigger price
                    trigger = getattr(strat, 'trigger_price', None)
                    entry = float(trigger) if trigger is not None else float(row['close'])

                    # Determine SL
                    if cfg['custom_sl'] == 'open':
                        sl = daily_open_prices.get(day_date, entry - sig * 3.0 * float(atr))
                    else:
                        sl = entry - sig * 3.0 * float(atr)

                    exit_p, exit_r, exit_t = simulate_exit(df, i, sig, sl, session, day_date)
                    pnl = (exit_p - entry) * sig - COST

                    all_trades.append({
                        'date': str(day_date), 'dow': DOW_NAMES[dow],
                        'time': row['time'].strftime('%H:%M'),
                        'strategy': cfg['label'], 'type': 'opening',
                        'session': session, 'direction': 'BUY' if sig == 1 else 'SELL',
                        'entry': entry, 'exit': exit_p, 'exit_r': exit_r, 'exit_t': exit_t,
                        'pnl': round(pnl, 2), 'atr': round(float(atr), 2),
                    })

    return all_trades


def print_report(df, trades):
    print()
    print('=' * 80)
    print(f'  WEEKLY SIMULATION {WEEK_START} to {WEEK_END}')
    print('=' * 80)

    # Daily session overview
    print()
    print(f'  {"Date":<12} {"Day":<4} {"AM Move":>8} {"PM Move":>8} {"AM Rng":>8} {"PM Rng":>8}')
    print(f'  {"-"*50}')
    for d in sorted(df['date'].unique()):
        dow = DOW_NAMES[pd.Timestamp(d).weekday()]
        am = df[(df['date'] == d) & (df['session'] == 'AM')]
        pm = df[(df['date'] == d) & (df['session'] == 'PM')]
        am_mv = f"{am['close'].iloc[-1] - am['open'].iloc[0]:+.1f}" if len(am) > 0 else 'N/A'
        pm_mv = f"{pm['close'].iloc[-1] - pm['open'].iloc[0]:+.1f}" if len(pm) > 0 else 'N/A'
        am_rg = f"{am['high'].max() - am['low'].min():.1f}" if len(am) > 0 else 'N/A'
        pm_rg = f"{pm['high'].max() - pm['low'].min():.1f}" if len(pm) > 0 else 'N/A'
        print(f'  {str(d):<12} {dow:<4} {am_mv:>8} {pm_mv:>8} {am_rg:>8} {pm_rg:>8}')

    if not trades:
        print('\n  No trades this week.')
        return

    # All trades
    print()
    print('=' * 80)
    print(f'  ALL TRADES THIS WEEK ({len(trades)} total)')
    print('=' * 80)
    print()
    print(f'  {"Date":<11} {"Day":<3} {"Time":<5} {"S":<2} {"Dir":<4} {"Strategy":<32} '
          f'{"Entry":>7} {"Exit":>7} {"Rst":<7} {"PnL":>6}')
    print(f'  {"-"*83}')

    for t in trades:
        win_mark = '+' if t['pnl'] > 0 else '-' if t['pnl'] < 0 else '0'
        print(f'  {t["date"][5:]:<11} {t["dow"]:<3} {t["time"]:<5} {t["session"]:<2} '
              f'{t["direction"]:<4} {t["strategy"]:<32} '
              f'{t["entry"]:>7.1f} {t["exit"]:>7.1f} {t["exit_r"]:<7} {t["pnl"]:>+6.2f}')

    total = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    wr = len(wins) / len(trades) * 100

    print(f'  {"-"*83}')
    print(f'  Total: {total:+.2f} pts | WR: {wr:.0f}% ({len(wins)}/{len(trades)}) | Avg: {total/len(trades):+.2f}')

    # By day
    print()
    print('  BY DAY:')
    print(f'  {"Date":<11} {"Day":<3} {"N":>3} {"W":>3} {"L":>3} {"WR":>5} {"PnL":>7}')
    print(f'  {"-"*38}')
    for d in sorted(df['date'].unique()):
        dt = [t for t in trades if t['date'] == str(d)]
        if not dt:
            dow = DOW_NAMES[pd.Timestamp(d).weekday()]
            print(f'  {str(d)[5:]:<11} {dow:<3}   0   0   0    n/a    0.00')
            continue
        dow = dt[0]['dow']
        dw = [t for t in dt if t['pnl'] > 0]
        dl = [t for t in dt if t['pnl'] <= 0]
        dwr = len(dw) / len(dt) * 100
        dpnl = sum(t['pnl'] for t in dt)
        print(f'  {str(d)[5:]:<11} {dow:<3} {len(dt):>3} {len(dw):>3} {len(dl):>3} {dwr:>4.0f}% {dpnl:>+7.2f}')

    # By strategy
    print()
    print('  BY STRATEGY:')
    print(f'  {"Strategy":<32} {"N":>3} {"WR":>5} {"PnL":>7}')
    print(f'  {"-"*50}')
    strat_names = sorted(set(t['strategy'] for t in trades))
    for s in strat_names:
        st = [t for t in trades if t['strategy'] == s]
        sw = [t for t in st if t['pnl'] > 0]
        swr = len(sw) / len(st) * 100 if st else 0
        spnl = sum(t['pnl'] for t in st)
        print(f'  {s:<32} {len(st):>3} {swr:>4.0f}% {spnl:>+7.2f}')

    # Summary
    print()
    print('=' * 80)
    print(f'  WEEK TOTAL: {total:+.2f} pts  |  {len(trades)} trades  |  WR {wr:.0f}%  |  Avg {total/len(trades):+.2f}/trade')
    print('=' * 80)


if __name__ == '__main__':
    df = fetch_and_prepare()
    trades = run_week(df)
    print_report(df, trades)
