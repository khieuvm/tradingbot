"""
Backtest improved combos on ALL timeframes (1m, 3m, 5m).
Uses BREAKOUT direction (high+0.1 / low-0.1 trigger) matching production logic.

Improvements tested:
  - CB + CI falling filter (Choppiness < 61.8 & falling)
  - CB + ADX<15 filter
  - CB + CI||ADX<20 combined
  - NR7 improved exit (BE@2.5pts, trail@3pts)
  - NR4 improved exit
  - NR7 PM-only
  - ORB relaxed
  - NR7 + CI filter
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from src.data_fetcher import DataFetcher

COST = 0.96

TF_PARAMS = {
    '5m': {
        'interval': '5m', 'days_back': 180,
        'cb_bars': 3, 'cb_thresh': 0.7, 'dedup': 5,
        'atr_min': 2.5, 'atr_max': 4.5,
        'nr7_lookback': 7, 'nr4_lookback': 4,
        'am_sl': 1.2, 'am_trail_act': 5, 'am_trail_mult': 2.0, 'am_max_hold': 24,
        'pm_sl': 1.0, 'pm_trail_act': 4, 'pm_trail_mult': 1.5, 'pm_max_hold': 12,
        'be_trigger': 4, 'be_atr_min': 3.5,
    },
    '3m': {
        'interval': '3m', 'days_back': 28,
        'cb_bars': 5, 'cb_thresh': 0.7, 'dedup': 8,
        'atr_min': 1.5, 'atr_max': 3.0,
        'nr7_lookback': 7, 'nr4_lookback': 4,
        'am_sl': 1.2, 'am_trail_act': 5, 'am_trail_mult': 2.0, 'am_max_hold': 40,
        'pm_sl': 1.0, 'pm_trail_act': 4, 'pm_trail_mult': 1.5, 'pm_max_hold': 20,
        'be_trigger': 4, 'be_atr_min': 2.0,
    },
    '1m': {
        'interval': '1m', 'days_back': 28,
        'cb_bars': 10, 'cb_thresh': 0.7, 'dedup': 15,
        'atr_min': 0.7, 'atr_max': 1.5,
        'nr7_lookback': 7, 'nr4_lookback': 4,
        'am_sl': 1.2, 'am_trail_act': 5, 'am_trail_mult': 2.0, 'am_max_hold': 120,
        'pm_sl': 1.0, 'pm_trail_act': 4, 'pm_trail_mult': 1.5, 'pm_max_hold': 60,
        'be_trigger': 4, 'be_atr_min': 1.0,
    },
}

NR_EXIT = {
    '5m': {'be_trigger': 2.5, 'be_atr_min': 2.5, 'am_trail_act': 3, 'pm_trail_act': 2.5},
    '3m': {'be_trigger': 2.5, 'be_atr_min': 1.5, 'am_trail_act': 3, 'pm_trail_act': 2.5},
    '1m': {'be_trigger': 2.0, 'be_atr_min': 0.7, 'am_trail_act': 2.5, 'pm_trail_act': 2.0},
}


def load_data(tf_key):
    params = TF_PARAMS[tf_key]
    fetcher = DataFetcher()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=params['days_back'])).strftime("%Y-%m-%d")
    df = fetcher.get_futures_ohlcv("VN30F1M", start, end, interval=params['interval'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
    df = df[((df['mins'] >= 540) & (df['mins'] < 690)) | ((df['mins'] >= 780) & (df['mins'] < 870))]
    if tf_key in ('1m', '3m'):
        df = df[df['time'] >= '2026-05-02']
    df = df.reset_index(drop=True)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['range'] = df['high'] - df['low']
    df['session'] = np.where(df['mins'] < 720, 'AM', 'PM')
    df['chop'] = ta.chop(df['high'], df['low'], df['close'], length=14)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14']
    return df


def simulate_breakout(df, signals_df, label, tf_key, exit_override=None):
    """
    Simulate with BREAKOUT direction: after signal bar, wait for price to break
    signal bar high+0.1 (BUY) or low-0.1 (SELL). Max 2 bars to trigger.
    This matches the production CB scanner logic.
    """
    params = TF_PARAMS[tf_key]
    if exit_override:
        p = {**params, **exit_override}
    else:
        p = params

    results = []
    for _, sig in signals_df.iterrows():
        i = int(sig['idx'])
        if i + 3 >= len(df):
            continue
        session = sig['session']
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < 0.1:
            continue

        sig_bar = df.iloc[i]
        buy_trigger = sig_bar['high'] + 0.1
        sell_trigger = sig_bar['low'] - 0.1

        # Look for breakout in next 2 bars
        direction = 0
        entry_bar_idx = None
        for k in range(1, 3):
            if i + k >= len(df):
                break
            bar = df.iloc[i + k]
            if bar['date'] != sig_bar['date']:
                break
            if bar['high'] >= buy_trigger:
                direction = 1
                entry_bar_idx = i + k
                break
            elif bar['low'] <= sell_trigger:
                direction = -1
                entry_bar_idx = i + k
                break

        if direction == 0 or entry_bar_idx is None:
            continue

        entry = buy_trigger if direction == 1 else sell_trigger

        if session == 'AM':
            sl_mult = p['am_sl']
            trail_act = p.get('am_trail_act', params['am_trail_act'])
            trail_mult = p['am_trail_mult']
            max_hold = p['am_max_hold']
        else:
            sl_mult = p['pm_sl']
            trail_act = p.get('pm_trail_act', params['pm_trail_act'])
            trail_mult = p['pm_trail_mult']
            max_hold = p['pm_max_hold']

        sl = entry - direction * sl_mult * atr
        cutoff = 685 if session == 'AM' else 865

        best_price = entry
        mfe = 0
        trail_active = False
        exit_p = None
        exit_r = None

        start_j = entry_bar_idx + 1
        for j in range(start_j, min(start_j + max_hold, len(df))):
            b = df.iloc[j]
            if b['date'] != sig_bar['date']:
                exit_p = df.iloc[j - 1]['close']
                exit_r = 'SESSION'
                break
            if b['mins'] >= cutoff:
                exit_p = b['close']
                exit_r = 'SESSION'
                break

            if direction == 1:
                if b['low'] <= sl:
                    exit_p = sl
                    exit_r = 'TRAIL' if trail_active else 'SL'
                    break
                best_price = max(best_price, b['high'])
                mfe = max(mfe, b['high'] - entry)
            else:
                if b['high'] >= sl:
                    exit_p = sl
                    exit_r = 'TRAIL' if trail_active else 'SL'
                    break
                best_price = min(best_price, b['low'])
                mfe = max(mfe, entry - b['low'])

            # BE
            be_trigger = p.get('be_trigger', 4)
            be_atr_min = p.get('be_atr_min', 3.5)
            if mfe >= be_trigger and atr >= be_atr_min:
                if direction == 1:
                    sl = max(sl, entry)
                else:
                    sl = min(sl, entry)

            # Trail
            if mfe >= trail_act:
                trail_active = True
                if direction == 1:
                    sl = max(sl, best_price - trail_mult * atr)
                else:
                    sl = min(sl, best_price + trail_mult * atr)
        else:
            exit_p = df.iloc[min(start_j + max_hold - 1, len(df) - 1)]['close']
            exit_r = 'MAX_HOLD'

        if exit_p is None:
            exit_p = df.iloc[-1]['close']
            exit_r = 'SESSION'

        pnl = direction * (exit_p - entry) - COST
        results.append({
            'pnl': pnl, 'mfe': mfe, 'exit': exit_r,
            'session': session, 'direction': direction,
            'mins': df.iloc[i]['mins'], 'atr': atr
        })

    if not results:
        print(f"    {label:<45} | NO SIGNALS")
        return None

    rdf = pd.DataFrame(results)
    trades = len(rdf)
    wins = (rdf['pnl'] > 0).sum()
    wr = wins / trades * 100
    gross_w = rdf[rdf['pnl'] > 0]['pnl'].sum()
    gross_l = abs(rdf[rdf['pnl'] < 0]['pnl'].sum())
    pf = gross_w / gross_l if gross_l > 0 else 999
    total_pnl = rdf['pnl'].sum()
    n_days = df['date'].nunique()
    pd_val = total_pnl / n_days

    sl_count = len(rdf[rdf['exit'] == 'SL'])
    trail_count = len(rdf[rdf['exit'] == 'TRAIL'])
    session_count = len(rdf[rdf['exit'] == 'SESSION'])
    maxh_count = len(rdf[rdf['exit'] == 'MAX_HOLD'])

    am = rdf[rdf['session'] == 'AM']
    pm = rdf[rdf['session'] == 'PM']
    am_wr = (am['pnl'] > 0).sum() / len(am) * 100 if len(am) > 0 else 0
    pm_wr = (pm['pnl'] > 0).sum() / len(pm) * 100 if len(pm) > 0 else 0
    am_pf = am[am['pnl']>0]['pnl'].sum() / abs(am[am['pnl']<0]['pnl'].sum()) if len(am) > 0 and am[am['pnl']<0]['pnl'].sum() != 0 else 0
    pm_pf = pm[pm['pnl']>0]['pnl'].sum() / abs(pm[pm['pnl']<0]['pnl'].sum()) if len(pm) > 0 and pm[pm['pnl']<0]['pnl'].sum() != 0 else 0

    mfe_2plus = (rdf['mfe'] >= 2).sum()
    mfe_4plus = (rdf['mfe'] >= 4).sum()
    sl_with_mfe2 = len(rdf[(rdf['exit'] == 'SL') & (rdf['mfe'] >= 2)])
    avg_mfe = rdf['mfe'].mean()
    avg_win = rdf[rdf['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = rdf[rdf['pnl'] < 0]['pnl'].mean() if (trades - wins) > 0 else 0

    print(f"    {label:<45} | T={trades:>4} WR={wr:>5.1f}% PF={pf:>5.2f} PnL={total_pnl:>+7.1f} P/D={pd_val:>+5.2f}")
    print(f"      AM: {len(am):>3}t WR={am_wr:>5.1f}% PF={am_pf:>4.1f} | PM: {len(pm):>3}t WR={pm_wr:>5.1f}% PF={pm_pf:>4.1f}")
    print(f"      Exits: SL={sl_count} TRAIL={trail_count} SESS={session_count} MAX={maxh_count} | AvgW={avg_win:>+.1f} AvgL={avg_loss:>+.1f}")
    print(f"      MFE: avg={avg_mfe:.1f} | ≥2:{mfe_2plus}({mfe_2plus/trades*100:.0f}%) ≥4:{mfe_4plus}({mfe_4plus/trades*100:.0f}%) | SL+MFE≥2: {sl_with_mfe2}/{sl_count if sl_count else 1}")

    return {
        'label': label, 'tf': tf_key, 'trades': trades, 'wr': wr, 'pf': pf,
        'total_pnl': total_pnl, 'pd': pd_val,
        'am_t': len(am), 'am_wr': am_wr, 'am_pf': am_pf,
        'pm_t': len(pm), 'pm_wr': pm_wr, 'pm_pf': pm_pf,
        'sl': sl_count, 'trail': trail_count, 'sess': session_count,
        'avg_mfe': avg_mfe, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'mfe2_pct': mfe_2plus / trades * 100,
        'sl_mfe2': sl_with_mfe2
    }


# === SIGNAL GENERATORS ===

def gen_cb_base(df, tf_key):
    params = TF_PARAMS[tf_key]
    signals = []
    last_sig = -999
    for i in range(14, len(df)):
        if i - last_sig < params['dedup']:
            continue
        row = df.iloc[i]
        mins = row['mins']
        if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
            continue
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < params['atr_min'] or atr > params['atr_max']:
            continue
        rsi = df['rsi'].iloc[i]
        if pd.isna(rsi) or rsi > 70:
            continue
        if i < params['cb_bars']:
            continue
        ranges = [df['range'].iloc[i - k] for k in range(1, params['cb_bars'] + 1)]
        if max(ranges) >= params['cb_thresh'] * atr:
            continue
        if i + 1 >= len(df):
            continue
        last_sig = i
        signals.append({'idx': i, 'direction': 0, 'session': row['session']})
    return pd.DataFrame(signals)


def gen_cb_ci(df, tf_key):
    """CB + CI<61.8 & falling."""
    params = TF_PARAMS[tf_key]
    signals = []
    last_sig = -999
    for i in range(14, len(df)):
        if i - last_sig < params['dedup']:
            continue
        row = df.iloc[i]
        mins = row['mins']
        if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
            continue
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < params['atr_min'] or atr > params['atr_max']:
            continue
        rsi = df['rsi'].iloc[i]
        if pd.isna(rsi) or rsi > 70:
            continue
        if i < params['cb_bars']:
            continue
        ranges = [df['range'].iloc[i - k] for k in range(1, params['cb_bars'] + 1)]
        if max(ranges) >= params['cb_thresh'] * atr:
            continue
        ci = df['chop'].iloc[i]
        ci_prev = df['chop'].iloc[i - 1]
        if pd.isna(ci) or pd.isna(ci_prev) or ci >= 61.8 or ci >= ci_prev:
            continue
        if i + 1 >= len(df):
            continue
        last_sig = i
        signals.append({'idx': i, 'direction': 0, 'session': row['session']})
    return pd.DataFrame(signals)


def gen_cb_adx15(df, tf_key):
    """CB + ADX<15."""
    params = TF_PARAMS[tf_key]
    signals = []
    last_sig = -999
    for i in range(14, len(df)):
        if i - last_sig < params['dedup']:
            continue
        row = df.iloc[i]
        mins = row['mins']
        if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
            continue
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < params['atr_min'] or atr > params['atr_max']:
            continue
        rsi = df['rsi'].iloc[i]
        if pd.isna(rsi) or rsi > 70:
            continue
        if i < params['cb_bars']:
            continue
        ranges = [df['range'].iloc[i - k] for k in range(1, params['cb_bars'] + 1)]
        if max(ranges) >= params['cb_thresh'] * atr:
            continue
        adx = df['adx'].iloc[i]
        if pd.isna(adx) or adx >= 15:
            continue
        if i + 1 >= len(df):
            continue
        last_sig = i
        signals.append({'idx': i, 'direction': 0, 'session': row['session']})
    return pd.DataFrame(signals)


def gen_cb_ci_or_adx(df, tf_key):
    """CB + (CI<61.8 falling OR ADX<20)."""
    params = TF_PARAMS[tf_key]
    signals = []
    last_sig = -999
    for i in range(14, len(df)):
        if i - last_sig < params['dedup']:
            continue
        row = df.iloc[i]
        mins = row['mins']
        if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
            continue
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < params['atr_min'] or atr > params['atr_max']:
            continue
        rsi = df['rsi'].iloc[i]
        if pd.isna(rsi) or rsi > 70:
            continue
        if i < params['cb_bars']:
            continue
        ranges = [df['range'].iloc[i - k] for k in range(1, params['cb_bars'] + 1)]
        if max(ranges) >= params['cb_thresh'] * atr:
            continue
        ci = df['chop'].iloc[i]
        ci_prev = df['chop'].iloc[i - 1]
        adx = df['adx'].iloc[i]
        ci_pass = (not pd.isna(ci) and not pd.isna(ci_prev) and ci < 61.8 and ci < ci_prev)
        adx_pass = (not pd.isna(adx) and adx < 20)
        if not (ci_pass or adx_pass):
            continue
        if i + 1 >= len(df):
            continue
        last_sig = i
        signals.append({'idx': i, 'direction': 0, 'session': row['session']})
    return pd.DataFrame(signals)


def gen_nr7(df, tf_key, pm_only=False):
    params = TF_PARAMS[tf_key]
    signals = []
    last_sig = -999
    for i in range(14, len(df)):
        if i - last_sig < params['dedup']:
            continue
        row = df.iloc[i]
        mins = row['mins']
        if pm_only:
            if not (795 <= mins <= 855):
                continue
        else:
            if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
                continue
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < params['atr_min'] or atr > params['atr_max']:
            continue
        if i < params['nr7_lookback']:
            continue
        cur_range = df['range'].iloc[i]
        prev_ranges = [df['range'].iloc[i - k] for k in range(1, params['nr7_lookback'])]
        if cur_range >= min(prev_ranges):
            continue
        if cur_range >= 0.8 * atr:
            continue
        if i + 1 >= len(df):
            continue
        last_sig = i
        signals.append({'idx': i, 'direction': 0, 'session': row['session']})
    return pd.DataFrame(signals)


def gen_nr4(df, tf_key):
    params = TF_PARAMS[tf_key]
    signals = []
    last_sig = -999
    for i in range(14, len(df)):
        if i - last_sig < params['dedup']:
            continue
        row = df.iloc[i]
        mins = row['mins']
        if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
            continue
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < params['atr_min'] or atr > params['atr_max']:
            continue
        if i < params['nr4_lookback']:
            continue
        cur_range = df['range'].iloc[i]
        prev_ranges = [df['range'].iloc[i - k] for k in range(1, params['nr4_lookback'])]
        if cur_range >= min(prev_ranges):
            continue
        if cur_range >= 0.7 * atr:
            continue
        if i + 1 >= len(df):
            continue
        last_sig = i
        signals.append({'idx': i, 'direction': 0, 'session': row['session']})
    return pd.DataFrame(signals)


def gen_nr7_ci(df, tf_key):
    """NR7 + CI<61.8 falling."""
    params = TF_PARAMS[tf_key]
    signals = []
    last_sig = -999
    for i in range(14, len(df)):
        if i - last_sig < params['dedup']:
            continue
        row = df.iloc[i]
        mins = row['mins']
        if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
            continue
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < params['atr_min'] or atr > params['atr_max']:
            continue
        if i < params['nr7_lookback']:
            continue
        cur_range = df['range'].iloc[i]
        prev_ranges = [df['range'].iloc[i - k] for k in range(1, params['nr7_lookback'])]
        if cur_range >= min(prev_ranges):
            continue
        if cur_range >= 0.8 * atr:
            continue
        ci = df['chop'].iloc[i]
        ci_prev = df['chop'].iloc[i - 1]
        if pd.isna(ci) or pd.isna(ci_prev) or ci >= 61.8 or ci >= ci_prev:
            continue
        if i + 1 >= len(df):
            continue
        last_sig = i
        signals.append({'idx': i, 'direction': 0, 'session': row['session']})
    return pd.DataFrame(signals)


def gen_orb(df, tf_key):
    """ORB relaxed: range < 1.2*ATR, look bars 3-12."""
    days = sorted(df['date'].unique())
    signals = []
    for day in days:
        day_df = df[df['date'] == day]
        for session_start, session_label in [(540, 'AM'), (780, 'PM')]:
            s_df = day_df[(day_df['mins'] >= session_start) & (day_df['mins'] < session_start + 150)]
            if len(s_df) < 5:
                continue
            first3 = s_df.iloc[:3]
            or_high = first3['high'].max()
            or_low = first3['low'].min()
            or_range = or_high - or_low
            atr = s_df['atr'].iloc[2] if not pd.isna(s_df['atr'].iloc[2]) else 3.0
            if or_range >= 1.2 * atr or or_range < 0.2:
                continue
            for j in range(3, min(13, len(s_df))):
                bar = s_df.iloc[j]
                idx_global = s_df.index[j]
                if bar['high'] > or_high + 0.1:
                    signals.append({'idx': idx_global, 'direction': 1, 'session': session_label})
                    break
                elif bar['low'] < or_low - 0.1:
                    signals.append({'idx': idx_global, 'direction': -1, 'session': session_label})
                    break
    return pd.DataFrame(signals)


def gen_inside_bar(df, tf_key):
    """Double Inside Bar."""
    params = TF_PARAMS[tf_key]
    signals = []
    last_sig = -999
    for i in range(14, len(df)):
        if i - last_sig < params['dedup']:
            continue
        row = df.iloc[i]
        mins = row['mins']
        if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
            continue
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < params['atr_min'] or atr > params['atr_max']:
            continue
        if i < 3:
            continue
        b0 = df.iloc[i]
        b1 = df.iloc[i - 1]
        b2 = df.iloc[i - 2]
        ib1 = b0['high'] < b1['high'] and b0['low'] > b1['low']
        ib2 = b1['high'] < b2['high'] and b1['low'] > b2['low']
        if not (ib1 and ib2):
            continue
        mother_range = b2['high'] - b2['low']
        if mother_range < 0.5 * atr or mother_range > 2.0 * atr:
            continue
        if i + 1 >= len(df):
            continue
        last_sig = i
        signals.append({'idx': i, 'direction': 0, 'session': row['session']})
    return pd.DataFrame(signals)


def simulate_orb(df, signals_df, label, tf_key, exit_override=None):
    """ORB already has direction from the breakout logic, use direct entry."""
    params = TF_PARAMS[tf_key]
    if exit_override:
        p = {**params, **exit_override}
    else:
        p = params

    results = []
    for _, sig in signals_df.iterrows():
        i = int(sig['idx'])
        if i + 2 >= len(df):
            continue
        direction = int(sig['direction'])
        session = sig['session']
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < 0.1:
            continue

        entry = df['close'].iloc[i]

        if session == 'AM':
            sl_mult = p['am_sl']
            trail_act = p.get('am_trail_act', params['am_trail_act'])
            trail_mult = p['am_trail_mult']
            max_hold = p['am_max_hold']
        else:
            sl_mult = p['pm_sl']
            trail_act = p.get('pm_trail_act', params['pm_trail_act'])
            trail_mult = p['pm_trail_mult']
            max_hold = p['pm_max_hold']

        sl = entry - direction * sl_mult * atr
        cutoff = 685 if session == 'AM' else 865
        best_price = entry
        mfe = 0
        trail_active = False
        exit_p = None
        exit_r = None

        for j in range(i + 1, min(i + 1 + max_hold, len(df))):
            b = df.iloc[j]
            if b['date'] != df.iloc[i]['date']:
                exit_p = df.iloc[j - 1]['close']
                exit_r = 'SESSION'
                break
            if b['mins'] >= cutoff:
                exit_p = b['close']
                exit_r = 'SESSION'
                break
            if direction == 1:
                if b['low'] <= sl:
                    exit_p = sl
                    exit_r = 'TRAIL' if trail_active else 'SL'
                    break
                best_price = max(best_price, b['high'])
                mfe = max(mfe, b['high'] - entry)
            else:
                if b['high'] >= sl:
                    exit_p = sl
                    exit_r = 'TRAIL' if trail_active else 'SL'
                    break
                best_price = min(best_price, b['low'])
                mfe = max(mfe, entry - b['low'])

            be_trigger = p.get('be_trigger', 4)
            be_atr_min = p.get('be_atr_min', 3.5)
            if mfe >= be_trigger and atr >= be_atr_min:
                if direction == 1:
                    sl = max(sl, entry)
                else:
                    sl = min(sl, entry)
            if mfe >= trail_act:
                trail_active = True
                if direction == 1:
                    sl = max(sl, best_price - trail_mult * atr)
                else:
                    sl = min(sl, best_price + trail_mult * atr)
        else:
            exit_p = df.iloc[min(i + max_hold, len(df) - 1)]['close']
            exit_r = 'MAX_HOLD'

        if exit_p is None:
            exit_p = df.iloc[-1]['close']
            exit_r = 'SESSION'

        pnl = direction * (exit_p - entry) - COST
        results.append({'pnl': pnl, 'mfe': mfe, 'exit': exit_r, 'session': session, 'direction': direction, 'mins': df.iloc[i]['mins'], 'atr': atr})

    if not results:
        print(f"    {label:<45} | NO SIGNALS")
        return None

    rdf = pd.DataFrame(results)
    trades = len(rdf)
    wins = (rdf['pnl'] > 0).sum()
    wr = wins / trades * 100
    gross_w = rdf[rdf['pnl'] > 0]['pnl'].sum()
    gross_l = abs(rdf[rdf['pnl'] < 0]['pnl'].sum())
    pf = gross_w / gross_l if gross_l > 0 else 999
    total_pnl = rdf['pnl'].sum()
    n_days = df['date'].nunique()
    pd_val = total_pnl / n_days
    sl_count = len(rdf[rdf['exit'] == 'SL'])
    trail_count = len(rdf[rdf['exit'] == 'TRAIL'])
    session_count = len(rdf[rdf['exit'] == 'SESSION'])
    maxh_count = len(rdf[rdf['exit'] == 'MAX_HOLD'])
    am = rdf[rdf['session'] == 'AM']
    pm = rdf[rdf['session'] == 'PM']
    am_wr = (am['pnl'] > 0).sum() / len(am) * 100 if len(am) > 0 else 0
    pm_wr = (pm['pnl'] > 0).sum() / len(pm) * 100 if len(pm) > 0 else 0
    am_pf = am[am['pnl']>0]['pnl'].sum() / abs(am[am['pnl']<0]['pnl'].sum()) if len(am) > 0 and am[am['pnl']<0]['pnl'].sum() != 0 else 0
    pm_pf = pm[pm['pnl']>0]['pnl'].sum() / abs(pm[pm['pnl']<0]['pnl'].sum()) if len(pm) > 0 and pm[pm['pnl']<0]['pnl'].sum() != 0 else 0
    avg_mfe = rdf['mfe'].mean()
    avg_win = rdf[rdf['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = rdf[rdf['pnl'] < 0]['pnl'].mean() if (trades - wins) > 0 else 0
    mfe_2plus = (rdf['mfe'] >= 2).sum()
    mfe_4plus = (rdf['mfe'] >= 4).sum()
    sl_with_mfe2 = len(rdf[(rdf['exit'] == 'SL') & (rdf['mfe'] >= 2)])

    print(f"    {label:<45} | T={trades:>4} WR={wr:>5.1f}% PF={pf:>5.2f} PnL={total_pnl:>+7.1f} P/D={pd_val:>+5.2f}")
    print(f"      AM: {len(am):>3}t WR={am_wr:>5.1f}% PF={am_pf:>4.1f} | PM: {len(pm):>3}t WR={pm_wr:>5.1f}% PF={pm_pf:>4.1f}")
    print(f"      Exits: SL={sl_count} TRAIL={trail_count} SESS={session_count} MAX={maxh_count} | AvgW={avg_win:>+.1f} AvgL={avg_loss:>+.1f}")
    print(f"      MFE: avg={avg_mfe:.1f} | ≥2:{mfe_2plus}({mfe_2plus/trades*100:.0f}%) ≥4:{mfe_4plus}({mfe_4plus/trades*100:.0f}%) | SL+MFE≥2: {sl_with_mfe2}/{sl_count if sl_count else 1}")

    return {
        'label': label, 'tf': tf_key, 'trades': trades, 'wr': wr, 'pf': pf,
        'total_pnl': total_pnl, 'pd': pd_val,
        'am_t': len(am), 'am_wr': am_wr, 'am_pf': am_pf,
        'pm_t': len(pm), 'pm_wr': pm_wr, 'pm_pf': pm_pf,
        'sl': sl_count, 'trail': trail_count, 'sess': session_count,
        'avg_mfe': avg_mfe, 'avg_win': avg_win, 'avg_loss': avg_loss,
    }


# === MAIN ===
if __name__ == '__main__':
    all_results = []

    for tf_key in ['5m', '3m', '1m']:
        print(f"\n{'='*100}")
        print(f"  TIMEFRAME: {tf_key} | Days back: {TF_PARAMS[tf_key]['days_back']} | ATR: {TF_PARAMS[tf_key]['atr_min']}-{TF_PARAMS[tf_key]['atr_max']}")
        print(f"  CB bars: {TF_PARAMS[tf_key]['cb_bars']} | Dedup: {TF_PARAMS[tf_key]['dedup']} | BE: {TF_PARAMS[tf_key]['be_trigger']}pts")
        print(f"{'='*100}")

        df = load_data(tf_key)
        n_days = df['date'].nunique()
        print(f"  Loaded: {len(df)} bars, {n_days} days")
        atr_valid = df['atr'].dropna()
        print(f"  ATR: mean={atr_valid.mean():.2f} med={atr_valid.median():.2f} range=[{atr_valid.min():.2f}, {atr_valid.max():.2f}]")
        print()

        # CB strategies
        print(f"  {'─'*50} CB STRATEGIES {'─'*35}")
        sigs = gen_cb_base(df, tf_key)
        r = simulate_breakout(df, sigs, f"CB Base", tf_key)
        if r: all_results.append(r)
        print()

        sigs = gen_cb_ci(df, tf_key)
        r = simulate_breakout(df, sigs, f"CB + CI<61.8 falling", tf_key)
        if r: all_results.append(r)
        print()

        sigs = gen_cb_adx15(df, tf_key)
        r = simulate_breakout(df, sigs, f"CB + ADX<15", tf_key)
        if r: all_results.append(r)
        print()

        sigs = gen_cb_ci_or_adx(df, tf_key)
        r = simulate_breakout(df, sigs, f"CB + (CI||ADX<20)", tf_key)
        if r: all_results.append(r)
        print()

        # NR strategies
        print(f"  {'─'*50} NR STRATEGIES {'─'*35}")
        nr_exit = NR_EXIT[tf_key]

        sigs = gen_nr7(df, tf_key)
        r = simulate_breakout(df, sigs, f"NR7 std exit", tf_key)
        if r: all_results.append(r)
        print()

        sigs = gen_nr7(df, tf_key)
        r = simulate_breakout(df, sigs, f"NR7 improved (BE@2.5, trail@3)", tf_key, exit_override=nr_exit)
        if r: all_results.append(r)
        print()

        sigs = gen_nr7(df, tf_key, pm_only=True)
        r = simulate_breakout(df, sigs, f"NR7 PM-only improved", tf_key, exit_override=nr_exit)
        if r: all_results.append(r)
        print()

        sigs = gen_nr7_ci(df, tf_key)
        r = simulate_breakout(df, sigs, f"NR7 + CI falling", tf_key, exit_override=nr_exit)
        if r: all_results.append(r)
        print()

        sigs = gen_nr4(df, tf_key)
        r = simulate_breakout(df, sigs, f"NR4 std exit", tf_key)
        if r: all_results.append(r)
        print()

        sigs = gen_nr4(df, tf_key)
        r = simulate_breakout(df, sigs, f"NR4 improved (BE@2.5, trail@3)", tf_key, exit_override=nr_exit)
        if r: all_results.append(r)
        print()

        # Other strategies
        print(f"  {'─'*50} OTHER {'─'*43}")
        sigs = gen_orb(df, tf_key)
        if len(sigs) > 0:
            r = simulate_orb(df, sigs, f"ORB relaxed", tf_key)
            if r: all_results.append(r)
        else:
            print(f"    {'ORB relaxed':<45} | NO SIGNALS")
        print()

        sigs = gen_inside_bar(df, tf_key)
        r = simulate_breakout(df, sigs, f"Double Inside Bar", tf_key, exit_override=nr_exit)
        if r: all_results.append(r)
        print()

    # Final summary
    print(f"\n{'='*110}")
    print(f"  FINAL SUMMARY — SORTED BY P/D (all timeframes)")
    print(f"{'='*110}")
    print(f"  {'Strategy':<40} {'TF':<4} {'T':>4} {'WR%':>6} {'PF':>6} {'P/D':>7} {'AM_WR':>6} {'PM_WR':>6} {'AvgW':>6} {'AvgL':>6} {'SL%':>5}")
    print(f"  {'─'*40} {'─'*4} {'─'*4} {'─'*6} {'─'*6} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*5}")
    for r in sorted(all_results, key=lambda x: -x['pd']):
        sl_pct = r['sl'] / r['trades'] * 100 if r['trades'] > 0 else 0
        print(f"  {r['label']:<40} {r['tf']:<4} {r['trades']:>4} {r['wr']:>5.1f}% {r['pf']:>5.2f} {r['pd']:>+6.2f} {r['am_wr']:>5.1f}% {r['pm_wr']:>5.1f}% {r['avg_win']:>+5.1f} {r['avg_loss']:>+5.1f} {sl_pct:>4.0f}%")
