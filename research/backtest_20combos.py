"""
Backtest 10 new combo strategies on VN30F1M 5m data.
Compares each to CB baseline (WR 65%, PF 4.5, +6.0/d).
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

# === DATA LOADING ===
fetcher = DataFetcher()
end = datetime.now().strftime("%Y-%m-%d")
start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
df = fetcher.get_futures_ohlcv("VN30F1M", start, end, interval="5m")
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df['date'] = df['time'].dt.date
df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
df = df[((df['mins'] >= 540) & (df['mins'] < 690)) | ((df['mins'] >= 780) & (df['mins'] < 870))]
df = df.reset_index(drop=True)

# Indicators
df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
df['rsi'] = ta.rsi(df['close'], length=14)
df['range'] = df['high'] - df['low']
df['session'] = np.where(df['mins'] < 720, 'AM', 'PM')

# ADX
adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
df['adx'] = adx_df['ADX_14']

# Bollinger Bands & Keltner Channel for TTM Squeeze
bb = ta.bbands(df['close'], length=20, std=2.0)
df['bb_upper'] = bb['BBU_20_2.0_2.0']
df['bb_lower'] = bb['BBL_20_2.0_2.0']

kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=1.5)
df['kc_upper'] = kc['KCUe_20_1.5']
df['kc_lower'] = kc['KCLe_20_1.5']

df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])

# Choppiness Index
df['chop'] = ta.chop(df['high'], df['low'], df['close'], length=14)

days = sorted(df['date'].unique())
n_days = len(days)
print(f"Data: {len(df)} bars, {n_days} days")
print(f"=" * 90)


# === SIMULATION ENGINE ===
def simulate(signals_df, label):
    """Simulate trades from signal list. Each signal has: idx, direction(1/-1), session."""
    results = []
    for _, sig in signals_df.iterrows():
        i = int(sig['idx'])
        if i + 2 >= len(df):
            continue
        direction = int(sig['direction'])
        session = sig['session']
        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < 1.0:
            continue

        entry = df['close'].iloc[i + 1]  # enter on next bar close

        # Session params
        if session == 'AM':
            sl_mult, trail_act, trail_mult, max_hold = 1.2, 5, 2.0, 24
        else:
            sl_mult, trail_act, trail_mult, max_hold = 1.0, 4, 1.5, 12

        sl = entry - direction * sl_mult * atr
        cutoff = 685 if session == 'AM' else 865  # 11:25 or 14:25

        best_price = entry
        mfe = 0
        trail_active = False
        exit_p = None
        exit_r = None

        for j in range(i + 2, min(i + 2 + max_hold, len(df))):
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
                cur_mfe = b['high'] - entry
                if b['low'] <= sl:
                    exit_p = sl
                    exit_r = 'TRAIL' if trail_active else 'SL'
                    break
                best_price = max(best_price, b['high'])
            else:
                cur_mfe = entry - b['low']
                if b['high'] >= sl:
                    exit_p = sl
                    exit_r = 'TRAIL' if trail_active else 'SL'
                    break
                best_price = min(best_price, b['low'])

            mfe = max(mfe, cur_mfe)

            # BE at 4pts
            if mfe >= 4 and atr >= 3.5:
                sl_new = entry
                if direction == 1:
                    sl = max(sl, sl_new)
                else:
                    sl = min(sl, sl_new)

            # Trail
            if mfe >= trail_act:
                trail_active = True
                if direction == 1:
                    sl = max(sl, best_price - trail_mult * atr)
                else:
                    sl = min(sl, best_price + trail_mult * atr)
        else:
            exit_p = df.iloc[min(i + 1 + max_hold, len(df) - 1)]['close']
            exit_r = 'MAX_HOLD'

        if exit_p is None:
            exit_p = df.iloc[-1]['close']
            exit_r = 'SESSION'

        pnl = direction * (exit_p - entry) - COST
        results.append({'pnl': pnl, 'mfe': mfe, 'exit': exit_r, 'session': session})

    if not results:
        print(f"  {label:<35} | {'NO SIGNALS':<50}")
        return

    rdf = pd.DataFrame(results)
    trades = len(rdf)
    wins = (rdf['pnl'] > 0).sum()
    wr = wins / trades * 100
    gross_w = rdf[rdf['pnl'] > 0]['pnl'].sum()
    gross_l = abs(rdf[rdf['pnl'] < 0]['pnl'].sum())
    pf = gross_w / gross_l if gross_l > 0 else 999
    total_pnl = rdf['pnl'].sum()
    pd_val = total_pnl / n_days

    print(f"  {label:<35} | T={trades:>4} | WR={wr:>5.1f}% | PF={pf:>5.2f} | PnL={total_pnl:>+7.1f} | P/D={pd_val:>+5.2f}/d | SL={len(rdf[rdf['exit']=='SL']):>3}")


# === COMBO 1: CB BASELINE ===
print(f"\n{'COMBO 1: CB Baseline (3-bar compression < 0.7*ATR)'}")
signals = []
last_sig = -999
for i in range(14, len(df)):
    if i - last_sig < 5:
        continue
    row = df.iloc[i]
    mins = row['mins']
    if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
        continue
    atr = df['atr'].iloc[i]
    if pd.isna(atr) or atr < 2.5 or atr > 4.5:
        continue
    rsi = df['rsi'].iloc[i]
    if pd.isna(rsi) or rsi > 70:
        continue
    if i < 3:
        continue
    ranges = [df['range'].iloc[i - k] for k in range(1, 4)]
    if max(ranges) >= 0.7 * atr:
        continue
    if i + 1 >= len(df):
        continue
    last_sig = i
    nxt = df.iloc[i + 1]
    direction = 1 if nxt['close'] > row['close'] else -1
    signals.append({'idx': i, 'direction': direction, 'session': row['session']})

simulate(pd.DataFrame(signals), "CB 3-bar < 0.7*ATR")


# === COMBO 2: NR7 ===
print(f"\n{'COMBO 2: NR7 Breakout (narrowest range in 7 bars)'}")
signals = []
last_sig = -999
for i in range(14, len(df)):
    if i - last_sig < 5:
        continue
    row = df.iloc[i]
    mins = row['mins']
    if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
        continue
    atr = df['atr'].iloc[i]
    if pd.isna(atr) or atr < 2.5 or atr > 4.5:
        continue
    if i < 7:
        continue
    cur_range = df['range'].iloc[i]
    prev_ranges = [df['range'].iloc[i - k] for k in range(1, 7)]
    if cur_range >= min(prev_ranges):  # must be THE smallest
        continue
    if cur_range >= 0.8 * atr:  # also filter by ATR
        continue
    if i + 1 >= len(df):
        continue
    last_sig = i
    nxt = df.iloc[i + 1]
    direction = 1 if nxt['close'] > row['close'] else -1
    signals.append({'idx': i, 'direction': direction, 'session': row['session']})

simulate(pd.DataFrame(signals), "NR7 < 0.8*ATR")


# === COMBO 3: NR4 ===
print(f"\n{'COMBO 3: NR4 Breakout (narrowest range in 4 bars)'}")
signals = []
last_sig = -999
for i in range(14, len(df)):
    if i - last_sig < 5:
        continue
    row = df.iloc[i]
    mins = row['mins']
    if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
        continue
    atr = df['atr'].iloc[i]
    if pd.isna(atr) or atr < 2.5 or atr > 4.5:
        continue
    if i < 4:
        continue
    cur_range = df['range'].iloc[i]
    prev_ranges = [df['range'].iloc[i - k] for k in range(1, 4)]
    if cur_range >= min(prev_ranges):
        continue
    if cur_range >= 0.7 * atr:
        continue
    if i + 1 >= len(df):
        continue
    last_sig = i
    nxt = df.iloc[i + 1]
    direction = 1 if nxt['close'] > row['close'] else -1
    signals.append({'idx': i, 'direction': direction, 'session': row['session']})

simulate(pd.DataFrame(signals), "NR4 < 0.7*ATR")


# === COMBO 4: ORB (Opening Range Breakout) ===
print(f"\n{'COMBO 4: ORB (Opening Range Breakout - first 3 bars)'}")
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
        if or_range >= 0.8 * atr or or_range < 0.3:
            continue
        # Look for breakout in bars 3-8
        for j in range(3, min(9, len(s_df))):
            bar = s_df.iloc[j]
            idx_global = s_df.index[j]
            if bar['high'] > or_high + 0.1:
                signals.append({'idx': idx_global, 'direction': 1, 'session': session_label})
                break
            elif bar['low'] < or_low - 0.1:
                signals.append({'idx': idx_global, 'direction': -1, 'session': session_label})
                break

simulate(pd.DataFrame(signals), "ORB 3-bar, range < 0.8*ATR")


# === COMBO 5: Inside Bar (Double IB) ===
print(f"\n{'COMBO 5: Inside Bar (Double Inside Bar breakout)'}")
signals = []
last_sig = -999
for i in range(14, len(df)):
    if i - last_sig < 5:
        continue
    row = df.iloc[i]
    mins = row['mins']
    if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
        continue
    atr = df['atr'].iloc[i]
    if pd.isna(atr) or atr < 2.5 or atr > 4.5:
        continue
    if i < 3:
        continue
    # Double inside bar: bar[i] inside bar[i-1], bar[i-1] inside bar[i-2]
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
    nxt = df.iloc[i + 1]
    direction = 1 if nxt['close'] > row['close'] else -1
    signals.append({'idx': i, 'direction': direction, 'session': row['session']})

simulate(pd.DataFrame(signals), "Double Inside Bar")


# === COMBO 6: TTM Squeeze Fire ===
print(f"\n{'COMBO 6: TTM Squeeze Fire (BB inside KC -> release)'}")
signals = []
last_sig = -999
for i in range(21, len(df)):
    if i - last_sig < 5:
        continue
    row = df.iloc[i]
    mins = row['mins']
    if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
        continue
    atr = df['atr'].iloc[i]
    if pd.isna(atr) or atr < 2.5 or atr > 4.5:
        continue
    # Squeeze fire: was ON, now OFF
    if pd.isna(df['squeeze_on'].iloc[i]) or pd.isna(df['squeeze_on'].iloc[i - 1]):
        continue
    was_squeeze = df['squeeze_on'].iloc[i - 1]
    now_no_squeeze = not df['squeeze_on'].iloc[i]
    if not (was_squeeze and now_no_squeeze):
        continue
    # Must have been in squeeze for >= 3 bars
    squeeze_count = 0
    for k in range(1, min(i, 20)):
        if df['squeeze_on'].iloc[i - k]:
            squeeze_count += 1
        else:
            break
    if squeeze_count < 3:
        continue
    if i + 1 >= len(df):
        continue
    last_sig = i
    nxt = df.iloc[i + 1]
    direction = 1 if nxt['close'] > row['close'] else -1
    signals.append({'idx': i, 'direction': direction, 'session': row['session']})

simulate(pd.DataFrame(signals), "TTM Squeeze Fire (3+ bars)")


# === COMBO 7: Multi-TF Compression (15m + 5m) ===
print(f"\n{'COMBO 7: Multi-TF CB (5m compression + 15m compression)'}")
# Resample to 15m
df15 = df.set_index('time').resample('15min').agg(
    {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
).dropna().reset_index()
df15['range'] = df15['high'] - df15['low']
df15['atr'] = ta.atr(df15['high'], df15['low'], df15['close'], length=14)

signals = []
last_sig = -999
for i in range(14, len(df)):
    if i - last_sig < 5:
        continue
    row = df.iloc[i]
    mins = row['mins']
    if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
        continue
    atr = df['atr'].iloc[i]
    if pd.isna(atr) or atr < 2.5 or atr > 4.5:
        continue
    rsi = df['rsi'].iloc[i]
    if pd.isna(rsi) or rsi > 70:
        continue
    if i < 3:
        continue
    ranges = [df['range'].iloc[i - k] for k in range(1, 4)]
    if max(ranges) >= 0.7 * atr:
        continue
    # Check 15m compression
    t = row['time'] if isinstance(row['time'], pd.Timestamp) else pd.Timestamp(row['time'])
    # Find matching 15m bars
    mask15 = df15['time'] <= t
    if mask15.sum() < 3:
        continue
    recent15 = df15[mask15].tail(3)
    atr15 = recent15['atr'].iloc[-1]
    if pd.isna(atr15):
        continue
    max_range15 = recent15['range'].iloc[-2:].max()  # last 2 completed 15m bars
    if max_range15 >= 0.7 * atr15:
        continue
    if i + 1 >= len(df):
        continue
    last_sig = i
    nxt = df.iloc[i + 1]
    direction = 1 if nxt['close'] > row['close'] else -1
    signals.append({'idx': i, 'direction': direction, 'session': row['session']})

simulate(pd.DataFrame(signals), "CB + 15m compressed")


# === COMBO 8: Donchian Channel Breakout ===
print(f"\n{'COMBO 8: Donchian Channel Breakout (12-bar)'}")
df['dc_high'] = df['high'].rolling(12).max()
df['dc_low'] = df['low'].rolling(12).min()
signals = []
last_sig = -999
for i in range(14, len(df)):
    if i - last_sig < 5:
        continue
    row = df.iloc[i]
    mins = row['mins']
    if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
        continue
    atr = df['atr'].iloc[i]
    if pd.isna(atr) or atr < 2.5 or atr > 4.5:
        continue
    dc_h = df['dc_high'].iloc[i - 1]  # previous bar's DC high
    dc_l = df['dc_low'].iloc[i - 1]
    if pd.isna(dc_h):
        continue
    # Breakout: close exceeds prior DC high/low
    if row['close'] > dc_h:
        direction = 1
    elif row['close'] < dc_l:
        direction = -1
    else:
        continue
    if i + 1 >= len(df):
        continue
    last_sig = i
    signals.append({'idx': i, 'direction': direction, 'session': row['session']})

simulate(pd.DataFrame(signals), "Donchian 12-bar breakout")


# === COMBO 9: Gap Fade ===
print(f"\n{'COMBO 9: Gap Fade (fade opening gap > 0.5*ATR)'}")
signals = []
for d_idx in range(1, len(days)):
    today = days[d_idx]
    yesterday = days[d_idx - 1]
    # Get prev PM close
    prev_df = df[(df['date'] == yesterday) & (df['session'] == 'PM')]
    if len(prev_df) == 0:
        continue
    prev_close = prev_df.iloc[-1]['close']
    # Get today AM open
    today_am = df[(df['date'] == today) & (df['session'] == 'AM')]
    if len(today_am) < 3:
        continue
    am_open = today_am.iloc[0]['open']
    gap = am_open - prev_close
    atr = today_am['atr'].iloc[2] if not pd.isna(today_am['atr'].iloc[2]) else 3.0
    gap_ratio = abs(gap) / atr
    if gap_ratio < 0.5 or gap_ratio > 2.5:
        continue
    # Fade the gap
    direction = -1 if gap > 0 else 1  # fade
    idx_global = today_am.index[0]
    signals.append({'idx': idx_global, 'direction': direction, 'session': 'AM'})

simulate(pd.DataFrame(signals), "Gap Fade 0.5-2.5*ATR")


# === COMBO 10: Choppiness + CB Filter ===
print(f"\n{'COMBO 10: CB + Choppiness Filter (CI < 61.8 & falling)'}")
signals = []
last_sig = -999
for i in range(14, len(df)):
    if i - last_sig < 5:
        continue
    row = df.iloc[i]
    mins = row['mins']
    if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
        continue
    atr = df['atr'].iloc[i]
    if pd.isna(atr) or atr < 2.5 or atr > 4.5:
        continue
    rsi = df['rsi'].iloc[i]
    if pd.isna(rsi) or rsi > 70:
        continue
    if i < 3:
        continue
    ranges = [df['range'].iloc[i - k] for k in range(1, 4)]
    if max(ranges) >= 0.7 * atr:
        continue
    # Choppiness filter
    ci = df['chop'].iloc[i]
    ci_prev = df['chop'].iloc[i - 1]
    if pd.isna(ci) or pd.isna(ci_prev):
        continue
    if ci >= 61.8:  # too choppy
        continue
    if ci >= ci_prev:  # not falling
        continue
    if i + 1 >= len(df):
        continue
    last_sig = i
    nxt = df.iloc[i + 1]
    direction = 1 if nxt['close'] > row['close'] else -1
    signals.append({'idx': i, 'direction': direction, 'session': row['session']})

simulate(pd.DataFrame(signals), "CB + CI<61.8 & falling")


# === COMBO 11: ADX < 20 + CB ===
print(f"\n{'COMBO 11: CB + ADX Compression (ADX < 20 in last 3 bars)'}")
signals = []
last_sig = -999
for i in range(14, len(df)):
    if i - last_sig < 5:
        continue
    row = df.iloc[i]
    mins = row['mins']
    if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
        continue
    atr = df['atr'].iloc[i]
    if pd.isna(atr) or atr < 2.5 or atr > 4.5:
        continue
    rsi = df['rsi'].iloc[i]
    if pd.isna(rsi) or rsi > 70:
        continue
    if i < 3:
        continue
    ranges = [df['range'].iloc[i - k] for k in range(1, 4)]
    if max(ranges) >= 0.7 * atr:
        continue
    # ADX filter: ADX < 20 in last 3 bars
    adx_vals = [df['adx'].iloc[i - k] for k in range(3)]
    if any(pd.isna(v) for v in adx_vals):
        continue
    if min(adx_vals) >= 20:  # none below 20
        continue
    if i + 1 >= len(df):
        continue
    last_sig = i
    nxt = df.iloc[i + 1]
    direction = 1 if nxt['close'] > row['close'] else -1
    signals.append({'idx': i, 'direction': direction, 'session': row['session']})

simulate(pd.DataFrame(signals), "CB + ADX<20 (3 bars)")


# === COMBO 12: CB + NR7 combined (tightest) ===
print(f"\n{'COMBO 12: CB + NR7 (compression AND narrowest of 7)'}")
signals = []
last_sig = -999
for i in range(14, len(df)):
    if i - last_sig < 5:
        continue
    row = df.iloc[i]
    mins = row['mins']
    if not ((555 <= mins <= 645) or (795 <= mins <= 855)):
        continue
    atr = df['atr'].iloc[i]
    if pd.isna(atr) or atr < 2.5 or atr > 4.5:
        continue
    if i < 7:
        continue
    # CB condition
    ranges_3 = [df['range'].iloc[i - k] for k in range(1, 4)]
    if max(ranges_3) >= 0.7 * atr:
        continue
    # NR7 condition
    cur_range = df['range'].iloc[i]
    prev_ranges_7 = [df['range'].iloc[i - k] for k in range(1, 7)]
    if cur_range >= min(prev_ranges_7):
        continue
    if i + 1 >= len(df):
        continue
    last_sig = i
    nxt = df.iloc[i + 1]
    direction = 1 if nxt['close'] > row['close'] else -1
    signals.append({'idx': i, 'direction': direction, 'session': row['session']})

simulate(pd.DataFrame(signals), "CB + NR7 combo")


print(f"\n{'=' * 90}")
print("BASELINE REFERENCE: CB 5m = 225 trades | WR 65% | PF 4.5 | +6.0/d")
print("Strategies with higher P/D than baseline are worth further investigation.")
print(f"{'=' * 90}")
