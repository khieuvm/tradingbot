"""AM SELL-only strategies with regime filter test."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import pandas_ta as ta

COST = 0.96

df = pd.read_parquet('data/vn30f1m_1m.parquet')
df['date'] = df['time'].dt.date
df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute
df['session'] = df['mins'].apply(lambda m: 'AM' if m < 720 else 'PM')
df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
df['ema8'] = ta.ema(df['close'], length=8)
df['ema21'] = ta.ema(df['close'], length=21)
df['ema50'] = ta.ema(df['close'], length=50)
adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
df['adx'] = adx_df['ADX_14']
df['di_plus'] = adx_df['DMP_14']
df['di_minus'] = adx_df['DMN_14']
df['rsi'] = ta.rsi(df['close'], length=14)

print('AM SELL-ONLY STRATEGIES with regime filter (682d)')
print('=' * 60)
print()


def report(trades, label):
    pnls = [t['pnl'] for t in trades]
    if not pnls:
        print(f'{label}: No trades')
        return
    wins = [p for p in pnls if p > 0]
    gl = abs(sum([p for p in pnls if p <= 0])) or 0.001
    print(f'{label}:')
    print(f'  N={len(pnls)}, WR={len(wins)/len(pnls)*100:.1f}%, PF={sum(wins)/gl:.2f}, '
          f'Total={sum(pnls):.1f}pts, /day={sum(pnls)/682:.2f}')
    # Win/loss avg
    if wins:
        print(f'  Avg win={np.mean(wins):.2f}, Avg loss={np.mean([p for p in pnls if p<=0]):.2f}')
    print()


# 1. Opening Momentum SELL + regime
trades1 = []
for date in df['date'].unique():
    am = df[(df['date'] == date) & (df['mins'] >= 555) & (df['mins'] <= 685)].reset_index(drop=True)
    if len(am) < 30:
        continue
    first_15 = am[am['mins'] <= 570]
    if len(first_15) < 10:
        continue
    ret = first_15['close'].iloc[-1] - first_15['open'].iloc[0]
    if ret >= -1.0:
        continue
    entry_bars = am[am['mins'] > 570]
    if len(entry_bars) < 5:
        continue
    idx0 = entry_bars.index[0]
    entry = am.loc[idx0, 'close']
    atr = am.loc[idx0, 'atr']
    ema50 = am.loc[idx0, 'ema50']
    if pd.isna(atr) or atr <= 0 or pd.isna(ema50):
        continue
    pve = (entry - ema50) / atr
    if pve >= -0.5:
        continue
    sl = entry + 3.0 * atr
    exit_price = entry
    for j in range(1, len(entry_bars)):
        bar = entry_bars.iloc[j]
        if bar['mins'] >= 685:
            exit_price = bar['close']
            break
        if bar['high'] >= sl:
            exit_price = sl
            break
    else:
        exit_price = entry_bars.iloc[-1]['close']
    pnl = (entry - exit_price) - COST
    trades1.append({'date': str(date), 'pnl': pnl})

report(trades1, '1. Opening Momentum AM SELL (first_15<-1pt) + regime')


# 2. EMA Trend SELL + regime + ADX>25
trades2 = []
for date in df['date'].unique():
    am = df[(df['date'] == date) & (df['mins'] >= 570) & (df['mins'] <= 685)].reset_index(drop=True)
    if len(am) < 20:
        continue
    bar0 = am.iloc[0]
    atr = bar0['atr']
    adx = bar0['adx']
    ema8 = bar0['ema8']
    ema21 = bar0['ema21']
    ema50 = bar0['ema50']
    if pd.isna(atr) or atr <= 0 or pd.isna(adx) or pd.isna(ema8) or pd.isna(ema21):
        continue
    if adx < 25:
        continue
    if ema8 >= ema21:
        continue
    if pd.isna(ema50):
        continue
    pve = (bar0['close'] - ema50) / atr
    if pve >= -0.5:
        continue
    entry = bar0['close']
    sl = entry + 3.0 * atr
    exit_price = entry
    for j in range(1, len(am)):
        bar = am.iloc[j]
        if bar['mins'] >= 685:
            exit_price = bar['close']
            break
        if bar['high'] >= sl:
            exit_price = sl
            break
    else:
        exit_price = am.iloc[-1]['close']
    pnl = (entry - exit_price) - COST
    trades2.append({'date': str(date), 'pnl': pnl})

report(trades2, '2. EMA Trend AM SELL (EMA8<EMA21, ADX>25) + regime')


# 3. ADX Trend Continuation (DI- > DI+, ADX > 25, regime)
trades3 = []
for date in df['date'].unique():
    am = df[(df['date'] == date) & (df['mins'] >= 560) & (df['mins'] <= 685)].reset_index(drop=True)
    if len(am) < 20:
        continue
    entered = False
    for j in range(5, min(45, len(am))):
        if entered:
            break
        bar = am.iloc[j]
        atr = bar['atr']
        adx = bar['adx']
        di_p = bar['di_plus']
        di_m = bar['di_minus']
        ema50 = bar['ema50']
        if pd.isna(atr) or atr <= 0 or pd.isna(adx):
            continue
        if adx < 25 or pd.isna(di_p) or pd.isna(di_m):
            continue
        if di_m <= di_p:
            continue
        if pd.isna(ema50):
            continue
        pve = (bar['close'] - ema50) / atr
        if pve >= -0.5:
            continue
        entered = True
        entry = bar['close']
        sl = entry + 3.0 * atr
        exit_price = entry
        for k in range(j + 1, len(am)):
            kbar = am.iloc[k]
            if kbar['mins'] >= 685:
                exit_price = kbar['close']
                break
            if kbar['high'] >= sl:
                exit_price = sl
                break
        else:
            exit_price = am.iloc[-1]['close']
        pnl = (entry - exit_price) - COST
        trades3.append({'date': str(date), 'pnl': pnl})

report(trades3, '3. ADX Trend Continuation AM SELL (DI->DI+, ADX>25, regime)')


# 4. Opening Range Breakdown SELL + regime
trades4 = []
for date in df['date'].unique():
    am = df[(df['date'] == date) & (df['mins'] >= 555) & (df['mins'] <= 685)].reset_index(drop=True)
    if len(am) < 30:
        continue
    opening = am[am['mins'] <= 570]
    if len(opening) < 5:
        continue
    range_low = opening['low'].min()
    atr = opening['atr'].iloc[-1]
    ema50 = opening['ema50'].iloc[-1]
    if pd.isna(atr) or atr <= 0 or pd.isna(ema50):
        continue
    pve = (opening['close'].iloc[-1] - ema50) / atr
    if pve >= -0.5:
        continue
    post = am[am['mins'] > 570]
    if len(post) < 5:
        continue
    traded = False
    for j in range(len(post)):
        if traded:
            break
        bar = post.iloc[j]
        if bar['low'] < range_low - 0.1:
            traded = True
            entry = range_low - 0.1
            sl = entry + 3.0 * atr
            exit_price = entry
            for k in range(j + 1, len(post)):
                kbar = post.iloc[k]
                if kbar['mins'] >= 685:
                    exit_price = kbar['close']
                    break
                if kbar['high'] >= sl:
                    exit_price = sl
                    break
            else:
                exit_price = post.iloc[-1]['close']
            pnl = (entry - exit_price) - COST
            trades4.append({'date': str(date), 'pnl': pnl})

report(trades4, '4. Opening Range Breakdown SELL + regime')


# 5. Same approaches but AM BUY + bullish regime
print()
print('AM BUY STRATEGIES with regime filter (price > EMA50 + 0.5*ATR)')
print('=' * 60)
print()

# Opening Momentum BUY + regime
trades5 = []
for date in df['date'].unique():
    am = df[(df['date'] == date) & (df['mins'] >= 555) & (df['mins'] <= 685)].reset_index(drop=True)
    if len(am) < 30:
        continue
    first_15 = am[am['mins'] <= 570]
    if len(first_15) < 10:
        continue
    ret = first_15['close'].iloc[-1] - first_15['open'].iloc[0]
    if ret <= 1.0:
        continue
    entry_bars = am[am['mins'] > 570]
    if len(entry_bars) < 5:
        continue
    idx0 = entry_bars.index[0]
    entry = am.loc[idx0, 'close']
    atr = am.loc[idx0, 'atr']
    ema50 = am.loc[idx0, 'ema50']
    if pd.isna(atr) or atr <= 0 or pd.isna(ema50):
        continue
    pve = (entry - ema50) / atr
    if pve <= 0.5:
        continue
    sl = entry - 3.0 * atr
    exit_price = entry
    for j in range(1, len(entry_bars)):
        bar = entry_bars.iloc[j]
        if bar['mins'] >= 685:
            exit_price = bar['close']
            break
        if bar['low'] <= sl:
            exit_price = sl
            break
    else:
        exit_price = entry_bars.iloc[-1]['close']
    pnl = (exit_price - entry) - COST
    trades5.append({'date': str(date), 'pnl': pnl})

report(trades5, '5. Opening Momentum AM BUY (first_15>+1pt) + regime')


# 6. EMA Trend BUY + regime
trades6 = []
for date in df['date'].unique():
    am = df[(df['date'] == date) & (df['mins'] >= 570) & (df['mins'] <= 685)].reset_index(drop=True)
    if len(am) < 20:
        continue
    bar0 = am.iloc[0]
    atr = bar0['atr']
    adx = bar0['adx']
    ema8 = bar0['ema8']
    ema21 = bar0['ema21']
    ema50 = bar0['ema50']
    if pd.isna(atr) or atr <= 0 or pd.isna(adx) or pd.isna(ema8) or pd.isna(ema21):
        continue
    if adx < 25:
        continue
    if ema8 <= ema21:
        continue
    if pd.isna(ema50):
        continue
    pve = (bar0['close'] - ema50) / atr
    if pve <= 0.5:
        continue
    entry = bar0['close']
    sl = entry - 3.0 * atr
    exit_price = entry
    for j in range(1, len(am)):
        bar = am.iloc[j]
        if bar['mins'] >= 685:
            exit_price = bar['close']
            break
        if bar['low'] <= sl:
            exit_price = sl
            break
    else:
        exit_price = am.iloc[-1]['close']
    pnl = (exit_price - entry) - COST
    trades6.append({'date': str(date), 'pnl': pnl})

report(trades6, '6. EMA Trend AM BUY (EMA8>EMA21, ADX>25) + regime')


# 7. ADX Trend Continuation BUY (DI+ > DI-, regime)
trades7 = []
for date in df['date'].unique():
    am = df[(df['date'] == date) & (df['mins'] >= 560) & (df['mins'] <= 685)].reset_index(drop=True)
    if len(am) < 20:
        continue
    entered = False
    for j in range(5, min(45, len(am))):
        if entered:
            break
        bar = am.iloc[j]
        atr = bar['atr']
        adx = bar['adx']
        di_p = bar['di_plus']
        di_m = bar['di_minus']
        ema50 = bar['ema50']
        if pd.isna(atr) or atr <= 0 or pd.isna(adx):
            continue
        if adx < 25 or pd.isna(di_p) or pd.isna(di_m):
            continue
        if di_p <= di_m:
            continue
        if pd.isna(ema50):
            continue
        pve = (bar['close'] - ema50) / atr
        if pve <= 0.5:
            continue
        entered = True
        entry = bar['close']
        sl = entry - 3.0 * atr
        exit_price = entry
        for k in range(j + 1, len(am)):
            kbar = am.iloc[k]
            if kbar['mins'] >= 685:
                exit_price = kbar['close']
                break
            if kbar['low'] <= sl:
                exit_price = sl
                break
        else:
            exit_price = am.iloc[-1]['close']
        pnl = (exit_price - entry) - COST
        trades7.append({'date': str(date), 'pnl': pnl})

report(trades7, '7. ADX Trend Continuation AM BUY (DI+>DI-, ADX>25, regime)')
