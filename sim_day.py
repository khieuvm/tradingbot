"""Simulate June 3 using compression breakout strategy (research_strategy)"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from research_strategy import *
import pandas as pd
import numpy as np
import pandas_ta as ta

# Load data covering June 3
df = load_data(days=30)
df = add_indicators(df)
df = detect_compression(df, n_bars=3, threshold=0.7)
df['rsi14'] = ta.rsi(df['close'], length=14)

# Target date
target_date = pd.Timestamp("2026-06-03").date()
df['date_val'] = df['time'].dt.date

unique_dates = sorted(df['date_val'].unique())
print(f"Available dates: {unique_dates[-5:]}")

if target_date not in [d for d in unique_dates]:
    print(f"June 3 not found, using: {unique_dates[-1]}")
    target_date = unique_dates[-1]

day_df = df[df['date_val'] == target_date].copy()
print(f"\nDate: {target_date} | Bars: {len(day_df)}")
print(f"Price: {day_df['low'].min():.1f} - {day_df['high'].max():.1f}")
print(f"ATR: {day_df['atr'].min():.2f} - {day_df['atr'].max():.2f}")

# Entry filter
filt = (df['atr'] <= 4.5) & (df['rsi14'] < 70)
filt = filt.fillna(False)

# Compression signals
maskAM = (df['compressed_3bar'] & (df['mins_in_day'] >= 9*60+15) & (df['mins_in_day'] <= 10*60+45))
maskAM = dedup_signals(maskAM, min_bars=5)
maskPM = (df['compressed_3bar'] & (df['mins_in_day'] >= 13*60+15) & (df['mins_in_day'] <= 14*60+15))
maskPM = dedup_signals(maskPM, min_bars=5)

maskAM_f = maskAM & filt
maskPM_f = maskPM & filt

# Find signals on target day
day_indices = day_df.index.tolist()
am_signals = [i for i in day_indices if maskAM_f.get(i, False)]
pm_signals = [i for i in day_indices if maskPM_f.get(i, False)]

print(f"\nCompression signals on {target_date}:")
print(f"  AM signals: {len(am_signals)}")
print(f"  PM signals: {len(pm_signals)}")

# Show all compression bars (even without filter)
all_compress = [i for i in day_indices if df.loc[i, 'compressed_3bar']]
print(f"  All compression bars (before filter/dedup): {len(all_compress)}")

# Print signal details
print("\n" + "="*85)
print(f"COMPRESSION BREAKOUT SIMULATION: {target_date}")
print("="*85)
print(f"\nStrategy: 3-bar compression < 0.7xATR")
print(f"Filter: ATR<=4.5, RSI(14)<70")
print(f"AM: SL=1.2xATR, trail 2.0xATR after +5pts | PM: SL=1.0xATR, trail 1.5xATR after +4pts")
print(f"Adaptive exit: BE at +4pts (ATR>=3.5), trail tighten at +8pts (1.2xATR)")

# Simulate each signal
COMMISSION = 0.87  # per side (total round trip = 1.74)

def sim_one_trade(df, signal_idx, sl_mult, trail_activate, trail_atr_mult, max_bars_hold,
                  be_trigger=4.0, be_atr_min=3.5, trail_tighten_at=8.0, trail_tighten_mult=1.2):
    """Simulate a single trade from compression signal."""
    row = df.loc[signal_idx]
    atr = row['atr']
    entry_price = row['close']

    # Direction from next bar
    if signal_idx + 1 >= len(df) or signal_idx + 1 not in df.index:
        return None
    next_bar = df.loc[signal_idx + 1]
    if next_bar['date'] != row['date'] or next_bar['session'] != row['session']:
        return None
    direction = 1 if next_bar['close'] > entry_price else -1

    sl_distance = sl_mult * atr
    sl_price = entry_price - direction * sl_distance
    best_price = entry_price
    trail_active = False
    be_applied = False
    use_be = (be_trigger > 0 and atr >= be_atr_min)
    use_tighten = (trail_tighten_at > 0 and atr >= be_atr_min)

    bars = []
    exit_price = None
    exit_reason = None

    for j in range(signal_idx + 1, min(signal_idx + 1 + max_bars_hold, len(df))):
        if j not in df.index:
            break
        bar = df.loc[j]
        if bar['date'] != row['date'] or bar['session'] != row['session']:
            exit_price = df.loc[j-1, 'close'] if j-1 in df.index else entry_price
            exit_reason = 'SESSION_CLOSE'
            break
        if (bar['session'] == 'PM' and bar['mins_in_day'] >= 14*60+25) or \
           (bar['session'] == 'AM' and bar['mins_in_day'] >= 11*60+25):
            exit_price = bar['close']
            exit_reason = 'SESSION_CLOSE'
            break

        bar_time = bar['time'].strftime("%H:%M")

        if direction == 1:
            if bar['low'] <= sl_price:
                exit_price = sl_price
                exit_reason = 'BE' if be_applied else 'SL'
                bars.append((bar_time, bar['close'], sl_price, 'EXIT ' + exit_reason))
                break
            if bar['high'] > best_price:
                best_price = bar['high']
            mfe = best_price - entry_price
            if use_be and not be_applied and mfe >= be_trigger:
                be_applied = True
                sl_price = max(sl_price, entry_price)
            if mfe >= trail_activate and not trail_active:
                trail_active = True
            cur_trail = trail_atr_mult
            if use_tighten and mfe >= trail_tighten_at:
                cur_trail = trail_tighten_mult
            if trail_active:
                new_sl = best_price - cur_trail * atr
                sl_price = max(sl_price, new_sl)
            if trail_active and bar['low'] <= sl_price:
                exit_price = sl_price
                exit_reason = 'TRAIL'
                bars.append((bar_time, bar['close'], sl_price, 'EXIT TRAIL'))
                break
        else:
            if bar['high'] >= sl_price:
                exit_price = sl_price
                exit_reason = 'BE' if be_applied else 'SL'
                bars.append((bar_time, bar['close'], sl_price, 'EXIT ' + exit_reason))
                break
            if bar['low'] < best_price:
                best_price = bar['low']
            mfe = entry_price - best_price
            if use_be and not be_applied and mfe >= be_trigger:
                be_applied = True
                sl_price = min(sl_price, entry_price)
            if mfe >= trail_activate and not trail_active:
                trail_active = True
            cur_trail = trail_atr_mult
            if use_tighten and mfe >= trail_tighten_at:
                cur_trail = trail_tighten_mult
            if trail_active:
                new_sl = best_price + cur_trail * atr
                sl_price = min(sl_price, new_sl)
            if trail_active and bar['high'] >= sl_price:
                exit_price = sl_price
                exit_reason = 'TRAIL'
                bars.append((bar_time, bar['close'], sl_price, 'EXIT TRAIL'))
                break

        bars.append((bar_time, bar['close'], sl_price, ''))

    if exit_price is None:
        exit_price = entry_price
        exit_reason = 'MAX_BARS'

    pnl = direction * (exit_price - entry_price) - 2 * COMMISSION
    mfe_final = (best_price - entry_price) if direction == 1 else (entry_price - best_price)

    return {
        'signal_time': row['time'].strftime("%H:%M"),
        'entry_price': entry_price,
        'direction': 'BUY' if direction == 1 else 'SELL',
        'atr': atr,
        'rsi': row['rsi14'],
        'sl_init': entry_price - direction * sl_distance,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl': pnl,
        'mfe': mfe_final,
        'bars_held': len(bars),
        'exit_time': bars[-1][0] if bars else row['time'].strftime("%H:%M"),
        'trail_active': trail_active,
        'be_applied': be_applied,
    }


print(f"\n{'#':<3} {'Signal':<7} {'Dir':<5} {'Entry':>7} {'ATR':>5} {'RSI':>5} "
      f"{'SL':>7} {'Exit':>7} {'ExitT':<7} {'Bars':>4} {'MFE':>5} {'PnL':>6} {'Reason':<12}")
print("-"*90)

all_trades = []
trade_num = 0

# AM trades
for i in am_signals:
    result = sim_one_trade(df, i, sl_mult=1.2, trail_activate=5, trail_atr_mult=2.0,
                           max_bars_hold=24)
    if result:
        trade_num += 1
        all_trades.append(result)
        print(f"{trade_num:<3} {result['signal_time']:<7} {result['direction']:<5} "
              f"{result['entry_price']:>7.1f} {result['atr']:>5.2f} {result['rsi']:>5.1f} "
              f"{result['sl_init']:>7.1f} {result['exit_price']:>7.1f} {result['exit_time']:<7} "
              f"{result['bars_held']:>4} {result['mfe']:>5.1f} {result['pnl']:>+6.1f} "
              f"{result['exit_reason']:<12}")

# PM trades
for i in pm_signals:
    result = sim_one_trade(df, i, sl_mult=1.0, trail_activate=4, trail_atr_mult=1.5,
                           max_bars_hold=12)
    if result:
        trade_num += 1
        all_trades.append(result)
        print(f"{trade_num:<3} {result['signal_time']:<7} {result['direction']:<5} "
              f"{result['entry_price']:>7.1f} {result['atr']:>5.2f} {result['rsi']:>5.1f} "
              f"{result['sl_init']:>7.1f} {result['exit_price']:>7.1f} {result['exit_time']:<7} "
              f"{result['bars_held']:>4} {result['mfe']:>5.1f} {result['pnl']:>+6.1f} "
              f"{result['exit_reason']:<12}")

# Summary
if all_trades:
    total_pnl = sum(t['pnl'] for t in all_trades)
    wins = [t for t in all_trades if t['pnl'] > 0]
    print(f"\n{'='*90}")
    print(f"TOTAL: {len(all_trades)} trades | Winners: {len(wins)} | WR: {len(wins)/len(all_trades)*100:.0f}%")
    print(f"Total PnL: {total_pnl:+.1f} pts | VND: {total_pnl*100000:+,.0f}")
    print(f"Avg MFE: {np.mean([t['mfe'] for t in all_trades]):.1f} pts")
else:
    print("\nNo trades on this day.")
    print("\nDebug - checking compression bars:")
    for i in day_indices:
        row = df.loc[i]
        t = row['time'].strftime("%H:%M")
        comp = row['compressed_3bar']
        atr_v = row['atr']
        rsi_v = row.get('rsi14', 0)
        sess = row['session']
        if comp:
            in_am = (row['mins_in_day'] >= 9*60+15) & (row['mins_in_day'] <= 10*60+45)
            in_pm = (row['mins_in_day'] >= 13*60+15) & (row['mins_in_day'] <= 14*60+15)
            filt_ok = (atr_v <= 4.5) and (not pd.isna(rsi_v)) and (rsi_v < 70)
            window = 'AM' if in_am else ('PM' if in_pm else 'OUT')
            print(f"  {t} | ATR={atr_v:.2f} RSI={rsi_v:.1f} | window={window} | filt={'OK' if filt_ok else 'FAIL'}")

# Also show price action for context
print(f"\n--- PRICE ACTION {target_date} ---")
print(f"{'Time':<6} {'Open':>7} {'High':>7} {'Low':>7} {'Close':>7} {'ATR':>5} {'Comp':<5}")
for i in day_indices:
    row = df.loc[i]
    t = row['time'].strftime("%H:%M")
    comp = '*' if row['compressed_3bar'] else ''
    print(f"{t:<6} {row['open']:>7.1f} {row['high']:>7.1f} {row['low']:>7.1f} "
          f"{row['close']:>7.1f} {row['atr']:>5.2f} {comp}")
