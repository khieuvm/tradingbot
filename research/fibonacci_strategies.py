"""
FIBONACCI STRATEGIES RESEARCH — VN30F1M Intraday
═══════════════════════════════════════════════════
Nghiên cứu các phương pháp giao dịch dựa trên Fibonacci:

1. FIB RETRACEMENT — Đợi pullback về Fib levels (38.2%, 50%, 61.8%) sau swing → entry
2. FIB EXTENSION — Dùng Fib mở rộng (127.2%, 161.8%) làm TP targets  
3. FIB + TREND — Fib retracement kết hợp EMA trend filter
4. INTRADAY FIB ZONES — Dùng session high/low trước đó tính Fib levels cho session sau
5. FIB + RSI CONFLUENCE — Fib level + RSI oversold/overbought = strong entry

Cost: 0.96 pts/trade (slippage 0.5 + commission 0.46)
Data: VN30F1M 5-minute bars
"""
import sys
sys.path.insert(0, r"E:\Trading")

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from itertools import product
from src.data_fetcher import DataFetcher

# ── Fetch data ────────────────────────────────────────────────────────────────
fetcher = DataFetcher(provider="VCI")
end_date = datetime(2026, 7, 25)
start_date = end_date - timedelta(days=365)

print("=" * 90)
print("FIBONACCI STRATEGIES RESEARCH — VN30F1M Intraday")
print("=" * 90)
print(f"\nFetching 5m data ({start_date.date()} to {end_date.date()})...", flush=True)

df = fetcher.get_futures_ohlcv("VN30F1M", start_date.strftime("%Y-%m-%d"),
                                end_date.strftime("%Y-%m-%d"), "5m")
print(f"  5m bars: {len(df)}", flush=True)

df = df.copy()
df["time"] = pd.to_datetime(df["time"])
df["date"] = df["time"].dt.date
df["mins"] = df["time"].dt.hour * 60 + df["time"].dt.minute
df["session"] = np.where(df["mins"] < 11*60+30, "AM", "PM")
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = df[col].astype(float)

# Indicators
df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
df["rsi"] = ta.rsi(df["close"], length=14)
df["ema8"] = ta.ema(df["close"], length=8)
df["ema21"] = ta.ema(df["close"], length=21)
df["ema50"] = ta.ema(df["close"], length=50)

dates = sorted(df["date"].unique())
print(f"Trading days: {len(dates)}\n", flush=True)

COST = 0.96
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXT = [1.0, 1.272, 1.618, 2.0, 2.618]

# ── Pre-extract sessions ──────────────────────────────────────────────────────
print("Pre-extracting sessions...", flush=True)
sessions = []
for d in dates:
    day_df = df[df["date"] == d].sort_values("time").reset_index(drop=True)
    for sess_name in ["AM", "PM"]:
        sess_df = day_df[day_df["session"] == sess_name].reset_index(drop=True)
        if len(sess_df) < 15:
            continue
        sessions.append({
            "date": d,
            "sess": sess_name,
            "n": len(sess_df),
            "open": sess_df["open"].values,
            "high": sess_df["high"].values,
            "low": sess_df["low"].values,
            "close": sess_df["close"].values,
            "volume": sess_df["volume"].values,
            "atr": sess_df["atr"].values,
            "rsi": sess_df["rsi"].values,
            "ema8": sess_df["ema8"].values,
            "ema21": sess_df["ema21"].values,
            "ema50": sess_df["ema50"].values,
            "mins": sess_df["mins"].values,
        })
print(f"Sessions: {len(sessions)}\n", flush=True)


def score_trades(pnls):
    """Score a set of trades"""
    n = len(pnls)
    if n < 5:
        return {"score": -999, "n": n, "wr": 0, "net": 0, "pf": 0, "total": 0}
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100
    net = np.mean(pnls)
    total = sum(pnls)
    gw = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gw / gl if gl > 0 else 99
    score = net * np.sqrt(n) * min(pf, 5) / 5
    return {"score": score, "n": n, "wr": wr, "net": net, "pf": pf, "total": total}


def find_swings(highs, lows, closes, lookback=5):
    """Find swing highs and swing lows using lookback period.
    Returns list of (index, price, type) where type is 'H' or 'L'
    """
    n = len(highs)
    swings = []
    for i in range(lookback, n - lookback):
        # Swing high
        if highs[i] == max(highs[i-lookback:i+lookback+1]):
            swings.append((i, highs[i], 'H'))
        # Swing low
        if lows[i] == min(lows[i-lookback:i+lookback+1]):
            swings.append((i, lows[i], 'L'))
    return sorted(swings, key=lambda x: x[0])


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: FIB RETRACEMENT PULLBACK
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("STRATEGY 1: FIB RETRACEMENT PULLBACK")
print("  Sau swing move, đợi price retrace về Fib level → entry theo hướng swing")
print("=" * 90, flush=True)

# Parameters to test
SWING_LOOKBACK = [3, 5, 7]          # Bars to confirm swing
MIN_SWING_PTS = [2.0, 3.0, 4.0]    # Min swing size in points
FIB_ENTRY = [0.382, 0.5, 0.618]    # Which fib level to enter
FIB_TOLERANCE = 0.3                  # Points tolerance around fib level
HOLD_BARS = [6, 8, 10, 12]          # Max hold
SL_MODES = ["swing", "next_fib"]    # SL at swing extreme or next fib level
SESS_FILTER = ["AM", "PM", "BOTH"]

total = len(SWING_LOOKBACK) * len(MIN_SWING_PTS) * len(FIB_ENTRY) * len(HOLD_BARS) * len(SL_MODES) * len(SESS_FILTER)
print(f"Combos: {total}\n", flush=True)

fib_retrace_results = []
cnt = 0

for swing_lb, min_swing, fib_entry, hold, sl_mode, sess_f in product(
    SWING_LOOKBACK, MIN_SWING_PTS, FIB_ENTRY, HOLD_BARS, SL_MODES, SESS_FILTER):
    cnt += 1
    pnls = []
    
    for s in sessions:
        if sess_f != "BOTH" and s["sess"] != sess_f:
            continue
        n = s["n"]
        if n < swing_lb * 2 + hold + 5:
            continue
        
        # Find swings
        swings = find_swings(s["high"], s["low"], s["close"], swing_lb)
        if len(swings) < 2:
            continue
        
        last_trade_bar = -hold  # Avoid overlapping trades
        
        for si in range(1, len(swings)):
            sw_idx, sw_price, sw_type = swings[si]
            
            # Need previous swing of opposite type for fib calculation
            prev_sw = None
            for pi in range(si-1, -1, -1):
                if swings[pi][2] != sw_type:
                    prev_sw = swings[pi]
                    break
            if prev_sw is None:
                continue
            
            prev_idx, prev_price, prev_type = prev_sw
            swing_size = abs(sw_price - prev_price)
            if swing_size < min_swing:
                continue
            
            # Calculate fib retracement level
            if sw_type == 'H':
                # Swing high → expect retrace down → BUY at fib level
                fib_price = sw_price - swing_size * fib_entry
                direction = 1  # BUY (price retraces down then continues up)
                sl_price = prev_price - 0.5 if sl_mode == "swing" else sw_price - swing_size * 0.786
            else:
                # Swing low → expect retrace up → SELL at fib level
                fib_price = sw_price + swing_size * fib_entry
                direction = -1  # SELL (price retraces up then continues down)
                sl_price = prev_price + 0.5 if sl_mode == "swing" else sw_price + swing_size * 0.786
            
            # Look for price reaching fib level after the swing
            entry_bar = None
            for b in range(sw_idx + 1, min(sw_idx + hold, n)):
                if b <= last_trade_bar:
                    continue
                if direction == 1:  # BUY: price must come down to fib_price
                    if s["low"][b] <= fib_price + FIB_TOLERANCE:
                        entry_bar = b
                        entry_price = fib_price
                        break
                else:  # SELL: price must come up to fib_price
                    if s["high"][b] >= fib_price - FIB_TOLERANCE:
                        entry_bar = b
                        entry_price = fib_price
                        break
            
            if entry_bar is None:
                continue
            if entry_bar + hold >= n:
                continue
            
            last_trade_bar = entry_bar + hold
            
            # Simulate trade
            best_pnl = 0
            exit_pnl = 0
            hit_sl = False
            
            for b in range(entry_bar + 1, min(entry_bar + hold + 1, n)):
                if direction == 1:
                    bar_pnl = s["close"][b] - entry_price
                    if s["low"][b] <= sl_price:
                        exit_pnl = sl_price - entry_price
                        hit_sl = True
                        break
                else:
                    bar_pnl = entry_price - s["close"][b]
                    if s["high"][b] >= sl_price:
                        exit_pnl = entry_price - sl_price
                        hit_sl = True
                        break
                best_pnl = max(best_pnl, bar_pnl)
                exit_pnl = bar_pnl
            
            pnls.append(exit_pnl - COST)
    
    stats = score_trades(pnls)
    if stats["n"] >= 5:
        fib_retrace_results.append({
            "swing_lb": swing_lb, "min_swing": min_swing, "fib_entry": fib_entry,
            "hold": hold, "sl_mode": sl_mode, "sess": sess_f, **stats
        })

fib_retrace_results.sort(key=lambda x: x["score"], reverse=True)
print(f"\nTop 10 Fib Retracement configs:")
print(f"{'Swing_LB':<9} {'Min_Sw':<7} {'Fib':<5} {'Hold':<5} {'SL':<9} {'Sess':<5} "
      f"{'N':<4} {'WR%':<6} {'Net':<7} {'PF':<5} {'Total':<7}")
print("-" * 80)
for r in fib_retrace_results[:10]:
    print(f"{r['swing_lb']:<9} {r['min_swing']:<7.1f} {r['fib_entry']:<5.3f} {r['hold']:<5} "
          f"{r['sl_mode']:<9} {r['sess']:<5} {r['n']:<4} {r['wr']:<6.1f} {r['net']:<7.2f} "
          f"{r['pf']:<5.2f} {r['total']:<7.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: FIB EXTENSION TARGET (Session-based)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STRATEGY 2: SESSION FIB ZONES")
print("  Dùng AM session high/low → tính Fib levels cho PM entry")
print("  Hoặc dùng first N bars → Fib retracement for rest of session")
print("=" * 90, flush=True)

# Group sessions by date to get AM → PM relationship
date_sessions = {}
for s in sessions:
    if s["date"] not in date_sessions:
        date_sessions[s["date"]] = {}
    date_sessions[s["date"]][s["sess"]] = s

# Strategy: AM range → PM Fib levels
OR_BARS_2 = [6, 9, 12, 18]       # Opening range bars (first N bars of session)
FIB_BUY = [0.382, 0.5, 0.618]    # Buy at this retracement of OR
FIB_SELL = [0.382, 0.5, 0.618]   # Sell at this retracement of OR
HOLD_2 = [6, 8, 10, 12]
SL_MULT_2 = [0.5, 0.7, 1.0]     # SL = x * OR range below/above fib
TARGET_FIB_EXT = [1.0, 1.272, 1.618]  # TP at fib extension of OR

total2 = len(OR_BARS_2) * len(FIB_BUY) * len(HOLD_2) * len(SL_MULT_2) * len(TARGET_FIB_EXT) * 2
print(f"Combos: {total2}\n", flush=True)

fib_session_results = []

for or_bars, fib_level, hold, sl_mult, tp_ext, sess_f in product(
    OR_BARS_2, FIB_BUY, HOLD_2, SL_MULT_2, TARGET_FIB_EXT, ["AM", "PM"]):
    
    pnls_buy = []
    pnls_sell = []
    
    for s in sessions:
        if s["sess"] != sess_f:
            continue
        n = s["n"]
        if n < or_bars + hold + 2:
            continue
        
        # Opening range
        or_high = s["high"][:or_bars].max()
        or_low = s["low"][:or_bars].min()
        or_range = or_high - or_low
        if or_range < 1.0:  # Min range
            continue
        
        # Fib levels from OR
        fib_buy_price = or_high - or_range * fib_level   # Buy on pullback within OR
        fib_sell_price = or_low + or_range * fib_level   # Sell on bounce within OR
        
        # Extension targets
        tp_buy = or_high + or_range * (tp_ext - 1.0)    # Buy target above OR high
        tp_sell = or_low - or_range * (tp_ext - 1.0)    # Sell target below OR low
        
        # SL
        sl_buy = fib_buy_price - or_range * sl_mult
        sl_sell = fib_sell_price + or_range * sl_mult
        
        last_buy_bar = -hold
        last_sell_bar = -hold
        
        # Scan for entries after OR is formed
        for b in range(or_bars, n - 1):
            # BUY: price pulls back to fib level
            if b > last_buy_bar + hold and s["low"][b] <= fib_buy_price + 0.2:
                entry = fib_buy_price
                pnl = 0
                for eb in range(b + 1, min(b + hold + 1, n)):
                    if s["low"][eb] <= sl_buy:
                        pnl = sl_buy - entry
                        break
                    if s["high"][eb] >= tp_buy:
                        pnl = tp_buy - entry
                        break
                    pnl = s["close"][eb] - entry
                pnls_buy.append(pnl - COST)
                last_buy_bar = b
            
            # SELL: price bounces to fib level
            if b > last_sell_bar + hold and s["high"][b] >= fib_sell_price - 0.2:
                entry = fib_sell_price
                pnl = 0
                for eb in range(b + 1, min(b + hold + 1, n)):
                    if s["high"][eb] >= sl_sell:
                        pnl = entry - sl_sell
                        break
                    if s["low"][eb] <= tp_sell:
                        pnl = entry - tp_sell
                        break
                    pnl = entry - s["close"][eb]
                pnls_sell.append(pnl - COST)
                last_sell_bar = b
    
    for direction, pnls in [("BUY", pnls_buy), ("SELL", pnls_sell)]:
        stats = score_trades(pnls)
        if stats["n"] >= 5:
            fib_session_results.append({
                "or_bars": or_bars, "fib": fib_level, "hold": hold,
                "sl_mult": sl_mult, "tp_ext": tp_ext, "sess": sess_f,
                "dir": direction, **stats
            })

fib_session_results.sort(key=lambda x: x["score"], reverse=True)
print(f"\nTop 10 Session Fib configs:")
print(f"{'OR':<4} {'Fib':<5} {'Hold':<5} {'SL_m':<5} {'TP_ext':<7} {'Sess':<5} {'Dir':<5} "
      f"{'N':<4} {'WR%':<6} {'Net':<7} {'PF':<5} {'Total':<7}")
print("-" * 80)
for r in fib_session_results[:10]:
    print(f"{r['or_bars']:<4} {r['fib']:<5.3f} {r['hold']:<5} {r['sl_mult']:<5.1f} "
          f"{r['tp_ext']:<7.3f} {r['sess']:<5} {r['dir']:<5} {r['n']:<4} {r['wr']:<6.1f} "
          f"{r['net']:<7.2f} {r['pf']:<5.2f} {r['total']:<7.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: FIB + EMA TREND CONFLUENCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STRATEGY 3: FIB + EMA TREND CONFLUENCE")
print("  Chỉ entry Fib retracement khi EMA trend confirms direction")
print("  BUY: EMA8 > EMA21 > EMA50 + price retrace to Fib → BUY")
print("  SELL: EMA8 < EMA21 < EMA50 + price retrace to Fib → SELL")
print("=" * 90, flush=True)

LOOKBACK_3 = [5, 8, 10, 14]       # Bars to find recent swing
FIB_3 = [0.382, 0.5, 0.618]       # Fib entry level
HOLD_3 = [6, 8, 10, 12]
ATR_SL_3 = [1.0, 1.2, 1.5]       # SL in ATR multiples
ATR_TP_3 = [2.0, 3.0, 4.0]       # TP in ATR multiples
SESS_3 = ["AM", "PM", "BOTH"]

total3 = len(LOOKBACK_3) * len(FIB_3) * len(HOLD_3) * len(ATR_SL_3) * len(ATR_TP_3) * len(SESS_3)
print(f"Combos: {total3}\n", flush=True)

fib_ema_results = []

for lb, fib_lv, hold, atr_sl, atr_tp, sess_f in product(
    LOOKBACK_3, FIB_3, HOLD_3, ATR_SL_3, ATR_TP_3, SESS_3):
    
    pnls = []
    
    for s in sessions:
        if sess_f != "BOTH" and s["sess"] != sess_f:
            continue
        n = s["n"]
        if n < lb + hold + 5:
            continue
        
        last_trade_bar = -hold
        
        for b in range(lb + 2, n - hold):
            if b <= last_trade_bar + hold:
                continue
            
            atr_val = s["atr"][b]
            if np.isnan(atr_val) or atr_val < 1.5:
                continue
            
            ema8 = s["ema8"][b]
            ema21 = s["ema21"][b]
            ema50 = s["ema50"][b]
            if np.isnan(ema8) or np.isnan(ema21) or np.isnan(ema50):
                continue
            
            # Determine trend from EMAs
            bullish = ema8 > ema21 > ema50
            bearish = ema8 < ema21 < ema50
            
            if not bullish and not bearish:
                continue
            
            # Find recent swing for fib calculation
            if bullish:
                # Find recent swing low and high
                recent_low = s["low"][b-lb:b].min()
                recent_high = s["high"][b-lb:b].max()
                swing_size = recent_high - recent_low
                if swing_size < 2.0:
                    continue
                
                # Fib retracement level (pullback from high)
                fib_price = recent_high - swing_size * fib_lv
                
                # Check if current price is near fib level
                if abs(s["close"][b] - fib_price) > 0.5:
                    continue
                
                # Also check RSI not overbought
                if s["rsi"][b] > 70:
                    continue
                
                # BUY entry
                entry = s["close"][b]
                sl = entry - atr_val * atr_sl
                tp = entry + atr_val * atr_tp
                direction = 1
                
            else:  # bearish
                recent_low = s["low"][b-lb:b].min()
                recent_high = s["high"][b-lb:b].max()
                swing_size = recent_high - recent_low
                if swing_size < 2.0:
                    continue
                
                # Fib retracement level (bounce from low)
                fib_price = recent_low + swing_size * fib_lv
                
                # Check if current price is near fib level
                if abs(s["close"][b] - fib_price) > 0.5:
                    continue
                
                # Also check RSI not oversold
                if s["rsi"][b] < 30:
                    continue
                
                # SELL entry
                entry = s["close"][b]
                sl = entry + atr_val * atr_sl
                tp = entry - atr_val * atr_tp
                direction = -1
            
            # Simulate
            pnl = 0
            for eb in range(b + 1, min(b + hold + 1, n)):
                if direction == 1:
                    if s["low"][eb] <= sl:
                        pnl = sl - entry
                        break
                    if s["high"][eb] >= tp:
                        pnl = tp - entry
                        break
                    pnl = s["close"][eb] - entry
                else:
                    if s["high"][eb] >= sl:
                        pnl = entry - sl
                        break
                    if s["low"][eb] <= tp:
                        pnl = entry - tp
                        break
                    pnl = entry - s["close"][eb]
            
            pnls.append(pnl - COST)
            last_trade_bar = b
    
    stats = score_trades(pnls)
    if stats["n"] >= 5:
        fib_ema_results.append({
            "lb": lb, "fib": fib_lv, "hold": hold,
            "atr_sl": atr_sl, "atr_tp": atr_tp, "sess": sess_f, **stats
        })

fib_ema_results.sort(key=lambda x: x["score"], reverse=True)
print(f"\nTop 10 Fib+EMA Trend configs:")
print(f"{'LB':<4} {'Fib':<5} {'Hold':<5} {'SL':<5} {'TP':<5} {'Sess':<5} "
      f"{'N':<4} {'WR%':<6} {'Net':<7} {'PF':<5} {'Total':<7}")
print("-" * 80)
for r in fib_ema_results[:10]:
    print(f"{r['lb']:<4} {r['fib']:<5.3f} {r['hold']:<5} {r['atr_sl']:<5.1f} "
          f"{r['atr_tp']:<5.1f} {r['sess']:<5} {r['n']:<4} {r['wr']:<6.1f} "
          f"{r['net']:<7.2f} {r['pf']:<5.2f} {r['total']:<7.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: FIB + RSI OVERSOLD/OVERBOUGHT CONFLUENCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STRATEGY 4: FIB + RSI CONFLUENCE")
print("  Entry khi price ở Fib level + RSI confirms (oversold → BUY, overbought → SELL)")
print("=" * 90, flush=True)

LOOKBACK_4 = [8, 10, 14, 18]     # Swing lookback
FIB_4 = [0.5, 0.618, 0.786]      # Deeper retracements
RSI_OB = [65, 70]                 # RSI overbought for SELL
RSI_OS = [30, 35]                 # RSI oversold for BUY
HOLD_4 = [6, 8, 10]
ATR_SL_4 = [1.0, 1.5]
ATR_TP_4 = [2.0, 3.0, 4.0]
SESS_4 = ["AM", "PM", "BOTH"]

total4 = len(LOOKBACK_4) * len(FIB_4) * len(RSI_OB) * len(RSI_OS) * len(HOLD_4) * len(ATR_SL_4) * len(ATR_TP_4) * len(SESS_4)
print(f"Combos: {total4}\n", flush=True)

fib_rsi_results = []

for lb, fib_lv, rsi_ob, rsi_os, hold, atr_sl, atr_tp, sess_f in product(
    LOOKBACK_4, FIB_4, RSI_OB, RSI_OS, HOLD_4, ATR_SL_4, ATR_TP_4, SESS_4):
    
    pnls = []
    
    for s in sessions:
        if sess_f != "BOTH" and s["sess"] != sess_f:
            continue
        n = s["n"]
        if n < lb + hold + 5:
            continue
        
        last_trade_bar = -hold
        
        for b in range(lb + 2, n - hold):
            if b <= last_trade_bar + hold:
                continue
            
            atr_val = s["atr"][b]
            rsi_val = s["rsi"][b]
            if np.isnan(atr_val) or np.isnan(rsi_val) or atr_val < 1.5:
                continue
            
            # Recent swing range
            recent_high = s["high"][b-lb:b].max()
            recent_low = s["low"][b-lb:b].min()
            swing_size = recent_high - recent_low
            if swing_size < 2.0:
                continue
            
            close = s["close"][b]
            direction = 0
            
            # BUY: price near fib support + RSI oversold
            fib_support = recent_high - swing_size * fib_lv
            if abs(close - fib_support) <= 0.5 and rsi_val <= rsi_os:
                direction = 1
                entry = close
                sl = entry - atr_val * atr_sl
                tp = entry + atr_val * atr_tp
            
            # SELL: price near fib resistance + RSI overbought
            fib_resistance = recent_low + swing_size * fib_lv
            if direction == 0 and abs(close - fib_resistance) <= 0.5 and rsi_val >= rsi_ob:
                direction = -1
                entry = close
                sl = entry + atr_val * atr_sl
                tp = entry - atr_val * atr_tp
            
            if direction == 0:
                continue
            
            # Simulate
            pnl = 0
            for eb in range(b + 1, min(b + hold + 1, n)):
                if direction == 1:
                    if s["low"][eb] <= sl:
                        pnl = sl - entry
                        break
                    if s["high"][eb] >= tp:
                        pnl = tp - entry
                        break
                    pnl = s["close"][eb] - entry
                else:
                    if s["high"][eb] >= sl:
                        pnl = entry - sl
                        break
                    if s["low"][eb] <= tp:
                        pnl = entry - tp
                        break
                    pnl = entry - s["close"][eb]
            
            pnls.append(pnl - COST)
            last_trade_bar = b
    
    stats = score_trades(pnls)
    if stats["n"] >= 5:
        fib_rsi_results.append({
            "lb": lb, "fib": fib_lv, "rsi_ob": rsi_ob, "rsi_os": rsi_os,
            "hold": hold, "atr_sl": atr_sl, "atr_tp": atr_tp, "sess": sess_f, **stats
        })

fib_rsi_results.sort(key=lambda x: x["score"], reverse=True)
print(f"\nTop 10 Fib+RSI configs:")
print(f"{'LB':<4} {'Fib':<5} {'OB':<4} {'OS':<4} {'Hold':<5} {'SL':<5} {'TP':<5} {'Sess':<5} "
      f"{'N':<4} {'WR%':<6} {'Net':<7} {'PF':<5} {'Total':<7}")
print("-" * 80)
for r in fib_rsi_results[:10]:
    print(f"{r['lb']:<4} {r['fib']:<5.3f} {r['rsi_ob']:<4} {r['rsi_os']:<4} "
          f"{r['hold']:<5} {r['atr_sl']:<5.1f} {r['atr_tp']:<5.1f} {r['sess']:<5} "
          f"{r['n']:<4} {r['wr']:<6.1f} {r['net']:<7.2f} {r['pf']:<5.2f} {r['total']:<7.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 5: AM→PM FIB PROJECTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STRATEGY 5: AM→PM FIB PROJECTION")
print("  Dùng AM session range → tính Fib extension/retracement cho PM entries")
print("=" * 90, flush=True)

FIB_PM_ENTRY = [0.382, 0.5, 0.618, 0.786]  # Retrace of AM range for PM entry
HOLD_5 = [6, 8, 10, 12]
SL_5 = [0.5, 0.7, 1.0]       # SL as fraction of AM range
TP_5 = [1.0, 1.272, 1.618]   # TP as fib extension of AM range

total5 = len(FIB_PM_ENTRY) * len(HOLD_5) * len(SL_5) * len(TP_5)
print(f"Combos: {total5}\n", flush=True)

fib_ampm_results = []

for fib_entry, hold, sl_frac, tp_ext in product(FIB_PM_ENTRY, HOLD_5, SL_5, TP_5):
    pnls_buy = []
    pnls_sell = []
    
    for d, d_sessions in date_sessions.items():
        if "AM" not in d_sessions or "PM" not in d_sessions:
            continue
        
        am = d_sessions["AM"]
        pm = d_sessions["PM"]
        
        am_high = am["high"].max()
        am_low = am["low"].min()
        am_range = am_high - am_low
        am_close = am["close"][-1]
        
        if am_range < 2.0:
            continue
        
        n = pm["n"]
        if n < hold + 3:
            continue
        
        # Determine AM direction
        am_bullish = am_close > (am_high + am_low) / 2
        
        if am_bullish:
            # AM was bullish → PM may retrace to fib level → BUY continuation
            fib_price = am_high - am_range * fib_entry
            sl = fib_price - am_range * sl_frac
            tp = am_high + am_range * (tp_ext - 1.0)
            
            # Find entry in PM
            for b in range(0, min(12, n - hold)):
                if pm["low"][b] <= fib_price + 0.3:
                    entry = fib_price
                    pnl = 0
                    for eb in range(b + 1, min(b + hold + 1, n)):
                        if pm["low"][eb] <= sl:
                            pnl = sl - entry
                            break
                        if pm["high"][eb] >= tp:
                            pnl = tp - entry
                            break
                        pnl = pm["close"][eb] - entry
                    pnls_buy.append(pnl - COST)
                    break
        else:
            # AM was bearish → PM may retrace up to fib level → SELL continuation
            fib_price = am_low + am_range * fib_entry
            sl = fib_price + am_range * sl_frac
            tp = am_low - am_range * (tp_ext - 1.0)
            
            for b in range(0, min(12, n - hold)):
                if pm["high"][b] >= fib_price - 0.3:
                    entry = fib_price
                    pnl = 0
                    for eb in range(b + 1, min(b + hold + 1, n)):
                        if pm["high"][eb] >= sl:
                            pnl = entry - sl
                            break
                        if pm["low"][eb] <= tp:
                            pnl = entry - tp
                            break
                        pnl = entry - pm["close"][eb]
                    pnls_sell.append(pnl - COST)
                    break
    
    for direction, pnls in [("BUY", pnls_buy), ("SELL", pnls_sell)]:
        stats = score_trades(pnls)
        if stats["n"] >= 5:
            fib_ampm_results.append({
                "fib": fib_entry, "hold": hold, "sl": sl_frac,
                "tp_ext": tp_ext, "dir": direction, **stats
            })

fib_ampm_results.sort(key=lambda x: x["score"], reverse=True)
print(f"\nTop 10 AM→PM Fib configs:")
print(f"{'Fib':<5} {'Hold':<5} {'SL':<5} {'TP_ext':<7} {'Dir':<5} "
      f"{'N':<4} {'WR%':<6} {'Net':<7} {'PF':<5} {'Total':<7}")
print("-" * 80)
for r in fib_ampm_results[:10]:
    print(f"{r['fib']:<5.3f} {r['hold']:<5} {r['sl']:<5.1f} "
          f"{r['tp_ext']:<7.3f} {r['dir']:<5} {r['n']:<4} {r['wr']:<6.1f} "
          f"{r['net']:<7.2f} {r['pf']:<5.2f} {r['total']:<7.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# OVERALL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("OVERALL SUMMARY — BEST FIBONACCI STRATEGY PER CATEGORY")
print("=" * 90)

all_best = []
if fib_retrace_results:
    best = fib_retrace_results[0]
    print(f"\n1. FIB RETRACEMENT PULLBACK:")
    print(f"   Best: swing_lb={best['swing_lb']}, min_swing={best['min_swing']}, "
          f"fib={best['fib_entry']}, hold={best['hold']}, sl={best['sl_mode']}, sess={best['sess']}")
    print(f"   N={best['n']}, WR={best['wr']:.1f}%, Net={best['net']:.2f}, PF={best['pf']:.2f}, Total={best['total']:.1f}")
    all_best.append(("Fib Retracement", best))

if fib_session_results:
    best = fib_session_results[0]
    print(f"\n2. SESSION FIB ZONES:")
    print(f"   Best: or_bars={best['or_bars']}, fib={best['fib']}, hold={best['hold']}, "
          f"sl_mult={best['sl_mult']}, tp_ext={best['tp_ext']}, sess={best['sess']}, dir={best['dir']}")
    print(f"   N={best['n']}, WR={best['wr']:.1f}%, Net={best['net']:.2f}, PF={best['pf']:.2f}, Total={best['total']:.1f}")
    all_best.append(("Session Fib", best))

if fib_ema_results:
    best = fib_ema_results[0]
    print(f"\n3. FIB + EMA TREND:")
    print(f"   Best: lb={best['lb']}, fib={best['fib']}, hold={best['hold']}, "
          f"atr_sl={best['atr_sl']}, atr_tp={best['atr_tp']}, sess={best['sess']}")
    print(f"   N={best['n']}, WR={best['wr']:.1f}%, Net={best['net']:.2f}, PF={best['pf']:.2f}, Total={best['total']:.1f}")
    all_best.append(("Fib+EMA", best))

if fib_rsi_results:
    best = fib_rsi_results[0]
    print(f"\n4. FIB + RSI CONFLUENCE:")
    print(f"   Best: lb={best['lb']}, fib={best['fib']}, rsi_ob={best['rsi_ob']}, "
          f"rsi_os={best['rsi_os']}, hold={best['hold']}, atr_sl={best['atr_sl']}, atr_tp={best['atr_tp']}, sess={best['sess']}")
    print(f"   N={best['n']}, WR={best['wr']:.1f}%, Net={best['net']:.2f}, PF={best['pf']:.2f}, Total={best['total']:.1f}")
    all_best.append(("Fib+RSI", best))

if fib_ampm_results:
    best = fib_ampm_results[0]
    print(f"\n5. AM→PM FIB PROJECTION:")
    print(f"   Best: fib={best['fib']}, hold={best['hold']}, sl={best['sl']}, "
          f"tp_ext={best['tp_ext']}, dir={best['dir']}")
    print(f"   N={best['n']}, WR={best['wr']:.1f}%, Net={best['net']:.2f}, PF={best['pf']:.2f}, Total={best['total']:.1f}")
    all_best.append(("AM→PM Fib", best))

print("\n" + "=" * 90)
print("RANKING (by score):")
all_best.sort(key=lambda x: x[1]["score"], reverse=True)
for i, (name, b) in enumerate(all_best, 1):
    print(f"  #{i} {name:<20} Score={b['score']:.2f}  N={b['n']}  WR={b['wr']:.1f}%  "
          f"Net={b['net']:.2f}  PF={b['pf']:.2f}  Total={b['total']:.1f}")
print("=" * 90)
print("\nDone!", flush=True)
