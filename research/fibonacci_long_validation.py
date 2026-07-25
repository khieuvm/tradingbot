"""
FIBONACCI STRATEGIES — LONG DATA VALIDATION (VN30 Index as proxy)
═══════════════════════════════════════════════════════════════════
VN30F1M chỉ có ~26 ngày data (rolling contract).
Dùng VN30 Index 5m (KBS ~8 tháng) làm proxy validation.
Cost adjusted: 0.96 pts/trade (same as futures)
"""
import sys
sys.path.insert(0, r"E:\Trading")

import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from itertools import product
from src.data_fetcher import DataFetcher

# ── Fetch VN30 Index (longer history) ─────────────────────────────────────────
fetcher = DataFetcher(provider="KBS")
end_date = datetime(2026, 7, 25)
start_date = end_date - timedelta(days=300)

print("=" * 90)
print("FIBONACCI STRATEGIES — LONG DATA (VN30 Index proxy)")
print("=" * 90)
print(f"\nFetching VN30 Index 5m data ({start_date.date()} to {end_date.date()})...", flush=True)

df = fetcher.get_historical_ohlcv("VN30", start_date.strftime("%Y-%m-%d"),
                                   end_date.strftime("%Y-%m-%d"), "5m")
print(f"  5m bars: {len(df)}", flush=True)

if df is None or len(df) < 100:
    print("Not enough data from VN30 index, trying VN30F1M with KBS...", flush=True)
    fetcher2 = DataFetcher(provider="KBS")
    df = fetcher2.get_futures_ohlcv("VN30F1M", start_date.strftime("%Y-%m-%d"),
                                     end_date.strftime("%Y-%m-%d"), "5m")
    print(f"  Fallback 5m bars: {len(df)}", flush=True)

df = df.copy()
df["time"] = pd.to_datetime(df["time"])
df["date"] = df["time"].dt.date
df["mins"] = df["time"].dt.hour * 60 + df["time"].dt.minute
df["session"] = np.where(df["mins"] < 11*60+30, "AM", "PM")
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = df[col].astype(float)

# Scale VN30 Index to futures-like point movements
# VN30 Index ~1300-1500, futures moves ~same absolute points
# But VN30 index values are much larger. We'll normalize by using
# percentage-based approach OR just use raw points (index moves similar to futures)
# Actually VN30 index is in hundreds (e.g. 1350) while futures is same (e.g. 1350)
# So the movements are the same scale - no adjustment needed.

# Indicators
df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
df["rsi"] = ta.rsi(df["close"], length=14)
df["ema8"] = ta.ema(df["close"], length=8)
df["ema21"] = ta.ema(df["close"], length=21)
df["ema50"] = ta.ema(df["close"], length=50)

dates = sorted(df["date"].unique())
print(f"Trading days: {len(dates)}", flush=True)

# Check ATR range to understand scale
valid_atr = df["atr"].dropna()
print(f"ATR stats: mean={valid_atr.mean():.2f}, median={valid_atr.median():.2f}, "
      f"min={valid_atr.min():.2f}, max={valid_atr.max():.2f}", flush=True)

COST = 0.96

# ── Pre-extract sessions ──────────────────────────────────────────────────────
print("\nPre-extracting sessions...", flush=True)
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

if len(sessions) < 20:
    print("ERROR: Not enough sessions for meaningful research. Exiting.", flush=True)
    sys.exit(1)


def score_trades(pnls):
    n = len(pnls)
    if n < 8:
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


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY A: FIB + EMA TREND CONFLUENCE (Best from short test)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("STRATEGY A: FIB + EMA TREND CONFLUENCE (Deep search)")
print("  EMA aligned + price at Fib retracement = entry")
print("=" * 90, flush=True)

LOOKBACK = [3, 5, 7, 8, 10, 14]
FIB_LV = [0.236, 0.382, 0.5, 0.618]
HOLD = [4, 6, 8, 10, 12, 15]
ATR_SL = [0.8, 1.0, 1.2, 1.5, 2.0]
ATR_TP = [2.0, 3.0, 4.0, 5.0]
SESS_F = ["AM", "PM", "BOTH"]
MIN_SWING = [1.5, 2.0, 3.0]  # Minimum swing size in points

total = len(LOOKBACK) * len(FIB_LV) * len(HOLD) * len(ATR_SL) * len(ATR_TP) * len(SESS_F) * len(MIN_SWING)
print(f"Combos: {total}\n", flush=True)

fib_ema_results = []
cnt = 0

for lb, fib_lv, hold, atr_sl, atr_tp, sess_f, min_sw in product(
    LOOKBACK, FIB_LV, HOLD, ATR_SL, ATR_TP, SESS_F, MIN_SWING):
    cnt += 1
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
            if np.isnan(atr_val) or atr_val < 1.0:
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
                recent_low = s["low"][b-lb:b].min()
                recent_high = s["high"][b-lb:b].max()
                swing_size = recent_high - recent_low
                if swing_size < min_sw:
                    continue
                
                # Fib retracement level (pullback from high)
                fib_price = recent_high - swing_size * fib_lv
                
                # Check if current price is near fib level
                if abs(s["close"][b] - fib_price) > atr_val * 0.3:
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
                if swing_size < min_sw:
                    continue
                
                # Fib retracement level (bounce from low)
                fib_price = recent_low + swing_size * fib_lv
                
                # Check if current price is near fib level
                if abs(s["close"][b] - fib_price) > atr_val * 0.3:
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
    if stats["n"] >= 8:
        fib_ema_results.append({
            "lb": lb, "fib": fib_lv, "hold": hold, "min_sw": min_sw,
            "atr_sl": atr_sl, "atr_tp": atr_tp, "sess": sess_f, **stats
        })

fib_ema_results.sort(key=lambda x: x["score"], reverse=True)
print(f"\nTop 15 Fib+EMA Trend configs (N>=8):")
print(f"{'LB':<4} {'Fib':<5} {'Hold':<5} {'MinSw':<6} {'SL':<5} {'TP':<5} {'Sess':<5} "
      f"{'N':<5} {'WR%':<6} {'Net':<8} {'PF':<6} {'Total':<8} {'Pts/d':<6}")
print("-" * 95)
for r in fib_ema_results[:15]:
    pts_per_day = r['total'] / len(dates) if len(dates) > 0 else 0
    print(f"{r['lb']:<4} {r['fib']:<5.3f} {r['hold']:<5} {r['min_sw']:<6.1f} "
          f"{r['atr_sl']:<5.1f} {r['atr_tp']:<5.1f} {r['sess']:<5} "
          f"{r['n']:<5} {r['wr']:<6.1f} {r['net']:<8.2f} {r['pf']:<6.2f} "
          f"{r['total']:<8.1f} {pts_per_day:<6.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY B: SESSION FIB ZONES (2nd best from short test)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STRATEGY B: SESSION FIB ZONES (Deep search)")
print("  First N bars form OR, then trade retracements to Fib levels of OR")
print("=" * 90, flush=True)

OR_BARS = [4, 6, 8, 9, 10, 12, 15]
FIB_2 = [0.382, 0.5, 0.618, 0.786]
HOLD_2 = [6, 8, 10, 12, 15]
SL_MULT = [0.3, 0.5, 0.7, 1.0]
TP_EXT = [1.0, 1.272, 1.618, 2.0]
SESS_2 = ["AM", "PM"]
DIR_2 = ["BUY", "SELL"]
MIN_OR = [1.5, 2.0, 3.0]  # Minimum OR range

total2 = len(OR_BARS) * len(FIB_2) * len(HOLD_2) * len(SL_MULT) * len(TP_EXT) * len(SESS_2) * len(DIR_2) * len(MIN_OR)
print(f"Combos: {total2}\n", flush=True)

fib_session_results = []

for or_bars, fib_level, hold, sl_mult, tp_ext, sess_f, direction_str, min_or in product(
    OR_BARS, FIB_2, HOLD_2, SL_MULT, TP_EXT, SESS_2, DIR_2, MIN_OR):
    
    pnls = []
    
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
        if or_range < min_or:
            continue
        
        if direction_str == "BUY":
            # Buy on pullback to fib level within OR
            fib_price = or_high - or_range * fib_level
            sl = fib_price - or_range * sl_mult
            tp = or_high + or_range * (tp_ext - 1.0)
        else:
            # Sell on bounce to fib level within OR
            fib_price = or_low + or_range * fib_level
            sl = fib_price + or_range * sl_mult
            tp = or_low - or_range * (tp_ext - 1.0)
        
        last_trade_bar = -hold
        
        for b in range(or_bars, min(or_bars + 15, n - hold)):
            if b <= last_trade_bar + hold:
                continue
            
            triggered = False
            if direction_str == "BUY" and s["low"][b] <= fib_price + 0.2:
                triggered = True
            elif direction_str == "SELL" and s["high"][b] >= fib_price - 0.2:
                triggered = True
            
            if not triggered:
                continue
            
            entry = fib_price
            pnl = 0
            for eb in range(b + 1, min(b + hold + 1, n)):
                if direction_str == "BUY":
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
            break  # Only 1 trade per session for this strategy
    
    stats = score_trades(pnls)
    if stats["n"] >= 8:
        fib_session_results.append({
            "or_bars": or_bars, "fib": fib_level, "hold": hold, "min_or": min_or,
            "sl_mult": sl_mult, "tp_ext": tp_ext, "sess": sess_f,
            "dir": direction_str, **stats
        })

fib_session_results.sort(key=lambda x: x["score"], reverse=True)
print(f"\nTop 15 Session Fib configs (N>=8):")
print(f"{'OR':<4} {'Fib':<5} {'Hold':<5} {'MinOR':<6} {'SL_m':<5} {'TP_ext':<7} "
      f"{'Sess':<5} {'Dir':<5} {'N':<5} {'WR%':<6} {'Net':<8} {'PF':<6} {'Total':<8} {'Pts/d':<6}")
print("-" * 100)
for r in fib_session_results[:15]:
    pts_per_day = r['total'] / len(dates) if len(dates) > 0 else 0
    print(f"{r['or_bars']:<4} {r['fib']:<5.3f} {r['hold']:<5} {r['min_or']:<6.1f} "
          f"{r['sl_mult']:<5.1f} {r['tp_ext']:<7.3f} {r['sess']:<5} {r['dir']:<5} "
          f"{r['n']:<5} {r['wr']:<6.1f} {r['net']:<8.2f} {r['pf']:<6.2f} "
          f"{r['total']:<8.1f} {pts_per_day:<6.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY C: FIB + MOMENTUM (RSI reclaim from Fib level)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("STRATEGY C: FIB MOMENTUM BOUNCE")
print("  Price hits Fib level + RSI showing momentum reversal")
print("  BUY: RSI crosses above threshold from below at Fib support")
print("  SELL: RSI crosses below threshold from above at Fib resistance")
print("=" * 90, flush=True)

LOOKBACK_C = [5, 8, 10, 14]
FIB_C = [0.382, 0.5, 0.618, 0.786]
RSI_BUY_MIN = [25, 30, 35, 40]    # RSI was below this (oversold at fib)
RSI_BUY_NOW = [35, 40, 45, 50]    # RSI now above this (reclaiming)
RSI_SELL_MAX = [60, 65, 70, 75]   # RSI was above (overbought at fib)
RSI_SELL_NOW = [50, 55, 60, 65]   # RSI now below (dropping)
HOLD_C = [6, 8, 10, 12]
ATR_SL_C = [1.0, 1.2, 1.5]
ATR_TP_C = [2.0, 3.0, 4.0]
SESS_C = ["AM", "PM", "BOTH"]

# Too many combos, reduce
total_c = len(LOOKBACK_C) * len(FIB_C) * len(HOLD_C) * len(ATR_SL_C) * len(ATR_TP_C) * len(SESS_C) * 4
print(f"Combos (approx): {total_c}\n", flush=True)

fib_momentum_results = []

# BUY signals
for lb, fib_lv, rsi_min, rsi_now, hold, atr_sl, atr_tp, sess_f in product(
    LOOKBACK_C, FIB_C, RSI_BUY_MIN, RSI_BUY_NOW, HOLD_C, ATR_SL_C, ATR_TP_C, SESS_C):
    
    if rsi_now <= rsi_min:  # Must cross upward
        continue
    
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
            if np.isnan(atr_val) or np.isnan(rsi_val) or atr_val < 1.0:
                continue
            
            # RSI reclaim condition: prev bar RSI below min, current above now
            if b < 2:
                continue
            rsi_prev = s["rsi"][b-1]
            if np.isnan(rsi_prev):
                continue
            
            if not (rsi_prev <= rsi_min and rsi_val >= rsi_now):
                continue
            
            # Fib level check
            recent_high = s["high"][b-lb:b].max()
            recent_low = s["low"][b-lb:b].min()
            swing = recent_high - recent_low
            if swing < 2.0:
                continue
            
            fib_support = recent_high - swing * fib_lv
            if abs(s["close"][b] - fib_support) > atr_val * 0.5:
                continue
            
            # BUY
            entry = s["close"][b]
            sl = entry - atr_val * atr_sl
            tp = entry + atr_val * atr_tp
            
            pnl = 0
            for eb in range(b + 1, min(b + hold + 1, n)):
                if s["low"][eb] <= sl:
                    pnl = sl - entry
                    break
                if s["high"][eb] >= tp:
                    pnl = tp - entry
                    break
                pnl = s["close"][eb] - entry
            
            pnls.append(pnl - COST)
            last_trade_bar = b
    
    stats = score_trades(pnls)
    if stats["n"] >= 8:
        fib_momentum_results.append({
            "lb": lb, "fib": fib_lv, "rsi_cond": f"BUY:{rsi_min}->{rsi_now}",
            "hold": hold, "atr_sl": atr_sl, "atr_tp": atr_tp, "sess": sess_f,
            "dir": "BUY", **stats
        })

# SELL signals
for lb, fib_lv, rsi_max, rsi_now, hold, atr_sl, atr_tp, sess_f in product(
    LOOKBACK_C, FIB_C, RSI_SELL_MAX, RSI_SELL_NOW, HOLD_C, ATR_SL_C, ATR_TP_C, SESS_C):
    
    if rsi_now >= rsi_max:
        continue
    
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
            if np.isnan(atr_val) or np.isnan(rsi_val) or atr_val < 1.0:
                continue
            
            if b < 2:
                continue
            rsi_prev = s["rsi"][b-1]
            if np.isnan(rsi_prev):
                continue
            
            if not (rsi_prev >= rsi_max and rsi_val <= rsi_now):
                continue
            
            # Fib level check
            recent_high = s["high"][b-lb:b].max()
            recent_low = s["low"][b-lb:b].min()
            swing = recent_high - recent_low
            if swing < 2.0:
                continue
            
            fib_resistance = recent_low + swing * fib_lv
            if abs(s["close"][b] - fib_resistance) > atr_val * 0.5:
                continue
            
            # SELL
            entry = s["close"][b]
            sl = entry + atr_val * atr_sl
            tp = entry - atr_val * atr_tp
            
            pnl = 0
            for eb in range(b + 1, min(b + hold + 1, n)):
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
    if stats["n"] >= 8:
        fib_momentum_results.append({
            "lb": lb, "fib": fib_lv, "rsi_cond": f"SELL:{rsi_max}->{rsi_now}",
            "hold": hold, "atr_sl": atr_sl, "atr_tp": atr_tp, "sess": sess_f,
            "dir": "SELL", **stats
        })

fib_momentum_results.sort(key=lambda x: x["score"], reverse=True)
print(f"\nTop 15 Fib Momentum configs:")
print(f"{'LB':<4} {'Fib':<5} {'RSI_cond':<14} {'Hold':<5} {'SL':<5} {'TP':<5} {'Sess':<5} "
      f"{'N':<5} {'WR%':<6} {'Net':<8} {'PF':<6} {'Total':<8}")
print("-" * 95)
for r in fib_momentum_results[:15]:
    print(f"{r['lb']:<4} {r['fib']:<5.3f} {r['rsi_cond']:<14} {r['hold']:<5} "
          f"{r['atr_sl']:<5.1f} {r['atr_tp']:<5.1f} {r['sess']:<5} "
          f"{r['n']:<5} {r['wr']:<6.1f} {r['net']:<8.2f} {r['pf']:<6.2f} {r['total']:<8.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# OVERALL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("OVERALL SUMMARY")
print("=" * 90)

all_best = []
if fib_ema_results:
    best = fib_ema_results[0]
    print(f"\nA. FIB + EMA TREND CONFLUENCE:")
    print(f"   Best: lb={best['lb']}, fib={best['fib']}, hold={best['hold']}, "
          f"min_sw={best['min_sw']}, atr_sl={best['atr_sl']}, atr_tp={best['atr_tp']}, sess={best['sess']}")
    print(f"   N={best['n']}, WR={best['wr']:.1f}%, Net={best['net']:.2f}, PF={best['pf']:.2f}, "
          f"Total={best['total']:.1f}, Pts/day={best['total']/len(dates):.2f}")
    all_best.append(("Fib+EMA", best))

if fib_session_results:
    best = fib_session_results[0]
    print(f"\nB. SESSION FIB ZONES:")
    print(f"   Best: or_bars={best['or_bars']}, fib={best['fib']}, hold={best['hold']}, "
          f"min_or={best['min_or']}, sl_mult={best['sl_mult']}, tp_ext={best['tp_ext']}, "
          f"sess={best['sess']}, dir={best['dir']}")
    print(f"   N={best['n']}, WR={best['wr']:.1f}%, Net={best['net']:.2f}, PF={best['pf']:.2f}, "
          f"Total={best['total']:.1f}, Pts/day={best['total']/len(dates):.2f}")
    all_best.append(("Session Fib", best))

if fib_momentum_results:
    best = fib_momentum_results[0]
    print(f"\nC. FIB MOMENTUM BOUNCE:")
    print(f"   Best: lb={best['lb']}, fib={best['fib']}, {best['rsi_cond']}, "
          f"hold={best['hold']}, atr_sl={best['atr_sl']}, atr_tp={best['atr_tp']}, sess={best['sess']}")
    print(f"   N={best['n']}, WR={best['wr']:.1f}%, Net={best['net']:.2f}, PF={best['pf']:.2f}, "
          f"Total={best['total']:.1f}, Pts/day={best['total']/len(dates):.2f}")
    all_best.append(("Fib Momentum", best))

print(f"\n{'='*90}")
print(f"RANKING (by score) — {len(dates)} trading days:")
all_best.sort(key=lambda x: x[1]["score"], reverse=True)
for i, (name, b) in enumerate(all_best, 1):
    ptsd = b['total'] / len(dates)
    print(f"  #{i} {name:<15} Score={b['score']:.2f}  N={b['n']}  WR={b['wr']:.1f}%  "
          f"Net={b['net']:.2f}  PF={b['pf']:.2f}  Total={b['total']:.1f}  Pts/day={ptsd:.2f}")
print("=" * 90)

# Profitability check
print("\nPROFITABILITY VERDICT:")
for name, b in all_best:
    ptsd = b['total'] / len(dates)
    if b['net'] > 1.0 and b['pf'] > 1.5 and b['wr'] > 50 and b['n'] >= 10:
        print(f"  [PASS] {name}: Net {b['net']:.2f}/trade, PF {b['pf']:.2f}, WR {b['wr']:.1f}%, "
              f"{ptsd:.2f} pts/day, {b['n']} trades")
    elif b['net'] > 0 and b['pf'] > 1.2:
        print(f"  [WEAK] {name}: Net {b['net']:.2f}/trade, PF {b['pf']:.2f}, WR {b['wr']:.1f}% "
              f"— marginal edge, needs more data")
    else:
        print(f"  [FAIL] {name}: Net {b['net']:.2f}/trade — not profitable after cost")

print("\nDone!", flush=True)
