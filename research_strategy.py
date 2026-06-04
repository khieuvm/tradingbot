"""
VN30F1M Edge Research: 9+ Point Move Analysis
==============================================
Research question: What conditions precede intraday moves of 9+ pts?
Follows edge-researcher skill methodology:
  1. Load 120+ days of data
  2. Find all 9+ pt moves within session rules
  3. Analyze pre-move conditions (compression, ATR, time, candle pattern)
  4. Propose entry strategies with simulated results
  5. Compare to random baseline

Intraday rules:
  - Sessions: 9:00-11:30 (AM), 13:00-14:30 (PM)
  - No entry after 14:15
  - All positions close by 14:28
  - Cost: 1.74 pts per trade (slippage 0.8 + commission 0.94)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# === DATA LOADING ===

def load_data(days=180):
    from src.data_fetcher import DataFetcher
    fetcher = DataFetcher()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"Loading VN30F1M 5m data: {start} to {end} ({days} days)...")
    df = fetcher.get_futures_ohlcv("VN30F1M", start, end, interval="5m")

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Filter session hours only
    h = df['time'].dt.hour
    m = df['time'].dt.minute
    mins = h * 60 + m
    in_session = ((mins >= 9*60) & (mins < 11*60+30)) | ((mins >= 13*60) & (mins < 14*60+30))
    df = df[in_session].reset_index(drop=True)

    print(f"  Loaded {len(df)} bars over {df['time'].dt.date.nunique()} trading days")
    return df

# === INDICATORS ===

def add_indicators(df):
    """Add ATR, ADX, EMA, body ratio, compression detection"""
    import pandas_ta as ta

    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df.iloc[:, 0]
    df['di_plus'] = adx_df.iloc[:, 1]
    df['di_minus'] = adx_df.iloc[:, 2]

    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)

    # Candle characteristics
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / df['range'].replace(0, np.nan)
    df['is_bull'] = df['close'] > df['open']

    # Time features
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['mins_in_day'] = df['hour'] * 60 + df['minute']
    df['date'] = df['time'].dt.date

    # Session label
    df['session'] = np.where(df['mins_in_day'] < 12*60, 'AM', 'PM')

    # ATR SMA for volatility regime
    df['atr_sma50'] = df['atr'].rolling(50).mean()
    df['atr_ratio'] = df['atr'] / df['atr_sma50']

    return df

# === FIND 9+ PT MOVES ===

def find_moves_9plus(df, target_pts=9, max_bars=24):
    """
    For each bar, check if price moves 9+ pts in either direction
    within the SAME session (max_bars forward looking).
    Records the move direction and how many bars it took.
    """
    results = []
    dates = df['date'].values
    sessions = df['session'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)

    for i in range(n - 1):
        entry_price = closes[i]
        entry_date = dates[i]
        entry_session = sessions[i]

        max_up = 0
        max_down = 0
        bars_to_up = 0
        bars_to_down = 0

        for j in range(i+1, min(i+1+max_bars, n)):
            # Must stay within same date AND same session
            if dates[j] != entry_date or sessions[j] != entry_session:
                break

            up_move = highs[j] - entry_price
            down_move = entry_price - lows[j]

            if up_move > max_up:
                max_up = up_move
                bars_to_up = j - i
            if down_move > max_down:
                max_down = down_move
                bars_to_down = j - i

        if max_up >= target_pts or max_down >= target_pts:
            direction = 'UP' if max_up >= max_down else 'DOWN'
            move_size = max(max_up, max_down)
            bars_to_target = bars_to_up if direction == 'UP' else bars_to_down

            results.append({
                'idx': i,
                'time': df.iloc[i]['time'],
                'date': entry_date,
                'session': entry_session,
                'direction': direction,
                'move_size': move_size,
                'bars_to_target': bars_to_target,
                'entry_price': entry_price,
                'atr': df.iloc[i]['atr'],
                'adx': df.iloc[i]['adx'],
                'body_ratio': df.iloc[i]['body_ratio'],
                'range': df.iloc[i]['range'],
                'hour': df.iloc[i]['hour'],
                'minute': df.iloc[i]['minute'],
                'is_bull': df.iloc[i]['is_bull'],
            })

    return pd.DataFrame(results)

# === COMPRESSION DETECTION ===

def detect_compression(df, n_bars=3, threshold=0.7):
    """
    Find bars where previous n_bars all had range < threshold * ATR.
    These are VCP/Micro-ORB setups.
    """
    ranges = df['range'].values
    atr = df['atr'].values
    compressed = np.zeros(len(df), dtype=bool)

    for i in range(n_bars, len(df)):
        max_range = max(ranges[i-n_bars:i])
        if atr[i] > 0 and max_range < threshold * atr[i]:
            compressed[i] = True

    df['compressed_3bar'] = compressed
    return df

# === ORB DETECTION ===

def detect_orb(df, n_bars=2, max_range=7):
    """
    Detect Opening Range Breakout:
    - First n_bars of each session form the range
    - Breakout = close above/below that range
    """
    df['is_orb_breakout'] = False
    df['orb_direction'] = ''
    df['orb_range'] = 0.0

    dates = df['date'].unique()
    for date in dates:
        for session in ['AM', 'PM']:
            mask = (df['date'] == date) & (df['session'] == session)
            session_df = df[mask]

            if len(session_df) < n_bars + 1:
                continue

            orb_bars = session_df.iloc[:n_bars]
            orb_high = orb_bars['high'].max()
            orb_low = orb_bars['low'].min()
            orb_range = orb_high - orb_low

            if orb_range > max_range or orb_range < 1.0:
                continue

            # Check subsequent bars for breakout
            for idx in session_df.index[n_bars:]:
                if df.loc[idx, 'close'] > orb_high:
                    df.loc[idx, 'is_orb_breakout'] = True
                    df.loc[idx, 'orb_direction'] = 'UP'
                    df.loc[idx, 'orb_range'] = orb_range
                    break
                elif df.loc[idx, 'close'] < orb_low:
                    df.loc[idx, 'is_orb_breakout'] = True
                    df.loc[idx, 'orb_direction'] = 'DOWN'
                    df.loc[idx, 'orb_range'] = orb_range
                    break

    return df

# === STRATEGY SIMULATION ===

def simulate_strategy(df, entries_mask, direction_col=None, sl_mult=1.2,
                      trail_activate=5, trail_atr_mult=2.0, cost=1.74,
                      max_bars_hold=24, entry_name="Strategy"):
    """
    Simulate trades with:
    - ATR-based SL
    - Trailing stop (activates after trail_activate pts profit)
    - Session close force exit
    - Slippage included in cost
    """
    trades = []
    entry_indices = df.index[entries_mask]

    for i in entry_indices:
        row = df.loc[i]

        # Skip if too late in session
        if row['session'] == 'PM' and row['mins_in_day'] >= 14*60+15:
            continue
        if row['session'] == 'AM' and row['mins_in_day'] >= 11*60+15:
            continue

        entry_price = row['close']
        atr = row['atr']
        if pd.isna(atr) or atr <= 0:
            continue

        sl_distance = sl_mult * atr

        # Determine direction
        if direction_col and direction_col in df.columns:
            dir_val = df.loc[i, direction_col]
            if dir_val == 'UP':
                direction = 1
            elif dir_val == 'DOWN':
                direction = -1
            else:
                # Breakout direction from next bar
                if i + 1 < len(df) and i+1 in df.index:
                    direction = 1 if df.loc[i+1, 'close'] > entry_price else -1
                else:
                    continue
        else:
            # Use breakout direction from next bar
            if i + 1 < len(df) and i+1 in df.index:
                direction = 1 if df.loc[i+1, 'close'] > entry_price else -1
            else:
                continue

        sl_price = entry_price - direction * sl_distance
        best_price = entry_price
        trail_active = False
        exit_price = None
        exit_reason = None
        exit_bar = 0

        # Forward simulate
        for j in range(i+1, min(i+1+max_bars_hold, len(df))):
            if j not in df.index:
                break
            bar = df.loc[j]

            # Session boundary check
            if bar['date'] != row['date'] or bar['session'] != row['session']:
                exit_price = df.loc[j-1, 'close'] if j-1 in df.index else entry_price
                exit_reason = 'SESSION_CLOSE'
                exit_bar = j - i
                break

            # Force close at 14:28 (PM) or 11:28 (AM — 2 min before close)
            if (bar['session'] == 'PM' and bar['mins_in_day'] >= 14*60+25) or \
               (bar['session'] == 'AM' and bar['mins_in_day'] >= 11*60+25):
                exit_price = bar['close']
                exit_reason = 'SESSION_CLOSE'
                exit_bar = j - i
                break

            if direction == 1:
                # Check SL
                if bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    exit_bar = j - i
                    break
                # Update trailing
                if bar['high'] > best_price:
                    best_price = bar['high']
                    profit = best_price - entry_price
                    if profit >= trail_activate and not trail_active:
                        trail_active = True
                    if trail_active:
                        trail_sl = best_price - trail_atr_mult * atr
                        sl_price = max(sl_price, trail_sl)
                # Check if trailing stop hit
                if trail_active and bar['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'TRAIL'
                    exit_bar = j - i
                    break
            else:
                # SHORT
                if bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    exit_bar = j - i
                    break
                if bar['low'] < best_price:
                    best_price = bar['low']
                    profit = entry_price - best_price
                    if profit >= trail_activate and not trail_active:
                        trail_active = True
                    if trail_active:
                        trail_sl = best_price + trail_atr_mult * atr
                        sl_price = min(sl_price, trail_sl)
                if trail_active and bar['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'TRAIL'
                    exit_bar = j - i
                    break

        if exit_price is None:
            last_idx = min(i + max_bars_hold, len(df) - 1)
            if last_idx in df.index:
                exit_price = df.loc[last_idx, 'close']
            else:
                exit_price = entry_price
            exit_reason = 'MAX_BARS'
            exit_bar = max_bars_hold

        pnl = direction * (exit_price - entry_price) - cost

        trades.append({
            'entry_time': row['time'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'pnl': pnl,
            'exit_reason': exit_reason,
            'bars_held': exit_bar,
            'atr': atr,
            'session': row['session'],
            'hour': row['hour'],
            'trail_activated': trail_active,
        })

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return trades_df, {}

    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    n_days = df['date'].nunique()
    metrics = {
        'name': entry_name,
        'trades': len(trades_df),
        'trades_per_day': len(trades_df) / n_days,
        'win_rate': len(wins) / len(trades_df) * 100,
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        'pf': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 999,
        'total_pnl': trades_df['pnl'].sum(),
        'pnl_per_day': trades_df['pnl'].sum() / n_days,
        'max_dd': compute_max_dd(trades_df['pnl'].values),
        'trail_pct': trades_df['trail_activated'].mean() * 100,
        'am_trades': len(trades_df[trades_df['session'] == 'AM']),
        'pm_trades': len(trades_df[trades_df['session'] == 'PM']),
        'session_close_pct': len(trades_df[trades_df['exit_reason'] == 'SESSION_CLOSE']) / len(trades_df) * 100,
    }

    return trades_df, metrics

def compute_max_dd(pnl_array):
    """Compute max drawdown from PnL series"""
    cum = np.cumsum(pnl_array)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return dd.min()

# === ANALYSIS ===

def analyze_pre_move_conditions(df, moves_df):
    """Analyze what conditions exist before 9+ pt moves"""
    print("\n" + "="*70)
    print("PHAN TICH DIEU KIEN TRUOC CAC MOVE 9+ PTS")
    print("="*70)

    print(f"\nTong so moves 9+ pts: {len(moves_df)}")
    print(f"  UP moves: {len(moves_df[moves_df['direction']=='UP'])}")
    print(f"  DOWN moves: {len(moves_df[moves_df['direction']=='DOWN'])}")

    n_days = df['date'].nunique()
    print(f"  Frequency: {len(moves_df)/n_days:.1f} moves/day")

    # By session
    print(f"\n--- By Session ---")
    for sess in ['AM', 'PM']:
        sub = moves_df[moves_df['session'] == sess]
        print(f"  {sess}: {len(sub)} moves ({len(sub)/n_days:.1f}/day), "
              f"avg size: {sub['move_size'].mean():.1f} pts, "
              f"avg bars: {sub['bars_to_target'].mean():.1f}")

    # By time of day
    print(f"\n--- By Time Window ---")
    time_windows = [
        ('09:00-09:30', 9*60, 9*60+30),
        ('09:30-10:00', 9*60+30, 10*60),
        ('10:00-10:45', 10*60, 10*60+45),
        ('10:45-11:30', 10*60+45, 11*60+30),
        ('13:00-13:30', 13*60, 13*60+30),
        ('13:30-14:00', 13*60+30, 14*60),
        ('14:00-14:30', 14*60, 14*60+30),
    ]

    for name, start_m, end_m in time_windows:
        sub = moves_df[(moves_df['hour']*60 + moves_df['minute'] >= start_m) &
                       (moves_df['hour']*60 + moves_df['minute'] < end_m)]
        if len(sub) > 0:
            print(f"  {name}: {len(sub):3d} moves | avg size: {sub['move_size'].mean():.1f} pts | avg bars: {sub['bars_to_target'].mean():.1f}")
        else:
            print(f"  {name}:   0 moves")

    # By ATR level
    print(f"\n--- By ATR Level ---")
    for lo, hi, label in [(0, 3.0, '<3.0'), (3.0, 3.5, '3.0-3.5'),
                           (3.5, 4.0, '3.5-4.0'), (4.0, 4.5, '4.0-4.5'), (4.5, 99, '>4.5')]:
        sub = moves_df[(moves_df['atr'] >= lo) & (moves_df['atr'] < hi)]
        if len(sub) > 0:
            print(f"  ATR {label}: {len(sub):3d} moves | avg size: {sub['move_size'].mean():.1f} pts | avg bars: {sub['bars_to_target'].mean():.1f}")

    # By body ratio (preceding bar)
    print(f"\n--- By Preceding Bar Body Ratio ---")
    for lo, hi, label in [(0, 0.3, 'Doji (<0.3)'), (0.3, 0.5, 'Small (0.3-0.5)'),
                           (0.5, 0.7, 'Medium (0.5-0.7)'), (0.7, 1.01, 'Strong (>0.7)')]:
        sub = moves_df[(moves_df['body_ratio'] >= lo) & (moves_df['body_ratio'] < hi)]
        if len(sub) > 0:
            print(f"  {label}: {len(sub):3d} moves | avg size: {sub['move_size'].mean():.1f} pts")

    # By ADX
    print(f"\n--- By ADX Level (Regime) ---")
    for lo, hi, label in [(0, 20, 'RANGING <20'), (20, 25, 'NORMAL 20-25'),
                           (25, 35, 'TRENDING 25-35'), (35, 99, 'STRONG >35')]:
        sub = moves_df[(moves_df['adx'] >= lo) & (moves_df['adx'] < hi)]
        if len(sub) > 0:
            print(f"  {label}: {len(sub):3d} moves | avg size: {sub['move_size'].mean():.1f} pts")

    # Compression before move
    compressed_moves = moves_df[moves_df['idx'].apply(lambda x: df.loc[x, 'compressed_3bar'] if x in df.index else False)]
    print(f"\n--- Compression Before Move ---")
    print(f"  Moves with 3-bar compression: {len(compressed_moves)} / {len(moves_df)} "
          f"({len(compressed_moves)/len(moves_df)*100:.1f}%)")

    compression_rate = df['compressed_3bar'].mean() * 100
    print(f"  Base rate of compression (any bar): {compression_rate:.1f}%")
    if len(compressed_moves) > 0 and compression_rate > 0:
        lift = (len(compressed_moves)/len(moves_df)*100) / compression_rate
        print(f"  Lift over random: {lift:.2f}x")


def run_strategy_comparison(df):
    """Test multiple entry strategies and compare"""
    print("\n" + "="*70)
    print("STRATEGY COMPARISON — ENTRY SIGNALS FOR 9+ PT TARGET")
    print("="*70)
    print(f"Data: {df['date'].nunique()} trading days, {len(df)} bars")
    print(f"Cost per trade: 1.74 pts | Trailing: 2.0x ATR after +5 pts")
    print(f"Max hold: 24 bars (2h) | Session rules enforced")
    print(f"SL: ATR-based | Force close at session end")
    print()

    results = []

    # Strategy 1: Compression Breakout (Micro-ORB)
    print("--- Strategy 1: 3-Bar Compression Breakout ---")
    mask1 = df['compressed_3bar'] & (df['mins_in_day'] >= 9*60+15) & (df['mins_in_day'] < 14*60+15)
    mask1_deduped = dedup_signals(mask1, min_bars=5)
    trades1, m1 = simulate_strategy(df, mask1_deduped, sl_mult=1.2,
                                     trail_activate=5, trail_atr_mult=2.0,
                                     entry_name="Compression 3-bar")
    if m1:
        results.append(m1)
        print_metrics(m1)

    # Strategy 2: ORB Breakout (first 2 bars)
    print("\n--- Strategy 2: Session ORB (2-bar) ---")
    mask2 = df['is_orb_breakout']
    trades2, m2 = simulate_strategy(df, mask2, direction_col='orb_direction', sl_mult=1.5,
                                     trail_activate=5, trail_atr_mult=2.0,
                                     entry_name="Session ORB")
    if m2:
        results.append(m2)
        print_metrics(m2)

    # Strategy 3: Compression + AM Prime Time only (9:15-10:45)
    print("\n--- Strategy 3: Compression + AM Prime (9:15-10:45) ---")
    mask3 = (df['compressed_3bar'] &
             (df['mins_in_day'] >= 9*60+15) & (df['mins_in_day'] <= 10*60+45))
    mask3_deduped = dedup_signals(mask3, min_bars=5)
    trades3, m3 = simulate_strategy(df, mask3_deduped, sl_mult=1.2,
                                     trail_activate=5, trail_atr_mult=2.0,
                                     entry_name="Compress+AM Prime")
    if m3:
        results.append(m3)
        print_metrics(m3)

    # Strategy 4: Compression + Trending regime (ADX>25)
    print("\n--- Strategy 4: Compression + TRENDING (ADX>25) ---")
    mask4 = (df['compressed_3bar'] & (df['adx'] > 25) &
             (df['mins_in_day'] >= 9*60+15) & (df['mins_in_day'] < 14*60+15))
    mask4_deduped = dedup_signals(mask4, min_bars=5)
    trades4, m4 = simulate_strategy(df, mask4_deduped, sl_mult=1.2,
                                     trail_activate=5, trail_atr_mult=2.0,
                                     entry_name="Compress+Trending")
    if m4:
        results.append(m4)
        print_metrics(m4)

    # Strategy 5: Compression + Small body (doji) + ATR optimal
    print("\n--- Strategy 5: Compression + Doji + ATR 3.0-4.2 ---")
    mask5 = (df['compressed_3bar'] & (df['body_ratio'] < 0.35) &
             (df['atr'] >= 3.0) & (df['atr'] <= 4.2) &
             (df['mins_in_day'] >= 9*60+15) & (df['mins_in_day'] < 14*60+15))
    mask5_deduped = dedup_signals(mask5, min_bars=5)
    trades5, m5 = simulate_strategy(df, mask5_deduped, sl_mult=1.0,
                                     trail_activate=5, trail_atr_mult=2.0,
                                     entry_name="Compress+Doji+ATR")
    if m5:
        results.append(m5)
        print_metrics(m5)

    # Strategy 6: ORB + Compression combined
    print("\n--- Strategy 6: ORB + Compression (combined) ---")
    mask6 = (mask1_deduped | mask2) & (df['mins_in_day'] >= 9*60+10) & (df['mins_in_day'] < 14*60+15)
    mask6_deduped = dedup_signals(mask6, min_bars=5)
    trades6, m6 = simulate_strategy(df, mask6_deduped, sl_mult=1.2,
                                     trail_activate=5, trail_atr_mult=2.0,
                                     entry_name="ORB+Compression")
    if m6:
        results.append(m6)
        print_metrics(m6)

    # Strategy 7: Compression + EMA trend alignment
    print("\n--- Strategy 7: Compression + EMA Trend Alignment ---")
    ema_bull = (df['close'] > df['ema20']) & (df['ema20'] > df['ema50'])
    ema_bear = (df['close'] < df['ema20']) & (df['ema20'] < df['ema50'])
    mask7 = (df['compressed_3bar'] & (ema_bull | ema_bear) &
             (df['mins_in_day'] >= 9*60+15) & (df['mins_in_day'] < 14*60+15))
    mask7_deduped = dedup_signals(mask7, min_bars=5)
    df['ema_direction'] = np.where(ema_bull, 'UP', np.where(ema_bear, 'DOWN', ''))
    trades7, m7 = simulate_strategy(df, mask7_deduped, direction_col='ema_direction',
                                     sl_mult=1.2, trail_activate=5, trail_atr_mult=2.0,
                                     entry_name="Compress+EMA")
    if m7:
        results.append(m7)
        print_metrics(m7)

    # Strategy 8: PM session compression (tighter params)
    print("\n--- Strategy 8: PM Compression (13:15-14:00, tight) ---")
    mask8 = (df['compressed_3bar'] &
             (df['mins_in_day'] >= 13*60+15) & (df['mins_in_day'] <= 14*60))
    mask8_deduped = dedup_signals(mask8, min_bars=5)
    trades8, m8 = simulate_strategy(df, mask8_deduped, sl_mult=1.0,
                                     trail_activate=4, trail_atr_mult=1.5,
                                     max_bars_hold=12,
                                     entry_name="PM Compress (tight)")
    if m8:
        results.append(m8)
        print_metrics(m8)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY — ALL STRATEGIES RANKED BY PnL")
    print("="*70)
    print(f"{'Strategy':<25} {'Trades':>6} {'T/Day':>5} {'WR%':>5} {'PF':>5} "
          f"{'AvgW':>6} {'AvgL':>6} {'PnL':>7} {'PnL/D':>6} {'SessCl%':>7}")
    print("-"*90)

    for m in sorted(results, key=lambda x: x['total_pnl'], reverse=True):
        print(f"{m['name']:<25} {m['trades']:>6} {m['trades_per_day']:>5.1f} "
              f"{m['win_rate']:>5.1f} {m['pf']:>5.2f} "
              f"{m['avg_win']:>6.1f} {m['avg_loss']:>6.1f} "
              f"{m['total_pnl']:>7.1f} {m['pnl_per_day']:>6.2f} {m['session_close_pct']:>7.1f}")

    return results


def dedup_signals(mask, min_bars=5):
    """Remove signals too close together (minimum min_bars apart)"""
    result = mask.copy()
    indices = mask[mask].index.tolist()
    last_signal = -min_bars - 1

    for idx in indices:
        if idx - last_signal < min_bars:
            result.iloc[idx] = False
        else:
            last_signal = idx

    return result


def print_metrics(m):
    """Print strategy metrics"""
    print(f"  Trades: {m['trades']} ({m['trades_per_day']:.1f}/day) | "
          f"WR: {m['win_rate']:.1f}% | PF: {m['pf']:.2f}")
    print(f"  Avg Win: +{m['avg_win']:.1f} | Avg Loss: {m['avg_loss']:.1f} | "
          f"Total PnL: {m['total_pnl']:+.1f} pts")
    print(f"  AM: {m['am_trades']} trades | PM: {m['pm_trades']} trades | "
          f"Trail activated: {m['trail_pct']:.0f}%")
    print(f"  Max DD: {m['max_dd']:.1f} pts | Session close exits: {m['session_close_pct']:.0f}%")


def monthly_breakdown(trades_df, name="Strategy"):
    """Show monthly PnL breakdown"""
    if len(trades_df) == 0:
        return
    trades_df = trades_df.copy()
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    monthly = trades_df.groupby('month').agg(
        trades=('pnl', 'count'),
        pnl=('pnl', 'sum'),
        wr=('pnl', lambda x: (x > 0).mean() * 100)
    )
    print(f"\n  Monthly breakdown ({name}):")
    for idx, row in monthly.iterrows():
        bar = "+" * int(max(0, row['pnl'])/3) + "-" * int(max(0, -row['pnl'])/3)
        print(f"    {idx}: {row['trades']:3.0f}T WR={row['wr']:4.1f}% PnL={row['pnl']:+7.1f} {bar}")


# === MAIN ===

if __name__ == "__main__":
    print("="*70)
    print("VN30F1M EDGE RESEARCH: 9+ PT INTRADAY MOVES")
    print("Following edge-researcher skill methodology")
    print("Sessions: 9:00-11:30 (AM) + 13:00-14:30 (PM)")
    print("Cost: 1.74 pts | No overnight | Close by 14:28")
    print("="*70)

    # Step 1: Load data
    df = load_data(days=180)

    # Step 2: Add indicators
    print("\nComputing indicators (ATR, ADX, EMA, body ratio)...")
    df = add_indicators(df)

    # Step 3: Detect patterns
    print("Detecting compression zones (3 bars, range < 0.7x ATR)...")
    df = detect_compression(df, n_bars=3, threshold=0.7)
    print(f"  Compression bars: {df['compressed_3bar'].sum()} ({df['compressed_3bar'].mean()*100:.1f}% of all bars)")

    print("Detecting Session ORB (2-bar, max range 7 pts)...")
    df = detect_orb(df, n_bars=2, max_range=7)
    print(f"  ORB breakouts: {df['is_orb_breakout'].sum()}")

    # Step 4: Find all 9+ pt moves (within same session)
    print("\nScanning for 9+ pt moves within sessions (max 24 bars = 2h)...")
    moves_df = find_moves_9plus(df, target_pts=9, max_bars=24)

    # Step 5: Analyze conditions before moves
    analyze_pre_move_conditions(df, moves_df)

    # Step 6: Test strategies
    results = run_strategy_comparison(df)

    # Step 7: Best strategy details
    if results:
        profitable = [r for r in results if r['total_pnl'] > 0]
        if profitable:
            best = max(profitable, key=lambda x: x['pf'])
            print(f"\n{'='*70}")
            print(f"BEST STRATEGY (by PF): {best['name']}")
            print(f"  PF={best['pf']:.2f} | WR={best['win_rate']:.1f}% | "
                  f"PnL={best['total_pnl']:+.1f} ({best['pnl_per_day']:+.2f}/day)")
            print(f"{'='*70}")

    print("\n" + "="*70)
    print("DONE — Next: run bt_robust.py for walk-forward validation")
    print("="*70)
