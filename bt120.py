"""Quick 120-day backtest: WR per combo."""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from src.data_fetcher import DataFetcher
from src.signals import COMBO_PRESETS, generate_combined_signals

SYMBOL = 'VN30F1M'
VN_TZ = timezone(timedelta(hours=7))
now = datetime.now(VN_TZ)
end_date = now.strftime('%Y-%m-%d')
start_date = (now - timedelta(days=120)).strftime('%Y-%m-%d')

COMMISSION = 0.94
SL_MULT = 1.5
TP_MULT = 3.0
MAX_HOLD = 30


def sim_fixed(sig_df):
    """Simulate with intraday-only constraint: force close at EOD (14:30)."""
    trades = []
    i = 0
    n = len(sig_df)
    # Pre-compute dates for EOD check
    if 'time' in sig_df.columns:
        dates = pd.to_datetime(sig_df['time']).dt.date
    else:
        dates = sig_df.index.date if hasattr(sig_df.index, 'date') else None

    while i < n:
        sig = int(sig_df.iloc[i].get('signal', 0))
        if sig == 0:
            i += 1
            continue
        atr = float(sig_df.iloc[i].get('atr', 0))
        if atr <= 0:
            i += 1
            continue
        entry = float(sig_df.iloc[i]['close'])
        entry_date = dates.iloc[i] if dates is not None else None
        d = sig
        sl = entry - d * SL_MULT * atr
        tp = entry + d * TP_MULT * atr
        exit_price = entry
        reason = 'TO'
        bars = 0
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            bar = sig_df.iloc[j]
            bars += 1
            # Force close if next bar is a different day (overnight)
            if dates is not None and dates.iloc[j] != entry_date:
                exit_price = float(sig_df.iloc[j - 1]['close'])
                reason = 'EOD'
                break
            if d == 1:
                if float(bar['low']) <= sl:
                    exit_price = sl; reason = 'SL'; break
                if float(bar['high']) >= tp:
                    exit_price = tp; reason = 'TP'; break
            else:
                if float(bar['high']) >= sl:
                    exit_price = sl; reason = 'SL'; break
                if float(bar['low']) <= tp:
                    exit_price = tp; reason = 'TP'; break
        else:
            exit_price = float(sig_df.iloc[min(i + MAX_HOLD, n - 1)]['close'])
        pnl = d * (exit_price - entry) - COMMISSION
        trades.append({'pnl': pnl, 'reason': reason})
        i += max(bars, 1) + 1
    return trades


fetcher = DataFetcher()
results = []

for tf in ['5m', '15m']:
    print(f'Fetching {tf}...', flush=True)
    df = fetcher.get_futures_ohlcv(SYMBOL, start_date, end_date, interval=tf)
    if df is None or df.empty:
        print(f'  No data for {tf}')
        continue
    print(f'  Got {len(df)} bars', flush=True)

    active_combos = [(n, p) for n, p in COMBO_PRESETS.items() if p.get('primary')]

    for combo_name, preset in active_combos:
        enabled = {c: True for c in
                   preset.get('primary', []) + preset.get('confirm', []) + preset.get('gate', [])}
        try:
            sig_df = generate_combined_signals(
                df.copy(), fast_ma=10, slow_ma=20, rsi_period=7,
                oversold=35, overbought=70,
                macd_fast=12, macd_slow=26, macd_signal=9,
                vol_mult=1.5, enabled=enabled, combo_mode=combo_name,
            )
        except Exception as e:
            print(f'  ERROR {combo_name}: {e}')
            continue

        trades = sim_fixed(sig_df)
        if not trades:
            continue

        combo_short = combo_name.split(':')[0].strip()
        n_t = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = n_t - wins
        wr = wins / n_t * 100
        total_pnl = sum(t['pnl'] for t in trades)
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins else 0
        avg_loss = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if losses else 0.01
        pf = (avg_win * wins) / max(0.01, avg_loss * losses)
        tp_hits = sum(1 for t in trades if t['reason'] == 'TP')
        sl_hits = sum(1 for t in trades if t['reason'] == 'SL')

        results.append({
            'combo': combo_short, 'tf': tf, 'trades': n_t, 'wins': wins,
            'wr': wr, 'pnl': total_pnl, 'pf': pf,
            'avg_win': avg_win, 'avg_loss': avg_loss,
            'tp': tp_hits, 'sl': sl_hits,
        })
        print(f'  {combo_short}/{tf}: {n_t}T WR={wr:.0f}% PnL={total_pnl:+.1f}', flush=True)

# Print results
rdf = pd.DataFrame(results)
if rdf.empty:
    print('No results!')
else:
    rdf = rdf.sort_values('pnl', ascending=False)

    print()
    print('=' * 115)
    print(f'  120-DAY BACKTEST: ALL COMBOS x [5m, 15m] | Fixed SL={SL_MULT}x / TP={TP_MULT}x ATR')
    print(f'  Period: {start_date} -> {end_date}')
    print('=' * 115)
    print(f'{"Combo":<8}{"TF":<5}{"Trades":<8}{"Wins":<6}{"WR%":<7}{"PF":<6}{"TP":<5}{"SL":<5}'
          f'{"AvgWin":<8}{"AvgLoss":<9}{"PnL(pts)":<11}{"VND(M)":<9}')
    print('-' * 115)

    for _, r in rdf.iterrows():
        vnd = r['pnl'] * 100_000 / 1e6
        print(f'{r["combo"]:<8}{r["tf"]:<5}{r["trades"]:<8}{r["wins"]:<6}'
              f'{r["wr"]:<7.1f}{r["pf"]:<6.2f}{r["tp"]:<5}{r["sl"]:<5}'
              f'{r["avg_win"]:<8.1f}{r["avg_loss"]:<9.1f}{r["pnl"]:<+11.1f}{vnd:<+9.1f}')

    print('=' * 115)

    # Summary: best TF per combo
    print()
    print('  === BEST TF PER COMBO (ranked by PnL) ===')
    print(f'{"Combo":<8}{"TF":<6}{"Trades":<8}{"WR%":<7}{"PF":<6}{"PnL":<11}{"VND(M)":<9}{"Verdict":<10}')
    print('-' * 75)
    best = rdf.loc[rdf.groupby('combo')['pnl'].idxmax()]
    best = best.sort_values('pnl', ascending=False)
    for _, r in best.iterrows():
        vnd = r['pnl'] * 100_000 / 1e6
        if r['wr'] >= 55 and r['pf'] >= 1.3:
            verdict = 'STRONG'
        elif r['wr'] >= 45 and r['pf'] >= 1.0:
            verdict = 'OK'
        elif r['pnl'] > 0:
            verdict = 'MARGINAL'
        else:
            verdict = 'WEAK'
        print(f'{r["combo"]:<8}{r["tf"]:<6}{r["trades"]:<8}{r["wr"]:<7.1f}'
              f'{r["pf"]:<6.2f}{r["pnl"]:<+11.1f}{vnd:<+9.1f}{verdict:<10}')
    print('-' * 75)
