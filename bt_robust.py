"""Walk-Forward Robust Backtest with Slippage & Monte Carlo validation."""
import sys, io, warnings, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from src.data_fetcher import DataFetcher
from src.signals import COMBO_PRESETS, generate_combined_signals

# ─── Configuration ───────────────────────────────────────────────────────────
SYMBOL = 'VN30F1M'
VN_TZ = timezone(timedelta(hours=7))
LOOKBACK_DAYS = 240
IS_DAYS = 60
OOS_DAYS = 30
STEP_DAYS = 30

COMMISSION = 0.94
SLIPPAGE_ENTRY = 0.5
SLIPPAGE_EXIT = 0.3
MAX_HOLD_DEFAULT = 30
SL_MULT_DEFAULT = 1.5
TP_MULT_DEFAULT = 3.0

MC_ITERATIONS = 100
OOS_PF_THRESHOLD = 1.3
OOS_WR_THRESHOLD = 50.0
MC_P_THRESHOLD = 0.05
MIN_TRADES_PER_WINDOW = 3

TIMEFRAMES = ['5m', '15m']


def load_combo_risk() -> dict:
    cfg_path = Path(__file__).parent / 'strategy_config.yaml'
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('combo_risk', {})
    return {}


COMBO_RISK = load_combo_risk()


def get_combo_params(combo_short: str, tf: str) -> dict:
    key = f"{combo_short}/{tf}"
    risk = COMBO_RISK.get(key, {})
    return {
        'sl_mult': risk.get('sl_atr_mult', SL_MULT_DEFAULT),
        'tp_mult': risk.get('tp_atr_mult', TP_MULT_DEFAULT),
        'max_hold': risk.get('max_hold', MAX_HOLD_DEFAULT),
        'trailing_offset': risk.get('trailing_offset', 0),
        'trailing_step': risk.get('trailing_step', 0),
    }


# ─── Core Simulation ─────────────────────────────────────────────────────────

def sim_robust(sig_df: pd.DataFrame, sl_mult: float = 1.5, tp_mult: float = 3.0,
               max_hold: int = 30, slippage_entry: float = SLIPPAGE_ENTRY,
               slippage_exit: float = SLIPPAGE_EXIT, commission: float = COMMISSION,
               trailing_offset: float = 0, trailing_step: float = 0) -> list[dict]:
    """Simulate trades with slippage, commission, and optional trailing SL."""
    trades = []
    i = 0
    n = len(sig_df)

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

        raw_entry = float(sig_df.iloc[i]['close'])
        entry_date = dates.iloc[i] if dates is not None else None
        d = sig  # 1=BUY, -1=SELL

        entry = raw_entry + d * slippage_entry
        sl = entry - d * sl_mult * atr
        tp = entry + d * tp_mult * atr

        use_trailing = trailing_offset > 0 and trailing_step > 0
        trail_activated = False
        best_price = entry

        exit_price = entry
        reason = 'TO'
        bars = 0

        for j in range(i + 1, min(i + max_hold + 1, n)):
            bar = sig_df.iloc[j]
            bars += 1

            if dates is not None and dates.iloc[j] != entry_date:
                exit_price = float(sig_df.iloc[j - 1]['close'])
                reason = 'EOD'
                break

            high_j = float(bar['high'])
            low_j = float(bar['low'])
            close_j = float(bar['close'])

            if use_trailing:
                if d == 1:
                    best_price = max(best_price, high_j)
                else:
                    best_price = min(best_price, low_j)

                unrealized = d * (best_price - entry)
                if not trail_activated and unrealized >= trailing_offset * atr:
                    trail_activated = True
                    new_sl = entry + d * trailing_step * atr
                    if d == 1 and new_sl > sl:
                        sl = new_sl
                    elif d == -1 and new_sl < sl:
                        sl = new_sl

                if trail_activated:
                    if d == 1:
                        new_trail = best_price - trailing_step * atr
                        if new_trail > sl:
                            sl = new_trail
                    else:
                        new_trail = best_price + trailing_step * atr
                        if new_trail < sl:
                            sl = new_trail

            if d == 1:
                if low_j <= sl:
                    exit_price = sl - slippage_exit
                    reason = 'SL'
                    break
                if high_j >= tp:
                    exit_price = tp - slippage_exit
                    reason = 'TP'
                    break
            else:
                if high_j >= sl:
                    exit_price = sl + slippage_exit
                    reason = 'SL'
                    break
                if low_j <= tp:
                    exit_price = tp + slippage_exit
                    reason = 'TP'
                    break
        else:
            exit_price = float(sig_df.iloc[min(i + max_hold, n - 1)]['close'])
            reason = 'TO'

        pnl = d * (exit_price - entry) - commission
        trades.append({
            'pnl': pnl,
            'reason': reason,
            'bars_held': bars,
            'entry': entry,
            'exit': exit_price,
            'direction': d,
        })
        i += max(bars, 1) + 1

    return trades


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {'n_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'total_pnl': 0, 'avg_win': 0, 'avg_loss': 0,
                'max_drawdown': 0, 'expectancy': 0, 'sharpe': 0}

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n = len(pnls)
    wr = len(wins) / n * 100 if n else 0
    total_pnl = sum(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean([abs(l) for l in losses]) if losses else 0.01
    pf = (sum(wins)) / max(0.01, abs(sum(losses)))

    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) else 0

    std_pnl = np.std(pnls) if len(pnls) > 1 else 1
    sharpe = (np.mean(pnls) / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0
    expectancy = np.mean(pnls)

    return {
        'n_trades': n,
        'win_rate': wr,
        'profit_factor': pf,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_dd,
        'expectancy': expectancy,
        'sharpe': sharpe,
    }


# ─── Walk-Forward Windows ────────────────────────────────────────────────────

def generate_walk_forward_windows(total_start, total_end,
                                  is_days=IS_DAYS, oos_days=OOS_DAYS,
                                  step_days=STEP_DAYS) -> list[dict]:
    windows = []
    cursor = total_start
    wid = 0
    while True:
        is_start = cursor
        is_end = cursor + timedelta(days=is_days - 1)
        oos_start = is_end + timedelta(days=1)
        oos_end = oos_start + timedelta(days=oos_days - 1)
        if oos_end > total_end:
            break
        windows.append({
            'window_id': wid,
            'is_start': is_start,
            'is_end': is_end,
            'oos_start': oos_start,
            'oos_end': oos_end,
        })
        cursor += timedelta(days=step_days)
        wid += 1
    return windows


def slice_window(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    if 'time' in df.columns:
        mask = (pd.to_datetime(df['time']).dt.date >= start_date) & \
               (pd.to_datetime(df['time']).dt.date <= end_date)
    else:
        mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    return df.loc[mask].copy().reset_index(drop=True)


# ─── Monte Carlo ─────────────────────────────────────────────────────────────

def monte_carlo_test(sig_df: pd.DataFrame, real_trades: list[dict],
                     sl_mult: float, tp_mult: float, max_hold: int,
                     n_iterations: int = MC_ITERATIONS,
                     trailing_offset: float = 0,
                     trailing_step: float = 0) -> dict:
    """Shuffle signal timing to test if combo has genuine alpha."""
    real_pnl = sum(t['pnl'] for t in real_trades)
    if not real_trades:
        return {'p_value': 1.0, 'real_pnl': 0, 'mean_random_pnl': 0,
                'std_random_pnl': 0, 'percentile_rank': 50}

    signal_mask = sig_df['signal'] != 0
    n_signals = int(signal_mask.sum())
    if n_signals < 2:
        return {'p_value': 1.0, 'real_pnl': real_pnl, 'mean_random_pnl': 0,
                'std_random_pnl': 0, 'percentile_rank': 50}

    signal_directions = sig_df.loc[signal_mask, 'signal'].values
    valid_indices = list(range(30, len(sig_df) - max_hold))
    if len(valid_indices) < n_signals:
        return {'p_value': 1.0, 'real_pnl': real_pnl, 'mean_random_pnl': 0,
                'std_random_pnl': 0, 'percentile_rank': 50}

    random_pnls = []
    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        shuffled = sig_df.copy()
        shuffled['signal'] = 0
        chosen_idx = rng.choice(valid_indices, size=min(n_signals, len(valid_indices)),
                                replace=False)
        chosen_idx.sort()
        dirs = rng.choice(signal_directions, size=len(chosen_idx), replace=True)
        for idx, d in zip(chosen_idx, dirs):
            shuffled.iloc[idx, shuffled.columns.get_loc('signal')] = d

        mc_trades = sim_robust(shuffled, sl_mult=sl_mult, tp_mult=tp_mult,
                               max_hold=max_hold, trailing_offset=trailing_offset,
                               trailing_step=trailing_step)
        random_pnls.append(sum(t['pnl'] for t in mc_trades))

    random_pnls = np.array(random_pnls)
    p_value = float(np.mean(random_pnls >= real_pnl))
    mean_rand = float(np.mean(random_pnls))
    std_rand = float(np.std(random_pnls)) if len(random_pnls) > 1 else 0
    percentile = float(np.mean(random_pnls < real_pnl) * 100)

    return {
        'p_value': p_value,
        'real_pnl': real_pnl,
        'mean_random_pnl': mean_rand,
        'std_random_pnl': std_rand,
        'percentile_rank': percentile,
    }


# ─── Decay Analysis ──────────────────────────────────────────────────────────

def decay_analysis(oos_window_metrics: list[dict]) -> dict:
    """Check if performance degrades over time across OOS windows."""
    pfs = [m['profit_factor'] for m in oos_window_metrics if m['n_trades'] >= MIN_TRADES_PER_WINDOW]
    if len(pfs) < 3:
        return {'slope': 0, 'degrading': False, 'first_half_pf': 0, 'second_half_pf': 0}

    x = np.arange(len(pfs), dtype=float)
    slope = float(np.polyfit(x, pfs, 1)[0])

    mid = len(pfs) // 2
    first_half = np.mean(pfs[:mid])
    second_half = np.mean(pfs[mid:])

    return {
        'slope': slope,
        'degrading': slope < -0.1,
        'first_half_pf': float(first_half),
        'second_half_pf': float(second_half),
    }


# ─── Grading ─────────────────────────────────────────────────────────────────

def grade_combo(oos_metrics: dict, mc_result: dict, decay_result: dict) -> str:
    oos_pf = oos_metrics.get('profit_factor', 0)
    oos_wr = oos_metrics.get('win_rate', 0)
    mc_p = mc_result.get('p_value', 1.0)
    degrading = decay_result.get('degrading', False)

    if oos_pf >= 1.5 and oos_wr >= 55 and mc_p < 0.03 and not degrading:
        return 'A'
    elif oos_pf >= OOS_PF_THRESHOLD and oos_wr >= OOS_WR_THRESHOLD and mc_p < MC_P_THRESHOLD:
        return 'B'
    elif oos_pf >= 1.0 and oos_wr >= 45:
        return 'C'
    else:
        return 'F'


# ─── Walk-Forward Orchestrator ───────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, combo_name: str, tf: str,
                     windows: list[dict]) -> dict:
    """Run full walk-forward analysis for one combo/TF pair."""
    combo_short = combo_name.split(':')[0].strip()
    params = get_combo_params(combo_short, tf)

    preset = COMBO_PRESETS.get(combo_name, {})
    enabled = {c: True for c in
               preset.get('primary', []) + preset.get('confirm', []) + preset.get('gate', [])}

    is_all_trades = []
    oos_all_trades = []
    oos_window_metrics = []

    for w in windows:
        is_df = slice_window(df, w['is_start'], w['is_end'])
        oos_df = slice_window(df, w['oos_start'], w['oos_end'])

        if len(is_df) < 50 or len(oos_df) < 20:
            continue

        try:
            is_sig = generate_combined_signals(
                is_df.copy(), fast_ma=10, slow_ma=20, rsi_period=7,
                oversold=35, overbought=70, macd_fast=12, macd_slow=26,
                macd_signal=9, vol_mult=1.5, enabled=enabled, combo_mode=combo_name)

            oos_sig = generate_combined_signals(
                oos_df.copy(), fast_ma=10, slow_ma=20, rsi_period=7,
                oversold=35, overbought=70, macd_fast=12, macd_slow=26,
                macd_signal=9, vol_mult=1.5, enabled=enabled, combo_mode=combo_name)
        except Exception:
            continue

        is_trades = sim_robust(is_sig, sl_mult=params['sl_mult'],
                               tp_mult=params['tp_mult'], max_hold=params['max_hold'],
                               trailing_offset=params['trailing_offset'],
                               trailing_step=params['trailing_step'])
        oos_trades = sim_robust(oos_sig, sl_mult=params['sl_mult'],
                                tp_mult=params['tp_mult'], max_hold=params['max_hold'],
                                trailing_offset=params['trailing_offset'],
                                trailing_step=params['trailing_step'])

        is_all_trades.extend(is_trades)
        oos_all_trades.extend(oos_trades)
        oos_window_metrics.append(compute_metrics(oos_trades))

    is_metrics = compute_metrics(is_all_trades)
    oos_metrics = compute_metrics(oos_all_trades)

    # Monte Carlo on aggregated OOS signals
    mc_result = {'p_value': 1.0, 'real_pnl': 0, 'mean_random_pnl': 0,
                 'std_random_pnl': 0, 'percentile_rank': 50}
    if oos_all_trades and oos_metrics['n_trades'] >= MIN_TRADES_PER_WINDOW:
        last_oos_window = None
        for w in reversed(windows):
            oos_df = slice_window(df, w['oos_start'], w['oos_end'])
            if len(oos_df) >= 20:
                last_oos_window = oos_df
                break
        if last_oos_window is not None:
            try:
                oos_sig = generate_combined_signals(
                    last_oos_window.copy(), fast_ma=10, slow_ma=20, rsi_period=7,
                    oversold=35, overbought=70, macd_fast=12, macd_slow=26,
                    macd_signal=9, vol_mult=1.5, enabled=enabled, combo_mode=combo_name)
                oos_trades_mc = sim_robust(oos_sig, sl_mult=params['sl_mult'],
                                           tp_mult=params['tp_mult'],
                                           max_hold=params['max_hold'],
                                           trailing_offset=params['trailing_offset'],
                                           trailing_step=params['trailing_step'])
                mc_result = monte_carlo_test(
                    oos_sig, oos_trades_mc,
                    sl_mult=params['sl_mult'], tp_mult=params['tp_mult'],
                    max_hold=params['max_hold'],
                    trailing_offset=params['trailing_offset'],
                    trailing_step=params['trailing_step'])
            except Exception:
                pass

    decay_result = decay_analysis(oos_window_metrics)
    grade = grade_combo(oos_metrics, mc_result, decay_result)

    return {
        'combo': combo_short,
        'combo_full': combo_name,
        'tf': tf,
        'is_metrics': is_metrics,
        'oos_metrics': oos_metrics,
        'mc_result': mc_result,
        'decay_result': decay_result,
        'grade': grade,
        'n_windows': len(oos_window_metrics),
        'params': params,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    now = datetime.now(VN_TZ)
    end_date = now.strftime('%Y-%m-%d')
    start_date = (now - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')

    print(f'Walk-Forward Robust Backtest: {start_date} -> {end_date}')
    print(f'  IS={IS_DAYS}d / OOS={OOS_DAYS}d / Step={STEP_DAYS}d')
    print(f'  Slippage: entry={SLIPPAGE_ENTRY}, exit={SLIPPAGE_EXIT}, commission={COMMISSION}')
    print(f'  MC iterations: {MC_ITERATIONS}')
    print()

    fetcher = DataFetcher()
    all_data = {}

    for tf in TIMEFRAMES:
        print(f'Fetching {tf} data ({LOOKBACK_DAYS} days)...', flush=True)
        df = fetcher.get_futures_ohlcv(SYMBOL, start_date, end_date, interval=tf)
        if df is None or df.empty:
            print(f'  ERROR: No data for {tf}')
            continue
        print(f'  Got {len(df)} bars', flush=True)
        all_data[tf] = df

    if not all_data:
        print('ERROR: No data fetched. Exiting.')
        return

    first_df = list(all_data.values())[0]
    if 'time' in first_df.columns:
        data_start = pd.to_datetime(first_df['time']).dt.date.min()
        data_end = pd.to_datetime(first_df['time']).dt.date.max()
    else:
        data_start = first_df.index.date.min()
        data_end = first_df.index.date.max()

    windows = generate_walk_forward_windows(data_start, data_end)
    print(f'  Generated {len(windows)} walk-forward windows')
    for w in windows:
        print(f'    W{w["window_id"]}: IS[{w["is_start"]} -> {w["is_end"]}] '
              f'OOS[{w["oos_start"]} -> {w["oos_end"]}]')
    print()

    active_combos = [(n, p) for n, p in COMBO_PRESETS.items() if p.get('primary')]
    results = []

    for tf in TIMEFRAMES:
        if tf not in all_data:
            continue
        df = all_data[tf]
        print(f'─── Testing TF={tf} ({len(active_combos)} combos) ───', flush=True)

        for combo_name, preset in active_combos:
            combo_short = combo_name.split(':')[0].strip()
            print(f'  {combo_short}/{tf}...', end=' ', flush=True)

            result = run_walk_forward(df, combo_name, tf, windows)
            results.append(result)

            oos = result['oos_metrics']
            mc = result['mc_result']
            print(f'OOS: {oos["n_trades"]}T WR={oos["win_rate"]:.0f}% '
                  f'PF={oos["profit_factor"]:.2f} '
                  f'MC_p={mc["p_value"]:.3f} → {result["grade"]}', flush=True)

    # Print final report
    print()
    print('=' * 130)
    print('  WALK-FORWARD ROBUSTNESS REPORT')
    print(f'  Period: {start_date} -> {end_date} | Windows: {len(windows)}')
    print(f'  Slippage: {SLIPPAGE_ENTRY}+{SLIPPAGE_EXIT} pts | Commission: {COMMISSION} pts')
    print('=' * 130)
    print(f'{"Combo":<8}{"TF":<5}{"Win":<4}'
          f'{"IS_Tr":<7}{"IS_WR":<7}{"IS_PF":<7}{"IS_PnL":<9}'
          f'{"OOS_Tr":<8}{"OOS_WR":<8}{"OOS_PF":<8}{"OOS_PnL":<10}'
          f'{"MC_p":<7}{"Decay":<7}{"Grade":<6}')
    print('─' * 130)

    results.sort(key=lambda r: r['oos_metrics']['total_pnl'], reverse=True)

    for r in results:
        is_m = r['is_metrics']
        oos_m = r['oos_metrics']
        mc = r['mc_result']
        decay = r['decay_result']
        decay_str = 'Yes' if decay['degrading'] else 'No'

        print(f'{r["combo"]:<8}{r["tf"]:<5}{r["n_windows"]:<4}'
              f'{is_m["n_trades"]:<7}{is_m["win_rate"]:<7.1f}{is_m["profit_factor"]:<7.2f}'
              f'{is_m["total_pnl"]:<+9.1f}'
              f'{oos_m["n_trades"]:<8}{oos_m["win_rate"]:<8.1f}{oos_m["profit_factor"]:<8.2f}'
              f'{oos_m["total_pnl"]:<+10.1f}'
              f'{mc["p_value"]:<7.3f}{decay_str:<7}{r["grade"]:<6}')

    print('=' * 130)

    # Summary
    grades = {'A': [], 'B': [], 'C': [], 'F': []}
    for r in results:
        grades[r['grade']].append(f'{r["combo"]}/{r["tf"]}')

    print()
    print('  SUMMARY:')
    print(f'    Grade A (STRONG):   {len(grades["A"])} combos - {", ".join(grades["A"]) or "none"}')
    print(f'    Grade B (PASS):     {len(grades["B"])} combos - {", ".join(grades["B"]) or "none"}')
    print(f'    Grade C (MARGINAL): {len(grades["C"])} combos - {", ".join(grades["C"]) or "none"}')
    print(f'    Grade F (FAIL):     {len(grades["F"])} combos - {", ".join(grades["F"]) or "none"}')
    print()
    print(f'  RECOMMENDATION: Keep only A+B combos ({len(grades["A"]) + len(grades["B"])} total)')
    print()

    # Save report
    report_path = Path(__file__).parent / 'logs' / 'robustness_report.json'
    report_path.parent.mkdir(exist_ok=True)
    report_data = {
        'generated': now.isoformat(),
        'config': {
            'lookback_days': LOOKBACK_DAYS,
            'is_days': IS_DAYS,
            'oos_days': OOS_DAYS,
            'slippage_entry': SLIPPAGE_ENTRY,
            'slippage_exit': SLIPPAGE_EXIT,
            'commission': COMMISSION,
            'mc_iterations': MC_ITERATIONS,
        },
        'results': [{
            'combo': r['combo'],
            'tf': r['tf'],
            'grade': r['grade'],
            'is_metrics': r['is_metrics'],
            'oos_metrics': r['oos_metrics'],
            'mc_p_value': r['mc_result']['p_value'],
            'decay_slope': r['decay_result']['slope'],
            'params': r['params'],
        } for r in results],
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f'  Report saved: {report_path}')


if __name__ == '__main__':
    main()
