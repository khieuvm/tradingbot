"""Base strategy class and shared exit engine for all 22 strategies."""
import numpy as np
import pandas as pd

COST = 0.96

EXIT_PRESETS = {
    'trend': {'sl_mult': 2.0, 'trail_pts': 5.0, 'trail_mult': 2.0, 'tp_mult': None,
              'be_trigger': 4.0, 'trail_tighten_pts': 8.0, 'trail_tighten_mult': 1.2,
              'max_hold_am': 30, 'max_hold_pm': 15},
    'trend_tight': {'sl_mult': 2.0, 'trail_pts': 8.0, 'trail_mult': 0.5, 'tp_mult': None,
                    'be_trigger': 4.0, 'trail_tighten_pts': None, 'trail_tighten_mult': None,
                    'max_hold_am': 30, 'max_hold_pm': 15},
    'mean_reversion': {'sl_mult': 1.0, 'trail_pts': 3.0, 'trail_mult': 1.0, 'tp_mult': 3.0,
                       'be_trigger': None, 'trail_tighten_pts': None, 'trail_tighten_mult': None,
                       'max_hold_am': 15, 'max_hold_pm': 8},
    'breakout': {'sl_mult': 1.5, 'trail_pts': 5.0, 'trail_mult': 1.8, 'tp_mult': 4.0,
                 'be_trigger': 4.0, 'trail_tighten_pts': None, 'trail_tighten_mult': None,
                 'max_hold_am': 24, 'max_hold_pm': 12},
    'reversal': {'sl_mult': 1.2, 'trail_pts': 4.0, 'trail_mult': 1.5, 'tp_mult': 3.5,
                 'be_trigger': 3.0, 'trail_tighten_pts': None, 'trail_tighten_mult': None,
                 'max_hold_am': 20, 'max_hold_pm': 10},
    'scalp': {'sl_mult': 0.8, 'trail_pts': 2.0, 'trail_mult': 0.8, 'tp_mult': 2.0,
              'be_trigger': None, 'trail_tighten_pts': None, 'trail_tighten_mult': None,
              'max_hold_am': 10, 'max_hold_pm': 6},
}

AM_CUTOFF = 685   # 11:25
PM_CUTOFF = 865   # 14:25


class BaseStrategy:
    """Abstract base for all 22 strategies."""
    name = "base"
    exit_type = "breakout"  # override in subclass

    def prepare(self, df):
        """Pre-compute expensive indicators once before detect loop. Override in subclass."""
        pass

    def detect(self, df, idx):
        """Detect signal at bar idx. Return +1 (BUY), -1 (SELL), or 0 (no signal)."""
        raise NotImplementedError

    def get_exit_params(self, session):
        preset = EXIT_PRESETS[self.exit_type]
        max_hold = preset['max_hold_am'] if session == 'AM' else preset['max_hold_pm']
        return {
            'sl_mult': preset['sl_mult'],
            'trail_pts': preset['trail_pts'],
            'trail_mult': preset['trail_mult'],
            'tp_mult': preset['tp_mult'],
            'be_trigger': preset.get('be_trigger'),
            'trail_tighten_pts': preset.get('trail_tighten_pts'),
            'trail_tighten_mult': preset.get('trail_tighten_mult'),
            'max_hold': max_hold,
        }


def simulate_trade(df, entry_idx, direction, atr, session, exit_params):
    """Simulate a single trade from entry_idx with given exit parameters.

    Returns: (exit_price, exit_reason, mfe, bars_held)
    """
    entry = df['close'].iloc[entry_idx]
    sl_mult = exit_params['sl_mult']
    trail_pts = exit_params['trail_pts']
    trail_mult = exit_params['trail_mult']
    tp_mult = exit_params['tp_mult']
    max_hold = exit_params['max_hold']
    be_trigger = exit_params.get('be_trigger')
    trail_tighten_pts = exit_params.get('trail_tighten_pts')
    trail_tighten_mult = exit_params.get('trail_tighten_mult')
    cutoff = AM_CUTOFF if session == 'AM' else PM_CUTOFF

    sl = entry - direction * sl_mult * atr
    tp = entry + direction * tp_mult * atr if tp_mult else None
    best_price = entry
    mfe = 0.0
    trail_active = False
    be_active = False

    for j in range(1, min(max_hold + 1, len(df) - entry_idx)):
        bar_idx = entry_idx + j
        bar = df.iloc[bar_idx]
        bar_h = bar['high']
        bar_l = bar['low']
        bar_mins = bar['mins']

        # Session end
        if bar_mins >= cutoff:
            exit_price = bar['close']
            return exit_price, 'SESSION', mfe, j

        # Update MFE and best price
        if direction == 1:
            cur_mfe = bar_h - entry
            if bar_h > best_price:
                best_price = bar_h
        else:
            cur_mfe = entry - bar_l
            if bar_l < best_price:
                best_price = bar_l
        mfe = max(mfe, cur_mfe)

        # Breakeven activation
        if be_trigger and not be_active and mfe >= be_trigger:
            be_active = True
            if direction == 1:
                sl = max(sl, entry)
            else:
                sl = min(sl, entry)

        # Trail activation
        if mfe >= trail_pts:
            trail_active = True
            active_trail_mult = trail_mult
            if trail_tighten_pts and trail_tighten_mult and mfe >= trail_tighten_pts:
                active_trail_mult = trail_tighten_mult
            if direction == 1:
                new_sl = best_price - active_trail_mult * atr
                sl = max(sl, new_sl)
            else:
                new_sl = best_price + active_trail_mult * atr
                sl = min(sl, new_sl)

        # Check SL
        if direction == 1:
            if bar_l <= sl:
                reason = 'TRAIL' if trail_active else ('BE' if be_active else 'SL')
                return sl, reason, mfe, j
        else:
            if bar_h >= sl:
                reason = 'TRAIL' if trail_active else ('BE' if be_active else 'SL')
                return sl, reason, mfe, j

        # Check TP
        if tp:
            if direction == 1 and bar_h >= tp:
                return tp, 'TP', mfe, j
            elif direction == -1 and bar_l <= tp:
                return tp, 'TP', mfe, j

    # Max hold exit
    exit_price = df['close'].iloc[min(entry_idx + max_hold, len(df) - 1)]
    return exit_price, 'MAX_HOLD', mfe, max_hold


def run_backtest(df, strategy, dedup_bars=5):
    """Run a full backtest for a strategy on prepared dataframe.

    Args:
        df: DataFrame with [time, open, high, low, close, volume, atr, session, mins, date]
        strategy: BaseStrategy instance
        dedup_bars: minimum bars between signals

    Returns:
        trades: list of trade dicts
    """
    strategy.prepare(df)

    trades = []
    last_signal_idx = -dedup_bars - 1
    min_idx = 60  # warmup

    for i in range(min_idx, len(df) - 2):
        if i - last_signal_idx < dedup_bars:
            continue

        session = df['session'].iloc[i]
        if session not in ('AM', 'PM'):
            continue

        mins = df['mins'].iloc[i]
        if session == 'AM' and not (555 <= mins <= 645):
            continue
        if session == 'PM' and not (795 <= mins <= 855):
            continue

        atr = df['atr'].iloc[i]
        if pd.isna(atr) or atr < 1.5:
            continue

        direction = strategy.detect(df, i)
        if direction == 0:
            continue

        exit_params = strategy.get_exit_params(session)
        exit_price, reason, mfe, bars_held = simulate_trade(
            df, i, direction, atr, session, exit_params)

        pnl = direction * (exit_price - df['close'].iloc[i]) - COST

        trades.append({
            'date': str(df['date'].iloc[i]),
            'time': str(df['time'].iloc[i]),
            'session': session,
            'mins': mins,
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry': df['close'].iloc[i],
            'exit': exit_price,
            'atr': atr,
            'mfe': mfe,
            'pnl': pnl,
            'reason': reason,
            'bars': bars_held,
        })
        last_signal_idx = i

    return trades


def compute_metrics(trades):
    """Compute strategy performance metrics from trade list."""
    if not trades:
        return {'n': 0, 'wr': 0, 'pf': 0, 'pnl': 0, 'avg_pnl': 0, 'avg_mfe': 0, 'pnl_per_day': 0}

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n = len(trades)
    total_pnl = sum(pnls)

    dates = set(t['date'] for t in trades)
    n_days = max(len(dates), 1)

    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001

    return {
        'n': n,
        'wr': len(wins) / n * 100 if n > 0 else 0,
        'pf': gross_win / gross_loss if gross_loss > 0 else 99.0,
        'pnl': total_pnl,
        'avg_pnl': total_pnl / n if n > 0 else 0,
        'avg_mfe': np.mean([t['mfe'] for t in trades]) if trades else 0,
        'pnl_per_day': total_pnl / n_days,
    }
