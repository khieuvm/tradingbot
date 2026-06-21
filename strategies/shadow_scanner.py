"""Shadow Scanner — run profitable strategies on 1m/3m/5m and emit Telegram alerts.

Profitable strategies (June 2026 regime, SELL-dominant):
  1m: CHoCH (ADX>15), Momentum PM SELL (mins>=825)
  3m: Momentum AM SELL (trend_tight), Heikin Ashi PM SELL (mins>=825)
  5m: Volume PM SELL (ADX>=25), Confluence SELL (3+ agree + ADX>=20)

Day-of-week optimized strategies (32d backtest, May-Jun 2026):
  Mon AM: ORB Breakdown SELL (PF 2.01)   | Mon PM: Divergence SELL (PF 4.00)
  Tue AM: Momentum SELL (PF 2.31)        | Tue PM: MACD BUY (PF 1.89)
  Wed AM: ORB Breakdown SELL (PF 11.26)  | Wed PM: MACD BUY (PF 1.56)
  Thu: SKIP (choppy, no edge)
  Fri AM: Heikin Ashi BUY (PF 1.52)      | Fri PM: Momentum SELL (PF 7.19)

Opening strategies:
  AM: ORB Breakdown SELL (Mon+Wed), Crabel Stretch SELL (Tue+Wed)
  PM ORB: SELL Mon (PF6.52), BUY Tue (PF1.84), BUY Wed (PF2.30), SELL Fri (PF4.30)

Each strategy runs independently with its own dedup timer.
Signals are shadow-only (Telegram alert, no trade).
"""
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pandas_ta as ta

VN_TZ = timezone(timedelta(hours=7))


def vn_now() -> datetime:
    return datetime.now(VN_TZ)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add all required indicators to raw OHLCV dataframe."""
    df = df.reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]

    if 'time' not in df.columns and 'datetime' in df.columns:
        df.rename(columns={'datetime': 'time'}, inplace=True)

    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date
    df['mins'] = df['time'].dt.hour * 60 + df['time'].dt.minute

    am_mask = (df['mins'] >= 540) & (df['mins'] <= 690)
    pm_mask = (df['mins'] >= 780) & (df['mins'] <= 870)
    df = df[am_mask | pm_mask].reset_index(drop=True)

    df['session'] = 'AM'
    df.loc[df['mins'] >= 780, 'session'] = 'PM'

    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['range'] = df['high'] - df['low']

    df['ema8'] = ta.ema(df['close'], length=8)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema50'] = ta.ema(df['close'], length=50)

    bb = ta.bbands(df['close'], length=20, std=2.0)
    if bb is not None and len(bb.columns) >= 3:
        df['bb_lower'] = bb.iloc[:, 0]
        df['bb_mid'] = bb.iloc[:, 1]
        df['bb_upper'] = bb.iloc[:, 2]
    else:
        df['bb_lower'] = df['bb_mid'] = df['bb_upper'] = np.nan

    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is not None:
        df['adx'] = adx_df.iloc[:, 0]
        df['di_plus'] = adx_df.iloc[:, 1]
        df['di_minus'] = adx_df.iloc[:, 2]
    else:
        df['adx'] = df['di_plus'] = df['di_minus'] = np.nan

    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd_df is not None:
        df['macd_line'] = macd_df.iloc[:, 0]
        df['macd_signal'] = macd_df.iloc[:, 2]
    else:
        df['macd_line'] = df['macd_signal'] = np.nan

    stoch_df = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    if stoch_df is not None:
        df['stoch_k'] = stoch_df.iloc[:, 0]
        df['stoch_d'] = stoch_df.iloc[:, 1]
    else:
        df['stoch_k'] = df['stoch_d'] = np.nan

    df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
    df['obv'] = ta.obv(df['close'], df['volume'])

    return df


class StrategyShadowScanner:
    """Manages all shadow strategy instances with dedup and signal emission."""

    # Dedup seconds per timeframe (bars * bar_duration)
    DEDUP_1M = 5 * 60       # 5 bars * 1 min = 5 min
    DEDUP_3M = 8 * 180      # 8 bars * 3 min = 24 min (matches backtest dedup_bars=8)
    DEDUP_5M = 5 * 300      # 5 bars * 5 min = 25 min

    def __init__(self):
        self._last_signal: dict[str, float] = {}
        self._strategies_1m = self._load_1m_strategies()
        self._strategies_3m = self._load_3m_strategies()
        self._strategies_5m = self._load_5m_strategies()
        self._confluence_strategies = self._load_confluence_strategies()
        self._opening_strategies = self._load_opening_strategies()

    def _load_opening_strategies(self):
        """Load ORB Breakdown, Crabel Stretch (AM), and PM ORB strategies."""
        result = {}
        try:
            from strategies.orb_breakdown import ORBBreakdownStrategy
            result['orb_breakdown_sell'] = {
                'instance': ORBBreakdownStrategy(or_bars=15, weekday_filter=[0, 2]),
                'label': 'ORB Breakdown AM SELL (Mon+Wed, PF3.25)',
                'session': 'AM',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip ORB Breakdown: {e}")

        try:
            from strategies.crabel_stretch import CrabelStretchStrategy
            result['crabel_stretch_sell'] = {
                'instance': CrabelStretchStrategy(stretch_sma=5, stretch_pct=0.5,
                                                   weekday_filter=[1, 2]),
                'label': 'Crabel Stretch AM SELL (Tue+Wed, WR100%)',
                'session': 'AM',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip Crabel Stretch: {e}")

        # PM ORB strategies (13:00-13:14 opening range)
        try:
            from strategies.pm_orb import PMORBStrategy
            result['pm_orb_sell_mon'] = {
                'instance': PMORBStrategy(or_bars=15, direction=-1, weekday_filter=[0]),
                'label': 'PM ORB SELL Mon (PF6.52, WR75%)',
                'session': 'PM',
            }
            result['pm_orb_buy_tue'] = {
                'instance': PMORBStrategy(or_bars=15, direction=1, weekday_filter=[1]),
                'label': 'PM ORB BUY Tue (PF1.84, WR40%)',
                'session': 'PM',
            }
            result['pm_orb_buy_wed'] = {
                'instance': PMORBStrategy(or_bars=15, direction=1, weekday_filter=[2]),
                'label': 'PM ORB BUY Wed (PF2.30, WR50%)',
                'session': 'PM',
            }
            result['pm_orb_sell_fri'] = {
                'instance': PMORBStrategy(or_bars=15, direction=-1, weekday_filter=[4]),
                'label': 'PM ORB SELL Fri (PF4.30, WR33%)',
                'session': 'PM',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip PM ORB: {e}")

        return result

    def _load_1m_strategies(self):
        """Load strategies that fire on 1m."""
        result = {}
        try:
            from strategies.choch import CHoCHStrategy
            result['choch_1m'] = {
                'instance': CHoCHStrategy(),
                'sessions': ['AM', 'PM'],
                'directions': [-1, 1],
                'label': 'CHoCH 1m (ADX>15)',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip CHoCH 1m: {e}")

        try:
            from strategies.momentum_trend import MomentumTrendStrategy
            result['momentum_1m_pm_sell'] = {
                'instance': MomentumTrendStrategy(),
                'sessions': ['PM'],
                'directions': [-1],
                'label': 'Momentum PM SELL 1m (mins>=825)',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip Momentum 1m: {e}")

        # RAW EXIT strategies (SL=3.0xATR, no trail, session exit)
        # Validated on full 682d data
        try:
            from strategies.momentum_trend import MomentumTrendStrategy
            result['raw_momentum_1m_pm_buy'] = {
                'instance': MomentumTrendStrategy(),
                'sessions': ['PM'],
                'directions': [1],
                'label': 'RAW Momentum PM BUY 1m (PF1.39, +1.55/d)',
                'regime_filter': 'buy',
                'exit_mode': 'raw',
            }
            result['raw_momentum_1m_pm_sell'] = {
                'instance': MomentumTrendStrategy(),
                'sessions': ['PM'],
                'directions': [-1],
                'label': 'RAW Momentum PM SELL 1m (PF1.44, +0.76/d)',
                'regime_filter': 'sell',
                'exit_mode': 'raw',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip RAW Momentum 1m: {e}")

        try:
            from strategies.macd_cross import MACDCrossStrategy
            result['raw_macd_1m_pm_buy'] = {
                'instance': MACDCrossStrategy(),
                'sessions': ['PM'],
                'directions': [1],
                'label': 'RAW MACD PM BUY 1m (PF1.30, +0.44/d)',
                'regime_filter': 'buy',
                'exit_mode': 'raw',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip RAW MACD 1m: {e}")

        try:
            from strategies.divergence import DivergenceStrategy
            result['raw_divergence_1m_am_sell'] = {
                'instance': DivergenceStrategy(),
                'sessions': ['AM'],
                'directions': [-1],
                'label': 'RAW Divergence AM SELL 1m (PF1.51, +0.06/d)',
                'regime_filter': 'sell',
                'exit_mode': 'raw',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip RAW Divergence 1m: {e}")

        # ── DOW-filtered strategies (best per weekday, 32d backtest) ──
        try:
            from strategies.momentum_trend import MomentumTrendStrategy
            result['dow_momentum_am_sell_tue'] = {
                'instance': MomentumTrendStrategy(),
                'sessions': ['AM'],
                'directions': [-1],
                'weekdays': [1],  # Tue only
                'label': 'DOW Momentum AM SELL Tue (PF2.31, N=14)',
                'exit_mode': 'raw',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip DOW Momentum AM SELL Tue: {e}")

        try:
            from strategies.momentum_trend import MomentumTrendStrategy
            result['dow_momentum_pm_sell_monfri'] = {
                'instance': MomentumTrendStrategy(),
                'sessions': ['PM'],
                'directions': [-1],
                'weekdays': [0, 4],  # Mon+Fri
                'label': 'DOW Momentum PM SELL Mon+Fri (PF7.19)',
                'exit_mode': 'raw',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip DOW Momentum PM SELL Mon+Fri: {e}")

        try:
            from strategies.divergence import DivergenceStrategy
            result['dow_divergence_pm_sell_mon'] = {
                'instance': DivergenceStrategy(),
                'sessions': ['PM'],
                'directions': [-1],
                'weekdays': [0],  # Mon only
                'label': 'DOW Divergence PM SELL Mon (PF4.00)',
                'exit_mode': 'raw',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip DOW Divergence PM SELL Mon: {e}")

        try:
            from strategies.macd_cross import MACDCrossStrategy
            result['dow_macd_pm_buy_tuewed'] = {
                'instance': MACDCrossStrategy(),
                'sessions': ['PM'],
                'directions': [1],
                'weekdays': [1, 2],  # Tue+Wed
                'label': 'DOW MACD PM BUY Tue+Wed (PF1.73)',
                'exit_mode': 'raw',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip DOW MACD PM BUY Tue+Wed: {e}")

        try:
            from strategies.heikin_ashi import HeikinAshiStrategy
            result['dow_heikin_ashi_am_buy_fri'] = {
                'instance': HeikinAshiStrategy(),
                'sessions': ['AM'],
                'directions': [1],
                'weekdays': [4],  # Fri only
                'label': 'DOW Heikin Ashi AM BUY Fri (PF1.52, N=11)',
                'exit_mode': 'raw',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip DOW Heikin Ashi AM BUY Fri: {e}")

        return result

    def _load_3m_strategies(self):
        """Load strategies that fire on 3m."""
        result = {}
        try:
            from strategies.momentum_trend import MomentumTrendStrategy
            result['momentum_3m_am_sell'] = {
                'instance': MomentumTrendStrategy(),
                'sessions': ['AM'],
                'directions': [-1],
                'label': 'Momentum AM SELL 3m (trend_tight)',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip Momentum 3m: {e}")

        try:
            from strategies.heikin_ashi import HeikinAshiStrategy
            result['heikin_ashi_3m_pm_sell'] = {
                'instance': HeikinAshiStrategy(),
                'sessions': ['PM'],
                'directions': [-1],
                'label': 'Heikin Ashi PM SELL 3m (monitor)',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip Heikin Ashi 3m: {e}")

        return result

    def _load_5m_strategies(self):
        """Load strategies that fire on 5m."""
        result = {}
        try:
            from strategies.volume import VolumeStrategy
            result['volume_5m_pm_sell'] = {
                'instance': VolumeStrategy(),
                'sessions': ['PM'],
                'directions': [-1],
                'label': 'Volume PM SELL 5m (ADX>=25)',
            }
        except Exception as e:
            print(f"  [SHADOW] Skip Volume 5m: {e}")

        return result

    def _load_confluence_strategies(self):
        """Load ALL 22 strategies for confluence SELL detection on 5m."""
        try:
            from strategies.backtest_all import get_all_strategies
            return get_all_strategies()
        except Exception as e:
            print(f"  [SHADOW] Skip confluence strategies: {e}")
            return []

    def _is_deduped(self, key: str, dedup_seconds: int) -> bool:
        last = self._last_signal.get(key, 0)
        return (time.time() - last) < dedup_seconds

    def _mark_signal(self, key: str):
        self._last_signal[key] = time.time()

    def scan_1m(self, df_raw: pd.DataFrame) -> list[dict]:
        """Scan 1m data for CHoCH and Momentum PM SELL signals."""
        if df_raw is None or len(df_raw) < 60:
            return []

        try:
            df = prepare_df(df_raw.copy())
        except Exception as e:
            print(f"  [SHADOW 1m] prepare_df error: {e}")
            return []

        if len(df) < 60:
            return []

        idx = len(df) - 2  # last complete bar
        if idx < 30:
            return []

        signals = []
        session = df['session'].iloc[idx]
        mins = df['mins'].iloc[idx]
        atr = df['atr'].iloc[idx]

        if pd.isna(atr) or atr < 1.5:
            return []

        # Time window for indicator-based strategies (AM 09:15-10:45, PM 13:15-14:15)
        in_indicator_window = True
        if session == 'AM' and not (555 <= mins <= 645):
            in_indicator_window = False
        if session == 'PM' and not (795 <= mins <= 855):
            in_indicator_window = False

        # Extended window for opening strategies
        # AM: 09:00-11:25 (ORB Breakdown, Crabel Stretch)
        # PM: 13:15-14:15 (PM ORB — after 15-min opening range)
        in_opening_window = (
            (session == 'AM' and 540 <= mins <= 685) or
            (session == 'PM' and 795 <= mins <= 855)
        )

        if not in_indicator_window and not in_opening_window:
            return []

        if in_indicator_window:
            for key, cfg in self._strategies_1m.items():
                if session not in cfg['sessions']:
                    continue
                if self._is_deduped(key, self.DEDUP_1M):
                    continue

                # Weekday filter for DOW-optimized strategies
                weekday_filter = cfg.get('weekdays')
                if weekday_filter is not None:
                    weekday = df['time'].iloc[idx].weekday()
                    if weekday not in weekday_filter:
                        continue

                try:
                    direction = cfg['instance'].detect(df, idx)
                except Exception:
                    continue

                if direction == 0:
                    continue
                if direction not in cfg['directions']:
                    continue

                # Regime filter for RAW exit strategies
                regime_filter = cfg.get('regime_filter')
                if regime_filter:
                    price_vs_ema50 = np.nan
                    if not pd.isna(df['ema50'].iloc[idx]) and not pd.isna(atr) and atr > 0:
                        price_vs_ema50 = (df['close'].iloc[idx] - df['ema50'].iloc[idx]) / atr
                    if pd.isna(price_vs_ema50):
                        continue
                    if regime_filter == 'sell' and price_vs_ema50 >= -0.5:
                        continue
                    if regime_filter == 'buy' and price_vs_ema50 <= 0.5:
                        continue

                self._mark_signal(key)
                exit_mode = cfg.get('exit_mode', 'standard')
                signals.append({
                    'strategy': key,
                    'label': cfg['label'],
                    'tf': '1m',
                    'direction': 'BUY' if direction == 1 else 'SELL',
                    'price': float(df['close'].iloc[idx]),
                    'atr': float(atr),
                    'adx': float(df['adx'].iloc[idx]) if not pd.isna(df['adx'].iloc[idx]) else 0,
                    'session': session,
                    'time': str(df['time'].iloc[idx]),
                    'mins': int(mins),
                    'exit_mode': exit_mode,
                })

        # Opening strategies (ORB, Crabel Stretch, PM ORB — weekday-filtered internally)
        if in_opening_window:
            for open_key, open_cfg in self._opening_strategies.items():
                # Session filter for opening strategies
                open_session = open_cfg.get('session')
                if open_session and open_session != session:
                    continue
                if self._is_deduped(open_key, self.DEDUP_1M * 10):
                    continue
                try:
                    strat = open_cfg['instance']
                    sig = strat.detect(df, idx)
                    if sig != 0:
                        self._mark_signal(open_key)
                        sl_price = getattr(strat, 'sl_price', None)
                        trigger = getattr(strat, 'trigger_price', None)
                        entry_price = float(trigger) if trigger is not None else float(df['close'].iloc[idx])
                        direction_str = 'BUY' if sig == 1 else 'SELL'
                        signals.append({
                            'strategy': open_key,
                            'label': open_cfg['label'],
                            'tf': '1m',
                            'direction': direction_str,
                            'price': entry_price,
                            'atr': float(atr),
                            'adx': float(df['adx'].iloc[idx]) if not pd.isna(df['adx'].iloc[idx]) else 0,
                            'session': session,
                            'time': str(df['time'].iloc[idx]),
                            'mins': int(mins),
                            'exit_mode': 'raw',
                            'custom_sl': float(sl_price) if sl_price is not None else None,
                        })
                except Exception:
                    pass

        return signals

    def scan_3m(self, df_raw: pd.DataFrame) -> list[dict]:
        """Scan 3m data for Momentum AM SELL and Heikin Ashi PM SELL."""
        if df_raw is None or len(df_raw) < 60:
            return []

        try:
            df = prepare_df(df_raw.copy())
        except Exception as e:
            print(f"  [SHADOW 3m] prepare_df error: {e}")
            return []

        if len(df) < 60:
            return []

        idx = len(df) - 2
        if idx < 30:
            return []

        signals = []
        session = df['session'].iloc[idx]
        mins = df['mins'].iloc[idx]
        atr = df['atr'].iloc[idx]

        if pd.isna(atr) or atr < 1.5:
            return []

        if session == 'AM' and not (555 <= mins <= 645):
            return []
        if session == 'PM' and not (795 <= mins <= 855):
            return []

        for key, cfg in self._strategies_3m.items():
            if session not in cfg['sessions']:
                continue
            if self._is_deduped(key, self.DEDUP_3M):
                continue

            try:
                direction = cfg['instance'].detect(df, idx)
            except Exception:
                continue

            if direction == 0:
                continue
            if direction not in cfg['directions']:
                continue

            self._mark_signal(key)
            signals.append({
                'strategy': key,
                'label': cfg['label'],
                'tf': '3m',
                'direction': 'BUY' if direction == 1 else 'SELL',
                'price': float(df['close'].iloc[idx]),
                'atr': float(atr),
                'adx': float(df['adx'].iloc[idx]) if not pd.isna(df['adx'].iloc[idx]) else 0,
                'session': session,
                'time': str(df['time'].iloc[idx]),
                'mins': int(mins),
            })

        return signals

    def scan_5m(self, df_raw: pd.DataFrame) -> list[dict]:
        """Scan 5m for Volume PM SELL and Confluence SELL."""
        if df_raw is None or len(df_raw) < 60:
            return []

        try:
            df = prepare_df(df_raw.copy())
        except Exception as e:
            print(f"  [SHADOW 5m] prepare_df error: {e}")
            return []

        if len(df) < 60:
            return []

        idx = len(df) - 2
        if idx < 30:
            return []

        signals = []
        session = df['session'].iloc[idx]
        mins = df['mins'].iloc[idx]
        atr = df['atr'].iloc[idx]

        if pd.isna(atr) or atr < 1.5:
            return []

        if session == 'AM' and not (555 <= mins <= 645):
            return []
        if session == 'PM' and not (795 <= mins <= 855):
            return []

        # Individual 5m strategies
        for key, cfg in self._strategies_5m.items():
            if session not in cfg['sessions']:
                continue
            if self._is_deduped(key, self.DEDUP_5M):
                continue

            try:
                direction = cfg['instance'].detect(df, idx)
            except Exception:
                continue

            if direction == 0:
                continue
            if direction not in cfg['directions']:
                continue

            self._mark_signal(key)
            signals.append({
                'strategy': key,
                'label': cfg['label'],
                'tf': '5m',
                'direction': 'BUY' if direction == 1 else 'SELL',
                'price': float(df['close'].iloc[idx]),
                'atr': float(atr),
                'adx': float(df['adx'].iloc[idx]) if not pd.isna(df['adx'].iloc[idx]) else 0,
                'session': session,
                'time': str(df['time'].iloc[idx]),
                'mins': int(mins),
            })

        # Confluence SELL: 3+ strategies agree on SELL + ADX >= 20
        confluence_key = 'confluence_5m_sell'
        if not self._is_deduped(confluence_key, self.DEDUP_5M):
            adx_val = float(df['adx'].iloc[idx]) if not pd.isna(df['adx'].iloc[idx]) else 0
            if adx_val >= 20:
                sell_names = []
                for strat in self._confluence_strategies:
                    try:
                        sig = strat.detect(df, idx)
                    except Exception:
                        continue
                    if sig == -1:
                        sell_names.append(strat.name)

                if len(sell_names) >= 3:
                    self._mark_signal(confluence_key)
                    signals.append({
                        'strategy': confluence_key,
                        'label': f'Confluence SELL 5m ({len(sell_names)} agree, ADX={adx_val:.0f})',
                        'tf': '5m',
                        'direction': 'SELL',
                        'price': float(df['close'].iloc[idx]),
                        'atr': float(atr),
                        'adx': adx_val,
                        'session': session,
                        'time': str(df['time'].iloc[idx]),
                        'mins': int(mins),
                        'confluence_strategies': sell_names,
                    })

        return signals

    def format_telegram_alert(self, signal: dict) -> str:
        """Format a shadow signal for Telegram."""
        dir_icon = "\U0001f534" if signal['direction'] == 'SELL' else "\U0001f7e2"
        strats = signal.get('confluence_strategies', [])
        strat_line = f"\nStrategies: {', '.join(strats)}" if strats else ""
        exit_mode = signal.get('exit_mode', 'standard')
        custom_sl = signal.get('custom_sl')
        if custom_sl is not None:
            exit_line = f"\nExit: SL=open({custom_sl:.1f}), hold to session end"
        elif exit_mode == 'raw':
            exit_line = "\nExit: SL=3.0xATR, hold to session end"
        else:
            exit_line = ""

        return (
            f"{dir_icon} <b>[SHADOW] {signal['label']}</b>\n"
            f"{'=' * 24}\n"
            f"Direction: {signal['direction']} | TF: {signal['tf']}\n"
            f"Price: <code>{signal['price']:.1f}</code> | ATR: {signal['atr']:.2f}\n"
            f"ADX: {signal['adx']:.1f} | Session: {signal['session']}\n"
            f"Time: <code>{signal['time']}</code>{strat_line}{exit_line}\n"
            f"<i>Shadow mode -- not trading</i>"
        )
