"""
CB (Compression Breakout) Signal Scanner
=========================================
Strategy: 3-bar compression < 0.7×ATR → breakout on next bar
Validated: 129d walk-forward, WR 67.7%, PF 4.87, +6.34 pts/day

Usage:
    py scanner.py
    py scanner.py --once
    py scanner.py --no-trade
"""

import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="Downcasting object dtype arrays", category=FutureWarning)

from combos import get_combo
from config import Config
from src.data_fetcher import DataFetcher
from src.notifier import TelegramNotifier
from src.portfolio_manager import PortfolioManager
from src.position_manager import PositionManager
from src.trade_logger import TradeLogger
from src.dnse_auth import create_authenticated_client
from src.dnse_executor import DnseExecutor
from ml.model import MLFilter

# --- CONFIG ---------------------------------------------------------------
from src.strategy_config import get_config, get_combo_config, get_session_params

_cfg = get_config()
SYMBOL = _cfg.get("symbol", "VN30F1M")
SCAN_INTERVAL = _cfg.get("scan_interval", 10)

_open_str = _cfg.get("market_open", "09:00")
_close_str = _cfg.get("market_close", "14:30")
MARKET_OPEN = tuple(int(x) for x in _open_str.split(":"))
MARKET_CLOSE = tuple(int(x) for x in str(_close_str).split(":"))

DAILY_DIR_LOSSES: dict = {}

VN_TZ = timezone(timedelta(hours=7))


def vn_now() -> datetime:
    return datetime.now(VN_TZ)


def is_trading_hours() -> bool:
    now = vn_now()
    if now.weekday() >= 5:
        return False
    current = (now.hour, now.minute)
    return (9, 0) <= current <= (11, 30) or (13, 0) <= current <= (14, 29)


def record_daily_loss(combo_short: str, direction: str):
    today = vn_now().strftime("%Y-%m-%d")
    if combo_short not in DAILY_DIR_LOSSES:
        DAILY_DIR_LOSSES[combo_short] = {}
    if today not in DAILY_DIR_LOSSES[combo_short]:
        DAILY_DIR_LOSSES[combo_short][today] = {}
    cur = DAILY_DIR_LOSSES[combo_short][today].get(direction, 0)
    DAILY_DIR_LOSSES[combo_short][today][direction] = cur + 1
    print(f"  [LOSS CAP] {combo_short} {direction} loss #{cur + 1} today")


# ─── Signal Quality Decay Tracking ──────────────────────────────────────────

class SignalTracker:
    """Track rolling WR per combo; auto-disable when WR < threshold over last N trades."""

    def __init__(self, window=20, disable_threshold=0.40,
                 disable_lookback=10, disable_duration_hours=24):
        self.window = window
        self.disable_threshold = disable_threshold
        self.disable_lookback = disable_lookback
        self.disable_duration_hours = disable_duration_hours
        self.history: dict[str, list[dict]] = {}
        self.disabled_until: dict[str, datetime] = {}

    def record(self, combo_short: str, pnl: float, direction: str):
        if combo_short not in self.history:
            self.history[combo_short] = []
        self.history[combo_short].append({"pnl": pnl, "direction": direction, "ts": vn_now()})
        if len(self.history[combo_short]) > self.window:
            self.history[combo_short] = self.history[combo_short][-self.window:]
        wr = self.rolling_win_rate(combo_short, self.disable_lookback)
        n = len(self.history[combo_short])
        if n >= self.disable_lookback and wr < self.disable_threshold:
            until = vn_now() + timedelta(hours=self.disable_duration_hours)
            self.disabled_until[combo_short] = until
            print(f"  [TRACKER] {combo_short} DISABLED until {until.strftime('%H:%M %d/%m')} "
                  f"(WR={wr * 100:.0f}% over last {self.disable_lookback} trades)")

    def is_disabled(self, combo_short: str) -> bool:
        until = self.disabled_until.get(combo_short)
        if until is None:
            return False
        if vn_now() >= until:
            del self.disabled_until[combo_short]
            print(f"  [TRACKER] {combo_short} re-enabled (cooldown expired)")
            return False
        return True

    def rolling_win_rate(self, combo_short: str, n: int = 10) -> float:
        history = self.history.get(combo_short, [])
        if not history:
            return 0.5
        recent = history[-n:]
        return sum(1 for t in recent if t["pnl"] > 0) / len(recent)


signal_tracker = SignalTracker()


# ─── Market Regime Detection ─────────────────────────────────────────────────

_last_regime: str = "UNKNOWN"


def detect_regime(df_15m: pd.DataFrame | None) -> dict:
    """Detect market regime from 15m data: TRENDING / RANGING / VOLATILE / NORMAL."""
    default = {"regime": "NORMAL", "adx": 0, "atr_ratio": 1.0, "trend_direction": 0}
    if df_15m is None or len(df_15m) < 60:
        return default
    try:
        import pandas_ta as ta
        adx_df = ta.adx(df_15m["high"], df_15m["low"], df_15m["close"], length=14)
        if adx_df is None or adx_df.empty:
            return default
        adx_val = float(adx_df.iloc[-1, 0])
        di_plus = float(adx_df.iloc[-1, 1])
        di_minus = float(adx_df.iloc[-1, 2])
        atr = ta.atr(df_15m["high"], df_15m["low"], df_15m["close"], length=14)
        if atr is None or len(atr) < 50:
            return default
        atr_ratio = float(atr.iloc[-1]) / float(atr.iloc[-50:].mean())
        if atr_ratio > 1.5:
            regime = "VOLATILE"
        elif adx_val > 25 and abs(di_plus - di_minus) > 10:
            regime = "TRENDING"
        elif adx_val < 20:
            regime = "RANGING"
        else:
            regime = "NORMAL"
        return {
            "regime": regime, "adx": adx_val, "atr_ratio": atr_ratio,
            "trend_direction": 1 if di_plus > di_minus else (-1 if di_minus > di_plus else 0),
        }
    except Exception:
        return default


# ─── CB Combo Instance ────────────────────────────────────────────────────────

_cb_combo = get_combo("CB")
_cb_last_signal_time: datetime | None = None
CB_DEDUP_SECONDS = _cb_combo.dedup_bars * 5 * 60  # bars × 5min


def detect_cb_signal(df_5m: pd.DataFrame) -> dict | None:
    """Dedup wrapper around combo.detect()."""
    global _cb_last_signal_time
    if _cb_last_signal_time and (vn_now() - _cb_last_signal_time).total_seconds() < CB_DEDUP_SECONDS:
        return None
    return _cb_combo.detect(df_5m)


# ─── NR4 Combo Instance ──────────────────────────────────────────────────────

_nr4_combo = get_combo("NR4")
_nr4_last_signal_time: datetime | None = None
NR4_DEDUP_SECONDS = _nr4_combo.dedup_bars * 5 * 60  # bars × 5min


def detect_nr4_signal(df_5m: pd.DataFrame) -> dict | None:
    """Dedup wrapper around NR4 combo.detect()."""
    global _nr4_last_signal_time
    if _nr4_last_signal_time and (vn_now() - _nr4_last_signal_time).total_seconds() < NR4_DEDUP_SECONDS:
        return None
    return _nr4_combo.detect(df_5m)


def check_nr4_3m_alignment(df_3m: pd.DataFrame | None, sig_time_str: str) -> bool:
    """Check if 3m also shows NR4 SHORT within ±15min of 5m signal.

    Used for strength assessment (confidence boost), not as a filter.
    """
    if df_3m is None or len(df_3m) < 20:
        return False
    try:
        sig_time = pd.to_datetime(sig_time_str)
    except Exception:
        return False

    # Check last 5 bars of 3m (= 15 min window)
    lookback = min(5, len(df_3m) - 14)
    for offset in range(lookback):
        idx = len(df_3m) - 1 - offset
        if idx < _nr4_combo.nr4_lookback:
            break
        row_time = pd.to_datetime(df_3m["time"].iloc[idx]) if "time" in df_3m.columns else df_3m.index[idx]
        if abs((row_time - sig_time).total_seconds()) > 900:
            continue
        # Check NR4 condition on this 3m bar
        sub = df_3m.iloc[max(0, idx - 19):idx + 1].reset_index(drop=True)
        result = _nr4_combo.detect(sub)
        if result is not None:
            return True
    return False


# ─── Main Scan Loop ───────────────────────────────────────────────────────────

def run_scan(fetcher: DataFetcher, notifier: TelegramNotifier, sent_alerts: dict,
             position_manager: PositionManager | None = None,
             portfolio_mgr: PortfolioManager | None = None,
             pending_signals: dict | None = None,
             ml_filter: MLFilter | None = None):
    """One CB scan cycle: detect compression, queue pending, confirm on breakout, enter."""
    global _last_regime, _cb_last_signal_time, _nr4_last_signal_time

    if pending_signals is None:
        pending_signals = {}

    print(f"\n[{vn_now().strftime('%H:%M:%S')}] Scanning {SYMBOL} (CB/5m)...")

    # --- Portfolio price update (1m, pandas_ta ATR) ---
    if portfolio_mgr and portfolio_mgr.n_open > 0:
        try:
            _now_pm = vn_now()
            _df_pm = fetcher.get_futures_ohlcv(
                SYMBOL,
                (_now_pm - timedelta(days=1)).strftime("%Y-%m-%d"),
                _now_pm.strftime("%Y-%m-%d"),
                interval="1m",
            )
            if _df_pm is not None and len(_df_pm) > 14:
                import pandas_ta as _ta_pm
                _atr_s = _ta_pm.atr(_df_pm["high"], _df_pm["low"], _df_pm["close"], length=14)
                _atr = float(_atr_s.iloc[-1]) if (_atr_s is not None and pd.notna(_atr_s.iloc[-1])) else 3.5
                _last = _df_pm.iloc[-1]
                portfolio_mgr.update_prices(
                    SYMBOL,
                    high=float(_last["high"]),
                    low=float(_last["low"]),
                    close=float(_last["close"]),
                    atr=_atr,
                    regime=_last_regime,
                )
        except Exception as _e:
            print(f"  [PM UPDATE ERROR] {_e}")
        # tick() handled by fast loop; run_scan only ticks when flat (elif below)
    elif portfolio_mgr:
        portfolio_mgr.tick()

    # --- Fetch 5m, 3m, and 15m data ---
    _now = vn_now()
    df_5m = None
    df_3m = None
    df_15m = None
    try:
        df_5m = fetcher.get_futures_ohlcv(
            SYMBOL, (_now - timedelta(days=5)).strftime("%Y-%m-%d"),
            _now.strftime("%Y-%m-%d"), interval="5m",
        )
    except Exception as e:
        print(f"  [5m] Fetch error: {e}")
    try:
        df_3m = fetcher.get_futures_ohlcv(
            SYMBOL, (_now - timedelta(days=3)).strftime("%Y-%m-%d"),
            _now.strftime("%Y-%m-%d"), interval="3m",
        )
    except Exception as e:
        print(f"  [3m] Fetch error: {e}")
    try:
        df_15m = fetcher.get_futures_ohlcv(
            SYMBOL, (_now - timedelta(days=10)).strftime("%Y-%m-%d"),
            _now.strftime("%Y-%m-%d"), interval="15m",
        )
    except Exception as e:
        print(f"  [15m] Fetch error: {e}")

    # --- Regime detection ---
    regime_info = detect_regime(df_15m)
    if regime_info["regime"] != _last_regime:
        print(f"  [REGIME] {_last_regime} → {regime_info['regime']} "
              f"(ADX={regime_info['adx']:.1f}, ATR_ratio={regime_info['atr_ratio']:.2f})")
        _last_regime = regime_info["regime"]

    volatile = regime_info["regime"] == "VOLATILE"
    if volatile:
        print(f"  [REGIME] VOLATILE — skip CB entries")

    # --- CB compression detection (last complete 5m bar) ---
    cb_sig = None
    if df_5m is not None and len(df_5m) >= 20 and not volatile:
        df_5m_complete = df_5m.iloc[:-1] if is_trading_hours() else df_5m
        cb_sig = detect_cb_signal(df_5m_complete)
        if cb_sig:
            print(f"  [CB] Compression @ {cb_sig['time']} "
                  f"ATR={cb_sig['atr']:.2f} RSI={cb_sig['rsi']:.0f} [{cb_sig['session']}]")

    if cb_sig and signal_tracker.is_disabled("CB"):
        print(f"  [CB] Disabled by decay tracker")
        cb_sig = None

    # --- Queue pending BUY + SELL triggers ---
    if cb_sig and portfolio_mgr and not portfolio_mgr.in_cooldown:
        _cb_last_signal_time = vn_now()
        for direction in ("BUY", "SELL"):
            pkey = f"CB_{direction}"
            if pkey in pending_signals:
                continue
            direction_int = 1 if direction == "BUY" else -1
            if (portfolio_mgr.current_direction != 0
                    and direction_int != portfolio_mgr.current_direction
                    and not portfolio_mgr.should_flip(direction_int, "5m", 1)):
                continue
            trigger = cb_sig["high"] + 0.1 if direction == "BUY" else cb_sig["low"] - 0.1
            alert_key = f"{SYMBOL}_CB_{direction}_{cb_sig['time']}"
            pending_signals[pkey] = {
                "direction": direction,
                "combo_short": "CB",
                "best_tf": "5m",
                "trigger_price": trigger,
                "alert_key": alert_key,
                "sig_price": cb_sig["price"],
                "sig_atr": cb_sig["atr"],
                "sig_time": cb_sig["time"],
                "sig_session": cb_sig["session"],
                "confidence": 1,
                "age": 0,
            }
            print(f"  [CB/{direction}] PENDING (trigger={trigger:.1f})")

    # --- Check pending for next-bar confirmation ---
    confirmed_items = []
    for pkey in list(pending_signals.keys()):
        if not pkey.startswith("CB_"):
            continue
        ps = pending_signals[pkey]
        direction = ps["direction"]
        trigger = ps["trigger_price"]

        if df_5m is None or df_5m.empty:
            del pending_signals[pkey]
            continue

        last_bar = df_5m.iloc[-1]
        confirmed = (
            (direction == "BUY" and float(last_bar["high"]) > trigger) or
            (direction == "SELL" and float(last_bar["low"]) < trigger)
        )

        if confirmed:
            confirmed_items.append(ps)
            del pending_signals[pkey]
            # Cancel opposite direction
            opp = "CB_SELL" if direction == "BUY" else "CB_BUY"
            pending_signals.pop(opp, None)
            print(f"  [CB/{direction}] CONFIRMED (trigger={trigger:.1f})")
        else:
            ps["age"] = ps.get("age", 0) + 1
            if ps["age"] >= 2:
                del pending_signals[pkey]
                print(f"  [CB/{direction}] EXPIRED (no confirmation)")

    # --- NR4 SHORT detection (last complete 5m bar) ---
    nr4_sig = None
    nr4_confidence = 1
    if df_5m is not None and len(df_5m) >= 20 and not volatile:
        df_5m_complete = df_5m.iloc[:-1] if is_trading_hours() else df_5m
        nr4_sig = detect_nr4_signal(df_5m_complete)
        if nr4_sig:
            # Check 3m alignment for strength assessment
            if check_nr4_3m_alignment(df_3m, nr4_sig["time"]):
                nr4_confidence = 2
            strength = "STRONG (3m aligned)" if nr4_confidence == 2 else "normal"
            print(f"  [NR4] NR4 compression @ {nr4_sig['time']} "
                  f"ATR={nr4_sig['atr']:.2f} [{nr4_sig['session']}] strength={strength}")

    if nr4_sig and signal_tracker.is_disabled("NR4"):
        print(f"  [NR4] Disabled by decay tracker")
        nr4_sig = None

    # --- NR4: queue SELL pending only (SHORT-only strategy) ---
    if nr4_sig and portfolio_mgr and not portfolio_mgr.in_cooldown:
        _nr4_last_signal_time = vn_now()
        pkey = "NR4_SELL"
        if pkey not in pending_signals:
            direction_int = -1
            if not (portfolio_mgr.current_direction != 0
                    and direction_int != portfolio_mgr.current_direction
                    and not portfolio_mgr.should_flip(direction_int, "5m", 1)):
                trigger = nr4_sig["low"] - 0.1
                alert_key = f"{SYMBOL}_NR4_SELL_{nr4_sig['time']}"
                pending_signals[pkey] = {
                    "direction": "SELL",
                    "combo_short": "NR4",
                    "best_tf": "5m",
                    "trigger_price": trigger,
                    "alert_key": alert_key,
                    "sig_price": nr4_sig["price"],
                    "sig_atr": nr4_sig["atr"],
                    "sig_time": nr4_sig["time"],
                    "sig_session": nr4_sig["session"],
                    "confidence": nr4_confidence,
                    "age": 0,
                }
                print(f"  [NR4/SELL] PENDING (trigger={trigger:.1f})")

    # --- Check NR4 pending for next-bar confirmation ---
    for pkey in list(pending_signals.keys()):
        if not pkey.startswith("NR4_"):
            continue
        ps = pending_signals[pkey]
        trigger = ps["trigger_price"]

        if df_5m is None or df_5m.empty:
            del pending_signals[pkey]
            continue

        last_bar = df_5m.iloc[-1]
        if float(last_bar["low"]) < trigger:
            confirmed_items.append(ps)
            del pending_signals[pkey]
            print(f"  [NR4/SELL] CONFIRMED (trigger={trigger:.1f})")
        else:
            ps["age"] = ps.get("age", 0) + 1
            if ps["age"] >= 2:
                del pending_signals[pkey]
                print(f"  [NR4/SELL] EXPIRED (no confirmation)")

    # --- Process confirmed: alert + portfolio entry ---
    any_signal = False
    for ps in confirmed_items:
        direction = ps["direction"]
        alert_key = ps["alert_key"]
        sig_atr = ps["sig_atr"]
        sig_price = ps["sig_price"]
        session = ps["sig_session"]

        if alert_key in sent_alerts:
            continue

        any_signal = True
        direction_int = 1 if direction == "BUY" else -1
        dir_icon = "\U0001f7e2" if direction == "BUY" else "\U0001f534"

        sl_mult = get_session_params().get(session, {}).get("sl_atr_mult", 1.2)
        combo_name = ps["combo_short"]
        tp_mult = get_combo_config(combo_name).get("tp_atr_mult", 4.0)
        sl = sig_price - direction_int * sl_mult * sig_atr
        tp = sig_price + direction_int * tp_mult * sig_atr
        current_price = float(df_5m.iloc[-1]["close"]) if df_5m is not None else sig_price

        sig_confidence = ps.get("confidence", 1)
        combo_desc = {
            "CB": "3-bar compression &lt; 0.7\u00d7ATR confirmed",
            "NR4": "NR4 SHORT \u2014 narrowest range in 4 bars confirmed",
        }.get(combo_name, f"{combo_name} signal confirmed")
        strength_tag = " | \u2b50 3m aligned" if sig_confidence >= 2 else ""

        msg = (
            f"{dir_icon} <b>{combo_name} {direction} [{ps['best_tf']}] \u2014 {SYMBOL}</b>\n"
            f"{'─' * 24}\n"
            f"{combo_desc}\n"
            f"Session: {session} | ATR: {sig_atr:.2f} pts{strength_tag}\n"
            f"Signal bar: <code>{ps['sig_time']}</code>\n"
            f"\n<b>Entry:</b> <code>{sig_price:,.1f}</code> (current: {current_price:,.1f})\n"
            f"SL: <code>{sl:,.1f}</code> ({sl_mult}\u00d7ATR = {sl_mult * sig_atr:.1f} pts)\n"
            f"TP: <code>{tp:,.1f}</code> ({tp_mult}\u00d7ATR = {tp_mult * sig_atr:.1f} pts)\n"
            f"R:R = <b>{tp_mult / sl_mult:.1f}:1</b>"
        )

        notifier.send(msg)
        sent_alerts[alert_key] = time.time()
        print(f"  [{combo_name}/{ps['best_tf']}] {direction} \u2192 entry={sig_price:.1f} SL={sl:.1f} TP={tp:.1f}")

        # Portfolio entry
        if portfolio_mgr:
            _now_e = vn_now()
            _mins_e = _now_e.hour * 60 + _now_e.minute
            _in_am = _mins_e < 11 * 60 + 30
            _entry_ok = (
                (_in_am and _mins_e < 11 * 60 + 25) or
                (not _in_am and 13 * 60 <= _mins_e < 14 * 60 + 15)
            )
            if not _entry_ok:
                print(f"  [TIME CUTOFF] No entry after {'11:25' if _in_am else '14:15'}")
                continue

            # pre_move_ratio for adaptive exit decision
            pre_move_ratio = 0.0
            if df_5m is not None and len(df_5m) >= 5 and sig_atr > 0:
                try:
                    _pre_close = float(df_5m.iloc[-2]["close"])
                    _pre_open = float(df_5m.iloc[-4]["open"])
                    pre_move_ratio = (_pre_close - _pre_open) * direction_int / sig_atr
                except Exception:
                    pass

            # ML filter check
            if ml_filter and ml_filter.is_active and df_5m is not None:
                ml_result = ml_filter.predict(df_5m, len(df_5m) - 1, direction_int)
                print(f"  [ML] P(win)={ml_result['p_win']:.3f} → {ml_result['reason']}")
                if not ml_result["should_enter"]:
                    continue

            if portfolio_mgr.should_flip(direction_int, "5m", 1):
                portfolio_mgr.execute_flip(current_price)

            if portfolio_mgr.can_open(direction_int):
                portfolio_mgr.open_position(
                    symbol=SYMBOL,
                    direction=direction_int,
                    entry_price=sig_price,
                    atr=sig_atr,
                    combo=combo_name,
                    timeframe=ps["best_tf"],
                    confidence=ps.get("confidence", 1),
                    pre_move_ratio=pre_move_ratio,
                )

    if not any_signal and not pending_signals:
        status = portfolio_mgr.status_str() if portfolio_mgr else "N/A"
        print(f"  No signal. Portfolio: {status}")

    # Cleanup alerts older than 30 min
    cutoff = time.time() - 1800
    for k in [k for k, t in sent_alerts.items() if t < cutoff]:
        del sent_alerts[k]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VN30F1M CB Scanner")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    parser.add_argument("--no-trade", action="store_true", help="Signal only, no real orders")
    parser.add_argument("--ml-shadow", action="store_true", help="Enable ML filter in shadow mode (log only)")
    args = parser.parse_args()

    fetcher = DataFetcher()
    notifier = TelegramNotifier()

    if not notifier.is_configured():
        print("WARNING: Telegram not configured. Alerts print to console only.")
    else:
        print(f"Telegram configured. Chat ID: {notifier.chat_id}")

    # --- DNSE Auto-Auth ---
    dnse_executor = None
    if not args.no_trade:
        try:
            if Config.DNSE_OTP_EMAIL and Config.DNSE_OTP_APP_PASSWORD:
                print("[DNSE] Authenticating with auto-OTP...")
                dnse_client = create_authenticated_client()
                dnse_executor = DnseExecutor(dnse_client, notifier)
                print(f"[DNSE] Ready. Contract: {dnse_executor.contract_symbol}")
                notifier.send(
                    f"\U0001f513 <b>DNSE Authenticated</b>\n"
                    f"Account: {Config.ACCOUNT_NO}\n"
                    f"Contract: {dnse_executor.contract_symbol}"
                )
            else:
                print("[DNSE] OTP email not configured. Signal-only mode.")
        except Exception as e:
            print(f"[DNSE] Auth failed: {e}. Signal-only mode.")
            notifier.send(f"\u26a0\ufe0f <b>DNSE Auth Failed</b>\n{e}\nRunning signal-only.")
    else:
        print("[MODE] Signal-only (--no-trade)")

    sp = get_session_params()
    am_p = sp.get("AM", {}); pm_p = sp.get("PM", {})
    ae = get_config().get("adaptive_exit", {})
    print(f"CB Scanner: {SYMBOL}")
    print(f"Strategy: CB/5m \u2014 3-bar compression < 0.7\u00d7ATR breakout")
    print(f"  AM: SL={am_p.get('sl_atr_mult',1.5)}\u00d7ATR, trail@{am_p.get('trail_activate_pts',7.0)}pts/{am_p.get('trail_atr_mult',2.5)}\u00d7ATR, max_hold={am_p.get('max_hold_bars',30)} bars")
    print(f"  PM: SL={pm_p.get('sl_atr_mult',1.2)}\u00d7ATR, trail@{pm_p.get('trail_activate_pts',6.0)}pts/{pm_p.get('trail_atr_mult',1.0)}\u00d7ATR, max_hold={pm_p.get('max_hold_bars',8)} bars")
    print(f"  Adaptive exit: AM pre_move/ATR > {ae.get('pre_move_ratio_threshold',0.8)} \u2192 MFE\u2265{ae.get('mfe_trigger_pts',2.0)}, exit@{ae.get('exit_target_pts',3.2)}pts")
    print(f"Scan interval: {SCAN_INTERVAL}s (positions) / 60s (signals)")
    print(f"Trading: {'LIVE' if dnse_executor else 'SIGNAL-ONLY'}")
    print("=" * 50)

    trade_logger = TradeLogger(log_dir="logs")
    portfolio_manager = PortfolioManager(
        notifier=notifier,
        logger=trade_logger,
        max_contracts=1,
        flip_cooldown=3,
        executor=dnse_executor,
    )

    def _on_close(combo: str, direction: str, pnl: float):
        signal_tracker.record(combo, pnl, direction)
        if pnl < 0:
            record_daily_loss(combo, direction)

    portfolio_manager.on_close_callback = _on_close

    # Legacy single-position manager (kept for compatibility)
    position_manager = PositionManager(
        notifier=notifier,
        logger=trade_logger,
        sl_atr_mult=1.2,
        tp_atr_mult=4.0,
    )

    sent_alerts: dict = {}
    pending_signals: dict = {}

    # --- ML Filter ---
    ml_cfg = get_config().get("ml_filter", {})
    if args.ml_shadow:
        ml_cfg["enabled"] = True
        ml_cfg["shadow_mode"] = True
    ml_filter = MLFilter(ml_cfg) if ml_cfg.get("enabled") else None
    if ml_filter and ml_filter.is_active:
        mode_str = "SHADOW" if ml_filter.shadow_mode else "LIVE"
        print(f"[ML] Filter active ({mode_str}), threshold={ml_filter.threshold:.2f}")

    if args.once:
        run_scan(fetcher, notifier, sent_alerts, position_manager,
                 portfolio_mgr=portfolio_manager, pending_signals=pending_signals,
                 ml_filter=ml_filter)
        return

    notifier.send(
        f"<b>CB Scanner Started</b>\n"
        f"Symbol: <code>{SYMBOL}</code>\n"
        f"Strategy: CB/5m — 3-bar compression breakout\n"
        f"Validated: WR 67.7%, PF 4.87, +6.34 pts/day\n"
        f"Portfolio: max 1 contract, cooldown 3 bars\n"
        f"Trading: {'LIVE' if dnse_executor else 'SIGNAL-ONLY'}\n"
        f"Status: {portfolio_manager.status_str()}"
    )

    _eod_sent_date: str = ""
    _last_full_scan: float = 0
    FULL_SCAN_INTERVAL = 60

    while True:
        try:
            now = vn_now()

            # EOD: force-close + daily summary
            today_str = now.strftime("%Y-%m-%d")
            after_close = (now.hour, now.minute) >= (14, 29)
            if after_close and now.weekday() < 5 and _eod_sent_date != today_str:
                _eod_sent_date = today_str
                if portfolio_manager.n_open > 0:
                    try:
                        _df_eod = fetcher.get_futures_ohlcv(SYMBOL, today_str, today_str, interval="1m")
                        if _df_eod is not None and len(_df_eod) > 0:
                            portfolio_manager.close_all(float(_df_eod.iloc[-1]["close"]), reason="EOD")
                    except Exception as _e:
                        print(f"  [EOD CLOSE ERROR] {_e}")
                summary = trade_logger.daily_summary(today_str)
                notifier.send(summary)
                print(f"[EOD] Summary sent for {today_str}")

            if not is_trading_hours():
                print(f"\r[{now.strftime('%H:%M:%S')}] Outside trading hours.", end="")
                time.sleep(60)
                continue

            # Fast loop: position SL/TP update every SCAN_INTERVAL (10s)
            if portfolio_manager.n_open > 0:
                try:
                    _now_f = vn_now()
                    _df_fast = fetcher.get_futures_ohlcv(
                        SYMBOL,
                        (_now_f - timedelta(days=1)).strftime("%Y-%m-%d"),
                        _now_f.strftime("%Y-%m-%d"),
                        interval="1m",
                    )
                    if _df_fast is not None and len(_df_fast) > 14:
                        import pandas_ta as _ta_fast
                        _atr_s = _ta_fast.atr(_df_fast["high"], _df_fast["low"],
                                              _df_fast["close"], length=14)
                        _atr = float(_atr_s.iloc[-1]) if (
                            _atr_s is not None and pd.notna(_atr_s.iloc[-1])
                        ) else 3.5
                        _last = _df_fast.iloc[-1]
                        portfolio_manager.update_prices(
                            SYMBOL,
                            high=float(_last["high"]),
                            low=float(_last["low"]),
                            close=float(_last["close"]),
                            atr=_atr,
                            regime=_last_regime,
                        )
                except Exception as _e:
                    print(f"  [FAST UPDATE ERROR] {_e}")
                portfolio_manager.tick()

            # Slow loop: full CB scan every 60s
            if time.time() - _last_full_scan >= FULL_SCAN_INTERVAL:
                _last_full_scan = time.time()
                run_scan(fetcher, notifier, sent_alerts, position_manager,
                         portfolio_mgr=portfolio_manager, pending_signals=pending_signals,
                         ml_filter=ml_filter)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            time.sleep(SCAN_INTERVAL)
            continue

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
