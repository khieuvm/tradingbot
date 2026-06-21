"""Shadow Performance Tracker — log RAW strategy signals and compute session-end P&L.

Records each shadow signal to JSONL with entry details. At session end, computes
hypothetical P&L (SL=3.0×ATR, hold to session end) for RAW exit strategies.
Provides rolling performance stats to decide when to go live.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

VN_TZ = timezone(timedelta(hours=7))
COST = 0.96
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "shadow_raw_signals.jsonl"


class ShadowTracker:
    """Track shadow signal performance with session-end P&L."""

    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._pending: list[dict] = []
        self._load_pending()

    def _load_pending(self):
        """Load signals from today that haven't been resolved yet."""
        if not LOG_FILE.exists():
            return
        today = datetime.now(VN_TZ).strftime("%Y-%m-%d")
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line.strip())
                    if rec.get("date") == today and rec.get("exit_price") is None:
                        self._pending.append(rec)
        except Exception:
            pass

    def log_signal(self, signal: dict):
        """Log a new shadow signal entry."""
        now = datetime.now(VN_TZ)
        record = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "strategy": signal["strategy"],
            "label": signal["label"],
            "tf": signal["tf"],
            "direction": signal["direction"],
            "session": signal["session"],
            "entry_price": signal["price"],
            "atr": signal["atr"],
            "adx": signal.get("adx", 0),
            "sl_price": self._calc_sl(signal),
            "exit_mode": signal.get("exit_mode", "standard"),
            "mins": signal.get("mins", 0),
            "exit_price": None,
            "exit_reason": None,
            "pnl": None,
        }
        self._pending.append(record)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _calc_sl(self, signal: dict) -> float:
        if signal.get('custom_sl') is not None:
            return signal['custom_sl']
        direction_mult = 1 if signal["direction"] == "BUY" else -1
        sl_mult = 3.0 if signal.get("exit_mode") == "raw" else 1.5
        return signal["price"] - direction_mult * sl_mult * signal["atr"]

    def check_exits(self, current_price: float, current_mins: int):
        """Check pending signals for SL hit or session end. Call every scan cycle."""
        if not self._pending:
            return []

        resolved = []
        still_pending = []

        am_end = 11 * 60 + 25  # 11:25
        pm_end = 14 * 60 + 25  # 14:25

        for rec in self._pending:
            if rec.get("exit_price") is not None:
                continue

            direction_mult = 1 if rec["direction"] == "BUY" else -1
            sl_price = rec["sl_price"]

            # Check SL hit
            sl_hit = (
                (rec["direction"] == "BUY" and current_price <= sl_price) or
                (rec["direction"] == "SELL" and current_price >= sl_price)
            )

            # Check session end
            session_end = (
                (rec["session"] == "AM" and current_mins >= am_end) or
                (rec["session"] == "PM" and current_mins >= pm_end)
            )

            if sl_hit:
                rec["exit_price"] = sl_price
                rec["exit_reason"] = "SL"
                rec["pnl"] = (sl_price - rec["entry_price"]) * direction_mult - COST
                resolved.append(rec)
            elif session_end:
                rec["exit_price"] = current_price
                rec["exit_reason"] = "SESSION"
                rec["pnl"] = (current_price - rec["entry_price"]) * direction_mult - COST
                resolved.append(rec)
            else:
                still_pending.append(rec)

        self._pending = still_pending

        if resolved:
            self._update_log(resolved)

        return resolved

    def _update_log(self, resolved: list[dict]):
        """Rewrite log file with resolved records updated."""
        if not LOG_FILE.exists():
            return

        all_records = []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    all_records.append(json.loads(line.strip()))
                except Exception:
                    continue

        for res in resolved:
            for i, rec in enumerate(all_records):
                if (rec.get("timestamp") == res["timestamp"]
                        and rec.get("strategy") == res["strategy"]
                        and rec.get("exit_price") is None):
                    all_records[i] = res
                    break

        with open(LOG_FILE, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec) + "\n")

    def get_stats(self, days: int = 14) -> dict:
        """Compute rolling performance stats from log file."""
        if not LOG_FILE.exists():
            return {"n": 0}

        cutoff = (datetime.now(VN_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")
        records = []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if rec.get("date", "") >= cutoff and rec.get("pnl") is not None:
                        records.append(rec)
                except Exception:
                    continue

        if not records:
            return {"n": 0}

        pnls = [r["pnl"] for r in records]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        n = len(pnls)
        wr = len(wins) / n * 100 if n > 0 else 0
        gross_win = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        pf = gross_win / gross_loss
        total_pnl = sum(pnls)
        per_day = total_pnl / max(days, 1)

        by_strategy = {}
        for r in records:
            key = r["strategy"]
            if key not in by_strategy:
                by_strategy[key] = []
            by_strategy[key].append(r["pnl"])

        strategy_stats = {}
        for key, strat_pnls in by_strategy.items():
            s_wins = [p for p in strat_pnls if p > 0]
            s_losses = [p for p in strat_pnls if p <= 0]
            s_gw = sum(s_wins) if s_wins else 0
            s_gl = abs(sum(s_losses)) if s_losses else 0.001
            strategy_stats[key] = {
                "n": len(strat_pnls),
                "wr": len(s_wins) / len(strat_pnls) * 100,
                "pf": s_gw / s_gl,
                "total_pnl": sum(strat_pnls),
            }

        return {
            "n": n,
            "wr": wr,
            "pf": pf,
            "total_pnl": total_pnl,
            "per_day": per_day,
            "by_strategy": strategy_stats,
            "days": days,
        }

    def format_stats_report(self) -> str:
        """Format performance stats for Telegram."""
        stats = self.get_stats(14)
        if stats["n"] == 0:
            return "[SHADOW TRACKER] No resolved signals yet"

        lines = [
            f"<b>[SHADOW TRACKER] 14-day Performance</b>",
            f"Trades: {stats['n']} | WR: {stats['wr']:.1f}% | PF: {stats['pf']:.2f}",
            f"Total PnL: {stats['total_pnl']:+.1f} pts | /day: {stats['per_day']:+.2f}",
            "",
        ]
        for key, s in stats.get("by_strategy", {}).items():
            lines.append(
                f"  {key}: N={s['n']} WR={s['wr']:.0f}% PF={s['pf']:.2f} PnL={s['total_pnl']:+.1f}"
            )

        return "\n".join(lines)

    @property
    def pending_count(self) -> int:
        return len(self._pending)
