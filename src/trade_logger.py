"""
Trade Logger - Log trades to daily files for analysis and EOD summaries.
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

POINT_VALUE = 100_000  # VND per point for VN30F futures


class TradeLogger:
    """Logs trade entries/exits and provides daily summaries."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _daily_file(self, date_str: str) -> Path:
        return self.log_dir / f"trades_{date_str}.jsonl"

    def log_entry(self, symbol: str, direction: int, entry_price: float,
                  sl: float, tp: float, atr: float, combo: str,
                  timeframe: str, confidence: int = 0,
                  pos_id: int = 0, timestamp: str | None = None,
                  **kwargs):
        """Log a new trade entry."""
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_str = ts[:10]
        record = {
            "type": "ENTRY",
            "timestamp": ts,
            "symbol": symbol,
            "direction": "BUY" if direction == 1 else "SELL",
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "atr": atr,
            "combo": combo,
            "timeframe": timeframe,
            "confidence": confidence,
            "pos_id": pos_id,
        }
        with open(self._daily_file(date_str), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_exit(self, symbol: str, direction: int, entry_price: float,
                 exit_price: float, reason: str, pnl_pts: float,
                 combo: str = "", timeframe: str = "",
                 pos_id: int = 0, timestamp: str | None = None):
        """Log a trade exit (SL hit, TP hit, FLIP, or manual)."""
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_str = ts[:10]
        record = {
            "type": "EXIT",
            "timestamp": ts,
            "symbol": symbol,
            "direction": "BUY" if direction == 1 else "SELL",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "reason": reason,
            "pnl_pts": pnl_pts,
            "pnl_vnd": pnl_pts * POINT_VALUE,
            "combo": combo,
            "timeframe": timeframe,
            "pos_id": pos_id,
        }
        with open(self._daily_file(date_str), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_event(self, event_type: str, details: dict,
                  timestamp: str | None = None):
        """Log a portfolio event (FLIP, COOLDOWN, SIGNAL_REJECTED, etc.)."""
        ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date_str = ts[:10]
        record = {"type": event_type, "timestamp": ts, **details}
        with open(self._daily_file(date_str), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def daily_summary(self, date_str: str) -> str:
        """Generate a formatted daily trade summary with per-combo breakdown."""
        fpath = self._daily_file(date_str)
        if not fpath.exists():
            return f"<b>\U0001f4ca EOD Report {date_str}</b>\nNo trades today."

        entries = []
        exits = []
        events = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["type"] == "ENTRY":
                    entries.append(rec)
                elif rec["type"] == "EXIT":
                    exits.append(rec)
                else:
                    events.append(rec)

        if not exits and not entries:
            return f"<b>\U0001f4ca EOD Report {date_str}</b>\nNo activity today."

        total_pnl_pts = sum(e.get("pnl_pts", 0) for e in exits)
        total_pnl_vnd = total_pnl_pts * POINT_VALUE
        wins = [e for e in exits if e.get("pnl_pts", 0) > 0]
        losses = [e for e in exits if e.get("pnl_pts", 0) <= 0]
        wr = len(wins) / max(1, len(exits)) * 100

        # Per-combo breakdown
        combo_pnl = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
        for e in exits:
            c = e.get("combo", "?")
            combo_pnl[c]["trades"] += 1
            combo_pnl[c]["pnl"] += e.get("pnl_pts", 0)
            if e.get("pnl_pts", 0) > 0:
                combo_pnl[c]["wins"] += 1

        # Exit reasons breakdown
        reason_counts = defaultdict(int)
        for e in exits:
            reason_counts[e.get("reason", "?")] += 1

        lines = [
            f"<b>\U0001f4ca EOD Report {date_str}</b>",
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
            f"",
            f"<b>SUMMARY:</b>",
            f"  Entries: {len(entries)} | Exits: {len(exits)}",
            f"  W/L: {len(wins)}/{len(losses)} (WR: {wr:.0f}%)",
            f"  Exits by: {', '.join(f'{r}={c}' for r, c in sorted(reason_counts.items()))}",
            f"",
            f"<b>\U0001f4b0 Net PnL: {total_pnl_pts:+.1f} pts ({total_pnl_vnd:+,.0f} VND)</b>",
        ]

        # Per-combo breakdown
        if combo_pnl:
            lines.append("")
            lines.append("<b>BY COMBO:</b>")
            for c in sorted(combo_pnl.keys(), key=lambda x: -combo_pnl[x]["pnl"]):
                s = combo_pnl[c]
                c_vnd = s["pnl"] * POINT_VALUE
                c_wr = s["wins"] / max(1, s["trades"]) * 100
                icon = "\u2705" if s["pnl"] > 0 else "\u274c"
                lines.append(
                    f"  {icon} {c}: {s['trades']} trades, "
                    f"WR {c_wr:.0f}%, {s['pnl']:+.1f} pts ({c_vnd:+,.0f} VND)"
                )

        # Trade details
        if exits:
            lines.append("")
            lines.append("<b>TRADES:</b>")
            for e in exits:
                pnl = e.get("pnl_pts", 0)
                pnl_vnd = pnl * POINT_VALUE
                icon = "\u2705" if pnl > 0 else "\u274c"
                combo = e.get("combo", "?")
                tf = e.get("timeframe", "?")
                time_str = e.get("timestamp", "")[11:16]
                lines.append(
                    f"  {icon} {time_str} {e['direction']} {combo}({tf}) "
                    f"@ {e['entry_price']:,.1f} \u2192 {e['exit_price']:,.1f} "
                    f"[{e['reason']}] {pnl:+.1f}pts ({pnl_vnd:+,.0f})"
                )

        # Open positions warning
        open_count = len(entries) - len(exits)
        if open_count > 0:
            lines.append(f"\n\u26a0\ufe0f {open_count} position(s) still open")

        return "\n".join(lines)

    def weekly_summary(self) -> str:
        """Generate summary of this week's trading from daily log files."""
        import glob
        files = sorted(glob.glob(str(self.log_dir / "trades_*.jsonl")))
        if not files:
            return "No trade logs found."

        # Last 5 trading days
        recent_files = files[-5:]
        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        daily_results = []

        for fpath in recent_files:
            date_str = Path(fpath).stem.replace("trades_", "")
            day_pnl = 0.0
            day_trades = 0
            day_wins = 0
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec["type"] == "EXIT":
                        pnl = rec.get("pnl_pts", 0)
                        day_pnl += pnl
                        day_trades += 1
                        if pnl > 0:
                            day_wins += 1

            total_pnl += day_pnl
            total_trades += day_trades
            total_wins += day_wins
            daily_results.append((date_str, day_trades, day_wins, day_pnl))

        lines = [
            "<b>\U0001f4c5 Weekly Summary</b>",
            "\u2500" * 20,
        ]
        for date_str, trades, wins, pnl in daily_results:
            vnd = pnl * POINT_VALUE
            icon = "\u2705" if pnl > 0 else "\u274c"
            lines.append(
                f"  {icon} {date_str}: {trades} trades, "
                f"{wins}W/{trades-wins}L, {pnl:+.1f}pts ({vnd:+,.0f})"
            )

        total_vnd = total_pnl * POINT_VALUE
        wr = total_wins / max(1, total_trades) * 100
        lines.extend([
            "",
            f"<b>TOTAL: {total_trades} trades, WR {wr:.0f}%</b>",
            f"<b>\U0001f4b0 {total_pnl:+.1f} pts ({total_vnd:+,.0f} VND)</b>",
        ])
        return "\n".join(lines)
