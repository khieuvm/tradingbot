---
description: "Run VN30F1M CB signal scanner and report current market signals. Shows regime and CB compression status."
argument-hint: ""
---

# Signal Scanner

Scan VN30F1M for current CB (Compression Breakout) signals.

## Execution Procedure

1. **Check session** — verify within trading hours (9:00-11:30 or 13:00-14:30 Vietnam time)
2. **Read skill** — load `skills/signal-scanner/SKILL.md` for context
3. **Fetch data** — get 5m and 15m VN30F1M OHLCV
4. **Detect regime** — run `detect_regime()` on 15m data (skip if VOLATILE)
5. **Run CB detection** — use `from combos import get_combo; combo = get_combo("CB"); combo.detect(df_5m)`
6. **Report** — output signal with entry/SL/TP or "no signal"

## Output

Report must include:
- Current time and session status (AM/PM/closed)
- Market regime (TRENDING/RANGING/VOLATILE/NORMAL)
- ATR value and ratio to SMA50
- CB compression status: detected or not, with params
- If signal: direction, trigger price, SL, TP, R:R
- If no signal: reason (ATR out of range, RSI too high, no compression, time window, regime block)
