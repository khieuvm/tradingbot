---
name: signal-scanner
description: Real-time VN30F1M intraday signal scanning using CB (Compression Breakout) strategy on 5m timeframe. Runs every 60 seconds during sessions (9:00-11:30, 13:00-14:30 UTC+7), checks regime filter, evaluates CB compression patterns, reports actionable entries with SL/TP. No overnight holds — all positions must close by 14:28. Use when user asks about current signals, entries, whether to trade now, or wants to check scanner status.
---

# VN30F1M Signal Scanner

Real-time intraday scanning engine for VN30F1M futures. Single strategy: CB (Compression Breakout) on 5m timeframe.

## Session Rules (Mandatory)

| Rule | Detail |
|------|--------|
| Morning session | 9:00–11:30 (UTC+7) |
| Afternoon session | 13:00–14:30 (UTC+7) |
| Lunch break | 11:30–13:00 — NO scanning, market closed |
| Last new entry | 14:15 — no entries after this time |
| Force flatten | 14:28 — close ALL open positions |
| No overnight | NEVER hold past session close |
| Daily loss cap | Stop all trading if PnL < -10 pts today |

## CB Strategy Detection

Signal: `scanner.py:detect_cb_compression()` on 5m bars

| Filter | Condition |
|--------|-----------|
| Compression | max range of 3 bars before signal < 0.7 × ATR(14) |
| ATR range | 2.5 ≤ ATR ≤ 4.5 |
| RSI filter | RSI(14) < 70 |
| Time (AM) | 09:15–10:45 |
| Time (PM) | 13:15–14:15 |
| Dedup | 25 min (5 bars) between signals |
| Regime | Skip in VOLATILE (ATR ratio > 1.5×SMA50) |
| Direction | BUY = signal bar high + 0.1, SELL = low - 0.1 |
| Confirmation | 2-scan expiry if trigger not hit |

## When to Use

- User asks "any signals right now?" or "should I enter?"
- Checking scanner status and recent signal history
- Debugging why signals are or aren't firing
- During live trading sessions ONLY (9:00-11:30, 13:00-14:30 UTC+7)

## Prerequisites

- Live market data feed via vnstock/KBS API
- `scanner.py` running (10s position update / 60s CB scan)
- `strategy_config.yaml` has CB configured
- Regime detection enabled

## Workflow

### Step 1: Check Session Status

Verify we're within trading hours (UTC+7):
- Morning: 9:00–11:30 → entry allowed 9:15–10:45
- Afternoon: 13:00–14:30 → entry allowed 13:15–14:15
- Lunch break 11:30–13:00: market CLOSED
- After 14:15: NO new entries
- After 14:28: ALL positions must be closed

### Step 2: Detect Current Regime

Skip CB signals when VOLATILE (5m ATR_ratio > 1.5×SMA50).

### Step 3: Evaluate CB Compression

`detect_cb_compression(df_5m)` checks:
1. 3-bar compression (range < 0.7×ATR)
2. ATR in valid range (2.5–4.5)
3. RSI < 70
4. Within allowed time window
5. Dedup: 25 min since last signal

### Step 4: Generate Entry

If compression detected:
- BUY trigger = signal bar high + 0.1
- SELL trigger = signal bar low - 0.1
- Next-bar confirmation: 2 scans to hit trigger, else expire

## Output Format

```
═══ CB SIGNAL ════════════════════════════════════════
Time: 09:35:00 | Regime: NORMAL (ATR=3.2)
──────────────────────────────────────────────────────
CB | 5m | BUY @ 1285.5 (high+0.1)
  SL: 1281.7 (-3.8 pts, 1.2×ATR)
  Trail: activates at +5 pts, then 2.0×ATR
  Compression: 3-bar range = 2.1 (< 0.7×3.2 = 2.24) ✓
  Dedup: Last signal 32 bars ago ✓
  Session: 1h55m remaining ✓
══════════════════════════════════════════════════════
```

## Key Principles

1. **Single strategy** — CB compression breakout only, 5m timeframe
2. **Regime-aware** — skip in VOLATILE
3. **Cost = 0.96 pts/trade** — slippage 0.5 + commission 0.46
4. **Dedup mandatory** — 25 min (5 bars) minimum between signals
5. **Time-of-day** — AM 09:15–10:45, PM 13:15–14:15 only
6. **No entries after 14:15** — not enough time for trailing
7. **Flatten by 14:28** — absolute, no exceptions
8. **1 contract max** — 40M account, single position

## Resources

- `scanner.py` — Main scanner loop + CB detection
- `strategy_config.yaml` — CB risk params (SL/TP, dedup)
- `src/portfolio_manager.py` — Position entry and exit management
- `bt_trail_sweep.py` — CB backtest reference (do not modify)
