---
name: signal-scanner
description: Real-time VN30F1M intraday signal scanning. Runs every 35 seconds during sessions (9:00-11:30, 13:00-14:30 UTC+7), checks regime filter, evaluates active combos against current price action, reports actionable entries with SL/TP. No overnight holds — all positions must close by 14:28. Use when user asks about current signals, entries, whether to trade now, or wants to check scanner status.
---

# VN30F1M Signal Scanner

Real-time intraday scanning engine for VN30F1M futures. Evaluates validated combo strategies during active sessions only.

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

## Intraday Signal Quality by Time

| Time Window | Quality | Entry Allowed | Notes |
|-------------|---------|---------------|-------|
| 9:00–9:15 | LOW | No | ORB forming, regime unstable |
| 9:15–10:45 | HIGH | Yes, full size | Prime trading window, best volume |
| 10:45–11:15 | MEDIUM | Yes, half size | Trend weakening before lunch |
| 11:15–11:30 | LOW | No | Close positions if not trailing |
| 13:00–13:15 | MEDIUM | Yes, half size | PM open, re-assess regime |
| 13:15–14:00 | MEDIUM-HIGH | Yes, full size | PM prime, lower vol than AM |
| 14:00–14:15 | LOW | Last chance | Only high-confidence signals |
| 14:15–14:28 | FORBIDDEN | No | Flatten only, no entries |

## When to Use

- User asks "any signals right now?" or "should I enter?"
- Checking scanner status and recent signal history
- Debugging why signals are or aren't firing
- Evaluating signal quality and confidence scoring
- During live trading sessions ONLY (9:00-11:30, 13:00-14:30 UTC+7)

## Prerequisites

- Live market data feed via broker API
- `scanner.py` running in background (35s loop)
- Active combos configured in `strategy_config.yaml`
- Regime detection enabled

## Workflow

### Step 1: Check Session Status

Verify we're within trading hours (UTC+7):
- Morning: 9:00–11:30 → entry allowed 9:15–11:15
- Afternoon: 13:00–14:30 → entry allowed 13:00–14:15
- Lunch break 11:30–13:00: market CLOSED, no data
- After 14:15: NO new entries, only manage existing positions
- After 14:28: ALL positions must be closed (force flatten)

### Step 2: Detect Current Regime

Run `detect_regime(df_15m)` to determine active combo set:
- TRENDING → trend combos (A, D, E, F, J, M, O, X, W)
- RANGING → mean-reversion combos (G+, C, K, V, R)
- VOLATILE → reduce or skip
- NORMAL → all combos

### Step 3: Check Signal Tracker

Verify combo isn't auto-disabled by `SignalTracker`:
- Rolling WR < 40% over last 10 trades → disabled 24h
- Apply confidence weight (0.5x–1.5x) based on recent performance

### Step 4: Evaluate Signals

For each active combo, call `generate_combined_signals()`:

```python
from src.signals import generate_combined_signals, COMBO_PRESETS
signals = generate_combined_signals(df, COMBO_PRESETS[combo], tf)
```

### Step 5: Generate Entry Report

For each triggered signal:
- Direction (BUY/SELL)
- Entry price (current + slippage estimate)
- SL level (from strategy_config.yaml, ATR-based)
- TP level (or trailing activation point)
- Combo name and confidence score
- Regime context
- Dedup check (minimum 5 bars between entries)

## Signal Quality Scoring

Each signal gets a composite confidence score:

| Factor | Weight | Source |
|--------|--------|--------|
| Combo grade (A/B/C) | 25% | backtest-validator |
| Regime alignment | 20% | regime-detector |
| Signal tracker confidence | 20% | Rolling performance |
| Time-of-day factor | 20% | AM prime=1.0, PM=0.85, edges=0.5 |
| ATR conditions | 15% | Prefer ATR 3.0–4.0 |

**Time-of-day multipliers:**
- 9:15–10:45 (AM prime): 1.0x
- 10:45–11:15: 0.7x
- 13:00–13:15 (PM open): 0.75x
- 13:15–14:00 (PM prime): 0.85x
- 14:00–14:15 (PM late): 0.5x
- All other times: 0.0x (no entries)

**Thresholds:**
- Score > 0.7 → HIGH confidence, full size
- Score 0.5–0.7 → MEDIUM confidence, half size
- Score < 0.5 → LOW confidence, skip

## Output Format

```
═══ SIGNAL ALERT ═══════════════════════════════════
Time: 09:35:00 | Regime: TRENDING (ADX=28.5)
─────────────────────────────────────────────────────
Combo D | 5m | BUY @ 1285.5
  SL: 1281.0 (-4.5 pts, 1.2x ATR)
  Trail: activates at +5 pts, then 2.0x ATR
  Confidence: 0.78 (HIGH) — Grade B, regime-aligned
  Signal tracker: 6W/3L last 9 trades (67% WR)
─────────────────────────────────────────────────────
Dedup: Last signal was 12 bars ago ✓
Session: 2h25m remaining ✓
Daily PnL: +3.2 pts (under -10 cap) ✓
═══════════════════════════════════════════════════════
```

## Integration

The scanner runs as `scanner.py` with a 35-second loop:
1. Fetch latest candle data
2. Detect regime
3. Loop through COMBO_TF_MAP (active combos only)
4. Check signal tracker for disabled combos
5. Generate signals → notify via Telegram
6. Log to `logs/scanner_YYYY-MM-DD.log`

## Key Principles

1. **Regime first** — never generate signals without regime check
2. **Cost-aware entries** — signal must have expected value > 1.74 pts cost
3. **Dedup is mandatory** — 5 bars minimum between same-combo signals
4. **Time-of-day decay** — reduce confidence approaching session end
5. **Daily loss cap** — stop scanning if daily PnL < -10 pts
6. **Grade-filtered** — only combos graded A or B go live
7. **No entries after 14:15** — not enough time for TP with trailing
8. **Flatten by 14:28** — absolute, no exceptions, no overnight holds
9. **Lunch break reset** — re-detect regime at 13:00, don't carry AM assumptions
10. **AM > PM** — morning session has better edge (higher volume, clearer trends)

## Resources

- `scanner.py` — Main scanner loop implementation
- `src/signals.py` — Signal generation engine
- `strategy_config.yaml` — Active combos, SL/TP, grades
- `src/portfolio_manager.py` — Position entry and management
