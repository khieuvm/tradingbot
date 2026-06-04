---
name: trade-postmortem
description: Record and analyze outcomes of VN30F1M intraday trades. Classify results as TRUE_POSITIVE, FALSE_POSITIVE, REGIME_MISMATCH, SESSION_KILL, or TIME_DECAY. Track per-combo rolling performance within sessions (9:00-11:30, 13:00-14:30 UTC+7). Feed results back to signal tracker. No overnight holds — all trades must close same day by 14:28.
---

# VN30F1M Trade Postmortem

Record, classify, and analyze intraday trade outcomes to maintain a feedback loop between live performance and strategy configuration.

## Session Context (Critical)

All trades are INTRADAY only. Key time constraints:
- Trades can only open: 9:15–11:15 (AM) or 13:00–14:15 (PM)
- Force close: 14:28 (no exceptions)
- Lunch break: 11:30–13:00 (close positions or hold through with tight SL)
- Session-killed trades: positions forced closed due to time, not signal

## When to Use

- After closing any trade (whether by SL/TP/trailing/session-force-close)
- End-of-day review after 14:30 (all positions should be flat)
- Weekly performance audit
- When a combo seems to be underperforming
- Before/after parameter changes to compare results
- When user asks "why did this trade fail?"

## Outcome Categories

| Category | Definition | Action |
|----------|-----------|--------|
| TRUE_POSITIVE | Signal correct, trade profitable after cost | None — working as designed |
| FALSE_POSITIVE | Signal fired but wrong direction, SL hit | Review signal conditions |
| REGIME_MISMATCH | Signal valid for different regime than actual | Improve regime detection |
| SESSION_KILL | Position force-closed at 14:28 or 11:30 (lunch) | Check entry time — too late? |
| TIME_DECAY | Didn't reach TP before session time pressure | Add time-of-day filter |
| SLIPPAGE_KILL | Would have been profitable but slippage ate edge | Improve entry timing |

## Workflow

### Step 1: Gather Trade Data

For each closed trade, record:
- Combo name and timeframe
- Entry time, price, direction
- Exit time, price, reason (TP/SL/trailing/session-close/manual)
- Realized PnL (including slippage 0.8 + commission 0.94 = 1.74 pts)
- Regime at entry vs regime at exit
- Max favorable excursion (MFE) and max adverse excursion (MAE)
- **Session context**: AM or PM? Time remaining when entered?
- **Was this a session-force-close?** (14:28 flatten or 11:30 lunch close)

### Step 2: Classify Outcome

Apply classification rules:

```python
def classify_trade(trade):
    if trade.pnl > 0:
        return "TRUE_POSITIVE"
    if trade.exit_reason == "session_close":
        return "SESSION_KILL"  # forced out by 14:28 or 11:30
    if trade.regime_at_entry != trade.regime_at_exit:
        return "REGIME_MISMATCH"
    if trade.mae > trade.sl * 1.5:
        return "FALSE_POSITIVE"  # clearly wrong direction
    if trade.mfe > trade.tp * 0.7 and trade.pnl < 0:
        return "SLIPPAGE_KILL"  # got close but didn't hold
    if trade.mfe < trade.tp * 0.5 and trade.holding_minutes > 60:
        return "TIME_DECAY"  # not enough time to work
    return "FALSE_POSITIVE"
```

### Step 3: Update Signal Tracker

Feed outcome back to `SignalTracker` in `scanner.py`:

```python
signal_tracker.on_close(combo=trade.combo, won=trade.pnl > 0)
```

Rolling metrics update:
- Last 20 trades per combo
- If WR < 40% over 10 trades → auto-disable 24h
- Confidence weight adjusts: 0.5x (cold) to 1.5x (hot)

### Step 4: Pattern Analysis

Look for systematic patterns across classified trades:

**By Combo:**
- Which combos have highest FALSE_POSITIVE rate?
- Which combos are REGIME_MISMATCH prone?

**By Time:**
- Are failures concentrated at session open/close?
- AM vs PM session performance difference?
- Entries after 10:30 AM — do they get SESSION_KILL'd?
- Entries after 14:00 — enough time for TP?
- Day-of-week effects?

**By Market Condition:**
- ATR at entry for winners vs losers
- ADX at entry for winners vs losers

### Step 5: Generate Recommendations

Based on accumulated postmortems:
- Suggest combo grade downgrades (B→C or C→F)
- Suggest regime filter adjustments
- Identify time-of-day patterns for conditional filtering
- Flag combos for full re-validation via backtest-validator

## Output Format

### Single Trade Postmortem

```
TRADE POSTMORTEM — 2026-06-03 10:15
────────────────────────────────────
Combo: D (5m) | Direction: BUY
Entry: 1285.5 @ 09:35 | Exit: 1281.0 @ 10:15
PnL: -4.5 pts (SL hit) | Cost: -1.74 pts | Net: -6.24 pts
Regime: TRENDING → TRENDING (no change)
MFE: +2.1 pts | MAE: -4.5 pts
Classification: FALSE_POSITIVE
────────────────────────────────────
Analysis: Signal fired on ADX=28 but DI+ was declining.
          Price reversed immediately — no follow-through.
Suggestion: Add DI+ momentum check (DI+ > DI+[3 bars ago])
```

### Weekly Summary

```
WEEKLY POSTMORTEM — Week of 2026-06-01
═══════════════════════════════════════
Total trades: 14 | Win: 8 | Loss: 6 | WR: 57%
Net PnL: +18.4 pts | Avg win: +6.2 | Avg loss: -4.1
PF: 1.52

By Outcome:
  TRUE_POSITIVE:    8 (57%)
  FALSE_POSITIVE:   4 (29%)
  REGIME_MISMATCH:  1 (7%)
  TIME_DECAY:       1 (7%)

By Combo:
  D:  5 trades, 4W/1L, +14.2 pts ← performing well
  J:  4 trades, 2W/2L, +2.1 pts  ← neutral
  X:  3 trades, 1W/2L, -3.8 pts  ← watch closely
  G+: 2 trades, 1W/1L, +5.9 pts  ← small sample

Action items:
  1. Combo X: 3 consecutive FP in afternoon → add PM time filter?
  2. REGIME_MISMATCH trade was during ADX transition → tighten threshold?
```

## Key Principles

1. **Every trade gets classified** — no exceptions, even winners
2. **MFE/MAE reveal truth** — a win that nearly hit SL first is fragile
3. **Regime context is mandatory** — without it, can't distinguish skill failure from market shift
4. **Session context is critical** — was there enough time for the trade to work?
5. **20-trade minimum** — don't change parameters on 3 trades of data
6. **Feedback loop must close** — postmortem → signal tracker → scanner
7. **Honest attribution** — don't blame the market, blame the signal
8. **AM vs PM split** — always analyze sessions separately (different dynamics)
9. **SESSION_KILL pattern** — if > 20% of trades are session-killed, entry time filter needed

## Resources

- `scanner.py:SignalTracker` — Rolling performance tracking
- `logs/` — Trade logs with entry/exit details
- `strategy_config.yaml` — Combo grades to update
- `src/portfolio_manager.py` — Trade execution details
