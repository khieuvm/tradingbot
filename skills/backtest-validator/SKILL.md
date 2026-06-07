---
name: backtest-validator
description: Run walk-forward validation with Monte Carlo permutation testing for VN30F1M intraday CB strategies. Sessions 9:00-11:30 and 13:00-14:30 UTC+7 only — no overnight holds, all simulated trades must close by 14:28. Grades strategies A/B/C/F based on OOS performance, signal randomness test, and decay. Use when validating a new rule, checking robustness of parameter changes, or doing monthly re-validation.
---

# VN30F1M Backtest Validator

Robust walk-forward validation for VN30F1M intraday trading strategies.

## Simulation Rules (Mandatory)

| Rule | Implementation |
|------|---------------|
| Sessions only | Signals only during 9:00–11:30 / 13:00–14:30 |
| No entry after 14:15 | Skip late signals |
| Force close 14:28 | Close open position at session end |
| Lunch break | No signals 11:30–13:00 |
| Cost | **0.96 pts/trade** (slippage 0.5 + commission 0.46) |
| AM/PM split | Always test AM and PM separately — different parameters |
| Direction | From NEXT bar (not current bar) — see CB spec |

## When to Use

- Before deploying any new rule/filter to live trading
- Testing parameter changes (SL multiplier, trail threshold, etc.)
- Validating the adaptive exit rule on new data
- Monthly re-validation of active strategies
- After live trading anomaly — check if something changed

## Core Philosophy

**"OOS is truth. IS performance is a hypothesis, not a result."**

A strategy passes when it demonstrates:
1. Stable OOS performance across multiple windows
2. Statistically significant edge vs random signals (Monte Carlo)
3. No decay trend over rolling windows
4. AM and PM performance aligned (one session shouldn't be carrying the other)

## Workflow

### Step 1: Define what you're testing

Be specific:
- "Does adaptive exit (pre_ratio > 0.8) hold on recent 30 days?"
- "Is CB 5m still valid after regime change in May?"
- NOT: "Is the strategy good?" (too vague)

### Step 2: Walk-Forward Setup

Rolling windows — 60-day IS / 30-day OOS, step 30 days:
```
Window 1: IS [Day 1-60]   → OOS [Day 61-90]
Window 2: IS [Day 31-90]  → OOS [Day 91-120]
Window 3: IS [Day 61-120] → OOS [Day 121-150]
~6 windows from 180 days of 5m data
```

For 3m/1m: only ~24 days available from API — walk-forward not viable, use full-period with caution.

### Step 3: Simulation Requirements

```python
COST = 0.96  # NOT 1.74 — corrected value
# AM params: SL=1.2xATR, trail@5pts/2.0xATR, max_hold=24 bars
# PM params: SL=1.0xATR, trail@4pts/2.5xATR, max_hold=12 bars
# BE trigger: +4 pts MFE when ATR >= 3.5
# Adaptive exit (AM): if pre_ratio > 0.8 → exit@4 (not trail)
# Direction: nxt['close'] > row['close'] → BUY, else SELL
```

### Step 4: Monte Carlo Test

100 shuffles of signal timing (preserve frequency, randomize timing):
- p-value = % of random runs that beat real PnL
- Require p < 0.05 for production deployment

### Step 5: AM/PM Breakdown (Required)

Always report AM and PM separately:
```
  AM: T | WR | PF | PnL/d | SL_count
  PM: T | WR | PF | PnL/d | SL_count
```

If PM is carrying AM (PM PF > 3x AM PF), investigate AM separately.

### Step 6: Grading

| Grade | OOS_PF | OOS_WR | MC_p | Decay |
|-------|--------|--------|------|-------|
| A | > 1.5 | > 50% | < 0.03 | No |
| B | > 1.3 | > 45% | < 0.05 | No |
| C | > 1.1 | > 40% | < 0.10 | Mild |
| F | < 1.0 | any | > 0.10 | Yes |

**Known baselines:**
- CB 5m (full 129d): IS PF ~4.5, OOS 30d: PF 1.72 → Grade B (solid)
- CB 5m with adaptive exit: +0.29/d improvement, validated on same data

## Output Format

```
BACKTEST REPORT — CB 5m — 2026-06-04
══════════════════════════════════════
Config: 3-bar compression, ATR 2.5-4.5, adaptive exit pre_ratio>0.8
Data: 129 trading days, cost 0.96/trade

FULL PERIOD:
  223 trades | WR 67.7% | PF 4.87 | +818.2 pts | +6.34/d

AM vs PM:
  AM: 142 | WR 66.2% | PF 4.20 | +453.3 | adapt_exits=21
  PM:  81 | WR 70.4% | PF 6.23 | +364.8

ADAPTIVE EXIT:
  Targets: 31 AM trades (pre_ratio>0.8)
  Exited@4pts: 21 | Avg PnL: +5.16 | Avg MFE: 6.5
  Missed 4pts (went to SL): 10

OOS (last 30d): 30 trades | WR 50.0% | PF 1.72 → Grade B
MC p-value: < 0.05 ✓

Verdict: Deploy. Apply adaptive exit rule to live sim_day.py.
```

## Red Flags

- OOS PF < 50% of IS PF → likely overfit
- MC p > 0.10 → can't distinguish from random
- PM PF > 4x AM PF → PM is compensating for broken AM edge
- MFE 0-2 WR > 30% → signal is firing on wrong direction too often
- Adapt exits hitting < 40% of targeted trades → threshold may be too high
- **SESSION exits > 80%** → trail never activates, holding too long without direction

## Resources

- `bt_trail_sweep.py` — CB trail activation parameter sweep (primary backtest reference)
- `bt_adaptive_3tf.py` — Adaptive exit backtest on 1m/3m/5m
- `debug_3m.py` / `debug_1m.py` — TF-specific deep analysis
- `strategy_config.yaml` — CB risk params and config
