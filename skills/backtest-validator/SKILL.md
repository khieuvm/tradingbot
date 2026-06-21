---
name: backtest-validator
description: Validate VN30F1M intraday trading strategies with walk-forward testing, Monte Carlo permutation, stress testing, and plateau detection. Sessions 9:00-11:30 and 13:00-14:30 UTC+7 — no overnight holds, close by 14:28. Grades strategies A/B/C/F. Use when validating new rules, checking robustness of parameter changes, detecting strategy stagnation, or doing monthly re-validation.
---

# VN30F1M Backtest Validator

Robust validation with stress-testing philosophy: **find strategies that break the least, not ones that profit the most on paper.**

## Simulation Rules (Mandatory)

| Rule | Implementation |
|------|---------------|
| Sessions only | Signals only during 9:00-11:30 / 13:00-14:30 |
| No entry after 14:15 | Skip late signals |
| Force close 14:28 | Close open position at session end |
| Lunch break | No signals 11:30-13:00 |
| Cost | **0.96 pts/trade** (slippage 0.5 + commission 0.46) |
| AM/PM split | Always test AM and PM separately |
| Direction | From breakout bar (not current bar) |

## When to Use

- Before deploying any new rule/filter to live trading
- Testing parameter changes (SL, trail, max_hold, HTF filter threshold)
- After `strategy-optimizer` or `mtf-filter-discovery` produces a candidate
- Monthly re-validation of active strategies
- When strategy performance appears to plateau
- After live trading anomaly

## Core Philosophy

**"OOS is truth. IS performance is a hypothesis, not a result."**

**"Seek plateaus, not peaks."** A strategy that profits with stop loss anywhere from 1.5-3.0 is robust. A strategy that only works at exactly 2.13 is fragile.

A strategy passes when it demonstrates:
1. Stable OOS performance across multiple windows
2. Statistical significance vs random signals (Monte Carlo p < 0.05)
3. No decay trend over rolling windows
4. Robustness under stress (1.5x cost, ±1 bar entry variation)

## Decision Gates

### GATE: Data Availability
- More than 100 days of data available for this TF?
- **YES** -> Use standard rolling walk-forward (60d IS / 30d OOS)
- **NO** (1m/3m only ~28 days) -> Use expanding-window with purged embargo. Report with caution flag.

### GATE: Monte Carlo Significance
- MC p-value < 0.05?
- **YES** -> Edge is statistically real. Continue to stress test.
- **NO, but PF > 1.5** -> Likely sample too small. Need more data before deployment.
- **NO, and PF < 1.5** -> Edge is indistinguishable from random. **Abandon.**

### GATE: Stress Test Survival
- Strategy still profitable at 1.5x cost (1.44 pts) AND ±1 bar entry?
- **YES** -> Robust. Proceed to grading.
- **NO** -> Edge is too thin for real-world execution. Either: improve exit to widen edge, or abandon.

### GATE: Plateau Detection
- Have 3+ consecutive parameter sweeps improved PF by < 0.05?
- **YES** -> **STOP TUNING.** This is a local optimum. Pivot structurally: try different indicator, different exit type, different timeframe, or different strategy entirely. Escalate to `edge-researcher`.
- **NO** -> Continue optimization, there's still alpha to extract.

### GATE: Grading Decision
- OOS Grade A or B?
- **YES** -> Deploy to production. Add to `signal-combiner` portfolio.
- **NO (Grade C)** + plateau detected -> Escalate to `edge-researcher` for structural pivot.
- **NO (Grade C)** + not plateau -> Continue tuning (there's room to improve).
- **NO (Grade F)** -> Abandon this combo.

## Workflow

### Step 1: Define Hypothesis
Be specific and falsifiable:
- "Does macd_cross AM/SELL with ll_5m>=1 filter maintain WR>60% OOS?"
- "Is momentum_trend PM/SELL still profitable at 1.5x cost?"
- NOT: "Is the strategy good?" (too vague)

### Step 2: Walk-Forward Validation

**Standard (>100 days data):** Rolling windows — 60d train / 30d test, step 30d:
```
Window 1: Train [Day 1-60]   -> Test [Day 61-90]
Window 2: Train [Day 31-90]  -> Test [Day 91-120]
Window 3: Train [Day 61-120] -> Test [Day 121-150]
```

**ML Walk-Forward (from strategy_filter.py):**
```python
TEST_DAYS = 60      # OOS test window
STEP_DAYS = 60      # Step between folds
EMBARGO_BARS = 14   # Purge 14 bars between train/test (no look-ahead)
MIN_TRAIN_SIGNALS = 100
MIN_TEST_SIGNALS = 20
```

### Step 3: Stress Testing

**Cost stress:**
```python
COST_STRESS = 1.44  # 1.5x normal (0.96 * 1.5)
# Re-run all trades with stressed cost
stressed_pnl = [raw_pnl - COST_STRESS for raw_pnl in raw_pnls]
# Strategy must still be net-profitable
```

**Entry timing stress:**
- Shift entry by +1 bar (simulates late fill / confirmation wait)
- Shift entry by -1 bar (simulates early fill / aggressive entry)
- Strategy should maintain PF > 1.0 under both shifts

**Worst-window analysis:**
- Find the worst rolling 30-day window
- Max drawdown in that window must be < 50% of total PnL
- If worst window has WR < 35%, investigate — regime-specific failure?

### Step 4: Parameter Sensitivity

Test key parameters at -20%, -10%, baseline, +10%, +20%:
```
SL mult:     [1.6, 1.8, 2.0, 2.2, 2.4] (baseline = 2.0)
Trail pts:   [6.4, 7.2, 8.0, 8.8, 9.6] (baseline = 8.0)
Max hold:    [16, 18, 20, 22, 24]        (baseline = 20)
```

**Pass criterion:** PF degrades < 30% at ±10% → parameter is in a plateau (good).
**Fail criterion:** PF degrades > 50% at ±10% → parameter is fragile (bad).

### Step 5: Monte Carlo Test

100 shuffles of signal timing (preserve frequency, randomize entry positions):
- p-value = % of random runs that beat real PnL
- Require p < 0.05 for production deployment
- If p = 0.06-0.10 and PF > 1.5: borderline — need more data, don't abandon yet

### Step 6: AM/PM Breakdown (Required)

Always report separately:
```
AM: Trades | WR | PF | PnL/d | SL_count | Trail% | Session_exit%
PM: Trades | WR | PF | PnL/d | SL_count | Trail% | Session_exit%
```

Red flag: PM PF > 4x AM PF → PM is compensating for broken AM edge.

### Step 7: Grading

| Grade | OOS_PF | OOS_WR | MC_p | Stress_pass | Decay |
|-------|--------|--------|------|-------------|-------|
| A | > 1.5 | > 55% | < 0.03 | Yes | No |
| B | > 1.3 | > 50% | < 0.05 | Yes | No |
| C | > 1.1 | > 45% | < 0.10 | Partial | Mild |
| F | < 1.0 | any | > 0.10 | No | Yes |

## CLI Commands

```bash
# CB trail sweep backtest (primary reference)
python -m backtest.engine --tf 5m

# Exit parameter grid sweep for any strategy
python -m strategies.optimize_exits --tf 5m

# ML walk-forward validation per strategy
python -m ml.strategy_filter --tf 5m
python -m ml.strategy_filter --tf 1m

# MTF filter discovery (validates HTF filter value)
python -m ml.mtf_indicator_discovery --tf 5m

# Analyze rejected trades (check if filter is too aggressive)
python -m ml.analyze_rejected --tf 5m
```

## Sample Size Requirements

| Confidence | Min Trades | Use Case |
|-----------|-----------|----------|
| Any statistical claim | 30 | Report with caveat |
| Deployment decision | 50 | Can deploy with monitoring |
| High confidence | 100+ | Full confidence deployment |
| Strategy comparison | 200+ | Reliable A/B comparison |

## Red Flags

- OOS PF < 50% of IS PF -> overfit
- MC p > 0.10 -> can't distinguish from random
- PM PF > 4x AM PF -> PM compensating for broken AM
- Parameter sensitivity > 50% at ±10% -> fragile, not in a plateau
- SESSION exits > 60% -> holding too long, max_hold or trail needs tightening
- 3+ sweeps with < 0.05 PF improvement -> **PLATEAU — stop tuning, pivot**
- Worst 30-day window has WR < 30% -> regime-dependent, needs regime filter

## Output Format

```
BACKTEST REPORT — [Strategy] [TF] [Session/Dir] — YYYY-MM-DD
================================================================
Config: [detection rule, filters, exit params]
Data: [N] trading days, cost 0.96/trade

FULL PERIOD:
  [N] trades | WR [X]% | PF [X] | +[X] pts | +[X]/d

STRESS TEST:
  1.5x cost:   PF [X] (pass/fail)
  +1 bar entry: PF [X] (pass/fail)
  Worst 30d:   [X] trades, WR [X]%, PnL [X]

PARAMETER SENSITIVITY:
  SL mult ±10%:  PF [X]-[X] (stable/fragile)
  Trail pts ±10%: PF [X]-[X] (stable/fragile)

OOS (last [N]d): [N] trades | WR [X]% | PF [X] -> Grade [A/B/C/F]
MC p-value: [X] (pass/fail)

Verdict: Deploy / Refine / Escalate / Abandon
```

## Resources

- `backtest/engine.py` — CB backtest engine (trail sweep, simulation)
- `strategies/optimize_exits.py` — Exit parameter grid sweep + simulate_trade_fast
- `ml/strategy_filter.py` — Walk-forward ML validation with purged embargo
- `ml/analyze_rejected.py` — Analyze what the filter misses
- `strategy_config.yaml` — Production parameters
- `skills/strategy-optimizer/SKILL.md` — Parent optimization workflow
- `skills/edge-researcher/SKILL.md` — Escalation target when plateau detected
