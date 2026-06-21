---
name: signal-combiner
description: Combine, deduplicate, and prioritize signals from multiple VN30F1M strategies. Detect contradictions, apply conviction scoring, manage single-contract constraint. Use when multiple strategies fire simultaneously or when designing multi-strategy portfolio allocation.
---

# VN30F1M Signal Combiner

Multiple strategies now produce signals (CB, momentum_trend, macd_cross, fibonacci, heikin_ashi, sr_horizontal, market_structure). With a single-contract account (40M, 1 max position), we must pick the best signal, skip contradictions, and avoid entering the same move twice.

## When to Use

- Live trading with multiple strategies active simultaneously
- Designing multi-strategy portfolio allocation weights
- Debugging conflicting signals (Strategy A says BUY, Strategy B says SELL)
- Analyzing signal overlap between strategies
- Determining conviction level for sizing/risk decisions

## Active Strategies

| Strategy | TF | Session/Dir | HTF Filter | WR | Trades/wk |
|----------|-----|------------|------------|-----|-----------|
| CB | 5m | ALL/ALL | regime != VOLATILE | 65% | 8-10 |
| momentum_trend | 1m | PM/SELL | ema8_5m > price | 68% | 0.8 |
| momentum_trend | 1m | PM/BUY | ema21_slope_5m > 0 | 58% | 1.3 |
| macd_cross | 1m | AM/SELL | ll_5m >= 1 | 71% | 0.2 |
| macd_cross | 1m | PM/BUY | ema21_slope_5m > 0 | 78% | 0.3 |
| fibonacci | 1m | PM/SELL | macd_hist_5m > 0.02 | 53% | 0.7 |
| heikin_ashi | 1m | AM/SELL | macd_line_5m < 5.95 | 56% | 0.7 |

## Deduplication Rules

### Rule 1: Temporal Dedup
- Two signals in the SAME direction within 5 bars of each other -> treat as one signal
- Take the FIRST signal (it has the better entry price)
- Log the duplicate for postmortem analysis

### Rule 2: Contradiction Detection
- Two signals on the SAME bar with OPPOSITE directions -> CONTRADICTION
- Action: **skip both** — market is indecisive
- Log with both strategy names for pattern analysis

### Rule 3: Confluence Boost
- Two or more signals on the SAME bar with SAME direction -> HIGH CONVICTION
- Take the signal from the highest-weighted strategy
- Mark as "confluence" in trade log (useful for postmortem)

## Conviction Scoring

```json
{
  "weights": {
    "CB_5m": 0.25,
    "momentum_trend_1m_PM_SELL": 0.20,
    "momentum_trend_1m_PM_BUY": 0.15,
    "macd_cross_1m_PM_BUY": 0.15,
    "macd_cross_1m_AM_SELL": 0.10,
    "fibonacci_1m_PM_SELL": 0.08,
    "heikin_ashi_1m_AM_SELL": 0.07
  },
  "min_conviction": 0.15,
  "dedup_bars": 5,
  "contradiction_window_bars": 3
}
```

Weights are proportional to walk-forward PF. Update monthly based on rolling 60-day performance.

## Decision Gates

### GATE: Contradiction Check
- Is this signal contradicted by another strategy within 3 bars?
- **YES** -> Check 15m trend alignment:
  - If 15m clearly favors one direction (ema_align_15m == +3 or -3) -> take aligned signal only
  - If 15m neutral -> skip both, log as contradiction
- **NO** -> Proceed to position check.

### GATE: Position Check
- Already holding a position?
- **YES** -> Only allow same-direction signal to tighten trail or confirm hold. Do NOT enter opposite direction (close first via normal exit logic).
- **NO** -> Enter the signal.

### GATE: HTF Filter
- Is an HTF filter configured for this strategy/session/direction?
- **YES** -> Apply filter. If signal fails filter -> skip (not a contradiction, just filtered).
- **NO** -> Use raw signal.

### GATE: Session Timing
- Is there enough time remaining in session for this strategy's typical holding period?
- **YES** -> Proceed.
- **NO** -> Skip. (e.g., don't enter a momentum_trend trade at 14:20 if avg hold is 30 min)

## Contradiction Resolution Logic

```
IF Strategy_A == BUY AND Strategy_B == SELL AND |bar_A - bar_B| <= 3:
    # Check 15m alignment
    ema_align = df_15m['ema_align_15m'].iloc[latest_15m_bar]
    IF ema_align >= 2:
        TAKE Strategy_A (BUY aligned with 15m uptrend)
    ELIF ema_align <= -2:
        TAKE Strategy_B (SELL aligned with 15m downtrend)
    ELSE:
        SKIP BOTH (15m indecisive)
```

## Output Schema

```json
{
  "timestamp": "2026-06-14T09:35:00+07:00",
  "session": "AM",
  "signals_raw": [
    {"strategy": "macd_cross", "direction": "SELL", "bar_idx": 1234, "atr": 3.2},
    {"strategy": "heikin_ashi", "direction": "SELL", "bar_idx": 1235, "atr": 3.2}
  ],
  "dedup_result": "confluence",
  "contradictions": [],
  "composite_score": 0.17,
  "htf_filter_pass": true,
  "action": "TAKE",
  "selected_strategy": "macd_cross",
  "reason": "2 strategies agree SELL (confluence), no contradiction, HTF filter passed"
}
```

## Priority Order (Tiebreaker)

When multiple signals pass all gates simultaneously:
1. **Highest walk-forward PF** in last 60 days
2. **CB always wins ties** — it has the longest track record and highest PF
3. **HTF-filtered signal beats unfiltered** — more selective = higher confidence
4. **Earlier bar wins** — signal that fired first gets priority

## Key Principles

1. **Single contract** — 40M account, only 1 position at a time, no pyramiding
2. **Dedup is mandatory** — same move entered twice = 2x cost for same PnL
3. **Contradiction = skip** — never "average" conflicting signals
4. **Weight by PF, not WR** — a strategy with WR 55% PF 3.0 beats WR 70% PF 1.2
5. **Update weights monthly** — strategies degrade; re-weight from rolling 60-day stats
6. **AM and PM have different optimal mixes** — don't combine AM-only and PM-only signals in same scoring
7. **CB is the anchor** — highest confidence, longest validation, always active

## Resources

- `scanner.py` — Main loop (currently CB + ML standalone)
- `combos/cb.py` — CB signal detection (reference implementation)
- `strategies/*.py` — All 7 strategy implementations
- `ml/scan_all_1m_combos.py` — Multi-strategy scan results
- `strategy_config.yaml` — Production parameters and combo weights
- `skills/strategy-optimizer/SKILL.md` — Upstream optimization workflow
- `skills/mtf-filter-discovery/SKILL.md` — HTF filter configuration
