---
description: "Review recent VN30F1M trading performance. Classify trades, analyze combo health, generate improvement recommendations."
argument-hint: "[period: today|week|month]"
---

# Performance Review

Analyze recent trading performance with postmortem classification.

## Arguments

```
$ARGUMENTS
```

**Argument interpretation**:
- `today`: review today's trades only
- `week`: review last 7 days (default if empty)
- `month`: review last 30 days
- A specific date (e.g., `2026-06-01`): review that day

## Execution Procedure

1. **Read skill** — load `skills/trade-postmortem/SKILL.md`
2. **Gather data** — read trade logs from `logs/` directory for specified period
3. **Classify trades** — categorize each as TRUE_POS/FALSE_POS/REGIME_MISMATCH/etc.
4. **Aggregate** — compute per-combo and overall statistics
5. **Pattern analysis** — look for time-of-day, regime, or combo-specific patterns
6. **Recommend** — suggest parameter tweaks, combo disables, or regime filter changes

## Output

Must include:
- Summary: total trades, WR, PF, net PnL
- Breakdown by outcome category
- Breakdown by combo (with rolling WR trend)
- Pattern insights (time, regime, ATR at entry)
- Action items (specific, implementable recommendations)
