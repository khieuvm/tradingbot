---
description: "Research new trading edges for VN30F1M from real data. Analyze patterns, indicators, compression zones, time effects."
argument-hint: "<topic>"
---

# Edge Researcher

Research new edges for VN30F1M intraday trading.

## Arguments

```
$ARGUMENTS
```

**Argument interpretation**:
- If a topic is provided (e.g., `volume spikes`, `lunch break patterns`, `ema pullback`): research that specific edge
- If empty: suggest research topics based on current gaps

## Execution Procedure

1. **Read skill** — load `skills/edge-researcher/SKILL.md`
2. **Frame hypothesis** — convert topic into testable question
3. **Load data** — minimum 120 days VN30F1M OHLCV
4. **Analyze** — measure occurrences, WR, MFE/MAE, EV
5. **Compare to random** — run basic Monte Carlo or chi-squared
6. **Report** — present findings with verdict (promising/marginal/dead)
7. **Next steps** — if promising, outline path to validation

## Key Constraints

- Must include slippage (1.74 pts cost) in all EV calculations
- Must measure frequency (need 1-3 signals/day for utility)
- Must compare to random baseline (is it better than noise?)
- Must check regime interaction (works in all regimes or just one?)
