---
description: "Research new trading edges for VN30F1M from real data. Analyze patterns, indicators, compression zones, time effects, MTF filters, multi-strategy combos."
argument-hint: "<topic: mtf|multi-strategy|htf-filter|all-combos|exit-tuning|...>"
---

# Edge Researcher

Research new edges for VN30F1M intraday trading.

## Arguments

```
$ARGUMENTS
```

**Argument interpretation**:
- If a topic is provided (e.g., `volume spikes`, `lunch break patterns`, `NR7`): research that specific edge
- `mtf` or `htf-filter`: run MTF filter discovery → see `skills/mtf-filter-discovery/SKILL.md`
- `multi-strategy` or `all-combos`: scan all strategy combos → see `skills/strategy-optimizer/SKILL.md`
- `exit-tuning`: optimize exit params for a strategy → `python -m strategies.optimize_exits`
- `signal-combine`: analyze signal overlap/contradiction → see `skills/signal-combiner/SKILL.md`
- If empty: suggest research topics based on current gaps

## Execution Procedure

1. **Read skill** — load `skills/edge-researcher/SKILL.md`
2. **Frame hypothesis** — convert topic into testable question
3. **Load data** — use `src/data_fetcher.py` for VN30F1M OHLCV
4. **Analyze** — write script in `research/`, measure occurrences, WR, MFE/MAE, EV
5. **Compare to random** — run basic Monte Carlo or chi-squared
6. **Report** — present findings with verdict (promising/marginal/dead)
7. **Next steps** — if promising, outline path to new combo in `combos/`

## Key Constraints

- Cost: **0.96 pts/trade** (slippage 0.5 + commission 0.46) — NOT 1.74
- Must measure frequency (need 1-3 signals/day for utility)
- Must compare to random baseline (is it better than noise?)
- Must check regime interaction (works in all regimes or just one?)
- New combos implement `combos/base.py:BaseCombo` interface
