---
description: "Run robust walk-forward backtest for VN30F1M combo strategies with Monte Carlo validation and grading."
argument-hint: "<combo_name|all>"
---

# Backtest Validator

Run walk-forward validation with Monte Carlo testing.

## Arguments

```
$ARGUMENTS
```

**Argument interpretation**:
- If a combo name is provided (e.g., `D`, `J`): validate that specific combo
- If `all`: validate all combos in COMBO_TF_MAP
- If empty: ask user which combo to validate

## Execution Procedure

1. **Read skill** — load `skills/backtest-validator/SKILL.md`
2. **Parse arguments** — determine target combo(s)
3. **Run bt_robust.py** — execute walk-forward validation

```bash
python bt_robust.py --combo $ARGUMENTS --tf 5m
```

4. **Present results** — show IS vs OOS metrics, MC p-value, decay analysis, grade
5. **Recommend** — suggest deploy/disable actions based on grades

## Output

Table format showing:
- Combo | TF | IS_WR | IS_PF | OOS_WR | OOS_PF | MC_pval | Decay | Grade
- Recommendation for each combo (deploy/monitor/disable)
- If grade F: explain why and suggest alternatives
