---
description: "Run CB trail sweep backtest on VN30F1M with AM/PM breakdown, MFE analysis, and exit reason stats."
argument-hint: "[5m|3m|1m|all]"
---

# Backtest Validator

Run CB compression breakout backtest with trail sweep analysis.

## Arguments

```
$ARGUMENTS
```

**Argument interpretation**:
- If a timeframe is provided (e.g., `5m`, `3m`, `1m`): test only that TF
- If `all`: test all timeframes (5m, 3m, 1m)
- If empty: default to 5m (primary)

## Execution Procedure

1. **Read skill** — load `skills/backtest-validator/SKILL.md`
2. **Run backtest** — execute CB trail sweep

```bash
python -m backtest.engine
```

3. **Present results** — show trail sweep comparison, AM/PM split, MFE distribution
4. **Compare to baseline** — CB 5m: WR 67.7%, PF 4.87, +6.34/d
5. **Recommend** — parameter changes if results differ from baseline

## Output

Table format showing:
- TF | Trail@Xpts | Trades | WR | PF | PnL | P/D | SL count
- AM vs PM breakdown per config
- MFE distribution (0-2, 2-4, 4-6, 6-9, 9+)
- Exit reasons (SL, BE, TRAIL, SESSION)
- Verdict: deploy/monitor/investigate
