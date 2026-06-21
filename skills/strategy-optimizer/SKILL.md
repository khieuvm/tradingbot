---
name: strategy-optimizer
description: Optimize VN30F1M multi-strategy portfolio. Run strategy combos, find best HTF filters, tune exit params, compare configs. Covers the full pipeline from raw signal generation through validated production config. Use when a new strategy is discovered, monthly re-optimization is needed, trade count is too low, or after scan_all_1m_combos reveals new combos.
---

# VN30F1M Strategy Optimizer

Full pipeline: generate signals on LTF -> apply HTF filter -> compute walk-forward stats -> compare configs -> deploy.

## When to Use

- New strategy discovered (from edge-researcher or external source)
- Monthly re-optimization of existing portfolio
- Trade count too low — need to find more profitable combos
- After `scan_all_1m_combos` reveals new tradeable combos
- When comparing session/direction variants for a strategy
- When tuning exit parameters per-strategy

## Strategy Landscape

| Strategy | Module | Best TF | Best Session/Dir | Notes |
|----------|--------|---------|-----------------|-------|
| market_structure | strategies.market_structure | 5m | AM/BUY | O(n^2) detect — slow on 1m |
| fibonacci | strategies.fibonacci | 5m, 1m | AM/SELL, PM/SELL | Rescuable with HTF filter |
| sr_horizontal | strategies.sr_horizontal | 5m | AM/SELL | Works in low-ADX (ranging) |
| heikin_ashi | strategies.heikin_ashi | 5m, 1m | AM/SELL | Needs short max_hold |
| reversal_patterns | strategies.reversal_patterns | 5m | AM/SELL | High signal count, low WR |
| momentum_trend | strategies.momentum_trend | 1m | PM/SELL, PM/BUY | Best 1m strategy |
| macd_cross | strategies.macd_cross | 1m | AM/SELL, PM/BUY | Best with ll_5m filter |

## Decision Gates

### GATE 1: Sample Size
- Combo has >= 50 trades over test period?
- **NO** -> Extend data window, merge AM+PM, or merge BUY+SELL to get more samples. If still < 30, abandon.
- **YES** -> Proceed to baseline test.

### GATE 2: Baseline Profitability
- Is baseline PF > 1.0 (without any HTF filter)?
- **NO** -> Combo is unprofitable at entry level. BUT: test HTF filter anyway (can rescue). If filtered PF > 1.3 with >= 30 trades -> proceed.
- **YES** -> Test HTF filter for improvement.

### GATE 3: HTF Filter Value
- Does HTF filter improve PF by > 0.3 compared to no filter?
- **NO** -> Filter is noise. Use unfiltered version if profitable, or abandon.
- **YES** -> Adopt filter. Proceed to robustness check.

### GATE 4: Robustness
- Is WR sensitivity to threshold < 5% when varying threshold by +/- 10%?
- **NO** -> Filter is fragile. Widen threshold or try different indicator.
- **YES** -> Robust. Deploy to production.

## CLI Commands

```bash
# Step 1: Scan all 1m strategy combos with 5m filter (6 strategies x 9 combos)
python -m ml.scan_all_1m_combos

# Step 2: Find best 15m filter for 5m strategies
python -m ml.mtf_indicator_discovery --tf 5m

# Step 3: Find best 5m filter for 1m strategies
python -m ml.mtf_1m_with_5m

# Step 4: ML walk-forward meta-labeling per strategy
python -m ml.strategy_filter --tf 5m
python -m ml.strategy_filter --tf 1m

# Step 5: Exit parameter grid sweep
python -m strategies.optimize_exits --tf 5m
python -m strategies.optimize_exits --tf 1m

# Step 6: Analyze rejected trades (why ML kills good trades)
python -m ml.analyze_rejected --tf 5m
```

## Workflow

### Phase 1: Discovery (find ALL profitable combos)
1. Run `ml.scan_all_1m_combos` — tests 54 combinations on 1m with best 5m filter
2. Run `ml.mtf_indicator_discovery --tf 5m` — finds 15m filters for 5m strategies
3. Collect results: which combos have filtered WR > 50% AND PnL > 0?

### Phase 2: Validation (stress-test each combo)
1. For each promising combo, run `ml.strategy_filter` — walk-forward with embargo
2. Check OOS performance: PF must be > 1.2 on unseen data
3. Test at 1.5x cost (1.44 pts): still profitable? → robust
4. Test worst 30-day rolling window: max drawdown acceptable?

### Phase 3: Exit Tuning (per-strategy optimization)
1. Run `strategies.optimize_exits` for each validated combo
2. Compare default exits vs optimized exits (side-by-side)
3. Apply entry filters (ADX, ATR thresholds) based on loss analysis
4. GATE: improved PF > 10%? YES → adopt custom exits. NO → keep defaults.

### Phase 4: Portfolio Integration
1. Combine all validated combos into unified portfolio
2. Check for signal overlap (same time, same direction from multiple strategies)
3. Estimate total trades/day and PnL/day
4. Update `strategy_config.yaml` with production params

## Output

Results saved to `ml/reports/`:
- `all_1m_combos_YYYY-MM-DD.json` — All combo scan results
- `mtf_discovery_5m_YYYY-MM-DD.json` — 15m filter results for 5m
- `strategy_filter_5m_YYYY-MM-DD.json` — ML walk-forward results
- `strategy_filter_1m_YYYY-MM-DD.json` — ML walk-forward results

## Key Principles

1. **Maximize P/D (PnL per day), not WR** — a 50% WR strategy with big winners beats 70% WR with tiny wins
2. **Minimum 50 trades** for any deployment decision
3. **Always test baseline first** — know what you're improving against
4. **AM and PM are separate markets** — always analyze and optimize them independently
5. **Cost 0.96 pts/trade** — every strategy must survive this drag
6. **No overnight holds** — all positions close by 14:28
7. **Seek plateaus, not peaks** — if performance only works at one exact parameter value, it's fragile

## Resources

- `ml/scan_all_1m_combos.py` — All-combos scanner (1m + 5m filter)
- `ml/mtf_indicator_discovery.py` — HTF indicator discovery (5m → 15m)
- `ml/mtf_1m_with_5m.py` — HTF indicator discovery (1m → 5m)
- `ml/strategy_filter.py` — Walk-forward ML meta-labeling
- `ml/analyze_rejected.py` — Rejected trade analysis
- `strategies/optimize_exits.py` — Exit parameter grid sweep
- `strategies/backtest_new.py` — Data loading, baseline backtesting
- `strategy_config.yaml` — Production parameters
